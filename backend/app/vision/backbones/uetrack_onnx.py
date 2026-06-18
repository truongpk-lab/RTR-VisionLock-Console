"""UETrack network spliced onto ONNX Runtime (TensorRT / CUDA EP) for fast inference.

The repo tracker's preprocessing (crop/resize) and postprocessing (hann window,
``decoder.cal_bbox``, map-back) are reused unchanged; only the two heavy calls
``forward_encoder`` + ``forward_decoder`` are replaced by ONE ONNX session. Build the
ONNX once with ``tools/export_uetrack_onnx.py``; the TensorRT EP then builds & caches
a per-device engine on first run.

Gating: missing ``onnxruntime`` or ``.onnx`` file (or no repo/torch for the pre/post)
flips ``available`` off, so :class:`ManagedTracker` falls back to OpenCV. This keeps
``uetrack_onnx`` a pure opt-in accelerator that never regresses a box without it.

I/O contract (see the export script): inputs ``template``/``search``/``template_anno``;
outputs ``score_map``/``size_map``/``offset_map``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional deployment dependency
    ort = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from .torch_tracker import UETrackBackbone


class _OnnxNetworkProxy:
    """Stand-in for ``UETrack.network``: encoder+decoder via ONNX, cal_bbox in torch.

    The repo's ``UETrack.track`` calls ``forward_encoder`` then ``forward_decoder``
    then ``decoder.cal_bbox``. We run one ONNX session for the first two (the heavy
    transformer) and keep ``cal_bbox`` (cheap tensor math) on the real decoder.
    """

    def __init__(self, network: Any, session: Any, device: Any) -> None:
        self._network = network
        self._session = session
        self._device = device
        self.decoder = network.decoder  # cal_bbox lives here
        self._pending: dict | None = None

    def __getattr__(self, name: str):
        # Only reached for attributes we do not define -> delegate to the real net.
        if name == "_network":
            raise AttributeError(name)
        return getattr(self._network, name)

    def forward_encoder(self, template_list, search_list, template_anno_list, text_src, task_index):
        feeds = {
            "template": template_list[0].detach().cpu().numpy(),
            "search": search_list[0].detach().cpu().numpy(),
            "template_anno": template_anno_list[0].detach().cpu().numpy(),
        }
        score_map, size_map, offset_map = self._session.run(None, feeds)
        self._pending = {
            "score_map": torch.from_numpy(score_map).to(self._device),
            "size_map": torch.from_numpy(size_map).to(self._device),
            "offset_map": torch.from_numpy(offset_map).to(self._device),
        }
        # The tracker ignores the feature lists on the inference path.
        return None, None, None

    def forward_decoder(self, feature, gt_score_map=None):
        return self._pending


class UETrackOnnxBackbone(UETrackBackbone):
    """UETrack with the heavy forward replaced by an ONNX Runtime session."""

    source = "uetrack_onnx"
    config_key = "tracker_uetrack_onnx"

    def _build(self, block: dict[str, Any], repo_path: str, checkpoint: Path):
        if ort is None:
            raise RuntimeError("onnxruntime not installed")
        onnx_path = self._resolve_path(block.get("onnx_path", ""))
        if onnx_path is None:
            raise RuntimeError("onnx_path not found (run tools/export_uetrack_onnx.py)")

        # Build the repo tracker (loads torch weights) purely for its pre/post code.
        tracker = super()._build(block, repo_path, checkpoint)
        session = self._make_session(block, onnx_path)
        real = tracker.network
        device = next(real.decoder.parameters()).device
        # The heavy FastiTPN encoder now runs in ONNX; free the torch copy so we do
        # not pay for it twice (matters on the 8GB Jetson). cal_bbox uses the decoder.
        try:
            real.encoder = None
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best-effort memory trim
            pass
        tracker.network = _OnnxNetworkProxy(real, session, device)
        param_name = str(block.get("param_name", self.default_param_name))
        self.kind = f"{self.source}:{param_name}"
        return tracker

    def _make_session(self, block: dict[str, Any], onnx_path: Path):
        requested = block.get("providers") or [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        available = set(ort.get_available_providers())
        cache_dir = Path(__file__).resolve().parents[3] / str(block.get("engine_cache_dir", "models/trt_cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        providers: list = []
        provider_options: list[dict] = []
        for name in requested:
            if name not in available:
                continue
            if name == "TensorrtExecutionProvider":
                providers.append(name)
                provider_options.append(
                    {
                        "trt_fp16_enable": bool(block.get("fp16", True)),
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(cache_dir),
                    }
                )
            else:
                providers.append(name)
                provider_options.append({})
        if not providers:  # nothing requested was available -> CPU
            providers, provider_options = ["CPUExecutionProvider"], [{}]
        return ort.InferenceSession(str(onnx_path), providers=providers, provider_options=provider_options)
