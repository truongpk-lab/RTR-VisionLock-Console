"""Export the UETrack network (FastiTPN encoder + center decoder) to ONNX.

Why: on Jetson the per-frame tracker inference is the dominant cost. Exporting the
heavy forward (forward_encoder + forward_decoder) to ONNX lets it run via the
ONNX Runtime TensorRT/CUDA Execution Provider (engine built & cached per device),
which is 2-4x faster than PyTorch eager. All the tracker's preprocessing
(crop/resize) and postprocessing (hann window, cal_bbox, map-back) stay in Python
and are reused unchanged by the `uetrack_onnx` backbone.

The exported graph maps exactly to the two heavy calls in
``lib/test/tracker/uetrack.py::track``:

    enc, _, _ = network.forward_encoder([template], [search], [template_anno], None, None)
    out       = network.forward_decoder(enc)
    -> (out["score_map"], out["size_map"], out["offset_map"])

Inputs are fixed-shape (the only way transformer trackers export cleanly). Channel
count follows cfg.DATA.MULTI_MODAL_VISION: the tracker duplicates RGB into 6 ch when
it is on, so we trace with whatever the patch-embed conv actually expects.

Run (on a machine with the repo + torch + the checkpoint):

    cd backend && PYTHONPATH= .venv/bin/python tools/export_uetrack_onnx.py \
        --repo /home/pk/code/UETrack \
        --checkpoint models/uetrack_tiny_rgb.pth \
        --param-name uetrack_tiny_rgb \
        --out models/uetrack_tiny_rgb.onnx

NOTE: this is the risky step. FastiTPN uses MoE / custom attention; if an op fails
to export, fix it here (replace the op, pin a shape, bump the opset) -- the runtime
backbone stays on PyTorch until a valid ONNX exists, so nothing regresses.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch


class _ExportNet(torch.nn.Module):
    """Wrap the two heavy forwards into one fixed-signature graph for ONNX."""

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, template, search, template_anno):
        enc, _, _ = self.net.forward_encoder([template], [search], [template_anno], None, None)
        out = self.net.forward_decoder(enc)
        return out["score_map"], out["size_map"], out["offset_map"]


def _infer_in_channels(net: torch.nn.Module, default: int) -> int:
    """Read the patch-embed conv in_channels so we trace with the right shape."""
    for module in net.modules():
        if isinstance(module, torch.nn.Conv2d):
            return int(module.in_channels)
    return default


def _build_net(repo: str, checkpoint: Path, param_name: str):
    for candidate in (repo, str(Path(repo) / "lib")):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    from lib.models.uetrack import build_uetrack_inference  # noqa: E402

    params = importlib.import_module("lib.test.parameter.uetrack").parameters(param_name)
    cfg = params.cfg
    net = build_uetrack_inference(cfg)
    state = torch.load(str(checkpoint), map_location="cpu")
    net.load_state_dict(state["net"] if "net" in state else state, strict=False)
    net.eval()
    return net, params


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="path to the cloned UETrack repo")
    ap.add_argument("--checkpoint", required=True, help="path to the .pth checkpoint")
    ap.add_argument("--param-name", default="uetrack_tiny_rgb")
    ap.add_argument("--out", required=True, help="output .onnx path")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = Path(__file__).resolve().parents[1] / checkpoint
    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out
    out.parent.mkdir(parents=True, exist_ok=True)

    net, params = _build_net(args.repo, checkpoint, args.param_name)
    cfg = params.cfg
    tsz = int(cfg.TEST.TEMPLATE_SIZE)
    ssz = int(cfg.TEST.SEARCH_SIZE)
    in_ch = _infer_in_channels(net, default=6 if cfg.DATA.MULTI_MODAL_VISION else 3)
    print(f"[export] template {in_ch}x{tsz}x{tsz}  search {in_ch}x{ssz}x{ssz}  opset {args.opset}")

    template = torch.randn(1, in_ch, tsz, tsz)
    search = torch.randn(1, in_ch, ssz, ssz)
    template_anno = torch.rand(1, 4)
    wrapper = _ExportNet(net).eval()

    with torch.no_grad():
        ref = wrapper(template, search, template_anno)

    torch.onnx.export(
        wrapper,
        (template, search, template_anno),
        str(out),
        input_names=["template", "search", "template_anno"],
        output_names=["score_map", "size_map", "offset_map"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[export] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # Parity: run the ONNX (CPU EP is fine for a correctness check) vs PyTorch.
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        onx = sess.run(
            None,
            {
                "template": template.numpy(),
                "search": search.numpy(),
                "template_anno": template_anno.numpy(),
            },
        )
        for name, r, o in zip(("score_map", "size_map", "offset_map"), ref, onx):
            r = r.detach().cpu().numpy()
            diff = float(np.abs(r - o).max())
            print(f"[parity] {name}: max|Δ| = {diff:.3e}  shape={tuple(o.shape)}")
    except Exception as exc:  # pragma: no cover - parity is best-effort
        print(f"[parity] skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
