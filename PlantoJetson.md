# Kế hoạch tối ưu VisionLock đạt ≥60 FPS trên Jetson Orin Nano 8GB

## Tiến độ tổng quan

| Phase | Nội dung | Máy | Trạng thái |
|---|---|---|---|
| P0 | Đo lường nền (per-stage timers) + uncap FPS benchmark | 4060 | ✅ (test 11✓, full 112✓) |
| P1 | Bỏ EVPTrack + Tier B → kiến trúc 2 tầng | 4060 | ✅ (pytest 109✓, build✓) |
| P2 | Tăng tốc tracker: ONNX + TensorRT-EP | 4060 | ✅ export+parity (IoU 1.0); TRT speedup chờ on-device |
| P3 | ReID theo cadence (cắt việc mỗi frame) | 4060 | ✅ (reid_interval+cache; full suite 110✓) |
| P4 | HW codec & offload CPU (device-agnostic) | 4060 | ☐ |
| P5 | Tinh chỉnh trên Jetson | `[ON-DEVICE]` | ☐ |

---

## Context — Vì sao làm việc này

Pipeline hiện chạy ổn trên RTX 4060 nhưng đích triển khai là **Jetson Orin Nano 8GB** (FP16 ~10 TFLOPs ≈ 1/12 của 4060, băng thông RAM 68 GB/s ≈ 1/4, RAM chia sẻ CPU+GPU). Mục tiêu **≥60 FPS** (ngân sách **16.6 ms/frame**) đòi hỏi: đo đúng bottleneck, dọn kiến trúc cho gọn, tăng tốc model bằng TensorRT, và cắt việc thừa mỗi frame.

**Quyết định đã chốt:**
1. Tăng tốc tracker bằng **ONNX Runtime + TensorRT Execution Provider** (engine tự build & cache theo từng máy → chạy được cả 4060 lẫn Orin). TRT engine native để dành sau nếu cần ép thêm.
2. **Bỏ hẳn EVPTrack và Tier B (refind)** → kiến trúc **2 tầng**: Tier A UETrack local + Tier C global re-acquire.
3. Phát triển & benchmark trên **RTX 4060 (máy dev)** trước, với **video import**. Các bước riêng Jetson được đánh dấu `[ON-DEVICE]`.

## Nguyên tắc thiết kế (bám CLAUDE.md)

- **Đo trước, tối ưu sau.** Phase 0 dựng đo lường + benchmark harness trước mọi tối ưu.
- **Thay đổi phẫu thuật, cộng thêm & gating bằng config.** Mỗi tối ưu bật/tắt qua config (`uncap_fps`, `reid_interval`, backbone name…), backbone mới đặt *cạnh* backbone cũ trong registry → không xung đột, dễ revert, dễ mở rộng.
- **Đơn giản nhất giải quyết được vấn đề.** Không thêm tính năng ngoài yêu cầu.
- **Mỗi bước có tiêu chí nghiệm thu rõ ràng** (số đo hoặc test xanh) trước khi sang bước sau.

**Lệnh chạy test backend (tránh plugin ROS):**
```bash
cd backend && PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

**Tiêu chí thành công cuối cùng:**
- `[ON-DEVICE]` Orin Nano: FPS lock-tracking ≥ 60 (đo bằng `metrics.fps` với video/feed).
- Không hồi quy độ chính xác: IoU bbox của tracker TRT so PyTorch trên 1 clip mẫu ≥ 0.9 trung bình; re-acquire vẫn re-lock đúng.
- Toàn bộ test backend xanh; frontend build (`tsc`) sạch.

---

## Phase 0 — Đo lường nền & benchmark harness  *(máy 4060)*

Mục tiêu: thấy được bottleneck thật và chạy video "hết tốc lực" để đo throughput.

### Bước 0.1 — Per-stage timers
- **Sửa:** `backend/app/core/metrics.py` — thêm trường vào `MetricState`: `tracker_ms`, `reid_ms`, `flow_ms`, `encode_ms` (+ vào `to_dict()`).
- **Sửa:** `backend/app/core/session.py` `_update_tracking` — bọc `time.perf_counter()` quanh `camera_motion.estimate` (→`flow_ms`), `tracker.track` (→`tracker_ms`), `memory.score` (→`reid_ms`); quanh `_encode_frame` (→`encode_ms`).
- **Deliverable:** UI/`/api/status` hiện 4 con số ms/stage.
- **Test:** `pytest test_metrics.py` xanh (mở rộng assert 4 trường mới); chạy app, xác nhận số liệu cập nhật.

### Bước 0.2 — Mở uncap FPS cho video benchmark
- **Sửa:** `backend/app/vision/camera.py` `open()` — thêm cờ config `camera.uncap_fps` (mặc định `false`, giữ nguyên hành vi portable). Khi `true`: với file cũng đặt `self._frame_interval = 0.0` → reader thread không sleep, loop chạy bằng tốc độ xử lý = throughput thật.
- **Sửa:** `backend/config/default.yaml` (camera) thêm `uncap_fps: false`; bật `true` trong `local.yaml` lúc benchmark.
- **Deliverable:** import video → FPS leo vượt FPS gốc của clip (đo trần pipeline).
- **Test:** mở rộng `backend/tests/test_video_import.py` thêm case `uncap_fps=true` ⇒ `_frame_interval == 0.0` cho file.

### Bước 0.3 — Ghi baseline
- **Deliverable (không sửa code):** chạy 1 clip mẫu trên 4060, ghi vào bảng dưới: `fps`, `tracker_ms`, `reid_ms`, `flow_ms`, `encode_ms`. Mốc so sánh cho mọi phase sau.

| Mốc đo | fps | tracker_ms | reid_ms | flow_ms | encode_ms | Ghi chú |
|---|---|---|---|---|---|---|
| Baseline (PyTorch, ReID mỗi frame) | _ | _ | _ | _ | _ | chờ đo |

**🚦 Cổng nghiệm thu Phase 0:** có số đo per-stage + chạy video uncapped + bảng baseline.

---

## Phase 1 — Dọn kiến trúc: bỏ EVPTrack + Tier B → 2 tầng  *(máy 4060)*

Mục tiêu: kiến trúc gọn, đúng SOTA (local tracker + global re-detector), trước khi gắn TRT.

> **Cơ sở SOTA:** LTMU (CVPR'20, *meta-updater* điều phối local-track vs re-detect), GlobalTrack/SAMURAI cho long-term = **local tracker + global re-detection**. Transformer tracker một luồng (OSTrack/UETrack) tự re-search trong search-region mỗi frame nên **Tier B (OpenCV CSRT refind) là dư thừa**. 2 tầng: confidence cao→UETrack; mất hẳn→global re-acquire.

### Bước 1.1 — Refactor policy 2 tầng
- **Sửa:** `backend/app/vision/policy.py` — bỏ `TrackMode.REFIND`; `PolicyDecision` bỏ `seed_refind`, `reinit_normal` (giữ `mode`, `state`, `reacquire`). Logic mới: STABLE (≥stable) / UNCERTAIN (uncertain..stable, vẫn chạy UETrack) / sau `lost_frames` hoặc `identity_lost_frames` → LOST + `reacquire=True`. Bỏ `refind_after`.
- **Test:** viết lại `backend/tests/test_policy.py` cho 2 tầng (bỏ các assert `REFIND`/`seed_refind`/`reinit_normal`; giữ case escalate-to-LOST, reset loss counter, identity-lost streak, configure giữ counter).

### Bước 1.2 — Gỡ refind tracker khỏi session
- **Sửa:** `backend/app/core/session.py` — bỏ `refind_tracker`, `_new_refind_tracker()`, nhánh refind trong `_apply_mode_transition`, `_widen_search_bbox`; chọn tracker rút gọn về `self.tracker`; bỏ key `refind_backbone` trong snapshot + docstring liên quan.

### Bước 1.3 — Xóa EVPTrack
- **Sửa:** `backend/app/vision/backbones/torch_tracker.py` xóa class `EVPTrackBackbone`; `backend/app/vision/backbones/__init__.py` bỏ import + entry `"evptrack"`.
- **Sửa config:** `backend/config/default.yaml` xóa block `tracker_evptrack`, bỏ `refind_backbone` + `wide_search_scale`/`refind_after`; `backend/config/local.yaml` bỏ `refind_backbone: opencv`.

### Bước 1.4 — Frontend
- **Sửa:** `frontend/src/app/pages/ModelSettings.tsx` bỏ toggle "EVPTrack (Tier B)"; `frontend/src/app/lib/trackingSession.tsx` bỏ field `refind_backbone`.

### Bước 1.5 — Cập nhật test còn lại
- **Sửa:** `backend/tests/test_backbones.py` bỏ `"evptrack"` khỏi assert/loop; rà `backend/tests/test_tracking_system.py` nếu tham chiếu refind.

**🚦 Cổng nghiệm thu Phase 1:** `pytest` backend xanh; `tsc`/build frontend sạch; chạy video → lock & track bình thường, mất mục tiêu → vào re-acquire (không còn REFIND).

---

## Phase 2 — Tăng tốc tracker: ONNX + TensorRT-EP  *(máy 4060, đòn lớn nhất)*

> **Cơ sở SOTA:** ONNX Runtime **TensorRT Execution Provider** với `trt_fp16_enable` + `trt_engine_cache_enable` → engine build/cache **tự động theo từng GPU** (giải quyết 4060 sm_89 ≠ Orin sm_87). FP16 cho transformer tracker thường 2–4× so PyTorch eager.

### Bước 2.1 — Export UETrack-tiny (FastiTPN) → ONNX  *(bước rủi ro nhất)*
- **Thêm:** `backend/tools/export_uetrack_onnx.py` — load checkpoint RGB (`models/uetrack_tiny_rgb.pth`) qua repo UETrack, `torch.onnx.export` với **input shape cố định** (template + search), opset đủ cao.
- **Xử lý vướng transformer:** attention/op không hỗ trợ → thay bằng op chuẩn; tắt FlashAttention; cố định mọi dynamic shape.
- **Test parity:** so output ONNX vs PyTorch trên vài frame: cosine logits > 0.99 / sai khác bbox < 1–2 px.

### Bước 2.2 — Backbone `uetrack_onnx` (đặt cạnh backbone cũ)
- **Thêm:** `backend/app/vision/backbones/uetrack_onnx.py` — class theo đúng contract `base.py` (`source`,`available`,`kind`,`init`,`track→TrackResult`). Chạy `onnxruntime.InferenceSession` providers `[TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider]`, FP16, bật engine-cache dir. Tự `available=False` khi thiếu onnxruntime/model → ManagedTracker tự rớt OpenCV.
- **Đăng ký:** thêm 1 dòng vào `BACKBONE_REGISTRY`.
- **Config:** thêm block `models.tracker_uetrack_onnx` (onnx_path, providers, fp16, cache_dir). **Giữ** `UETrackBackbone` PyTorch để đối chứng/fallback.
- **Test:** `test_backbones.py` thêm `"uetrack_onnx"` vào registry; build/init không lỗi khi thiếu model (rớt OpenCV).

### Bước 2.3 — Bật & đo
- **Sửa:** `local.yaml` đặt `tracking.normal_backbone: uetrack_onnx`. TRT-EP build engine lần chạy đầu (chậm 1 lần), cache lại.
- **Deliverable:** so `tracker_ms` & `fps` trước/sau; xác nhận IoU parity ≥ fa0.9 trên clip.

**🚦 Cổng nghiệm thu Phase 2:** `tracker_ms` giảm rõ rệt, FPS tăng, không hồi quy bám.

**Kết quả thực thi (4060):**
- Export OK: `models/uetrack_tiny_rgb.onnx` (40.3 MB), inputs 6-kênh `template[1,6,112,112]`/`search[1,6,224,224]`/`template_anno[1,4]`, outputs `score/size/offset_map`. Parity tensor max|Δ| ~1e-6.
- Backbone `uetrack_onnx` splice ONNX vào tracker repo (tái dùng pre/post, free encoder torch để tiết kiệm RAM). Parity mức tracker **IoU=1.0** so PyTorch trên chuỗi synthetic.
- **CHẶN đo tốc độ trên 4060:** `onnxruntime` cài là bản **CPU-only** (không có CUDA/TensorRT EP). Muốn đo tăng tốc TRT phải `pip install onnxruntime-gpu` + TensorRT → để dành **on-device/khi user đồng ý đổi môi trường**. Hạ tầng đã sẵn (providers list + engine cache + fp16), chỉ cần EP.

---

## Phase 3 — Cắt việc mỗi frame: ReID theo cadence  *(máy 4060)*

> **Vấn đề:** `memory.score()` chạy deep ReID **mỗi frame** (`session.py:723`) — embed sâu 60 lần/giây chỉ để chấm identity.
> **Cơ sở SOTA:** template/appearance update **thưa** (STARK/MixFormer cập nhật template mỗi N frame). Áp dụng tương tự cho ReID.

### Bước 3.1 — Deep ReID theo nhịp + cache
- **Sửa:** `backend/app/core/session.py` `_update_tracking` — deep embed (full fused) chạy mỗi `identity.reid_interval` frame **hoặc** khi rớt UNCERTAIN; frame giữa chỉ chạy phần handcrafted rẻ (`FusedEncoder`) để giữ drift-catch nhạy. Cache feature fused gần nhất.
- **An toàn:** trên frame "cached", **không** cho `consider_update`/`consolidate` admit bằng score cũ (tránh `negative_margin` cũ làm hỏng bank). Memory chỉ học trên frame có deep embed tươi.
- **Config:** thêm `identity.reid_interval` (vd 3–5) + `identity.reid_on_uncertain: true`.
- **Test:** `test_encoders.py`/`test_tracking_system.py` thêm case cadence: deep chỉ gọi mỗi N frame (đếm mock), drift vẫn bị bắt khi identity tụt.
- **Deliverable:** `reid_ms` trung bình giảm ~1/N; FPS tăng; vẫn bắt được drift/đổi mục tiêu trên clip thử.

**🚦 Cổng nghiệm thu Phase 3:** FPS đạt/áp sát mục tiêu trên 4060; bám không hồi quy.

---

## Phase 4 — HW codec & offload CPU (chuẩn bị Jetson, device-agnostic)

### Bước 4.1 — JPEG encode tách backend
- **Sửa:** `backend/app/core/session.py` `_encode_frame` — trừu tượng hoá encoder: dùng NVJPEG `[ON-DEVICE]` khi có, else `cv2.imencode`. Giữ `stream_max_hz`/`stream_max_width`.
- **Test:** snapshot stream vẫn ra JPEG hợp lệ; `encode_ms` giảm trên thiết bị có NVJPEG.

### Bước 4.2 — Optical flow nhẹ hơn
- **Sửa config:** `motion.camera` cho phép `flow_downscale`↓, `max_features`↓, `flow_interval`↑; chừa hook chạy **VPI HW optical flow** `[ON-DEVICE]`.
- **Test:** `flow_ms` giảm; EKF gate vẫn ổn (test_ekf.py xanh).

---

## Phase 5 — Tinh chỉnh trên Jetson  `[ON-DEVICE]` *(sau khi port)*

- **5.1 Power/clocks (gần như miễn phí ~1.7×):** `sudo nvpmodel -m 0` (MAXN/Super) + `sudo jetson_clocks`. JetPack 6.1+ bật **Super mode** (Orin Nano 8GB 40→67 TOPS). Đo lại trước khi tối ưu thêm.
- **5.2 Capture HW decode:** GStreamer `nvv4l2decoder`/`nvjpegdec` thay decode CPU.
- **5.3 INT8 (nếu FP16 chưa đủ 60fps):** calibrate ReID (`reid_mbnv3.onnx`) + (tùy chọn) tracker; validate accuracy chặt vì INT8 hại localization.
- **5.4 Ngân sách RAM 8GB:** kiểm `tegrastats`; không load net deep thứ 2 (Tier B đã bỏ); SAM2 chỉ init/refine.
- **Nghiệm thu cuối:** `metrics.fps` ≥ 60 khi lock-tracking; accuracy đạt; `tegrastats` không nghẽn RAM.

---

## Rủi ro & Rollback

| Rủi ro | Giảm thiểu |
|---|---|
| Export ONNX transformer fail (op lạ) | Cô lập ở Bước 2.1 (chỉ script, chưa đụng runtime); nếu kẹt → fallback: FP16 PyTorch + giữ `uetrack` cũ. Backbone mới *cộng thêm*, không thay thế. |
| TRT engine không chạy trên Orin | TRT-EP build on-device tự động; engine cache per-device. |
| ReID cadence làm chậm bắt drift | `reid_on_uncertain` ép deep ngay khi confidence tụt; handcrafted vẫn chạy mỗi frame. |
| Bỏ Tier B giảm robustness | UETrack tự re-search local; mất hẳn → Tier C global re-acquire (đúng SOTA LTMU/SAMURAI). |
| 60fps không đạt chỉ bằng FP16 | Lối thoát: INT8 (5.3) + giảm search-size tracker + VPI flow. |

Mọi tối ưu **gating bằng config / additive** → revert = đổi 1 dòng config hoặc xoá 1 entry registry.

## Tóm tắt thứ tự thực thi
**P0** đo lường+uncap → **P1** bỏ EVPTrack/Tier B (2 tầng) → **P2** ONNX+TRT-EP tracker → **P3** ReID cadence → **P4** HW codec/flow (device-agnostic) → **P5** `[ON-DEVICE]` Jetson tuning. Mỗi phase qua cổng nghiệm thu mới sang phase sau.
