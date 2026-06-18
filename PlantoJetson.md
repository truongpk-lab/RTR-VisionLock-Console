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
| P6 | Đo bottleneck (=flow) + hysteresis chống flapping + expose flow UI + dọn slider chết | 4060 | ✅ (full suite 112✓, build✓) |

---

## Việc cần làm tiếp (mở)

Làm được ngay trên **4060** (dùng cả cho Jetson):
- [ ] **P4.3 — `process_max_width`**: hạ độ phân giải XỬ LÝ (đòn FPS lớn nhất cho nguồn 1080p, chi tiết ở Phase 4). *Ưu tiên cao.*
- [ ] **Preset `local.yaml`** sẵn cho 2 chế độ: *cam tĩnh* (`motion.camera.enabled:false` + `identity.reid_interval:3` → ~78fps) và *cam động* (`flow_interval:2` + `flow_downscale:0.4` + `reid_interval:3` → ~59fps). *(tùy chọn)*

`[ON-DEVICE]` khi cắm **Jetson Orin Nano** (theo checklist Phase 5):
- [ ] Super mode + `jetson_clocks` → đo lại.
- [ ] Cài `onnxruntime-gpu` + TensorRT → bật backbone `uetrack_onnx` (TRT-EP) + ReID GPU.
- [ ] VPI HW optical flow (P4.2) nếu cam động; NVJPEG encode (P4.1).
- [ ] INT8 (P5.3) nếu FP16 chưa đủ 60fps; theo dõi `tegrastats` (RAM 8GB).

> Ước lượng: port "thô" ~8–15fps → tối ưu đủ gói ~45–65fps (xem bảng ở Phase 5). 60fps khả thi nhưng sát.

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

**Đo thật trên RTX 4060 — video 1080p@30fps (đoạn ~120 frame, đường `_update_tracking` đầy đủ, đều STABLE):**

| Cấu hình | proc/frame | FPS tối đa |
|---|---|---|
| Nặng nhất (flow mỗi frame + reid mỗi frame) | 24.9 ms | 40 |
| local.yaml hiện tại (flow interval 2) | 23.3 ms | 43 |
| flow OFF (cam tĩnh) | 20.5 ms | 49 |
| **cam tĩnh: flow OFF + reid_interval 3** | 12.8 ms | **78** |
| **cam động: flow int2 + ds0.4 + reid 3** | 16.8 ms | **59** |

> **Kết luận đo (P6.1) — đã đính chính:**
> - Con số "flow 14ms" báo lúc đầu là **artifact** của micro-bench nạp frame random khác nhau mỗi lần. Với **video/cam thật (frame liên tục)** flow chỉ ~2.5–4.5ms; **không stage nào áp đảo tuyệt đối** trên 4060.
> - **`reid_interval` là đòn mạnh nhất** vì mỗi frame STABLE deep ReID chạy **2 lần** (chấm điểm + `update_ram` ghi memory); cadence cắt ~7–8ms.
> - **30fps thấy trên app = pacing video**, không phải trần xử lý (4060 xử lý 40–78fps tùy cấu hình).
> - ⇒ Đã **bỏ cài onnxruntime-gpu** (deep net rẻ, không đáng). Đã expose 4 control flow (enabled/interval/downscale/max_features) ra Model Settings + thêm hysteresis chống flapping.

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

### Bước 4.2 — Optical flow nhẹ hơn  ✅ (control đã expose ở P6)
- **Đã làm (P6):** expose `motion.camera.{enabled, flow_interval, flow_downscale, max_features}` ra Model Settings để chỉnh sống. **Đòn lớn nhất: `enabled:false` cho cam TĨNH** (bỏ trọn stage CPU nặng nhất — clip truy đuổi gần như cam đứng yên ⇒ tắt là hợp lý).
- **Còn lại `[ON-DEVICE]`:** hook **VPI HW optical flow** trên Orin (nếu vẫn cần ego-motion mà muốn rẻ).
- **Test:** `flow_ms` giảm; EKF gate vẫn ổn (test_ekf.py xanh).

### Bước 4.3 — `process_max_width`: hạ độ phân giải XỬ LÝ  *(đòn FPS lớn cho nguồn 1080p + Jetson)* — CHƯA LÀM
- **Vấn đề (đo thật):** video nguồn **1920×1080** → flow + đặc trưng HSV thủ công chạy trên frame/crop lớn nên tốn; tracker/ReID thì luôn resize về 224 (không phụ thuộc độ phân giải). Hiện `stream_max_width` chỉ hạ ảnh **gửi UI**, KHÔNG hạ ảnh **xử lý**.
- **Sửa:** thêm cờ `runtime.process_max_width` (vd 960). Trong `_loop`/`_dispatch_frame` ([session.py](backend/app/core/session.py)) downscale frame **một lần** trước khi vào pipeline; nhớ **scale ngược bbox** về toạ độ gốc khi trả ra UI/lưu. Mặc định 0 = giữ nguyên (không đổi hành vi).
- **Lợi:** cắt chi phí flow + HSV ~ theo bình phương tỉ lệ; gần như không ảnh hưởng độ bám. Dùng được cả 4060 lẫn Jetson.
- **Test:** bbox trả về vẫn đúng toạ độ gốc (scale round-trip); `flow_ms` giảm; lock vẫn ổn trên clip.

---

## Phase 5 — Tinh chỉnh trên Jetson  `[ON-DEVICE]` *(sau khi port)*

- **5.1 Power/clocks (gần như miễn phí ~1.7×):** `sudo nvpmodel -m 0` (MAXN/Super) + `sudo jetson_clocks`. JetPack 6.1+ bật **Super mode** (Orin Nano 8GB 40→67 TOPS). Đo lại trước khi tối ưu thêm.
- **5.2 Capture HW decode:** GStreamer `nvv4l2decoder`/`nvjpegdec` thay decode CPU.
- **5.3 INT8 (nếu FP16 chưa đủ 60fps):** calibrate ReID (`reid_mbnv3.onnx`) + (tùy chọn) tracker; validate accuracy chặt vì INT8 hại localization.
- **5.4 Ngân sách RAM 8GB:** kiểm `tegrastats`; không load net deep thứ 2 (Tier B đã bỏ); SAM2 chỉ init/refine.
- **Nghiệm thu cuối:** `metrics.fps` ≥ 60 khi lock-tracking; accuracy đạt; `tegrastats` không nghẽn RAM.

**Ước lượng FPS khi port sang Orin Nano 8GB** (suy từ đo 4060, sai số rộng — CPU A78 chậm ~4–6×, GPU ~4–7×, BW ~4×; flow CPU @1080p là rủi ro lớn nhất):

| Mức tối ưu trên Orin | FPS ước lượng |
|---|---|
| Port "thô" (PyTorch, ReID+flow CPU, không TRT, power mặc định) | ~8–15 |
| + Super mode + jetson_clocks | ~14–22 |
| + TRT FP16 tracker + ReID lên GPU | ~25–40 |
| + flow OFF/VPI + reid cadence + `process_max_width` | **~45–65** |
| + INT8 (đòn cuối nếu còn thiếu) | đẩy tiếp |

⇒ **60fps khả thi nhưng SÁT**, cần gần như cả gói. Bỏ được **optical flow CPU** là cú nhảy lớn nhất trên Jetson.

**Checklist đo on-device (làm tuần tự, đọc per-stage timers đã có sẵn sau mỗi bước):**
1. `sudo nvpmodel -m 0` (MAXN/Super, JetPack 6.1+) + `sudo jetson_clocks` → đo lại trước tiên.
2. Cài `onnxruntime-gpu` + TensorRT cho JetPack → bật `tracking.normal_backbone: uetrack_onnx` (TRT-EP tự build engine lần đầu) + ReID providers `[TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider]`.
3. Bật `camera.uncap_fps` + chạy video 1080p → đọc `tracker_ms/reid_ms/flow_ms/encode_ms` để biết stage nào phình.
4. Cam tĩnh: `motion.camera.enabled:false`; cam động: `flow_interval` + `process_max_width` (4.3) + VPI (4.2).
5. `identity.reid_interval: 3`; `stable_hysteresis` giữ ~0.08.
6. Còn thiếu → INT8 (5.3) + giảm search-size tracker.
7. Theo dõi `tegrastats` (RAM 8GB + throttle nhiệt).

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
