# PlanModel.md — Kế hoạch: bộ chọn 2 tracker (UETrack ⇆ ORTrack-D-DeiT), tách riêng & chống xung đột

> Tài liệu **lập kế hoạch**, chưa sửa code. Phạm vi đã thu gọn theo yêu cầu: mục *Model
> Settings* chỉ cần chọn **1 trong 2** backbone Tier A để **thực nghiệm so sánh**:
> **UETrack** (hiện tại) và **ORTrack-D-DeiT**. Trọng tâm: **kiểm tra kĩ xung đột** và
> **tách riêng** sao cho (a) đổi qua lại không xung đột khi thử nghiệm, (b) khi chọn
> được model thắng thì **đưa lên Jetson dễ nhất**. Nguồn số liệu ở [§7](#7-tài-liệu-tham-khảo).

---

## 0. TL;DR

- **UETrack có cấu trúc riêng?** Là repo riêng (dòng OSTrack), nhưng trong dự án được nạp
  động qua *adapter dùng chung* [`StarkLineageBackbone`](backend/app/vision/backbones/torch_tracker.py);
  `UETrackBackbone` chỉ là subclass ~8 dòng. ORTrack cùng họ → thêm cũng chỉ ~8 dòng.
- **Chọn 1 trong 2 có dễ?** **Code: dễ.** **Vận hành: có 2 trục xung đột thật phải xử lý**
  (xem [§3](#3-phân-tích-xung-đột-kiểm-tra-kĩ)):
  1. **Namespace gói `lib`** — cả 2 repo đều có gói `lib` với import tuyệt đối → *không thể
     cùng nạp trong 1 tiến trình*. ⇒ giải bằng **nạp-một-lúc-một-repo + dọn sạch khi switch**.
  2. **Phiên bản dependency** (ORTrack ghim torch 1.10/CUDA 10.2/timm 0.9.10 cũ vs UETrack
     torch hiện đại) → nếu thật sự kẹt version thì *không live-switch được trong 1 process*.
- **Chiến lược "tách riêng" khuyến nghị:** mỗi tracker là **một clone repo riêng, giữ
  nguyên gốc (không sửa)**; backend chỉ giữ *một* repo sống tại một thời điểm và **dọn
  `sys.modules`/`sys.path` khi đổi model**. Lợi ích kép: đổi model lúc thử nghiệm không
  xung đột, và khi xong **chỉ cần copy đúng 1 repo + checkpoint (+ export TRT) lên Jetson**
  — trên Jetson chỉ chạy 1 model nên **không còn xung đột nào**.

---

## 1. Cấu trúc tracking hiện tại (vì sao thêm model là dễ về code)

Adapter dùng chung cho cả họ OSTrack/PyTracking:

| Lớp | File | Vai trò |
|---|---|---|
| `BACKBONE_REGISTRY` | [backbones/\_\_init\_\_.py](backend/app/vision/backbones/__init__.py) | map `tên → constructor`; thêm tracker = +1 dòng. |
| `ManagedTracker` | [backbones/\_\_init\_\_.py](backend/app/vision/backbones/__init__.py) | sở hữu 1 backbone + fallback OpenCV, `warmup()`, `reinit()`, `close()`. |
| `StarkLineageBackbone` | [torch_tracker.py](backend/app/vision/backbones/torch_tracker.py) | adapter động: nạp repo qua `importlib` từ `repo_path`, gọi `initialize(rgb,{"init_bbox":[...]})` + `track(rgb)→{"target_bbox":[...]}`. Lỗi → `available=False` → fallback. |
| `UETrackBackbone` | [torch_tracker.py:179](backend/app/vision/backbones/torch_tracker.py#L179) | subclass ~8 dòng chỉ điền tên module/class/param. |

`tracking.normal_backbone` (trong [default.yaml](backend/config/default.yaml)) chọn backbone
theo tên; [session.py:399-401](backend/app/core/session.py#L399) **rebuild khi tên đổi** —
nên cơ chế đổi model lúc chạy đã có sẵn. ORTrack là fork OSTrack với đúng giao diện
`initialize()/track()` → thêm `ORTrackBackbone` là một subclass tương tự.

---

## 2. So sánh 2 model (cho mục tiêu ≥60 FPS, RGB-only)

| | **UETrack-T** (hiện tại) | **ORTrack-D-DeiT** |
|---|---|---|
| Paper / venue | "UETrack: Unified & Efficient…", **CVPR 2026** (arXiv 2603.01412) | "Occlusion-Robust ViT for Real-Time UAV Tracking", **CVPR 2025** (arXiv 2504.09228) |
| Lineage | OSTrack one-stream (Fast-iTPN), MoE chia sẻ backbone + chưng cất TAD | OSTrack one-stream (DeiT-Tiny), bản **chưng cất** ORTrack→ORTrack-D |
| Điểm mạnh | **Unified + Efficient**, cân bằng tốc độ/độ chính xác | **Bền với che khuất** (ORR: bất biến đặc trưng dưới mask) |
| Params / FLOPs | 6M / 1.8 G | **5.3M / 1.5 GMac** (nhẹ hơn chút) |
| FPS GPU gốc | 221 (2080Ti) | 292 (TitanX*) |
| FPS Jetson AGX Xavier | **77** | *chưa báo cáo* (params nhỏ → kỳ vọng cao, **phải tự đo**) |
| LaSOT AUC | **63.4** | 54.6 (tinh chỉnh cho UAV) |
| Modality / interface | RGB-only OK / chuẩn OSTrack | **RGB-only thuần** / chuẩn OSTrack |
| State giữa frame | Tĩnh (template) | Tĩnh |
| License | (kiểm tra repo) | **MIT** |
| ONNX/TensorRT | Có sẵn đường `uetrack_onnx` trong dự án | Không chính thức (ViT thuần → export khả thi, tự làm) |
| Rủi ro riêng | — | **Ghim torch 1.10/CUDA 10.2/timm 0.9.10 cũ** → xem [§3.2](#32-trục-2-xung-đột-phiên-bản-dependency-torchtimm) |

(\* ORTrack đo trên TitanX — chậm hơn 2080Ti → quy đổi sẽ còn cao hơn; về compute hai model
cùng hạng "siêu nhẹ".) **Đọc nhanh:** hai model FPS tương đương nhau; UETrack chính xác hơn
trên SOT tổng quát, ORTrack bền hơn khi che khuất và nhẹ hơn chút. Đây chính là lý do đáng
để **so sánh trực tiếp** trên phần cứng của bạn.

---

## 3. Phân tích xung đột (kiểm tra kĩ)

Có **hai trục xung đột độc lập**. Trục 1 đã *xác nhận bằng kiểm tra repo*; trục 2 là rủi
ro cần verify khi clone ORTrack.

### 3.1 Trục 1 — Xung đột namespace gói `lib` (đã xác nhận)

Bằng chứng từ repo UETrack local (`/home/pk/code/UETrack`):
- Top-level có gói **`lib/__init__.py`**.
- Mọi import nội bộ là **tuyệt đối theo `lib.`**: `from lib.test.tracker.basetracker import …`,
  `from lib.models.uetrack import …`, `from lib.utils.box_ops import …` (xác nhận trong
  `lib/test/tracker/uetrack.py`).

ORTrack là fork OSTrack ⇒ **cũng có gói `lib`** với import `lib.*` tuyệt đối.

Hệ quả: adapter `_build()` chèn `repo_path` + `repo_path/lib` vào `sys.path` rồi
`import lib.test.tracker.<x>` ([torch_tracker.py:100-130](backend/app/vision/backbones/torch_tracker.py#L100)).
Khi đã import `lib` của UETrack, Python cache `sys.modules["lib"]` với `__path__` trỏ vào
`UETrack/lib`. Đổi sang ORTrack → `import lib.test.tracker.ortrack` **vẫn tra trong
`UETrack/lib`** → `ModuleNotFoundError` → `available=False` → **fallback OpenCV âm thầm**.

**Điểm chốt quan trọng:** đường switch hiện tại
([session.py:399-401](backend/app/core/session.py#L399)) **chỉ thay `self.tracker`, KHÔNG
dọn `sys.modules`/`sys.path`** (và `patch_config` không gọi `close()` trên tracker cũ).
Đây chính là chỗ phải vá.

### 3.2 Trục 2 — Xung đột phiên bản dependency (torch/timm)

ORTrack ghim **torch 1.10.0+cu102, torchvision 0.10.0, timm 0.9.10, Python 3.8** (rất cũ),
trong khi venv dự án chạy torch hiện đại (CUDA 12, cho SAM2/YOLO trên RTX 4060). Hai phiên
bản torch **không thể cùng tồn tại trong một tiến trình** — `sys.modules` dọn được, nhưng
*không* dọn được "hai bản torch".

Vì ORTrack chỉ là ViT thuần + timm, **khả năng cao chạy được trên torch hiện đại** với sửa
nhỏ/không cần sửa; rủi ro chính là **API timm trôi giữa 0.9.10 và bản hiện tại**. Bắt buộc
**verify bằng cách clone + import + chạy thử** trước khi tin vào live-switch.

→ Quyết định phụ thuộc kết quả verify:
- **Nếu ORTrack chạy được trên torch/timm của dự án** ⇒ live-switch trong 1 backend OK
  (chỉ cần xử lý Trục 1).
- **Nếu ORTrack đòi timm/torch khác xung khắc** ⇒ **không ép sống chung trong 1 process**;
  benchmark mỗi model ở **process/venv riêng** (xem [§3.4](#34-phương-án-cô-lập-khuyến-nghị--so-sánh)).

### 3.3 Có collider nào ngoài `lib` không?

Adapter chỉ import `lib.test.parameter.<x>` và `lib.test.tracker.<x>`; các import nội bộ
đều dưới `lib.*`. UETrack còn có thư mục top-level `tracking/`, `uetrack/` nhưng đó là
script train/test CLI và package model con (`lib.models.uetrack`) — **không** vào đường
inference. ⇒ Collider thực tế cần dọn là **`lib` + `lib.*`** (và bất kỳ module nào có
`__file__` nằm trong `repo_path` cũ, để chắc chắn).

### 3.4 Phương án cô lập (khuyến nghị) + so sánh

**KHUYẾN NGHỊ — "nạp một-lúc-một-repo + dọn sạch khi switch" (giữ repo gốc nguyên vẹn):**

- Mỗi tracker là **một clone riêng** (`repo_path` cấu hình theo model), **không sửa code gốc**.
- Khi đổi `normal_backbone`: **tear down hoàn toàn** backbone cũ *trước khi* build cái mới:
  1. `torch.cuda.empty_cache()` (đã có trong `close()`),
  2. gỡ khỏi `sys.path` các entry của repo cũ (`repo_path`, `repo_path/lib`),
  3. `pop` khỏi `sys.modules` mọi key `== "lib"` hoặc bắt đầu `"lib."`, **cộng** mọi module
     có `__file__` nằm trong `repo_path` cũ.
- Switch là thao tác *thỉnh thoảng* (bấm trên UI) nên chi phí rebuild chấp nhận được.

*Vì sao đây là lựa chọn "tách riêng" tốt nhất cho mục tiêu của bạn:*
- **Không sửa repo gốc** ⇒ re-sync upstream dễ, và **đưa lên Jetson chỉ là copy đúng 1 repo
  thắng cuộc + checkpoint** (Jetson chỉ chạy 1 model ⇒ *không còn xung đột Trục 1 hay 2*).
- Chỉ phải vá **một chỗ trong code dự án** (logic dọn khi switch), không đụng tracker logic.

**Các phương án khác (và vì sao không chọn cho ca 2-model này):**

| Phương án | Ưu | Nhược | Kết luận |
|---|---|---|---|
| **(A) Dọn-khi-switch** *(chọn)* | repo gốc nguyên vẹn; Jetson copy 1 repo; vá 1 chỗ | switch tốn rebuild; *không* giải Trục 2 nếu version xung khắc | **Khuyến nghị**; nếu Trục 2 xung khắc → kết hợp (C) |
| (B) Vendor + đổi tên gói `lib`→`uetrack_lib`/`ortrack_lib` | hai model sống chung vĩnh viễn 1 process | phải sửa **mọi** import `lib.*` trong cả 2 repo; khó re-sync; thừa (không bao giờ cần cả 2 cùng lúc) | Không cần |
| (C) Mỗi model một **venv + tiến trình riêng** | cô lập tuyệt đối cả Trục 1 lẫn Trục 2 | thêm IPC/độ trễ; vận hành nặng hơn | Dùng **khi Trục 2 xung khắc** (benchmark từng model ở run riêng) |

> Tóm lại: **(A) cho trường hợp ORTrack chạy được trên torch dự án; (C) nếu không.** Cả
> hai đều giữ repo riêng biệt và Jetson chỉ nhận 1 model ⇒ đúng tinh thần "tách riêng".

---

## 4. Việc cần làm

### 4.1 Backend

1. **`ORTrackBackbone`** trong [torch_tracker.py](backend/app/vision/backbones/torch_tracker.py):
   ```python
   class ORTrackBackbone(StarkLineageBackbone):
       source = "ortrack"; config_key = "tracker_ortrack"
       default_param_module = "lib.test.parameter.ortrack"
       default_tracker_module = "lib.test.tracker.ortrack"
       default_tracker_class = "ORTrack"     # VERIFY tên class trong repo
       default_param_name = "ortrack_d_deit" # VERIFY tên yaml ở experiments/ortrack/
   ```
2. Đăng ký `"ortrack": ORTrackBackbone` vào `BACKBONE_REGISTRY`.
3. Block config `tracker_ortrack` trong [default.yaml](backend/config/default.yaml) (repo_path,
   checkpoint = weight student ORTrack-D, param_name, enabled).
4. **Logic dọn-khi-switch (Trục 1)** — phần cốt lõi:
   - Trong `StarkLineageBackbone`, lưu lại danh sách `sys.path` đã chèn + `repo_path`.
   - Mở rộng `ManagedTracker.close()` (hiện chỉ `empty_cache()`) để gọi một hàm
     `_purge_repo(repo_path)`: gỡ entry `sys.path` + `pop` `sys.modules` (`lib`, `lib.*`,
     và module có `__file__` trong `repo_path`).
   - Trong [session.py](backend/app/core/session.py): **trước** khi
     `self.tracker = self._new_normal_tracker()` (dòng 401), gọi
     `if self.tracker is not None: self.tracker.close()` để dọn repo cũ. (Hiện switch
     **không** close → chính là nguồn xung đột.)
5. **Verify Trục 2:** clone ORTrack, `import` + chạy thử trên torch/timm của venv dự án.
   - Chạy được → giữ phương án (A).
   - Không → chuyển ORTrack sang venv/tiến trình riêng (phương án C); bộ chọn lúc đó là
     "chọn cấu hình để chạy benchmark", không phải live-switch trong 1 process.

### 4.2 Frontend

1. Thêm `kind: "select"` (+ `options`) vào `FieldSpec` của
   [ConfigForm.tsx](frontend/src/app/components/config/ConfigForm.tsx) (dùng `select.tsx`/
   `radio-group.tsx` có sẵn).
2. Trong [ModelSettings.tsx](frontend/src/app/pages/ModelSettings.tsx): field
   `tracking.normal_backbone` kiểu select với **2 lựa chọn**: `uetrack | ortrack`
   (giữ `uetrack_onnx` như tuỳ chọn tăng tốc nếu muốn).
3. Badge backbone đang chạy đã có ở
   [RightPanel.tsx:50](frontend/src/app/components/RightPanel.tsx#L50) (đọc
   `debug.tracker_backend`) — quan sát để phát hiện fallback OpenCV (model nạp lỗi).

### 4.3 Tiêu chí xác minh (per CLAUDE.md §4)

- [ ] Chọn `ortrack` trên UI → log `tracker_backend = ortrack`, **không** rơi OpenCV.
- [ ] **Đổi qua lại `uetrack ⇆ ortrack` nhiều lần trong 1 phiên không lỗi import `lib`**
      (test đúng Trục 1; nếu chưa vá dọn-khi-switch sẽ thấy fallback OpenCV ở lần đổi thứ 2).
- [ ] ORTrack import/chạy được trên torch/timm dự án (Trục 2) — hoặc đã chốt chạy venv riêng.
- [ ] Đo FPS thực (panel timing đã có) cho cả 2 ở cùng cảnh/độ phân giải → điền lại bảng §2.

---

## 5. Định hướng sau thử nghiệm (gọn)

- **Không** ensemble (chạy cả 2 cùng lúc) — FPS sụp tuyến tính, vô ích với mục tiêu tốc độ.
- Chọn **một model thắng** làm mặc định; nếu cần FPS production → export ONNX/TensorRT FP16
  (mẫu sẵn: đường `uetrack_onnx`). **Trên Jetson chỉ deploy 1 model** ⇒ không còn xung đột.
- (Tuỳ chọn, sau này) nếu muốn tận dụng *điểm mạnh khác nhau* (UETrack chính xác / ORTrack
  bền che khuất) mà vẫn giữ FPS một model → **online switching**: router chọn 1 trong 2 theo
  tình huống (confidence ở [policy.py](backend/app/vision/policy.py), tuổi mất dấu ở reacquire),
  **không** chạy đồng thời.

---

## 6. Rủi ro & câu hỏi mở

- **Trục 1 (`lib`)** — đã xác nhận; bắt buộc vá dọn-khi-switch (§4.1.4), nếu không lần đổi
  model thứ hai sẽ âm thầm về OpenCV.
- **Trục 2 (torch/timm)** — phải verify ORTrack trên torch dự án; nếu xung khắc thì
  benchmark ở process/venv riêng (§3.4-C). Đây là rủi ro lớn nhất chưa kiểm chứng được trên
  dev box (ORTrack chưa clone).
- **Tên `param_name`/class ORTrack** — phải xác nhận trong repo (`experiments/ortrack/*.yaml`,
  `lib/test/tracker/ortrack.py`).
- **FPS Jetson** — số trong paper là AGX **Xavier**; ORTrack *chưa* có số Jetson; nếu bạn
  dùng AGX **Orin** thì nhanh hơn nhưng *phải tự đo*.
- **Độ chính xác ORTrack** — LaSOT AUC 54.6 (tối ưu UAV) thấp hơn UETrack-T 63.4; mạnh khi
  cảnh giống không ảnh / nhiều che khuất, yếu nếu cần SOT tổng quát chính xác cao.
- **Câu hỏi cho bạn:** ORTrack có cần chạy venv riêng không là điều quyết định kiến trúc bộ
  chọn (live-switch 1 process vs chọn-rồi-chạy process riêng). Cần kết quả verify Trục 2 mới
  chốt — bạn muốn tôi clone ORTrack về để kiểm tra trước khi code chứ?

---

## 7. Tài liệu tham khảo

- **OSTrack** (nền one-stream): Ye et al., ECCV 2022 — arXiv:2203.11991 · https://github.com/botaoye/OSTrack
- **UETrack** (CVPR 2026): arXiv:2603.01412 · https://huggingface.co/kangben258/UETrack · repo local `/home/pk/code/UETrack/`
- **ORTrack** (CVPR 2025): arXiv:2504.09228 · https://github.com/wuyou3474/ORTrack · CVF PDF (openaccess.thecvf.com/content/CVPR2025/…)
- **Tối ưu Jetson / TensorRT ViT FP16:** Nota AI Jetson Orin NX case study · NVIDIA/TensorRT issue #4599
