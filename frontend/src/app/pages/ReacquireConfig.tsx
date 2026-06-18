import { Radar } from "lucide-react";
import { ConfigForm } from "../components/config/ConfigForm";

export function ReacquireConfig() {
  return (
    <ConfigForm
      title="Re-acquire Config"
      description="Tìm lại mục tiêu khi LOST: tần suất detect, xác nhận, và trọng số ghép điểm."
      icon={Radar}
      sections={[
        {
          heading: "Detect & confirm",
          fields: [
            { path: ["reacquire", "match_label"], label: "Match label", kind: "bool", hint: "Chỉ xét vật cùng nhãn với mục tiêu" },
            { path: ["reacquire", "detect_hz"], label: "Detect Hz", min: 1, max: 10, step: 1, hint: "Tần suất chạy YOLO toàn khung khi LOST" },
            { path: ["reacquire", "confirm_frames"], label: "Confirm frames", min: 1, max: 8, step: 1, hint: "Số lần xác nhận liên tiếp trước khi re-lock" },
            { path: ["reacquire", "confirm_iou_gate"], label: "Confirm IoU gate", min: 0.0, max: 0.8, step: 0.01 },
            { path: ["reacquire", "reacquire_threshold"], label: "Re-acquire threshold", min: 0.3, max: 0.95, step: 0.01 },
            { path: ["reacquire", "top_k"], label: "Top-K hypotheses", min: 1, max: 4, step: 1 },
          ],
        },
        {
          heading: "Scoring weights (identity + motion + detector + mask)",
          fields: [
            { path: ["reacquire", "identity_weight"], label: "Identity weight", min: 0, max: 1, step: 0.01 },
            { path: ["reacquire", "motion_weight"], label: "Motion weight", min: 0, max: 1, step: 0.01 },
            { path: ["reacquire", "detector_weight"], label: "Detector weight", min: 0, max: 1, step: 0.01 },
            { path: ["reacquire", "mask_weight"], label: "Mask weight", min: 0, max: 1, step: 0.01 },
          ],
        },
      ]}
    />
  );
}
