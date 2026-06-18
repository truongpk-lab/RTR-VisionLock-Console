import { SlidersHorizontal } from "lucide-react";
import { ConfigForm } from "../components/config/ConfigForm";

export function ThresholdConfig() {
  return (
    <ConfigForm
      title="Threshold Config"
      description="Ngưỡng quyết định trạng thái bám: STABLE / UNCERTAIN / LOST và bắt drift."
      icon={SlidersHorizontal}
      sections={[
        {
          heading: "Confidence state machine",
          fields: [
            { path: ["thresholds", "stable_threshold"], label: "Stable threshold", min: 0.3, max: 0.95, step: 0.01, hint: "≥ ngưỡng này → STABLE (bám bằng Tier A)" },
            { path: ["thresholds", "uncertain_threshold"], label: "Uncertain threshold", min: 0.2, max: 0.9, step: 0.01, hint: "< ngưỡng này → UNCERTAIN (Tier B re-find)" },
            { path: ["thresholds", "lost_frames"], label: "Lost frames", min: 1, max: 30, step: 1, hint: "Số khung dưới uncertain liên tiếp → LOST" },
            { path: ["thresholds", "identity_lost_frames"], label: "Identity-lost frames", min: 1, max: 60, step: 1, hint: "Khung ok nhưng identity thấp → ép LOST" },
          ],
        },
        {
          heading: "Identity & re-detect gates",
          fields: [
            { path: ["thresholds", "min_similarity"], label: "Min similarity (drift catch)", min: 0.2, max: 0.9, step: 0.01 },
            { path: ["thresholds", "reacquire_threshold"], label: "Re-acquire threshold", min: 0.3, max: 0.95, step: 0.01, hint: "reid tối thiểu để re-lock khi LOST" },
            { path: ["thresholds", "mask_iou_threshold"], label: "Mask IoU threshold", min: 0.2, max: 0.9, step: 0.01 },
            { path: ["thresholds", "kalman_max_error"], label: "Kalman max error (px)", min: 20, max: 200, step: 1 },
          ],
        },
      ]}
    />
  );
}
