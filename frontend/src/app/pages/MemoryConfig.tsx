import { BrainCircuit } from "lucide-react";
import { ConfigForm } from "../components/config/ConfigForm";

export function MemoryConfig() {
  return (
    <ConfigForm
      title="Memory Config"
      description="Bộ nhớ đặc trưng để phân biệt vật cùng label: số slot, ngưỡng nạp (SAMURAI), tốc độ hợp nhất."
      icon={BrainCircuit}
      sections={[
        {
          heading: "Banks (RAM short / WRM working / DRM anchor / negative)",
          fields: [
            { path: ["memory", "ram_slots"], label: "RAM slots", min: 1, max: 32, step: 1 },
            { path: ["memory", "wrm_slots"], label: "WRM slots", min: 1, max: 32, step: 1 },
            { path: ["memory", "drm_slots"], label: "DRM (anchor) slots", min: 1, max: 32, step: 1 },
            { path: ["identity", "negative_slots"], label: "Negative slots", min: 1, max: 32, step: 1 },
            { path: ["memory", "working_promote_every"], label: "Working promote every", min: 1, max: 60, step: 1 },
            { path: ["memory", "long_term_min_margin"], label: "Long-term min margin", min: 0, max: 0.6, step: 0.01 },
            { path: ["identity", "min_margin"], label: "Identity min margin", min: 0, max: 0.5, step: 0.01 },
          ],
        },
        {
          heading: "SAMURAI memory admission gates",
          fields: [
            { path: ["samurai", "memory_admission", "min_affinity"], label: "Min affinity", min: 0, max: 1, step: 0.01 },
            { path: ["samurai", "memory_admission", "min_positive"], label: "Min positive", min: 0, max: 1, step: 0.01 },
            { path: ["samurai", "memory_admission", "max_negative"], label: "Max negative", min: 0, max: 1, step: 0.01 },
            { path: ["samurai", "memory_admission", "min_motion"], label: "Min motion", min: 0, max: 1, step: 0.01 },
            { path: ["samurai", "memory_admission", "min_margin"], label: "Min margin", min: 0, max: 0.6, step: 0.01 },
          ],
        },
      ]}
    />
  );
}
