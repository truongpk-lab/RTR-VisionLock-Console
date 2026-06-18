import { Settings } from "lucide-react";
import { ConfigForm } from "../components/config/ConfigForm";

export function ModelSettings() {
  return (
    <ConfigForm
      title="Model Settings"
      description="Bật/tắt detector, segmenter, tracker; thông số suy luận; và preset tối ưu FPS."
      icon={Settings}
      presets={[
        {
          label: "FPS Preset (nhẹ)",
          patch: {
            runtime: { refine_during_tracking: false, stream_max_width: 800, stream_jpeg_quality: 65 },
            models: { proposal: { input_size: 512 } },
            motion: { camera: { flow_interval: 2 } },
          },
        },
        {
          label: "Quality Preset (nét)",
          patch: {
            runtime: { stream_max_width: 1280, stream_jpeg_quality: 80 },
            models: { proposal: { input_size: 640 } },
            motion: { camera: { flow_interval: 1 } },
          },
        },
      ]}
      sections={[
        {
          heading: "Detector (YOLO)",
          fields: [
            { path: ["models", "proposal", "enabled"], label: "Detector enabled", kind: "bool" },
            { path: ["models", "proposal", "conf"], label: "Confidence", min: 0.05, max: 0.9, step: 0.01 },
            { path: ["models", "proposal", "iou"], label: "NMS IoU", min: 0.1, max: 0.9, step: 0.01 },
            { path: ["models", "proposal", "input_size"], label: "Input size", min: 320, max: 1280, step: 32 },
          ],
        },
        {
          heading: "Segmenter (SAM2) & trackers",
          fields: [
            { path: ["models", "segmenter", "enabled"], label: "Segmenter enabled", kind: "bool" },
            { path: ["models", "tracker_uetrack", "enabled"], label: "UETrack (Tier A) enabled", kind: "bool" },
            { path: ["runtime", "refine_during_tracking"], label: "SAM2 refine while tracking", kind: "bool", hint: "Tắt để tăng FPS (UETrack bám bằng box)" },
          ],
        },
        {
          heading: "Streaming & motion (FPS)",
          fields: [
            { path: ["runtime", "stream_max_width"], label: "Stream max width", min: 480, max: 1920, step: 20 },
            { path: ["runtime", "stream_jpeg_quality"], label: "Stream JPEG quality", min: 30, max: 95, step: 1 },
            { path: ["motion", "camera", "flow_interval"], label: "Optical-flow interval", min: 1, max: 5, step: 1 },
          ],
        },
      ]}
    />
  );
}
