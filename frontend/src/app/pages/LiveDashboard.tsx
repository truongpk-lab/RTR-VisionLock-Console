import { CameraViewport } from "../components/CameraViewport";
import { LeftPanel } from "../components/LeftPanel";
import { RightPanelMetrics } from "../components/RightPanel";
import { BottomConsole } from "../components/BottomConsole";

export function LiveDashboard() {
  return (
    <div className="flex h-full w-full bg-[#05090f] overflow-hidden">
      {/* Left Control Panel */}
      <LeftPanel />

      {/* Main Center Area */}
      <div className="flex-1 flex flex-col min-w-0 relative border-r border-slate-800/60">
        
        {/* Camera Area */}
        <div className="flex-1 p-4 flex items-center justify-center bg-[#030508] relative overflow-hidden">
          <div className="w-full h-full relative border border-slate-800 bg-black rounded-sm overflow-hidden shadow-[0_0_20px_rgba(0,0,0,0.5)]">
            <CameraViewport />
          </div>
        </div>

        {/* Bottom Timeline & Logs */}
        <div className="h-48 border-t border-slate-800/60 bg-[#0a0f16]">
          <BottomConsole />
        </div>
      </div>

      {/* Right Metrics Panel */}
      <RightPanelMetrics />
    </div>
  );
}
