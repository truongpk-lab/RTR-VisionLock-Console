import { Cpu, Wifi, ActivitySquare, Camera } from "lucide-react";
import { useTrackingSession } from "../lib/trackingSession";

export function Topbar() {
  const { connected, session } = useTrackingSession();
  const metrics = session.metrics;
  const stateColor = session.state === "ERROR" || session.state === "LOST" ? "text-rose-400" : session.state === "SEARCHING" ? "text-amber-400" : "text-cyan-400";
  return (
    <header className="h-14 flex-shrink-0 bg-[#0a0f16] border-b border-slate-800/60 flex items-center justify-between px-4 z-10">
      <div className="flex items-center gap-4">
        <div className="text-xs font-mono text-slate-400 tracking-widest hidden sm:block">
          OP: OMNI_WATCH
        </div>
      </div>
      
      <div className="flex items-center gap-3 md:gap-6">
        <StatusBadge icon={ActivitySquare} label="FPS" value={metrics.fps.toFixed(1)} color="text-emerald-400" />
        <StatusBadge icon={Wifi} label="LATENCY" value={`${metrics.latency_ms.toFixed(0)}ms`} color="text-cyan-400" />
        <StatusBadge icon={Cpu} label="GPU" value={metrics.gpu} color="text-amber-400" />
        <StatusBadge icon={Camera} label="CAM" value={connected ? metrics.camera : "OFFLINE"} color={metrics.camera === "ACTIVE" ? "text-emerald-400" : "text-slate-400"} />
        
        <div className="h-6 w-px bg-slate-800 mx-2 hidden md:block"></div>
        
        <div className="flex items-center gap-2 px-3 py-1 bg-cyan-950/30 border border-cyan-900/50 rounded-sm">
          <div className={`w-2 h-2 rounded-full ${connected ? "bg-cyan-400" : "bg-slate-500"} animate-pulse`}></div>
          <span className={`text-xs font-mono font-bold tracking-widest ${stateColor}`}>{connected ? session.state : "OFFLINE"}</span>
        </div>
      </div>
    </header>
  );
}

function StatusBadge({ icon: Icon, label, value, color }: { icon: any, label: string, value: string, color: string }) {
  return (
    <div className="flex items-center gap-2">
      <Icon size={14} className="text-slate-500" />
      <div className="flex flex-col">
        <span className="text-[9px] font-mono text-slate-500 uppercase tracking-widest leading-none mb-0.5">{label}</span>
        <span className={`text-xs font-mono font-bold leading-none ${color}`}>{value}</span>
      </div>
    </div>
  );
}
