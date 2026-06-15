import { Target, Fingerprint, ActivitySquare, BrainCircuit } from "lucide-react";
import { useTrackingSession } from "../lib/trackingSession";

export function RightPanelMetrics() {
  const { session } = useTrackingSession();
  const metrics = session.metrics;
  const memory = session.memory;
  return (
    <div className="w-64 flex-shrink-0 border-l border-slate-800/60 bg-[#0a0f16] flex flex-col z-10 overflow-y-auto">
      <div className="p-4 border-b border-slate-800/60 bg-slate-900/20">
        <h2 className="text-xs font-mono font-bold text-slate-300 uppercase tracking-widest mb-4 flex items-center gap-2">
          <ActivitySquare size={14} className="text-cyan-500" />
          Primary Metrics
        </h2>
        
        <div className="space-y-4">
          <ProgressMetric label="Track Score" value={metrics.track_score * 100} color="bg-cyan-500" />
          <ProgressMetric label="Confidence" value={metrics.confidence * 100} color="bg-emerald-500" />
          <ProgressMetric label="Similarity" value={metrics.similarity * 100} color="bg-blue-500" />
        </div>
      </div>

      <div className="p-4 border-b border-slate-800/60">
        <h2 className="text-xs font-mono font-bold text-slate-300 uppercase tracking-widest mb-4 flex items-center gap-2">
          <BrainCircuit size={14} className="text-purple-400" />
          Analytics
        </h2>
        
        <div className="grid grid-cols-2 gap-3">
          <MetricCard label="Mask IoU" value={metrics.mask_iou.toFixed(2)} />
          <MetricCard label="Kalman Err" value={`${metrics.kalman_error.toFixed(1)}px`} good />
          <MetricCard label="Motion" value={metrics.motion.slice(0, 4)} />
          <MetricCard label="Candidates" value={String(metrics.candidates)} highlight={session.state === "SEARCHING"} />
        </div>
      </div>

      <div className="p-4 flex-1 bg-slate-900/10">
        <h2 className="text-xs font-mono font-bold text-slate-300 uppercase tracking-widest mb-4 flex items-center gap-2">
          <Fingerprint size={14} className="text-amber-500" />
          Identity Memory
        </h2>
        
        <div className="space-y-3">
          <div className="p-2 border border-slate-800 rounded bg-[#030508] flex items-center gap-3">
            <div className="w-10 h-10 rounded bg-slate-800 flex items-center justify-center overflow-hidden border border-slate-700">
               {/* Tiny thumbnail representation */}
               <Target size={16} className="text-slate-500" />
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">Base ID</span>
              <span className="text-xs font-mono text-slate-300">{memory.base_id}</span>
            </div>
          </div>
          
          <div className="flex justify-between items-center text-[10px] font-mono">
            <span className="text-slate-500">Features Extracted:</span>
            <span className="text-cyan-400">{memory.feature_dim.toLocaleString()}-dim</span>
          </div>
          <div className="flex justify-between items-center text-[10px] font-mono">
            <span className="text-slate-500">Memory Bank:</span>
            <span className="text-slate-300">{memory.ram_slots} / {memory.ram_capacity} slots</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function ProgressMetric({ label, value, color }: { label: string, value: number, color: string }) {
  const safeValue = Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex justify-between items-center">
        <span className="text-[10px] font-mono text-slate-400 uppercase tracking-widest">{label}</span>
        <span className="text-xs font-mono font-medium text-slate-200">{safeValue.toFixed(1)}%</span>
      </div>
      <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${safeValue}%` }} />
      </div>
    </div>
  );
}

function MetricCard({ label, value, trend, good, highlight }: any) {
  return (
    <div className={`p-2.5 rounded border ${highlight ? 'bg-amber-950/20 border-amber-900/50' : 'bg-[#030508] border-slate-800'} flex flex-col gap-1`}>
      <span className="text-[9px] font-mono text-slate-500 uppercase tracking-widest leading-none">{label}</span>
      <div className="flex items-end justify-between">
        <span className={`text-sm font-mono font-medium leading-none ${highlight ? 'text-amber-400' : 'text-slate-200'}`}>{value}</span>
        {trend && (
          <span className={`text-[9px] font-mono leading-none ${good ? 'text-emerald-400' : 'text-amber-400'}`}>
            {trend}
          </span>
        )}
      </div>
    </div>
  );
}
