import { TerminalSquare, AlertTriangle, Info, CheckCircle2 } from "lucide-react";
import { useTrackingSession } from "../lib/trackingSession";

export function BottomConsole() {
  const { session } = useTrackingSession();
  const logs = session.logs.length
    ? [...session.logs].reverse()
    : [
        { time: "--:--:--.---", level: "INFO", message: "Backend offline. Start the RTR VisionLock API to stream live logs.", module: "UI", state: "OFFLINE" },
      ];
  const timeline = session.timeline.length ? session.timeline.slice(-8) : logs.slice(0, 1);

  return (
    <div className="h-full flex flex-col">
      {/* Timeline Strip */}
      <div className="h-8 border-b border-slate-800/60 flex items-center px-4 gap-2 overflow-hidden relative bg-slate-900/30">
        <div className="text-[10px] font-mono text-slate-500 uppercase tracking-widest absolute left-4 z-10 bg-[#0a0f16] pr-2">
          TIMELINE
        </div>
        <div className="flex-1 ml-20 relative h-full flex items-center">
           {/* Timeline background line */}
           <div className="absolute left-0 right-0 h-px bg-slate-800"></div>
           
           {/* Timeline Events */}
           {timeline.map((item, index) => (
             <div
               key={`${item.time}-${index}`}
               className={`absolute w-2 h-2 rounded-full top-1/2 -translate-y-1/2 shadow-[0_0_5px_currentColor] ${
                 item.level === "ERROR" ? "bg-rose-500 text-rose-500" : item.level === "WARN" ? "bg-amber-500 text-amber-500" : item.level === "SUCCESS" ? "bg-emerald-500 text-emerald-500" : "bg-cyan-500 text-cyan-500"
               }`}
               style={{ left: `${10 + index * 10}%` }}
             />
           ))}
           
           {/* Current Time Indicator */}
           <div className="absolute left-[45%] top-0 bottom-0 w-px bg-cyan-400">
             <div className="absolute -top-1 -translate-x-1/2 text-[8px] font-mono text-cyan-400 bg-[#0a0f16] px-1">NOW</div>
           </div>
        </div>
      </div>

      {/* Log Console */}
      <div className="flex-1 flex flex-col bg-[#030508] relative overflow-hidden">
        <div className="absolute top-2 left-4 flex items-center gap-2 opacity-50 pointer-events-none">
          <TerminalSquare size={12} className="text-slate-500" />
          <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">SYS_LOG</span>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4 pt-6 font-mono text-xs flex flex-col-reverse gap-1.5">
          {logs.map((log, i) => (
            <div key={i} className="flex items-start gap-3 hover:bg-slate-800/30 px-2 py-1 rounded transition-colors group">
              <span className="text-slate-600 flex-shrink-0 group-hover:text-slate-500">{log.time}</span>
              <span className="text-slate-500 w-16 flex-shrink-0">[{log.module}]</span>
              
              {log.level === 'INFO' && <Info size={14} className="text-blue-400 mt-0.5 flex-shrink-0" />}
              {log.level === 'WARN' && <AlertTriangle size={14} className="text-amber-400 mt-0.5 flex-shrink-0" />}
              {log.level === 'ERROR' && <AlertTriangle size={14} className="text-rose-500 mt-0.5 flex-shrink-0" />}
              {log.level === 'SUCCESS' && <CheckCircle2 size={14} className="text-emerald-400 mt-0.5 flex-shrink-0" />}
              
              <span className={`
                ${log.level === 'INFO' ? 'text-slate-300' : ''}
                ${log.level === 'WARN' ? 'text-amber-200' : ''}
                ${log.level === 'ERROR' ? 'text-rose-400' : ''}
                ${log.level === 'SUCCESS' ? 'text-emerald-300' : ''}
              `}>
                {log.message}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
