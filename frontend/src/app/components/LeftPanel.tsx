import { Play, Square, Crosshair, Lock, RotateCcw, ScanSearch, Upload } from "lucide-react";
import { useRef, useState, type ChangeEvent } from "react";
import { useTrackingSession } from "../lib/trackingSession";

export function LeftPanel() {
  const { connected, session, startCamera, stopCamera, uploadVideo, selectTarget, pickTarget, resetTracking, forceReacquire, applyPrompt } = useTrackingSession();
  const [prompt, setPrompt] = useState("Silver SUV, plate ending in 8X");
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [video, setVideo] = useState<{ path: string; name: string } | null>(null);
  const [importing, setImporting] = useState(false);
  const trackingState = session.state;

  const onPickVideo = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = ""; // let the operator re-pick the same file later
    if (!file) return;
    setImporting(true);
    try {
      setVideo(await uploadVideo(file));
    } catch (error) {
      console.error(error);
    } finally {
      setImporting(false);
    }
  };

  const runVideo = async () => {
    if (!video) return;
    if (session.metrics.camera === "ACTIVE") await stopCamera();
    await startCamera(video.path);
  };

  return (
    <div className="w-64 flex-shrink-0 border-r border-slate-800/60 bg-[#0a0f16] flex flex-col z-10 overflow-y-auto">
      <div className="p-4 border-b border-slate-800/60">
        <h2 className="text-xs font-mono font-bold text-slate-300 uppercase tracking-widest mb-4">
          Target Controls
        </h2>
        
        <div className="flex flex-col gap-2">
          <ActionButton icon={Play} label="START CAMERA" onClick={() => startCamera()} active={session.metrics.camera === "ACTIVE"} disabled={!connected} />
          <ActionButton icon={Square} label="STOP CAMERA" onClick={stopCamera} variant="danger" disabled={!connected} />

          <input ref={fileInputRef} type="file" accept="video/*" className="hidden" onChange={onPickVideo} />
          <ActionButton
            icon={Upload}
            label={importing ? "IMPORTING…" : "IMPORT VIDEO"}
            onClick={() => fileInputRef.current?.click()}
            disabled={!connected || importing}
          />
          {video && (
            <>
              <div className="px-3 text-[10px] font-mono text-slate-500 truncate" title={video.name}>
                {video.name}
              </div>
              <ActionButton icon={Play} label="RUN VIDEO" onClick={runVideo} variant="cyan" disabled={!connected} />
            </>
          )}

          <div className="h-px bg-slate-800 my-2"></div>
          
          <ActionButton
            icon={Crosshair}
            label="SELECT TARGET"
            onClick={selectTarget}
            active={trackingState === "TARGET_SELECTION" || trackingState === "POINT_PROMPT" || trackingState === "CANDIDATE_TRACKING"}
            disabled={!connected}
          />
          <ActionButton
            icon={Lock}
            label="LOCK TARGET"
            onClick={() => pickTarget(session.selected_candidate_id || session.candidate_boxes[0]?.id)}
            active={trackingState === "LEARNING_TARGET" || trackingState === "LOCKED_TRACKING" || trackingState === "STABLE"}
            variant="cyan"
            disabled={!connected || (trackingState !== "CANDIDATE_TRACKING" && trackingState !== "TARGET_SELECTION")}
          />
          <ActionButton 
            icon={RotateCcw} 
            label="RESET TRACKING" 
            onClick={resetTracking} 
            disabled={!connected}
          />
          <ActionButton 
            icon={ScanSearch} 
            label="FORCE RE-ACQUIRE" 
            onClick={forceReacquire}
            active={trackingState === "SEARCHING" || trackingState === "LOST"}
            variant="amber" 
            disabled={!connected}
          />
        </div>
      </div>

      <div className="p-4 flex-1">
        <h2 className="text-xs font-mono font-bold text-slate-300 uppercase tracking-widest mb-4">
          Text-Guided Tracking
        </h2>
        <div className="space-y-3">
          <div className="flex flex-col gap-1.5">
            <label className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">
              Target Description
            </label>
            <textarea 
              className="w-full h-24 bg-[#030508] border border-slate-800 rounded px-3 py-2 text-sm text-slate-300 placeholder:text-slate-600 focus:outline-none focus:border-cyan-800/50 resize-none font-mono"
              placeholder="e.g. 'Red sedan', 'Person in blue jacket'"
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
            />
          </div>
          <button
            className="w-full py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-300 text-xs font-mono uppercase tracking-widest rounded transition-colors border border-slate-700"
            onClick={() => applyPrompt(prompt)}
            disabled={!connected}
          >
            Apply Prompt
          </button>
        </div>
      </div>
    </div>
  );
}

function ActionButton({ icon: Icon, label, onClick, active, variant = "default", disabled }: any) {
  let baseClass = "flex items-center gap-3 w-full px-3 py-2 text-xs font-mono font-medium uppercase tracking-wider rounded border transition-all duration-200";
  
  if (active) {
    if (variant === "cyan") baseClass += " bg-cyan-950/40 text-cyan-400 border-cyan-900/50";
    else if (variant === "amber") baseClass += " bg-amber-950/40 text-amber-400 border-amber-900/50";
    else baseClass += " bg-slate-800 text-slate-200 border-slate-700";
  } else {
    baseClass += " bg-transparent border-transparent hover:bg-slate-800/50 text-slate-400 hover:text-slate-300";
  }

  if (variant === "danger" && !active) {
    baseClass += " hover:bg-rose-950/30 hover:text-rose-400";
  }

  return (
    <button className={`${baseClass} disabled:opacity-50 disabled:cursor-not-allowed`} onClick={onClick} disabled={disabled}>
      <Icon size={14} className={active ? "opacity-100" : "opacity-70"} />
      {label}
    </button>
  );
}
