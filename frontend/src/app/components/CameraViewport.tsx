import { useState, useEffect, useRef, useCallback } from "react";
import type { MouseEvent } from "react";
import { Crosshair, Maximize2, Minimize2 } from "lucide-react";
import { useTrackingSession } from "../lib/trackingSession";

export function CameraViewport() {
  const [frame, setFrame] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const viewportRef = useRef<HTMLDivElement>(null);
  const { connected, session, pickTarget, segmentTarget } = useTrackingSession();
  const trackingState = session.state;
  const pointPrompt = trackingState === "POINT_PROMPT";
  const selecting = pointPrompt || trackingState === "CANDIDATE_TRACKING" || trackingState === "TARGET_SELECTION";

  // Animate kalman prediction slightly
  useEffect(() => {
    const interval = setInterval(() => {
      setFrame(f => f + 1);
    }, 50);
    return () => clearInterval(interval);
  }, []);

  // Toggle real fullscreen on the camera viewport element. Works in the Electron
  // desktop shell and in a browser; ESC exits via the platform default, and the
  // fullscreenchange listener keeps our UI state in sync.
  const toggleFullscreen = useCallback((event?: MouseEvent) => {
    event?.stopPropagation();
    const el = viewportRef.current;
    if (!el) return;
    if (document.fullscreenElement) {
      document.exitFullscreen?.();
    } else {
      el.requestFullscreen?.();
    }
  }, []);

  useEffect(() => {
    const onChange = () => setIsFullscreen(document.fullscreenElement === viewportRef.current);
    document.addEventListener("fullscreenchange", onChange);
    // Explicit ESC handling so it exits even if focus is on an inner element.
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && document.fullscreenElement) document.exitFullscreen?.();
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("fullscreenchange", onChange);
      document.removeEventListener("keydown", onKey);
    };
  }, []);

  const offsetX = Math.sin(frame * 0.1) * 2;
  const offsetY = Math.cos(frame * 0.1) * 2;
  const hasLiveFrame = connected && session.frame;
  const frameSrc = hasLiveFrame
    ? `data:image/jpeg;base64,${session.frame}`
    : "https://images.unsplash.com/photo-1472146936668-d987bf0a6e38?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxhZXJpYWwlMjBjaXR5JTIwaW50ZXJzZWN0aW9uJTIwbmlnaHQlMjBkcm9uZXxlbnwxfHx8fDE3ODE1MDQ1NTd8MA&ixlib=rb-4.1.0&q=80&w=1080";

  function handleViewportClick(event: MouseEvent<HTMLDivElement>) {
    if (!connected || !selecting) return;
    const [frameWidth, frameHeight] = session.frame_size;
    if (!frameWidth || !frameHeight) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = Math.round(((event.clientX - rect.left) / rect.width) * frameWidth);
    const y = Math.round(((event.clientY - rect.top) / rect.height) * frameHeight);
    if (pointPrompt) {
      // Click-to-segment: SAM/GrabCut returns one object box at the click.
      segmentTarget({ x, y });
    } else {
      // Legacy auto mode: click a moving candidate box to pick it.
      pickTarget(undefined, { x, y });
    }
  }

  return (
    <div
      ref={viewportRef}
      className={`absolute inset-0 w-full h-full bg-black ${selecting ? "cursor-crosshair" : ""}`}
      onClick={handleViewportClick}
    >
      {/* Background Video/Image Feed */}
      <img
        src={frameSrc}
        alt="Camera Feed"
        className={`w-full h-full opacity-80 ${isFullscreen ? "object-contain" : "object-cover"}`}
      />
      
      {/* Dark overlay for better UI contrast */}
      <div className="absolute inset-0 bg-slate-900/40 mix-blend-multiply"></div>

      {/* Tactical Grid */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(14,165,233,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(14,165,233,0.05)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none">
      </div>
      
      {/* Center Crosshair */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-cyan-500/30 pointer-events-none">
        <Crosshair size={120} strokeWidth={0.5} />
      </div>

      {/* Overlays based on state */}
      {hasLiveFrame && (
        <LiveOverlays session={session} />
      )}

      {!hasLiveFrame && trackingState === "STABLE" && (
        <StableOverlays offsetX={offsetX} offsetY={offsetY} />
      )}
      
      {!hasLiveFrame && trackingState === "SEARCHING" && (
        <ReacquiringOverlays frame={frame} />
      )}
      
      {!hasLiveFrame && trackingState === "TARGET_SELECTION" && (
        <div className="absolute inset-0 cursor-crosshair">
          <div className="absolute top-1/3 left-1/3 w-32 h-24 border border-cyan-400/50 bg-cyan-400/10"></div>
        </div>
      )}

      {/* Fullscreen toggle (top-left) */}
      <button
        onClick={toggleFullscreen}
        title={isFullscreen ? "Exit fullscreen (Esc)" : "Fullscreen camera"}
        className="absolute top-4 left-4 z-20 flex items-center gap-2 px-2.5 py-1.5 bg-slate-900/80 hover:bg-slate-800 border border-slate-600/60 hover:border-cyan-700/60 text-slate-300 hover:text-cyan-300 font-mono text-[10px] uppercase tracking-widest rounded-sm backdrop-blur-sm transition-colors"
      >
        {isFullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
        <span className="hidden sm:inline">{isFullscreen ? "Exit" : "Fullscreen"}</span>
      </button>

      {/* ESC hint while in fullscreen */}
      {isFullscreen && (
        <div className="absolute bottom-4 right-4 z-20 px-3 py-1 bg-slate-900/80 border border-slate-600/50 text-slate-400 font-mono text-[10px] uppercase tracking-widest rounded-sm backdrop-blur-sm pointer-events-none">
          Press ESC to exit
        </div>
      )}

      {/* Status Overlay UI */}
      <div className="absolute top-4 right-4 flex items-center gap-2">
         {trackingState === "SEARCHING" ? (
           <div className="px-3 py-1 bg-amber-500/20 border border-amber-500/50 text-amber-400 font-mono text-xs uppercase tracking-widest animate-pulse flex items-center gap-2 rounded-sm backdrop-blur-sm">
             <span className="w-2 h-2 bg-amber-400 rounded-full"></span>
             TARGET LOST - SEARCHING
           </div>
         ) : trackingState === "POINT_PROMPT" ? (
           <div className="px-3 py-1 bg-cyan-500/20 border border-cyan-500/50 text-cyan-300 font-mono text-xs uppercase tracking-widest flex items-center gap-2 rounded-sm backdrop-blur-sm">
             <span className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></span>
             CLICK OBJECT · {session.segmenter.model_ready ? "SAM" : "GRABCUT"}
           </div>
         ) : trackingState === "CANDIDATE_TRACKING" ? (
           <div className="px-3 py-1 bg-cyan-500/20 border border-cyan-500/50 text-cyan-300 font-mono text-xs uppercase tracking-widest flex items-center gap-2 rounded-sm backdrop-blur-sm">
             <span className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></span>
             CLICK A TARGET
           </div>
         ) : trackingState === "LEARNING_TARGET" ? (
           <div className="px-3 py-1 bg-emerald-500/20 border border-emerald-500/50 text-emerald-300 font-mono text-xs uppercase tracking-widest flex items-center gap-2 rounded-sm backdrop-blur-sm">
             <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></span>
             LEARNING TARGET
           </div>
         ) : trackingState === "STABLE" || trackingState === "LOCKED_TRACKING" ? (
           <div className="px-3 py-1 bg-cyan-500/20 border border-cyan-500/50 text-cyan-400 font-mono text-xs uppercase tracking-widest flex items-center gap-2 rounded-sm backdrop-blur-sm">
             <span className="w-2 h-2 bg-cyan-400 rounded-full"></span>
             TRK-LOCKED
           </div>
         ) : (
           <div className="px-3 py-1 bg-slate-800/80 border border-slate-600/50 text-slate-300 font-mono text-xs uppercase tracking-widest rounded-sm backdrop-blur-sm">
             {connected ? trackingState : "BACKEND OFFLINE"}
           </div>
         )}
      </div>
      
      <div className="absolute bottom-4 left-4 text-cyan-500/60 font-mono text-[10px] tracking-widest">
        FOV: 45.2 deg | Z: 1.0x | PTZ: 0.0, 0.0
      </div>
    </div>
  );
}

function LiveOverlays({ session }: { session: ReturnType<typeof useTrackingSession>["session"] }) {
  const [frameWidth, frameHeight] = session.frame_size;
  if (!frameWidth || !frameHeight) return null;

  const toStyle = (bbox: [number, number, number, number]) => ({
    left: `${(bbox[0] / frameWidth) * 100}%`,
    top: `${(bbox[1] / frameHeight) * 100}%`,
    width: `${(bbox[2] / frameWidth) * 100}%`,
    height: `${(bbox[3] / frameHeight) * 100}%`,
  });

  return (
    <div className="absolute inset-0 pointer-events-none">
      {session.candidate_boxes.map((candidate) => (
        <div
          key={candidate.id}
          className={`absolute border border-dashed ${
            session.selected_candidate_id === candidate.id ? "border-amber-400 bg-amber-400/20" : "border-cyan-400/60 bg-cyan-400/10"
          }`}
          style={toStyle(candidate.bbox)}
        >
          <div className="absolute -top-4 left-0 bg-slate-900/90 text-cyan-300 text-[8px] font-mono px-1">
            {candidate.id}: {(candidate.reid_score ?? candidate.score).toFixed(2)}
          </div>
        </div>
      ))}

      {session.target_bbox && (
        <div
          className={`absolute border-2 ${
            session.state === "LEARNING_TARGET"
              ? "border-emerald-400 bg-emerald-400/10 shadow-[0_0_10px_rgba(52,211,153,0.5)]"
              : "border-cyan-400 bg-cyan-400/10 shadow-[0_0_10px_rgba(34,211,238,0.5)]"
          }`}
          style={toStyle(session.target_bbox)}
        >
          {session.state === "LEARNING_TARGET" ? (
            <div className="absolute -top-5 left-0 bg-emerald-400 text-black text-[9px] font-mono font-bold px-1 uppercase tracking-wider">
              LEARNING {session.learning.samples} · {session.learning.elapsed.toFixed(1)}/{session.learning.duration.toFixed(1)}s
            </div>
          ) : (
            <div className="absolute -top-5 left-0 bg-cyan-400 text-black text-[9px] font-mono font-bold px-1 uppercase tracking-wider">
              {session.memory.base_id} {session.metrics.track_score.toFixed(2)}
            </div>
          )}
        </div>
      )}

      {session.kalman_bbox && (
        <div
          className="absolute border border-dashed border-amber-400/70 bg-amber-400/5"
          style={toStyle(session.kalman_bbox)}
        >
          <div className="absolute -bottom-4 right-0 text-amber-400/70 text-[8px] font-mono uppercase tracking-widest">
            PREDICT
          </div>
        </div>
      )}
    </div>
  );
}

function StableOverlays({ offsetX, offsetY }: { offsetX: number, offsetY: number }) {
  // Hardcoded target position for demo
  const tx = 45; // %
  const ty = 55; // %
  const tw = 12; // %
  const th = 16; // %
  
  return (
    <div className="absolute inset-0 pointer-events-none">
      {/* Main Bounding Box */}
      <div 
        className="absolute border-2 border-cyan-400 bg-cyan-400/10 shadow-[0_0_10px_rgba(34,211,238,0.5)] transition-all duration-75"
        style={{ left: `${tx}%`, top: `${ty}%`, width: `${tw}%`, height: `${th}%` }}
      >
        {/* Corner Accents */}
        <div className="absolute -top-1 -left-1 w-2 h-2 border-t-2 border-l-2 border-cyan-300"></div>
        <div className="absolute -top-1 -right-1 w-2 h-2 border-t-2 border-r-2 border-cyan-300"></div>
        <div className="absolute -bottom-1 -left-1 w-2 h-2 border-b-2 border-l-2 border-cyan-300"></div>
        <div className="absolute -bottom-1 -right-1 w-2 h-2 border-b-2 border-r-2 border-cyan-300"></div>
        
        {/* Label */}
        <div className="absolute -top-5 left-0 bg-cyan-400 text-black text-[9px] font-mono font-bold px-1 uppercase tracking-wider">
          ID:8842 0.92
        </div>
      </div>
      
      {/* Segmentation Mask Fake Overlay */}
      <div 
        className="absolute bg-emerald-500/30 blur-sm rounded-[40%_60%_70%_30%/40%_50%_60%_50%] transition-all duration-75"
        style={{ left: `${tx + 1}%`, top: `${ty + 1}%`, width: `${tw - 2}%`, height: `${th - 2}%` }}
      ></div>

      {/* Kalman Prediction Dashed Box */}
      <div 
        className="absolute border border-dashed border-amber-400/70 bg-amber-400/5 transition-all duration-75"
        style={{ 
          left: `calc(${tx}% + ${offsetX}px)`, 
          top: `calc(${ty}% + ${offsetY}px)`, 
          width: `${tw}%`, 
          height: `${th}%` 
        }}
      >
        <div className="absolute -bottom-4 right-0 text-amber-400/70 text-[8px] font-mono uppercase tracking-widest">
          PREDICT
        </div>
      </div>

      {/* Motion Vector Line */}
      <svg className="absolute inset-0 w-full h-full overflow-visible">
        <line 
          x1={`${tx + tw/2}%`} 
          y1={`${ty + th/2}%`} 
          x2={`calc(${tx + tw/2}% + ${offsetX * 10}px)`} 
          y2={`calc(${ty + th/2}% + ${offsetY * 10}px)`} 
          stroke="rgba(34,211,238,0.8)" 
          strokeWidth="1.5"
          markerEnd="url(#arrowhead)"
        />
        <defs>
          <marker id="arrowhead" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <polygon points="0 0, 6 3, 0 6" fill="rgba(34,211,238,0.8)" />
          </marker>
        </defs>
      </svg>
    </div>
  );
}

function ReacquiringOverlays({ frame }: { frame: number }) {
  // Simulate candidate boxes
  const candidates = [
    { x: 30, y: 40, w: 10, h: 14, score: 0.65 },
    { x: 50, y: 60, w: 11, h: 15, score: 0.72 },
    { x: 65, y: 35, w: 9, h: 12, score: 0.45 },
    { x: 45, y: 55, w: 12, h: 16, score: 0.81, isBest: true },
  ];

  const searchRadius = 25 + Math.sin(frame * 0.05) * 5;

  return (
    <div className="absolute inset-0 pointer-events-none">
      {/* Search Area */}
      <div 
        className="absolute border border-amber-500/30 rounded-full bg-amber-500/5 -translate-x-1/2 -translate-y-1/2"
        style={{ left: '45%', top: '55%', width: `${searchRadius}%`, height: `${searchRadius * 1.5}%` }}
      >
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-[1px] bg-amber-500/20 rotate-45"></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-[1px] bg-amber-500/20 -rotate-45"></div>
      </div>

      {/* Candidates */}
      {candidates.map((c, i) => (
        <div 
          key={i}
          className={`absolute border border-dashed ${c.isBest ? 'border-amber-400 bg-amber-400/20 z-10' : 'border-slate-500 bg-slate-500/10'}`}
          style={{ left: `${c.x}%`, top: `${c.y}%`, width: `${c.w}%`, height: `${c.h}%` }}
        >
          <div className={`absolute -top-4 left-0 text-[8px] font-mono px-1 ${c.isBest ? 'bg-amber-400 text-black' : 'bg-slate-700 text-slate-300'}`}>
            C{i}: {c.score.toFixed(2)}
          </div>
        </div>
      ))}
    </div>
  );
}
