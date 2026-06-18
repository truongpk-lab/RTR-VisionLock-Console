import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { apiUrl, wsUrl } from "./api";

export type TrackingState =
  | "INIT"
  | "CAMERA_READY"
  | "TARGET_SELECTION"
  | "POINT_PROMPT"
  | "CANDIDATE_TRACKING"
  | "LEARNING_TARGET"
  | "LOCKED_TRACKING"
  | "TRACKING"
  | "STABLE"
  | "UNCERTAIN"
  | "LOST"
  | "SEARCHING"
  | "REACQUIRED"
  | "STOPPED"
  | "ERROR"
  | "OFFLINE";

export interface CandidateBox {
  id: string;
  track_id?: string;
  bbox: [number, number, number, number];
  score: number;
  class_id?: number | null;
  class_name?: string;
  source?: "yolo" | "opencv";
  refined?: boolean;
  mask_quality?: number;
  identity_score?: number;
  negative_margin?: number;
  similarity?: number;
  motion?: number;
  motion_score?: number;
  reid_score?: number;
  is_distractor?: boolean;
  mask_polygon?: [number, number][] | null;
}

export interface SessionMetrics {
  fps: number;
  latency_ms: number;
  gpu: string;
  camera: string;
  track_score: number;
  confidence: number;
  similarity: number;
  mask_iou: number;
  kalman_error: number;
  motion: string;
  candidates: number;
}

export interface SessionLog {
  time: string;
  module: string;
  level: string;
  message: string;
  state: string;
}

export interface SessionTracking {
  mode: string;
  confidence_state: "LOCKED" | "UNCERTAIN" | "LOST";
  normal_backbone: string;
  refind_backbone: string;
  tracker_fallback: boolean;
  refind_fallback: boolean;
  reacquire: { confirming: number; need: number; detect_hz: number };
}

export interface SessionDebug {
  tracker_backend: string;
  tracker_fallback: boolean;
  refind_backend: string;
  refind_fallback: boolean;
  proposal_source: string;
  lost_age_sec: number;
  negative_similarity: number;
  positive_negative_margin: number;
  reacquire_score: number;
  ego_motion_ok: boolean;
  ego_motion_inlier_ratio: number;
}

export interface SessionSnapshot {
  app: string;
  state: TrackingState;
  frame: string;
  frame_size: [number, number];
  target_bbox: [number, number, number, number] | null;
  kalman_bbox: [number, number, number, number] | null;
  target_mask?: [number, number][] | null;
  candidate_boxes: CandidateBox[];
  selected_candidate_id: string | null;
  learning: {
    active: boolean;
    samples: number;
    elapsed: number;
    duration: number;
  };
  proposal: {
    backend: string;
    model_ready: boolean;
    path?: string;
    input_size?: number;
    conf?: number;
    iou?: number;
    last_error?: string;
  };
  segmenter: {
    backend: string;
    model_ready: boolean;
    checkpoint?: string;
    refine_interval?: number;
    video_memory_window?: number;
    last_error?: string;
  };
  tracking?: SessionTracking;
  debug?: SessionDebug;
  metrics: SessionMetrics;
  memory: {
    base_id: string;
    feature_dim: number;
    ram_slots: number;
    ram_capacity: number;
    drm_slots: number;
    drm_capacity: number;
    positive_slots?: number;
    negative_slots?: number;
    negative_capacity?: number;
    identity_backend?: string;
    identity_margin?: number;
    ram_enabled: boolean;
    drm_enabled: boolean;
  };
  logs: SessionLog[];
  timeline: SessionLog[];
  prompt: string;
}

interface TrackingContextValue {
  connected: boolean;
  session: SessionSnapshot;
  startCamera: (source?: string) => Promise<void>;
  stopCamera: () => Promise<void>;
  uploadVideo: (file: File) => Promise<{ path: string; name: string }>;
  selectTarget: () => Promise<void>;
  segmentTarget: (point: { x: number; y: number }) => Promise<void>;
  selectBox: (bbox: [number, number, number, number]) => Promise<void>;
  pickTarget: (candidateId?: string, point?: { x: number; y: number }) => Promise<void>;
  lockTarget: (candidateId?: string, point?: { x: number; y: number }) => Promise<void>;
  resetTracking: () => Promise<void>;
  forceReacquire: () => Promise<void>;
  applyPrompt: (prompt: string) => Promise<void>;
  getConfig: () => Promise<Record<string, any>>;
  patchConfig: (patch: Record<string, any>) => Promise<Record<string, any>>;
  saveConfig: () => Promise<{ saved: string }>;
}

const emptyMetrics: SessionMetrics = {
  fps: 0,
  latency_ms: 0,
  gpu: "N/A",
  camera: "OFFLINE",
  track_score: 0,
  confidence: 0,
  similarity: 0,
  mask_iou: 0,
  kalman_error: 0,
  motion: "IDLE",
  candidates: 0,
};

const fallbackSession: SessionSnapshot = {
  app: "RTR VisionLock Console",
  state: "OFFLINE",
  frame: "",
  frame_size: [0, 0],
  target_bbox: null,
  kalman_bbox: null,
  candidate_boxes: [],
  selected_candidate_id: null,
  learning: { active: false, samples: 0, elapsed: 0, duration: 2.5 },
  proposal: { backend: "opencv", model_ready: false },
  segmenter: { backend: "grabcut", model_ready: false, checkpoint: "", refine_interval: 8 },
  metrics: emptyMetrics,
  memory: {
    base_id: "TGT-8842-A",
    feature_dim: 1024,
    ram_slots: 0,
    ram_capacity: 8,
    drm_slots: 0,
    drm_capacity: 8,
    positive_slots: 0,
    negative_slots: 0,
    negative_capacity: 8,
    identity_backend: "hsv_shape",
    identity_margin: 0,
    ram_enabled: true,
    drm_enabled: true,
  },
  logs: [],
  timeline: [],
  prompt: "",
};

const TrackingContext = createContext<TrackingContextValue | null>(null);

async function apiPost(path: string, body?: unknown) {
  const response = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`${path} failed with ${response.status}`);
  }
  return response.json();
}

async function apiGet(path: string) {
  const response = await fetch(apiUrl(path));
  if (!response.ok) throw new Error(`${path} failed with ${response.status}`);
  return response.json();
}

async function apiPatch(path: string, body: unknown) {
  const response = await fetch(apiUrl(path), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${path} failed with ${response.status}`);
  return response.json();
}

export function TrackingSessionProvider({ children }: { children: ReactNode }) {
  const [connected, setConnected] = useState(false);
  const [session, setSession] = useState<SessionSnapshot>(fallbackSession);

  useEffect(() => {
    let closed = false;
    let retry: number | undefined;
    let socket: WebSocket | undefined;

    function connect() {
      socket = new WebSocket(wsUrl("/ws/session"));

      socket.onopen = () => setConnected(true);
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data) as SessionSnapshot;
        setSession({ ...fallbackSession, ...data });
      };
      socket.onclose = () => {
        setConnected(false);
        if (!closed) retry = window.setTimeout(connect, 1200);
      };
      socket.onerror = () => {
        setConnected(false);
        socket?.close();
      };
    }

    connect();
    return () => {
      closed = true;
      if (retry) window.clearTimeout(retry);
      socket?.close();
    };
  }, []);

  const refreshFromAction = useCallback((next: SessionSnapshot) => {
    setSession((current) => ({ ...current, ...next, frame: current.frame || next.frame }));
  }, []);

  const value = useMemo<TrackingContextValue>(
    () => ({
      connected,
      session,
      startCamera: async (source?: string) => refreshFromAction(await apiPost("/api/camera/start", { source: source || undefined })),
      stopCamera: async () => refreshFromAction(await apiPost("/api/camera/stop")),
      uploadVideo: async (file: File) => {
        // Raw-body upload (no multipart): the backend streams these bytes to a
        // temp file and returns its path, which RUN VIDEO feeds to startCamera.
        const response = await fetch(apiUrl(`/api/camera/upload?filename=${encodeURIComponent(file.name)}`), {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream" },
          body: file,
        });
        if (!response.ok) {
          throw new Error(`/api/camera/upload failed with ${response.status}`);
        }
        return response.json() as Promise<{ path: string; name: string }>;
      },
      selectTarget: async () => refreshFromAction(await apiPost("/api/target/select")),
      segmentTarget: async (point: { x: number; y: number }) =>
        refreshFromAction(await apiPost("/api/target/segment", { point })),
      selectBox: async (bbox: [number, number, number, number]) =>
        refreshFromAction(await apiPost("/api/target/box", { bbox })),
      pickTarget: async (candidateId?: string, point?: { x: number; y: number }) =>
        refreshFromAction(await apiPost("/api/target/pick", { candidate_id: candidateId, point })),
      lockTarget: async (candidateId?: string, point?: { x: number; y: number }) =>
        refreshFromAction(await apiPost("/api/target/lock", { candidate_id: candidateId, point })),
      resetTracking: async () => refreshFromAction(await apiPost("/api/tracking/reset")),
      forceReacquire: async () => refreshFromAction(await apiPost("/api/reacquire/force")),
      applyPrompt: async (prompt: string) => refreshFromAction(await apiPost("/api/prompt/apply", { prompt })),
      getConfig: async () => apiGet("/api/config"),
      patchConfig: async (patch: Record<string, any>) => apiPatch("/api/config", patch),
      saveConfig: async () => apiPost("/api/config/save"),
    }),
    [connected, refreshFromAction, session],
  );

  return <TrackingContext.Provider value={value}>{children}</TrackingContext.Provider>;
}

export function useTrackingSession() {
  const context = useContext(TrackingContext);
  if (!context) {
    throw new Error("useTrackingSession must be used inside TrackingSessionProvider");
  }
  return context;
}
