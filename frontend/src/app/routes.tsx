import { createHashRouter } from "react-router";
import { Layout } from "./components/Layout";
import { LiveDashboard } from "./pages/LiveDashboard";
import { ThresholdConfig } from "./pages/ThresholdConfig";
import { ReacquireConfig } from "./pages/ReacquireConfig";
import { MemoryConfig } from "./pages/MemoryConfig";
import { ModelSettings } from "./pages/ModelSettings";

function DummyPage({ title }: { title: string }) {
  return (
    <div className="flex-1 h-full flex items-center justify-center text-slate-500 font-mono text-sm tracking-widest uppercase">
      {title} Module Offline
    </div>
  );
}

export const router = createHashRouter([
  {
    path: "/",
    Component: Layout,
    children: [
      { index: true, Component: LiveDashboard },
      { path: "target", element: <DummyPage title="Target Setup" /> },
      { path: "threshold", Component: ThresholdConfig },
      { path: "reacquire", Component: ReacquireConfig },
      { path: "memory", Component: MemoryConfig },
      { path: "logs", element: <DummyPage title="Logs & Diagnostics" /> },
      { path: "settings", Component: ModelSettings },
    ],
  },
]);
