import { createHashRouter } from "react-router";
import { Layout } from "./components/Layout";
import { LiveDashboard } from "./pages/LiveDashboard";

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
      { path: "threshold", element: <DummyPage title="Threshold Config" /> },
      { path: "reacquire", element: <DummyPage title="Re-acquire Config" /> },
      { path: "memory", element: <DummyPage title="Memory Config" /> },
      { path: "logs", element: <DummyPage title="Logs & Diagnostics" /> },
      { path: "settings", element: <DummyPage title="Model Settings" /> },
    ],
  },
]);
