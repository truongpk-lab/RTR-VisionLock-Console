import { NavLink } from "react-router";
import { 
  Activity, 
  Crosshair, 
  SlidersHorizontal, 
  Radar, 
  BrainCircuit, 
  Terminal, 
  Settings 
} from "lucide-react";
import { Topbar } from "./Topbar";
import { PageTransition } from "./PageTransition";
import { TrackingSessionProvider } from "../lib/trackingSession";

const navItems = [
  { path: "/", icon: Activity, label: "Live Console" },
  { path: "/target", icon: Crosshair, label: "Target Setup" },
  { path: "/threshold", icon: SlidersHorizontal, label: "Threshold Config" },
  { path: "/reacquire", icon: Radar, label: "Re-acquire Config" },
  { path: "/memory", icon: BrainCircuit, label: "Memory Config" },
  { path: "/logs", icon: Terminal, label: "Logs & Diagnostics" },
  { path: "/settings", icon: Settings, label: "Model Settings" },
];

export function Layout() {
  return (
    <TrackingSessionProvider>
    <div className="flex h-screen w-full bg-[#05090f] text-slate-300 font-sans overflow-hidden selection:bg-cyan-900 selection:text-cyan-50">
      {/* Sidebar */}
      <div className="w-16 md:w-64 flex-shrink-0 bg-[#0a0f16] border-r border-slate-800/60 flex flex-col z-20">
        <div className="h-14 flex items-center justify-center md:justify-start md:px-4 border-b border-slate-800/60">
          <div className="w-8 h-8 rounded bg-cyan-950 border border-cyan-800/50 flex items-center justify-center text-cyan-400">
            <Crosshair size={18} />
          </div>
          <span className="hidden md:block ml-3 font-mono font-bold text-sm tracking-wider text-slate-100">
            RTR VisionLock
          </span>
        </div>
        
        <div className="flex-1 py-4 flex flex-col gap-1 overflow-y-auto px-2">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center px-3 py-2.5 rounded-md transition-colors ${
                  isActive
                    ? "bg-cyan-950/40 text-cyan-400 border border-cyan-900/50"
                    : "text-slate-400 hover:bg-slate-800/50 hover:text-slate-200 border border-transparent"
                }`
              }
            >
              <item.icon size={18} className="flex-shrink-0" />
              <span className="hidden md:block ml-3 text-sm font-medium tracking-wide">
                {item.label}
              </span>
            </NavLink>
          ))}
        </div>

        <div className="p-4 border-t border-slate-800/60">
          <div className="hidden md:flex flex-col gap-1 text-[10px] font-mono text-slate-600 uppercase tracking-widest">
            <span>SYS.OK.</span>
            <span>MEM: 48%</span>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <Topbar />
        <main className="flex-1 overflow-hidden relative">
          <PageTransition />
        </main>
      </div>
    </div>
    </TrackingSessionProvider>
  );
}
