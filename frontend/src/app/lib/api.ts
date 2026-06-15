// Resolves where the backend lives across the three run modes:
//   * Desktop shell (Electron): preload injects window.rtrDesktop.apiBase with
//     the dynamically chosen port -> talk to it directly.
//   * Browser dev (vite): use relative paths so the dev proxy forwards them.
//   * Static file:// without the bridge: fall back to the default backend port.

declare global {
  interface Window {
    rtrDesktop?: { apiBase: string; isDesktop: boolean };
  }
}

const desktopBase = typeof window !== "undefined" ? window.rtrDesktop?.apiBase : undefined;
const isFileProtocol = typeof window !== "undefined" && window.location.protocol === "file:";

export const API_BASE = desktopBase ?? (isFileProtocol ? "http://127.0.0.1:8000" : "");

export function wsUrl(path: string): string {
  if (API_BASE) {
    return `${API_BASE.replace(/^http/, "ws")}${path}`;
  }
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}${path}`;
}

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}
