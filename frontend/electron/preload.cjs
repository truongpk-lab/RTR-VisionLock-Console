// Bridges the backend base URL (chosen at runtime by the main process) into the
// renderer so the frontend can reach the API/WebSocket regardless of which port
// was free and whether the page was loaded from file:// or the dev server.
const { contextBridge } = require("electron");

const arg = process.argv.find((a) => a.startsWith("--rtr-api-base="));
const apiBase = arg ? arg.replace("--rtr-api-base=", "") : "http://127.0.0.1:8000";

contextBridge.exposeInMainWorld("rtrDesktop", {
  apiBase,
  isDesktop: true,
});
