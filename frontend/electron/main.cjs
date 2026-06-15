// RTR VisionLock Console - Electron desktop shell.
//
// Responsibilities:
//   * Enforce a single running instance (focus existing window instead of
//     opening a second one -> avoids port / state conflicts).
//   * Pick a free TCP port and launch the Python backend on it, then kill the
//     backend cleanly when the app quits (no orphaned uvicorn = no port clash).
//   * Load the built frontend (production) or the Vite dev server (RTR_DEV=1)
//     and hand the chosen backend port to the renderer via the preload bridge.

const { app, BrowserWindow, shell } = require("electron");
const { spawn } = require("child_process");
const net = require("net");
const path = require("path");
const http = require("http");

const isDev = process.env.RTR_DEV === "1";
const projectRoot = path.resolve(__dirname, "..", "..");
const backendDir = path.join(projectRoot, "backend");
const frontendDir = path.resolve(__dirname, "..");
const devServerUrl = process.env.RTR_DEV_URL || "http://127.0.0.1:5173";

let mainWindow = null;
let backendProcess = null;
let backendPort = 8000;
let shuttingDown = false;

function findFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

function backendPython() {
  // Prefer the project venv; fall back to the system interpreter.
  const venvPy = path.join(backendDir, ".venv", "Scripts", "python.exe");
  return require("fs").existsSync(venvPy) ? venvPy : "python";
}

function startBackend(port) {
  return new Promise((resolve, reject) => {
    backendProcess = spawn(backendPython(), ["-m", "app.main"], {
      cwd: backendDir,
      env: {
        ...process.env,
        RTR_HOST: "127.0.0.1",
        RTR_PORT: String(port),
        RTR_RELOAD: "0",
        PYTHONUNBUFFERED: "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true,
    });

    backendProcess.stdout.on("data", (d) => process.stdout.write(`[backend] ${d}`));
    backendProcess.stderr.on("data", (d) => process.stderr.write(`[backend] ${d}`));
    backendProcess.on("error", reject);
    backendProcess.on("exit", (code) => {
      backendProcess = null;
      if (!shuttingDown && code !== 0) {
        console.error(`Backend exited unexpectedly with code ${code}`);
      }
    });

    waitForHealth(port, 40, resolve, reject);
  });
}

function waitForHealth(port, attemptsLeft, resolve, reject) {
  const req = http.get({ host: "127.0.0.1", port, path: "/health", timeout: 1000 }, (res) => {
    res.resume();
    if (res.statusCode === 200) resolve();
    else retryHealth(port, attemptsLeft, resolve, reject);
  });
  req.on("error", () => retryHealth(port, attemptsLeft, resolve, reject));
  req.on("timeout", () => {
    req.destroy();
    retryHealth(port, attemptsLeft, resolve, reject);
  });
}

function retryHealth(port, attemptsLeft, resolve, reject) {
  if (attemptsLeft <= 0) {
    reject(new Error("Backend did not become healthy in time."));
    return;
  }
  setTimeout(() => waitForHealth(port, attemptsLeft - 1, resolve, reject), 500);
}

function stopBackend() {
  if (!backendProcess) return;
  const proc = backendProcess;
  backendProcess = null;
  try {
    if (process.platform === "win32") {
      spawn("taskkill", ["/pid", String(proc.pid), "/T", "/F"], { windowsHide: true });
    } else {
      proc.kill("SIGTERM");
    }
  } catch (err) {
    console.error("Failed to stop backend:", err);
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 680,
    backgroundColor: "#05090f",
    show: false,
    title: "RTR VisionLock Console",
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      // The renderer talks to a trusted local backend; allow it to bridge the port.
      additionalArguments: [`--rtr-api-base=http://127.0.0.1:${backendPort}`],
    },
  });

  mainWindow.once("ready-to-show", () => mainWindow.show());

  // Open external links in the system browser, never in-app.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  if (isDev) {
    mainWindow.loadURL(devServerUrl);
    mainWindow.webContents.openDevTools({ mode: "detach" });
  } else {
    mainWindow.loadFile(path.join(frontendDir, "dist", "index.html"));
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(async () => {
    try {
      backendPort = await findFreePort();
      await startBackend(backendPort);
    } catch (err) {
      console.error("Failed to start backend:", err);
      // Still open the window; the UI shows an OFFLINE state and keeps retrying.
    }
    createWindow();

    app.on("activate", () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  });
}

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  shuttingDown = true;
  stopBackend();
});

process.on("exit", stopBackend);
