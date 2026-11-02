import React from "react";
import ReactDOM from "react-dom/client";
import { invoke } from "@tauri-apps/api/core";
import { attachConsole } from "@tauri-apps/plugin-log";
import App from "./App";
import "./theme.css";

function reportCrash(source: string, message: string, detail?: string) {
  try {
    void invoke("write_crash_report", { source, message, detail }).catch(
      () => {},
    );
  } catch {
    // Browser preview without the Tauri runtime.
  }
}

window.addEventListener("error", (event) => {
  reportCrash("window", event.message, event.error?.stack ?? "");
});

window.addEventListener("unhandledrejection", (event) => {
  const reason = event.reason;
  reportCrash(
    "unhandledrejection",
    reason instanceof Error ? reason.message : String(reason),
    reason instanceof Error ? (reason.stack ?? "") : "",
  );
});

try {
  void attachConsole().catch(() => {});
} catch {
  // Browser preview without the Tauri runtime.
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
