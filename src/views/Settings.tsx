import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  ChevronDown,
  Cpu,
  Download,
  FileCode2,
  FolderOpen,
  Info,
  Loader2,
  RefreshCw,
} from "lucide-react";
import { ThemeToggle } from "../components/ThemeToggle";
import { useAppInfo } from "../hooks/useAppInfo";
import {
  readReconstructionSettings,
  TRAINING_QUALITY_OPTIONS,
  TRAINING_STEPS_MAX,
  TRAINING_STEPS_MIN,
  trainingQualityPreset,
  writeReconstructionSettings,
  type TrainingQuality,
} from "../lib/reconstructionSettings";
import {
  FALLBACK_VERSION,
  useUpdater,
} from "../lib/updater";

interface Diagnostics {
  app: string;
  version: string;
  os: string;
  arch: string;
  cpuCores: number;
  gpu: string | null;
  python: string | null;
  logDir: string;
  crashDir: string;
}

export function Settings() {
  const info = useAppInfo();
  const [gpuBudget, setGpuBudget] = useState("8GB");
  const [settings, setSettings] = useState(() => readReconstructionSettings());
  const [stepsDraft, setStepsDraft] = useState(() => String(settings.steps));
  const [mixedPrecision, setMixedPrecision] = useState(true);
  const [pythonPath, setPythonPath] = useState("");
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null);
  const updater = useUpdater({ autoCheck: true, checkDelay: 1400 });
  const updateState = updater.state;
  const currentVersion = updateState.currentVersion ?? FALLBACK_VERSION;
  const updateBusy =
    updateState.status === "checking" ||
    updateState.status === "downloading" ||
    updateState.status === "installing";

  let updateStatusText = "未检查";
  if (updateState.status === "checking") {
    updateStatusText = "正在检查更新…";
  } else if (updateState.status === "latest") {
    updateStatusText = "已是最新版本";
  } else if (updateState.status === "available") {
    updateStatusText = `发现新版本 v${updateState.version ?? ""}`;
  } else if (updateState.status === "downloading") {
    updateStatusText =
      updateState.progress && updateState.progress > 0
        ? `已下载 ${updateState.progress}%`
        : `正在下载 v${updateState.version ?? ""}`;
  } else if (updateState.status === "installing") {
    updateStatusText = "正在安装，完成后自动重启";
  } else if (updateState.status === "error") {
    updateStatusText =
      updateState.message === "unsupported"
        ? "当前环境不支持在线更新"
        : "检查更新失败";
  }

  const detectEnvironment = async () => {
    try {
      const result = await invoke<Diagnostics>("get_diagnostics");
      setDiagnostics(result);
      setPythonPath(result.python ?? "");
    } catch {
      setDiagnostics(null);
    }
  };

  const openLogs = async () => {
    try {
      await invoke("open_log_dir");
    } catch {
      // Browser preview without the Tauri runtime.
    }
  };

  const commitSteps = () => {
    const parsed = Number(stepsDraft);
    const steps = Number.isFinite(parsed)
      ? Math.min(
          TRAINING_STEPS_MAX,
          Math.max(TRAINING_STEPS_MIN, Math.round(parsed)),
        )
      : settings.steps;
    setStepsDraft(String(steps));
    setSettings((current) => {
      const next = { ...current, steps };
      writeReconstructionSettings(next);
      return next;
    });
  };

  return (
    <>
      <div className="page-header">
        <h1>设置</h1>
      </div>

      <div className="settings-grid">
        <div className="settings-stack">
          <section className="panel">
            <div className="panel__header">
              <h2 className="panel__title">外观</h2>
            </div>
            <div className="field-row">
              <span className="field-row__label">主题模式</span>
              <ThemeToggle />
            </div>
          </section>

          <section className="panel">
            <div className="panel__header">
              <h2 className="panel__title">引擎</h2>
              <FileCode2 size={16} className="panel__icon" aria-hidden="true" />
            </div>
            <div className="field-row field-row--column">
              <span className="field-row__label">Python 环境</span>
              <div className="field-row__control">
                <input
                  className="input field-row__input"
                  value={pythonPath}
                  onChange={(event) => setPythonPath(event.currentTarget.value)}
                  placeholder="未配置"
                />
                <button
                  className="btn btn--secondary"
                  type="button"
                  onClick={detectEnvironment}
                >
                  检测
                </button>
              </div>
            </div>
            <div className="field-row">
              <span className="field-row__label">引擎版本</span>
              <span className="info-row__value mono">{info.engine}</span>
            </div>
          </section>
        </div>

        <div className="settings-stack">
          <section className="panel">
            <div className="panel__header">
              <h2 className="panel__title">重建</h2>
              <Cpu size={16} className="panel__icon" aria-hidden="true" />
            </div>
            <div className="field-row">
              <span className="field-row__label">显存预算</span>
              <div className="select-wrap">
                <select
                  className="input select"
                  value={gpuBudget}
                  onChange={(event) => setGpuBudget(event.currentTarget.value)}
                  aria-label="显存预算"
                >
                  <option>6GB</option>
                  <option>8GB</option>
                  <option>10GB</option>
                  <option>12GB+</option>
                </select>
                <ChevronDown size={16} className="select-chevron" aria-hidden="true" />
              </div>
            </div>
            <div className="field-row">
              <span className="field-row__label">训练画质</span>
              <div className="select-wrap">
                <select
                  className="input select"
                  value={settings.quality}
                  onChange={(event) => {
                    const quality = event.currentTarget
                      .value as TrainingQuality;
                    const next = {
                      ...trainingQualityPreset(quality),
                      steps: settings.steps,
                    };
                    setStepsDraft(String(settings.steps));
                    setSettings(next);
                    writeReconstructionSettings(next);
                  }}
                  aria-label="训练画质"
                >
                  {TRAINING_QUALITY_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <ChevronDown size={16} className="select-chevron" aria-hidden="true" />
              </div>
            </div>
            <div className="field-row">
              <span className="field-row__label">训练步数</span>
              <input
                className="input field-row__number"
                type="number"
                min={TRAINING_STEPS_MIN}
                max={TRAINING_STEPS_MAX}
                step={100}
                value={stepsDraft}
                onChange={(event) => setStepsDraft(event.currentTarget.value)}
                onBlur={commitSteps}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.currentTarget.blur();
                  }
                }}
                aria-label="训练步数"
              />
            </div>
            <div className="field-row">
              <span className="field-row__label">混合精度</span>
              <label className="switch">
                <input
                  type="checkbox"
                  checked={mixedPrecision}
                  onChange={(event) => setMixedPrecision(event.currentTarget.checked)}
                />
                <span className="switch__track">
                  <span className="switch__thumb" />
                </span>
              </label>
            </div>
          </section>

          <section className="panel">
            <div className="panel__header">
              <h2 className="panel__title">关于</h2>
              <Info size={16} className="panel__icon" aria-hidden="true" />
            </div>
            <div className="info-list">
              <div className="info-row">
                <span className="info-row__label">应用</span>
                <span className="info-row__value">{info.name}</span>
              </div>
              <div className="info-row">
                <span className="info-row__label">版本</span>
                <span className="info-row__value mono">{currentVersion}</span>
              </div>
              <div className="info-row">
                <span className="info-row__label">算法</span>
                <span className="info-row__value">{info.engine}</span>
              </div>
              <div className="info-row">
                <span className="info-row__label">显存预算</span>
                <span className="info-row__value mono">{info.gpuBudget}</span>
              </div>
              <div className="info-row">
                <span className="info-row__label">平台</span>
                <span className="info-row__value">{info.targetPlatform}</span>
              </div>
              {diagnostics && (
                <>
                  <div className="info-row">
                    <span className="info-row__label">GPU</span>
                    <span className="info-row__value mono">
                      {diagnostics.gpu ?? "未检测到 NVIDIA GPU"}
                    </span>
                  </div>
                  <div className="info-row">
                    <span className="info-row__label">Python</span>
                    <span className="info-row__value mono">
                      {diagnostics.python ?? "未找到"}
                    </span>
                  </div>
                  <div className="info-row">
                    <span className="info-row__label">日志目录</span>
                    <span className="info-row__value mono">
                      {diagnostics.logDir}
                    </span>
                  </div>
                  <div className="info-row">
                    <span className="info-row__label">崩溃目录</span>
                    <span className="info-row__value mono">
                      {diagnostics.crashDir}
                    </span>
                  </div>
                </>
              )}
            </div>
            <div className="update-block">
              <div className="update-block__row">
                <span className="update-block__label">更新状态</span>
                <strong
                  className={`update-block__value${
                    updateState.status === "error"
                      ? " update-block__value--error"
                      : ""
                  }`}
                >
                  {updateStatusText}
                </strong>
              </div>
              {updateState.status === "available" && updateState.body ? (
                <div className="update-notes">{updateState.body}</div>
              ) : null}
              {updateState.status === "downloading" ? (
                <div className="update-download">
                  <div
                    className={`progress update-progress${
                      updateState.progress ? "" : " indeterminate"
                    }`}
                  >
                    <div
                      className="progress__bar"
                      style={{ width: `${updateState.progress ?? 38}%` }}
                    />
                  </div>
                  <span className="update-download__meta">
                    {updateState.progress
                      ? `已下载 ${updateState.progress}%`
                      : `正在下载 v${updateState.version ?? ""}`}
                  </span>
                </div>
              ) : null}
              {updateState.status === "error" &&
              updateState.message !== "unsupported" ? (
                <div className="update-error">{updateState.message}</div>
              ) : null}
            </div>
            <div className="panel__footer">
              <button
                className="btn btn--ghost"
                type="button"
                onClick={openLogs}
              >
                <FolderOpen size={16} />
                打开日志目录
              </button>
              <button
                className="btn btn--ghost"
                type="button"
                onClick={() => void updater.checkNow()}
                disabled={updateBusy}
              >
                {updateState.status === "checking" ? (
                  <Loader2 size={16} className="spin" />
                ) : (
                  <RefreshCw size={16} />
                )}
                {updateState.status === "checking" ? "检查中" : "检查更新"}
              </button>
              {updateState.status === "available" ? (
                <button
                  className="btn btn--primary"
                  type="button"
                  onClick={() => void updater.install()}
                  disabled={updateBusy}
                >
                  <Download size={16} />
                  立即更新
                </button>
              ) : null}
            </div>
          </section>
        </div>
      </div>
    </>
  );
}
