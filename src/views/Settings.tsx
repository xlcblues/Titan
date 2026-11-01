import { useState } from "react";
import { ChevronDown, Cpu, FileCode2, Gauge, Info } from "lucide-react";
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

export function Settings() {
  const info = useAppInfo();
  const [gpuBudget, setGpuBudget] = useState("8GB");
  const [settings, setSettings] = useState(() => readReconstructionSettings());
  const [stepsDraft, setStepsDraft] = useState(() => String(settings.steps));
  const [mixedPrecision, setMixedPrecision] = useState(true);

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
                  defaultValue=""
                  placeholder="未配置"
                />
                <button className="btn btn--secondary" type="button">
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
                <span className="info-row__value mono">{info.version}</span>
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
            </div>
            <div className="panel__footer">
              <button className="btn btn--ghost" type="button">
                <Gauge size={16} />
                打开日志目录
              </button>
            </div>
          </section>
        </div>
      </div>
    </>
  );
}
