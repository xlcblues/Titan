import { useState } from "react";
import { ChevronDown, Cpu, FileCode2, Gauge, Info } from "lucide-react";
import { ThemeToggle } from "../components/ThemeToggle";
import { useAppInfo } from "../hooks/useAppInfo";

export function Settings() {
  const info = useAppInfo();
  const [gpuBudget, setGpuBudget] = useState("8GB");
  const [resolution, setResolution] = useState("1280 × 720");
  const [mixedPrecision, setMixedPrecision] = useState(true);

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
              <span className="field-row__label">训练分辨率</span>
              <div className="select-wrap">
                <select
                  className="input select"
                  value={resolution}
                  onChange={(event) => setResolution(event.currentTarget.value)}
                  aria-label="训练分辨率"
                >
                  <option>960 × 540</option>
                  <option>1280 × 720</option>
                  <option>1920 × 1080</option>
                </select>
                <ChevronDown size={16} className="select-chevron" aria-hidden="true" />
              </div>
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
