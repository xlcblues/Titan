import {
  Boxes,
  Camera,
  Download,
  Grid3x3,
  Play,
  Upload,
} from "lucide-react";
import { LogoMark } from "../components/Logo";
import { StatusPill } from "../components/StatusPill";

const MODES = [
  { value: "camera", label: "相机路径", icon: Camera },
  { value: "points", label: "点云", icon: Grid3x3 },
  { value: "mesh", label: "网格", icon: Boxes },
] as const;

const TASKS = [
  { name: "SfM 位姿估计", status: "等待", tone: "neutral" as const },
  { name: "TFRF 训练", status: "等待", tone: "neutral" as const },
  { name: "渲染导出", status: "等待", tone: "neutral" as const },
];

export function Scene() {
  return (
    <>
      <div className="scene-toolbar">
        <div className="segmented" role="group" aria-label="场景显示模式">
          {MODES.map(({ value, label, icon: Icon }, index) => (
            <button
              key={value}
              type="button"
              aria-pressed={index === 0}
              title={label}
            >
              <Icon size={16} />
              <span>{label}</span>
            </button>
          ))}
        </div>
        <div className="scene-toolbar__actions">
          <button className="icon-btn" type="button" title="导入图片" aria-label="导入图片">
            <Upload size={16} />
          </button>
          <button className="icon-btn" type="button" title="开始重建" aria-label="开始重建" disabled>
            <Play size={16} />
          </button>
          <button className="icon-btn" type="button" title="导出渲染" aria-label="导出渲染" disabled>
            <Download size={16} />
          </button>
        </div>
      </div>

      <div className="scene-workspace">
        <section className="panel scene-panel">
          <div className="panel__header">
            <h2 className="panel__title">任务</h2>
            <StatusPill tone="neutral" label="未开始" />
          </div>
          <div className="panel__body task-list">
            {TASKS.map((task) => (
              <div className="task-item" key={task.name}>
                <span className="task-item__name">{task.name}</span>
                <span className="task-item__state">
                  <StatusPill tone={task.tone} label={task.status} />
                </span>
              </div>
            ))}
          </div>
        </section>

        <section className="scene-stage" aria-label="场景画布">
          <div className="scene-stage__inner">
            <div className="scene-grid" aria-hidden="true" />
            <div className="scene-symbol" aria-hidden="true">
              <LogoMark size={72} />
            </div>
            <span className="pill scene-hint">暂无场景数据</span>
            <span className="scene-hud mono">
              0 相机 · 0 点 · 0 步
            </span>
            <span className="scene-ratio mono">16:9</span>
          </div>
        </section>

        <div className="scene-side">
          <section className="panel scene-panel">
            <div className="panel__header">
              <h2 className="panel__title">参数</h2>
            </div>
            <div className="metric-list">
              <div className="metric-row">
                <span className="metric-row__label">分辨率</span>
                <span className="metric-row__value mono">1280 × 720</span>
              </div>
              <div className="metric-row">
                <span className="metric-row__label">Patch 采样</span>
                <span className="metric-row__value mono">4096 px</span>
              </div>
              <div className="metric-row">
                <span className="metric-row__label">混合精度</span>
                <span className="metric-row__value mono">FP16</span>
              </div>
            </div>
          </section>

          <section className="panel scene-panel">
            <div className="panel__header">
              <h2 className="panel__title">指标</h2>
            </div>
            <div className="metric-list">
              <div className="metric-row">
                <span className="metric-row__label">PSNR</span>
                <span className="metric-row__value mono">—</span>
              </div>
              <div className="metric-row">
                <span className="metric-row__label">SSIM</span>
                <span className="metric-row__value mono">—</span>
              </div>
              <div className="metric-row">
                <span className="metric-row__label">显存</span>
                <span className="metric-row__value mono">0 MB</span>
              </div>
              <div className="metric-row">
                <span className="metric-row__label">剩余时间</span>
                <span className="metric-row__value mono">—</span>
              </div>
            </div>
            <div className="metric-progress">
              <div className="progress" role="progressbar" aria-valuenow={0} aria-valuemin={0} aria-valuemax={100}>
                <div className="progress__bar" style={{ width: "0%" }} />
              </div>
            </div>
          </section>
        </div>
      </div>
    </>
  );
}
