import { useCallback, useState } from "react";
import {
  Boxes,
  Camera,
  Download,
  Grid3x3,
  Play,
  Upload,
  type LucideIcon,
} from "lucide-react";
import { SceneCanvas, type SceneStats } from "../components/SceneCanvas";
import { StatusPill } from "../components/StatusPill";
import type { SceneMode } from "../lib/demoScene";

const MODES: { value: SceneMode; label: string; icon: LucideIcon }[] = [
  { value: "camera", label: "相机路径", icon: Camera },
  { value: "points", label: "点云", icon: Grid3x3 },
  { value: "mesh", label: "网格", icon: Boxes },
];

const TASKS = [
  { name: "SfM 位姿估计", status: "完成", tone: "success" as const },
  { name: "TFRF 训练", status: "进行中", tone: "accent" as const },
  { name: "渲染导出", status: "等待", tone: "neutral" as const },
];

export function Scene() {
  const [mode, setMode] = useState<SceneMode>("points");
  const [stats, setStats] = useState<SceneStats>({
    pointCount: 0,
    cameraCount: 0,
    selected: null,
    fps: 0,
  });
  const handleStats = useCallback((next: SceneStats) => setStats(next), []);
  const activeMode = MODES.find((item) => item.value === mode) ?? MODES[0];

  return (
    <>
      <div className="scene-toolbar">
        <div className="segmented" role="group" aria-label="场景显示模式">
          {MODES.map(({ value, label, icon: Icon }) => (
            <button
              key={value}
              type="button"
              aria-pressed={mode === value}
              onClick={() => setMode(value)}
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
            <StatusPill tone="accent" label="预览中" />
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

        <section className="scene-stage" aria-label="三维重建场景">
          <SceneCanvas mode={mode} onStats={handleStats} />
          <span className="scene-selection mono">
            选中 {stats.selected ?? "—"}
          </span>
          <span className="scene-chip">
            <span className="scene-chip__dot" aria-hidden="true" />
            {activeMode.label}
          </span>
          <span className="scene-hud mono">
            {stats.cameraCount} 相机 · {stats.pointCount.toLocaleString()} 点 ·{" "}
            {Math.round(stats.fps)} FPS
          </span>
          <span className="scene-ratio mono">TITAN M3</span>
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
              <div className="metric-row">
                <span className="metric-row__label">显存预算</span>
                <span className="metric-row__value mono">8 GB</span>
              </div>
            </div>
          </section>

          <section className="panel scene-panel">
            <div className="panel__header">
              <h2 className="panel__title">指标</h2>
            </div>
            <div className="metric-list">
              <div className="metric-row">
                <span className="metric-row__label">点云</span>
                <span className="metric-row__value mono">
                  {stats.pointCount.toLocaleString()}
                </span>
              </div>
              <div className="metric-row">
                <span className="metric-row__label">相机</span>
                <span className="metric-row__value mono">
                  {stats.cameraCount}
                </span>
              </div>
              <div className="metric-row">
                <span className="metric-row__label">选中点</span>
                <span className="metric-row__value mono">
                  {stats.selected ?? "—"}
                </span>
              </div>
              <div className="metric-row">
                <span className="metric-row__label">FPS</span>
                <span className="metric-row__value mono">
                  {Math.round(stats.fps)}
                </span>
              </div>
            </div>
            <div className="metric-progress">
              <div className="progress" role="progressbar" aria-valuenow={32} aria-valuemin={0} aria-valuemax={100}>
                <div className="progress__bar" style={{ width: "32%" }} />
              </div>
            </div>
          </section>
        </div>
      </div>
    </>
  );
}
