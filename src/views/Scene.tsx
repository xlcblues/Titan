import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  ArrowLeft,
  Boxes,
  Camera,
  ChevronLeft,
  ChevronRight,
  Download,
  Focus,
  Grid3x3,
  LoaderCircle,
  type LucideIcon,
} from "lucide-react";
import { convertFileSrc, invoke, isTauri } from "@tauri-apps/api/core";
import {
  SceneCanvas,
  type SceneCanvasHandle,
  type SceneStats,
} from "../components/SceneCanvas";
import { StatusPill } from "../components/StatusPill";
import type { SceneMode, TitanSceneData } from "../lib/demoScene";

type RenderFilter = "all" | "train" | "eval";
type RenderItem = { name: string; path: string; view: string };

const RENDER_PAGE_SIZE = 18;

const RENDER_FILTERS: { value: RenderFilter; label: string }[] = [
  { value: "all", label: "全部" },
  { value: "train", label: "训练" },
  { value: "eval", label: "留出" },
];

const MODES: {
  value: SceneMode;
  label: string;
  icon: LucideIcon;
  realScene?: boolean;
}[] = [
  { value: "camera", label: "相机路径", icon: Camera },
  { value: "points", label: "点云", icon: Grid3x3 },
  { value: "mesh", label: "网格", icon: Boxes, realScene: false },
];

function formatMetric(
  value: number | string | undefined,
  digits: number,
  suffix = "",
) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return `${value.toFixed(digits)}${suffix}`;
}

function configNumber(value: unknown, fallback: number) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function viewLabel(view: string | undefined) {
  if (view === "train") return "训练视图";
  if (view === "eval") return "留出视图";
  return "渲染";
}

export function Scene() {
  const [searchParams] = useSearchParams();
  const projectId = searchParams.get("project") ?? "";
  const canvasRef = useRef<SceneCanvasHandle | null>(null);
  const [mode, setMode] = useState<SceneMode>("points");
  const [renderFilter, setRenderFilter] = useState<RenderFilter>("all");
  const [renderPage, setRenderPage] = useState(0);
  const [stats, setStats] = useState<SceneStats>({
    pointCount: 0,
    cameraCount: 0,
    selected: null,
    fps: 0,
  });
  const [scene, setScene] = useState<TitanSceneData | null>(null);
  const [sceneLoading, setSceneLoading] = useState(false);
  const [sceneError, setSceneError] = useState<string | null>(null);
  const handleStats = useCallback((next: SceneStats) => setStats(next), []);
  const activeMode = MODES.find((item) => item.value === mode) ?? MODES[0];

  useEffect(() => {
    let mounted = true;
    if (!isTauri() || !projectId) {
      setScene(null);
      setSceneLoading(false);
      setSceneError(null);
      return undefined;
    }
    setSceneLoading(true);
    setSceneError(null);
    invoke<TitanSceneData | null>("get_scene", { projectId })
      .then((result) => {
        if (mounted) setScene(result ?? null);
      })
      .catch((error) => {
        if (mounted) setSceneError(String(error));
      })
      .finally(() => {
        if (mounted) setSceneLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, [projectId]);

  useEffect(() => {
    if (scene && mode === "mesh") setMode("points");
  }, [scene, mode]);

  useEffect(() => {
    setRenderPage(0);
  }, [renderFilter, scene]);

  const config = scene?.config ?? null;
  const poseSource =
    scene?.poseSource ??
    (typeof config?.pose_source === "string" ? config.pose_source : "orbit");
  const realPose = poseSource === "colmap" || poseSource === "llff";
  const configRows = config
    ? [
        {
          label: "分辨率",
          value: `${configNumber(config.resolution, 192)} px`,
        },
        {
          label: "训练步数",
          value: `${configNumber(config.steps, 1200)} 步`,
        },
        {
          label: "训练视角",
          value: `${configNumber(config.train_views, 16)} 视角`,
        },
        {
          label: "Patch 采样",
          value: `${configNumber(config.patch_size, 1024)} px`,
        },
        {
          label: "显存预算",
          value: `${configNumber(config.gpu_budget, 8)} GB`,
        },
      ]
    : [
        { label: "分辨率", value: "1280 × 720" },
        { label: "Patch 采样", value: "4096 px" },
        { label: "混合精度", value: "FP16" },
        { label: "显存预算", value: "8 GB" },
      ];

  const metrics = scene?.metrics;
  const metricRows = scene
    ? [
        { label: "PSNR", value: formatMetric(metrics?.psnr, 2, " dB") },
        { label: "SSIM", value: formatMetric(metrics?.ssim, 3) },
        { label: "Loss", value: formatMetric(metrics?.loss, 4) },
        { label: "显存峰值", value: formatMetric(metrics?.vramMB, 0, " MB") },
        { label: "留出视图", value: formatMetric(metrics?.views, 0, " 视图") },
      ]
    : [
        {
          label: "点云",
          value: stats.pointCount.toLocaleString(),
        },
        { label: "相机", value: stats.cameraCount.toLocaleString() },
        { label: "选中点", value: stats.selected ?? "—" },
        { label: "FPS", value: `${Math.round(stats.fps)}` },
      ];

  const progress = scene ? 100 : 32;
  const pipeline = scene
    ? [
        {
          name: "SfM 位姿估计",
          label: realPose ? "完成" : "假设位姿",
          tone: realPose ? "success" as const : "warning" as const,
        },
        { name: "TFRF 训练", label: "完成", tone: "success" as const },
        { name: "渲染导出", label: "完成", tone: "success" as const },
      ]
    : [
        {
          name: "SfM 位姿估计",
          label: "完成",
          tone: "success" as const,
        },
        { name: "TFRF 训练", label: "进行中", tone: "accent" as const },
        { name: "渲染导出", label: "等待", tone: "neutral" as const },
      ];

  const renderItems: RenderItem[] = [
    ...(scene?.renders ?? []).map((render) => ({
      name: render.name,
      path: render.path,
      view: render.view,
    })),
  ];
  const filteredRenders = renderItems.filter(
    (render) => renderFilter === "all" || render.view === renderFilter,
  );
  const renderPageCount = Math.max(
    1,
    Math.ceil(filteredRenders.length / RENDER_PAGE_SIZE),
  );
  const clampedRenderPage = Math.min(renderPage, renderPageCount - 1);
  const pageRenders = filteredRenders.slice(
    clampedRenderPage * RENDER_PAGE_SIZE,
    (clampedRenderPage + 1) * RENDER_PAGE_SIZE,
  );
  const projectBase = (scene?.projectDir ?? "").replace(/\\/g, "/");
  const renderSrc = (path: string) => convertFileSrc(`${projectBase}/${path}`);

  return (
    <>
      <div className="scene-toolbar">
        <div className="scene-toolbar__start">
          <Link
            className="icon-btn"
            to={projectId ? `/projects/${projectId}` : "/projects"}
            title="返回项目"
            aria-label="返回项目"
          >
            <ArrowLeft size={16} />
          </Link>
          <div className="segmented" role="group" aria-label="场景显示模式">
            {MODES.map(({ value, label, icon: Icon, realScene }) => (
              <button
                key={value}
                type="button"
                aria-pressed={mode === value}
                onClick={() => setMode(value)}
                title={realScene === false && scene ? "网格模式暂未开放" : label}
                disabled={realScene === false && Boolean(scene)}
              >
                <Icon size={16} />
                <span>{label}</span>
              </button>
            ))}
          </div>
        </div>
        <div className="scene-toolbar__actions">
          <button
            className="icon-btn"
            type="button"
            onClick={() => canvasRef.current?.resetView()}
            title="聚焦场景"
            aria-label="聚焦场景"
          >
            <Focus size={16} />
          </button>
          <button
            className="icon-btn"
            type="button"
            title="导出渲染"
            aria-label="导出渲染"
            disabled
          >
            <Download size={16} />
          </button>
        </div>
      </div>

      <div className="scene-workspace">
        <section className="panel scene-panel">
          <div className="panel__header">
            <h2 className="panel__title">任务</h2>
            <StatusPill
              tone={scene ? "success" : "accent"}
              label={scene ? "重建完成" : "预览中"}
            />
          </div>
          <div className="panel__body task-list">
            {pipeline.map((task) => (
              <div className="task-item" key={task.name}>
                <span className="task-item__name">{task.name}</span>
                <span className="task-item__state">
                  <StatusPill tone={task.tone} label={task.label} />
                </span>
              </div>
            ))}
          </div>
        </section>

        <section className="scene-stage" aria-label="三维重建场景">
          <SceneCanvas
            ref={canvasRef}
            mode={mode}
            onStats={handleStats}
            scene={scene}
          />
          {sceneLoading && (
            <div className="scene-loading">
              <LoaderCircle className="spin" size={18} />
              正在加载场景
            </div>
          )}
          {sceneError && !sceneLoading && (
            <div className="scene-notice">读取场景失败：{sceneError}</div>
          )}
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
          <span className="scene-ratio mono">TITAN M4</span>
        </section>

        <div className="scene-side">
          <section className="panel scene-panel">
            <div className="panel__header">
              <h2 className="panel__title">参数</h2>
            </div>
            <div className="metric-list">
              {configRows.map((row) => (
                <div className="metric-row" key={row.label}>
                  <span className="metric-row__label">{row.label}</span>
                  <span className="metric-row__value mono">{row.value}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="panel scene-panel">
            <div className="panel__header">
              <h2 className="panel__title">指标</h2>
            </div>
            <div className="metric-list">
              {metricRows.map((row) => (
                <div className="metric-row" key={row.label}>
                  <span className="metric-row__label">{row.label}</span>
                  <span className="metric-row__value mono">{row.value}</span>
                </div>
              ))}
            </div>
            <div className="metric-progress">
              <div
                className="progress"
                role="progressbar"
                aria-valuenow={progress}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <div className="progress__bar" style={{ width: `${progress}%` }} />
              </div>
            </div>
          </section>

          {scene && renderItems.length > 0 && (
            <section className="panel scene-panel scene-panel--render">
              <div className="panel__header">
                <h2 className="panel__title">渲染</h2>
                <StatusPill tone="accent" label={`${filteredRenders.length} 张`} />
              </div>
              <div className="render-toolbar">
                <div
                  className="segmented render-filter"
                  role="group"
                  aria-label="渲染分类"
                >
                  {RENDER_FILTERS.map(({ value, label }) => (
                    <button
                      key={value}
                      type="button"
                      aria-pressed={renderFilter === value}
                      onClick={() => setRenderFilter(value)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="panel__body render-list">
                {pageRenders.map((render, index) => {
                  const absoluteIndex =
                    clampedRenderPage * RENDER_PAGE_SIZE + index + 1;
                  return (
                    <figure
                      className="render-thumb"
                      key={`${render.view}-${render.name}-${absoluteIndex}`}
                    >
                      <div className="render-thumb__frame">
                        <img
                          src={renderSrc(render.path)}
                          alt={render.name}
                          loading="lazy"
                        />
                      </div>
                      <figcaption className="render-thumb__meta">
                        <strong>{viewLabel(render.view)}</strong>
                        <span className="mono">
                          {String(absoluteIndex).padStart(2, "0")}
                        </span>
                      </figcaption>
                    </figure>
                  );
                })}
                {pageRenders.length === 0 && (
                  <div className="render-empty">当前分类暂无渲染</div>
                )}
              </div>
              {renderPageCount > 1 && (
                <div className="render-pager">
                  <button
                    className="icon-btn"
                    type="button"
                    onClick={() => setRenderPage(clampedRenderPage - 1)}
                    disabled={clampedRenderPage === 0}
                    title="上一页"
                    aria-label="上一页"
                  >
                    <ChevronLeft size={16} />
                  </button>
                  <span className="render-pager__info mono">
                    第 {clampedRenderPage + 1} / {renderPageCount} 页
                  </span>
                  <button
                    className="icon-btn"
                    type="button"
                    onClick={() => setRenderPage(clampedRenderPage + 1)}
                    disabled={clampedRenderPage >= renderPageCount - 1}
                    title="下一页"
                    aria-label="下一页"
                  >
                    <ChevronRight size={16} />
                  </button>
                </div>
              )}
            </section>
          )}
        </div>
      </div>
    </>
  );
}
