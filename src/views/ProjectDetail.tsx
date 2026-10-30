import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  ArrowLeft,
  Cpu,
  Eye,
  FolderOpen,
  ImagePlus,
  LoaderCircle,
  Pause,
  Play,
  RotateCcw,
  Square,
  Upload,
} from "lucide-react";
import { convertFileSrc } from "@tauri-apps/api/core";
import type { UnlistenFn } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open } from "@tauri-apps/plugin-dialog";
import { StatusPill } from "../components/StatusPill";
import { useProjectImages, useProjects } from "../hooks/useProjects";
import { useTasks } from "../hooks/useTasks";
import { projectStatusMeta } from "../utils/projectMeta";

function formatBytes(size: number | null) {
  if (size == null) return "—";
  if (size < 1024 * 1024) return `${Math.round(size / 1024)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function taskStatusMeta(status: string | undefined) {
  switch (status) {
    case "running":
      return { label: "训练中", tone: "accent" as const };
    case "paused":
      return { label: "已暂停", tone: "warning" as const };
    case "done":
      return { label: "已完成", tone: "success" as const };
    case "failed":
      return { label: "失败", tone: "danger" as const };
    case "cancelled":
      return { label: "已取消", tone: "neutral" as const };
    default:
      return { label: "未开始", tone: "neutral" as const };
  }
}

function stageLabel(stage: string | undefined) {
  const labels: Record<string, string> = {
    data: "数据准备",
    sfm: "位姿估计",
    training: "TFRF 训练",
    render: "渲染导出",
    done: "完成",
    error: "错误",
    cancel: "已取消",
  };
  return stage ? (labels[stage] ?? stage) : "等待开始";
}

function formatMetric(value: number | undefined, digits: number, suffix = "") {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value.toFixed(digits)}${suffix}`
    : "—";
}

export function ProjectDetail() {
  const { id } = useParams<{ id: string }>();
  const projectId = id ?? "";
  const { projects, refresh: refreshProjects } = useProjects();
  const { images, loading, error, importImages } = useProjectImages(projectId);
  const {
    task,
    loading: taskLoading,
    error: taskError,
    start,
    pause,
    resume,
    cancel,
  } = useTasks(projectId);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const project = projects.find((item) => item.id === projectId);
  const status = projectStatusMeta(project?.status ?? "idle");
  const taskMeta = taskStatusMeta(task?.status);

  const startTask = useCallback(async () => {
    try {
      await start();
      setNotice("重建任务已启动");
    } catch (err) {
      setNotice(`启动失败：${err}`);
    }
  }, [start]);

  const runPause = useCallback(async () => {
    try {
      await pause();
    } catch (err) {
      setNotice(`暂停失败：${err}`);
    }
  }, [pause]);

  const runResume = useCallback(async () => {
    try {
      await resume();
    } catch (err) {
      setNotice(`恢复失败：${err}`);
    }
  }, [resume]);

  const runCancel = useCallback(async () => {
    try {
      await cancel();
    } catch (err) {
      setNotice(`取消失败：${err}`);
    }
  }, [cancel]);

  const runImport = useCallback(
    async (paths: string[]) => {
      if (!projectId || paths.length === 0) return;
      setBusy(true);
      setNotice(null);
      try {
        const added = await importImages(paths);
        setNotice(added.length > 0 ? `已导入 ${added.length} 张图片` : "没有新增图片");
        void refreshProjects();
      } catch (err) {
        setNotice(`导入失败：${err}`);
      } finally {
        setBusy(false);
      }
    },
    [projectId, importImages, refreshProjects],
  );

  useEffect(() => {
    let unlisten: UnlistenFn | undefined;
    let mounted = true;
    const init = async () => {
      try {
        unlisten = await getCurrentWebview().onDragDropEvent((event) => {
          const drag = event.payload;
          if (drag.type === "enter" || drag.type === "over") {
            setDragging(true);
          } else if (drag.type === "leave") {
            setDragging(false);
          } else if (drag.type === "drop") {
            setDragging(false);
            void runImport(drag.paths);
          }
        });
      } catch {
        if (mounted) setNotice("浏览器预览不支持系统拖拽导入");
      }
    };
    void init();
    return () => {
      mounted = false;
      unlisten?.();
    };
  }, [runImport]);

  useEffect(() => {
    if (task?.status === "done") void refreshProjects();
  }, [task?.status, refreshProjects]);

  const pickImages = async () => {
    try {
      const selected = await open({
        multiple: true,
        filters: [
          {
            name: "Images",
            extensions: ["jpg", "jpeg", "png", "webp", "bmp", "gif"],
          },
        ],
      });
      if (!selected) return;
      const paths = Array.isArray(selected) ? selected : [selected];
      void runImport(paths);
    } catch (err) {
      setNotice(`选择失败：${err}`);
    }
  };

  return (
    <>
      <div className="page-header project-header">
        <div className="project-header__title">
          <Link
            className="icon-btn"
            to="/projects"
            title="返回项目列表"
            aria-label="返回项目列表"
          >
            <ArrowLeft size={16} />
          </Link>
          <div className="project-header__text">
            <h1>{project?.name ?? "项目"}</h1>
            <span className="project-header__meta mono">
              {images.length} 张图片
            </span>
          </div>
        </div>
        <div className="page-header__actions">
          <StatusPill tone={status.tone} label={status.label} />
        </div>
      </div>

      <section
        className={dragging ? "dropzone dropzone--active" : "dropzone"}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            void pickImages();
          }
        }}
      >
        <span className="dropzone__icon" aria-hidden="true">
          {busy ? <LoaderCircle className="spin" size={20} /> : <Upload size={20} />}
        </span>
        <div className="dropzone__copy">
          <strong>拖拽图片到此处导入</strong>
          <span>支持 JPG、PNG、WEBP、BMP、GIF</span>
        </div>
        <button
          className="btn btn--primary"
          type="button"
          onClick={pickImages}
          disabled={busy}
        >
          <FolderOpen size={16} />
          选择图片
        </button>
      </section>

      {notice && <div className="import-notice">{notice}</div>}

      <section className="panel task-panel">
        <div className="panel__header">
          <h2 className="panel__title">重建任务</h2>
          <StatusPill
            tone={taskMeta.tone}
            label={taskLoading ? "读取中" : taskMeta.label}
          />
        </div>
        <div className="panel__body task-panel__body">
          <div className="task-runner">
            <div className="task-runner__actions">
              <button
                className="btn btn--primary"
                type="button"
                onClick={() => void startTask()}
                disabled={
                  images.length === 0 ||
                  task?.status === "running" ||
                  task?.status === "paused"
                }
              >
                <Cpu size={16} />
                开始重建
              </button>
              {task?.status === "running" && (
                <button
                  className="btn btn--secondary"
                  type="button"
                  onClick={() => void runPause()}
                >
                  <Pause size={16} />
                  暂停
                </button>
              )}
              {task?.status === "paused" && (
                <button
                  className="btn btn--secondary"
                  type="button"
                  onClick={() => void runResume()}
                >
                  <Play size={16} />
                  继续
                </button>
              )}
              {(task?.status === "running" || task?.status === "paused") && (
                <button
                  className="btn btn--ghost"
                  type="button"
                  onClick={() => void runCancel()}
                >
                  <Square size={16} />
                  取消
                </button>
              )}
              {task?.status === "failed" && (
                <button
                  className="btn btn--secondary"
                  type="button"
                  onClick={() => void startTask()}
                >
                  <RotateCcw size={16} />
                  重试
                </button>
              )}
              {task?.status === "done" && (
                <Link
                  className="btn btn--secondary"
                  to={`/scene?project=${projectId}`}
                >
                  <Eye size={16} />
                  查看场景
                </Link>
              )}
            </div>
            <div className="task-runner__progress">
              <div className="task-stage">
                <span className="task-stage__label">
                  {stageLabel(task?.stage)}
                </span>
                <span className="task-stage__value mono">
                  {task ? `${Math.round(task.progress * 100)}%` : "—"}
                </span>
              </div>
              <div
                className="task-progress"
                role="progressbar"
                aria-valuenow={Math.round((task?.progress ?? 0) * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <div className="progress">
                  <div
                    className="progress__bar"
                    style={{ width: `${(task?.progress ?? 0) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
          <div className="task-metrics">
            <div className="metric-row">
              <span className="metric-row__label">Loss</span>
              <span className="metric-row__value mono">
                {formatMetric(task?.latest?.loss, 4)}
              </span>
            </div>
            <div className="metric-row">
              <span className="metric-row__label">PSNR</span>
              <span className="metric-row__value mono">
                {formatMetric(task?.latest?.psnr, 2, " dB")}
              </span>
            </div>
            <div className="metric-row">
              <span className="metric-row__label">SSIM</span>
              <span className="metric-row__value mono">
                {formatMetric(task?.latest?.ssim, 3)}
              </span>
            </div>
            <div className="metric-row">
              <span className="metric-row__label">显存</span>
              <span className="metric-row__value mono">
                {formatMetric(task?.latest?.vramMB, 0, " MB")}
              </span>
            </div>
          </div>
          {taskError && <div className="task-error">{taskError}</div>}
        </div>
      </section>

      <section className="panel thumb-panel">
        <div className="panel__header">
          <h2 className="panel__title">图片</h2>
          <StatusPill tone="neutral" label={`${images.length} 张`} />
        </div>
        <div className="panel__body thumb-panel__body">
          {error ? (
            <div className="empty-state">
              <span>读取图片失败：{error}</span>
            </div>
          ) : loading ? (
            <div className="empty-state">
              <LoaderCircle className="spin empty-state__icon" size={22} />
              <span>正在读取图片</span>
            </div>
          ) : images.length === 0 ? (
            <div className="empty-state">
              <ImagePlus className="empty-state__icon" size={24} />
              <span>导入图片后将在重建中使用</span>
            </div>
          ) : (
            <div className="thumb-grid">
              {images.map((image) => (
                <figure className="thumb" key={image.id}>
                  <div className="thumb__frame">
                    <img
                      src={convertFileSrc(image.path)}
                      alt={image.filename ?? "导入图片"}
                      loading="lazy"
                    />
                  </div>
                  <figcaption className="thumb__meta">
                    <span className="thumb__name" title={image.filename ?? undefined}>
                      {image.filename ?? "未命名图片"}
                    </span>
                    <span className="mono">{formatBytes(image.size)}</span>
                  </figcaption>
                </figure>
              ))}
            </div>
          )}
        </div>
      </section>
    </>
  );
}
