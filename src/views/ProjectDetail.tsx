import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  ArrowLeft,
  FolderOpen,
  ImagePlus,
  LoaderCircle,
  Upload,
} from "lucide-react";
import { convertFileSrc } from "@tauri-apps/api/core";
import type { UnlistenFn } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open } from "@tauri-apps/plugin-dialog";
import { StatusPill } from "../components/StatusPill";
import { useProjectImages, useProjects } from "../hooks/useProjects";
import { projectStatusMeta } from "../utils/projectMeta";

function formatBytes(size: number | null) {
  if (size == null) return "—";
  if (size < 1024 * 1024) return `${Math.round(size / 1024)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

export function ProjectDetail() {
  const { id } = useParams<{ id: string }>();
  const projectId = id ?? "";
  const { projects, refresh: refreshProjects } = useProjects();
  const { images, loading, error, importImages } = useProjectImages(projectId);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const project = projects.find((item) => item.id === projectId);
  const status = projectStatusMeta(project?.status ?? "idle");

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
