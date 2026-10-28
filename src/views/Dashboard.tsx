import { Link } from "react-router-dom";
import {
  ArrowRight,
  FolderKanban,
  Plus,
  ScanLine,
  Upload,
} from "lucide-react";
import { useState } from "react";
import { CreateProjectDialog } from "../components/CreateProjectDialog";
import { StatusPill } from "../components/StatusPill";
import { useAppInfo } from "../hooks/useAppInfo";
import { useProjects } from "../hooks/useProjects";
import { formatUpdated, projectStatusMeta } from "../utils/projectMeta";

export function Dashboard() {
  const info = useAppInfo();
  const { projects, loading, error, createProject } = useProjects();
  const [dialogOpen, setDialogOpen] = useState(false);
  const recent = projects.slice(0, 3);

  return (
    <>
      <div className="page-header">
        <h1>工作台</h1>
        <div className="page-header__actions">
          <button
            className="btn btn--primary"
            type="button"
            onClick={() => setDialogOpen(true)}
          >
            <Plus size={16} />
            新建项目
          </button>
        </div>
      </div>

      <div className="dashboard-grid">
        <div className="dashboard-main">
          <section className="quick-start" aria-label="新建重建项目">
            <div className="quick-start__icon">
              <Upload size={22} />
            </div>
            <div className="quick-start__text">
              <h2 className="quick-start__title">新建重建项目</h2>
            </div>
            <button
              className="btn btn--primary"
              type="button"
              onClick={() => setDialogOpen(true)}
            >
              <FolderKanban size={16} />
              创建项目
            </button>
          </section>

          <section className="panel">
            <div className="panel__header">
              <h2 className="panel__title">最近项目</h2>
              <Link className="panel__link" to="/projects">
                全部项目
                <ArrowRight size={14} />
              </Link>
            </div>
            <div className="panel__body">
              {error ? (
                <div className="empty-state">
                  <span>读取项目失败：{error}</span>
                </div>
              ) : loading ? (
                <div className="empty-state">
                  <span>正在读取项目</span>
                </div>
              ) : recent.length === 0 ? (
                <div className="empty-state">
                  <FolderKanban size={24} className="empty-state__icon" />
                  <span>还没有项目</span>
                  <button
                    className="btn"
                    type="button"
                    onClick={() => setDialogOpen(true)}
                  >
                    <Plus size={16} />
                    创建第一个项目
                  </button>
                </div>
              ) : (
                recent.map((project) => {
                  const status = projectStatusMeta(project.status);
                  return (
                    <Link
                      key={project.id}
                      className="row"
                      to={`/projects/${encodeURIComponent(project.id)}`}
                      aria-label={`打开项目 ${project.name}`}
                    >
                      <span className="row__icon" aria-hidden="true">
                        <FolderKanban size={16} />
                      </span>
                      <span className="row__main">
                        <span className="row__name">{project.name}</span>
                        <span className="row__meta mono">
                          {project.imageCount} 张图片
                        </span>
                      </span>
                      <StatusPill tone={status.tone} label={status.label} />
                      <span className="row__time">
                        {formatUpdated(project.updatedAt)}
                      </span>
                    </Link>
                  );
                })
              )}
            </div>
          </section>
        </div>

        <div className="dashboard-side">
          <section className="panel">
            <div className="panel__header">
              <h2 className="panel__title">环境</h2>
              <StatusPill tone="success" label="运行正常" />
            </div>
            <div className="info-list">
              <div className="info-row">
                <span className="info-row__label">引擎</span>
                <span className="info-row__value">{info.engine}</span>
              </div>
              <div className="info-row">
                <span className="info-row__label">显存预算</span>
                <span className="info-row__value mono">{info.gpuBudget}</span>
              </div>
              <div className="info-row">
                <span className="info-row__label">版本</span>
                <span className="info-row__value mono">{info.version}</span>
              </div>
              <div className="info-row">
                <span className="info-row__label">平台</span>
                <span className="info-row__value">{info.targetPlatform}</span>
              </div>
            </div>
          </section>

          <section className="panel">
            <div className="panel__header">
              <h2 className="panel__title">任务</h2>
            </div>
            <div className="empty-state">
              <ScanLine size={24} className="empty-state__icon" />
              <span>暂无重建任务</span>
            </div>
          </section>
        </div>
      </div>

      <CreateProjectDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        onSubmit={(name) => createProject(name)}
      />
    </>
  );
}
