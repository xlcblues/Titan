import { Link } from "react-router-dom";
import {
  ArrowRight,
  FolderKanban,
  Plus,
  ScanLine,
  Upload,
} from "lucide-react";
import { StatusPill } from "../components/StatusPill";
import { useAppInfo } from "../hooks/useAppInfo";

const RECENT_PROJECTS = [
  {
    id: "p-01",
    name: "书房桌景",
    images: 64,
    status: "已完成",
    tone: "success" as const,
    updated: "今天 10:24",
  },
  {
    id: "p-02",
    name: "客厅多视角",
    images: 120,
    status: "训练中",
    tone: "accent" as const,
    updated: "昨天 16:40",
  },
  {
    id: "p-03",
    name: "产品白底组",
    images: 32,
    status: "待处理",
    tone: "neutral" as const,
    updated: "10 月 24 日",
  },
];

export function Dashboard() {
  const info = useAppInfo();

  return (
    <>
      <div className="page-header">
        <h1>工作台</h1>
        <div className="page-header__actions">
          <Link className="btn btn--primary" to="/projects">
            <Plus size={16} />
            新建项目
          </Link>
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
            <Link className="btn btn--primary" to="/projects">
              <FolderKanban size={16} />
              创建项目
            </Link>
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
              {RECENT_PROJECTS.map((project) => (
                <Link
                  key={project.id}
                  className="row"
                  to="/scene"
                  aria-label={`打开项目 ${project.name}`}
                >
                  <span className="row__icon" aria-hidden="true">
                    <FolderKanban size={16} />
                  </span>
                  <span className="row__main">
                    <span className="row__name">{project.name}</span>
                    <span className="row__meta mono">
                      {project.images} 张图片
                    </span>
                  </span>
                  <StatusPill tone={project.tone} label={project.status} />
                  <span className="row__time">{project.updated}</span>
                </Link>
              ))}
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
    </>
  );
}
