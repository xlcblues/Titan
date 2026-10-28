import { useState } from "react";
import {
  Eye,
  FolderKanban,
  MoreHorizontal,
  Plus,
  Search,
} from "lucide-react";
import { Link } from "react-router-dom";
import { StatusPill } from "../components/StatusPill";

type Filter = "all" | "active" | "done";

const PROJECTS = [
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
  {
    id: "p-04",
    name: "户外雕塑环拍",
    images: 96,
    status: "已完成",
    tone: "success" as const,
    updated: "10 月 18 日",
  },
];

const FILTERS: { value: Filter; label: string }[] = [
  { value: "all", label: "全部" },
  { value: "active", label: "进行中" },
  { value: "done", label: "已完成" },
];

export function Projects() {
  const [filter, setFilter] = useState<Filter>("all");

  const visible = PROJECTS.filter((project) => {
    if (filter === "active") return project.tone !== "success";
    if (filter === "done") return project.tone === "success";
    return true;
  });

  return (
    <>
      <div className="page-header">
        <h1>项目</h1>
        <div className="page-header__actions">
          <button className="btn btn--primary" type="button">
            <Plus size={16} />
            新建项目
          </button>
        </div>
      </div>

      <div className="toolbar">
        <div className="search">
          <Search size={16} aria-hidden="true" />
          <input className="input" type="search" placeholder="搜索项目" />
        </div>
        <div className="segmented" role="group" aria-label="项目状态筛选">
          {FILTERS.map((item) => (
            <button
              key={item.value}
              type="button"
              aria-pressed={filter === item.value}
              onClick={() => setFilter(item.value)}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <section className="panel">
        <table className="table">
          <thead>
            <tr>
              <th>项目</th>
              <th>图片</th>
              <th>状态</th>
              <th>更新时间</th>
              <th aria-label="操作" />
            </tr>
          </thead>
          <tbody>
            {visible.map((project) => (
              <tr key={project.id}>
                <td>
                  <div className="table-name">
                    <span className="row__icon" aria-hidden="true">
                      <FolderKanban size={16} />
                    </span>
                    <span className="row-name">{project.name}</span>
                  </div>
                </td>
                <td className="mono">{project.images}</td>
                <td>
                  <StatusPill tone={project.tone} label={project.status} />
                </td>
                <td className="table-muted">{project.updated}</td>
                <td>
                  <div className="row-actions">
                    <Link
                      className="icon-btn"
                      to="/scene"
                      title="查看场景"
                      aria-label={`查看 ${project.name} 场景`}
                    >
                      <Eye size={16} />
                    </Link>
                    <button
                      className="icon-btn"
                      type="button"
                      title="更多操作"
                      aria-label={`${project.name} 更多操作`}
                    >
                      <MoreHorizontal size={16} />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}
