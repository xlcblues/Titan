import { useMemo, useState } from "react";
import {
  Eye,
  FolderKanban,
  Plus,
  Search,
} from "lucide-react";
import { Link } from "react-router-dom";
import { CreateProjectDialog } from "../components/CreateProjectDialog";
import { StatusPill } from "../components/StatusPill";
import { useProjects } from "../hooks/useProjects";
import { formatUpdated, projectStatusMeta } from "../utils/projectMeta";

type Filter = "all" | "active" | "done";

const FILTERS: { value: Filter; label: string }[] = [
  { value: "all", label: "全部" },
  { value: "active", label: "进行中" },
  { value: "done", label: "已完成" },
];

export function Projects() {
  const { projects, loading, error, createProject } = useProjects();
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");
  const [dialogOpen, setDialogOpen] = useState(false);

  const visible = useMemo(() => {
    const keyword = query.trim().toLowerCase();
    return projects.filter((project) => {
      const statusMatch = filter === "all" || project.status === filter;
      const queryMatch =
        keyword.length === 0 || project.name.toLowerCase().includes(keyword);
      return statusMatch && queryMatch;
    });
  }, [projects, filter, query]);

  return (
    <>
      <div className="page-header">
        <h1>项目</h1>
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

      <div className="toolbar">
        <div className="search">
          <Search size={16} aria-hidden="true" />
          <input
            className="input"
            type="search"
            placeholder="搜索项目"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
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
        {error ? (
          <div className="empty-state">
            <span>读取项目失败：{error}</span>
          </div>
        ) : loading ? (
          <div className="empty-state">
            <span>正在读取项目</span>
          </div>
        ) : visible.length === 0 ? (
          <div className="empty-state">
            <FolderKanban size={24} className="empty-state__icon" />
            <span>没有匹配的项目</span>
          </div>
        ) : (
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
              {visible.map((project) => {
                const status = projectStatusMeta(project.status);
                return (
                  <tr key={project.id}>
                    <td>
                      <Link
                        className="table-name table-name--link"
                        to={`/projects/${encodeURIComponent(project.id)}`}
                      >
                        <span className="row__icon" aria-hidden="true">
                          <FolderKanban size={16} />
                        </span>
                        <span className="row-name">{project.name}</span>
                      </Link>
                    </td>
                    <td className="mono">{project.imageCount}</td>
                    <td>
                      <StatusPill tone={status.tone} label={status.label} />
                    </td>
                    <td className="table-muted">
                      {formatUpdated(project.updatedAt)}
                    </td>
                    <td>
                      <div className="row-actions">
                        <Link
                          className="icon-btn"
                          to={`/projects/${encodeURIComponent(project.id)}`}
                          title="打开项目"
                          aria-label={`打开 ${project.name}`}
                        >
                          <Eye size={16} />
                        </Link>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </section>

      <CreateProjectDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        onSubmit={(name) => createProject(name)}
      />
    </>
  );
}
