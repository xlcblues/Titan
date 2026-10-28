import { NavLink } from "react-router-dom";
import {
  Box,
  FolderKanban,
  HelpCircle,
  LayoutDashboard,
  Settings,
} from "lucide-react";
import { LogoWordmark } from "./Logo";
import { useAppInfo } from "../hooks/useAppInfo";

const NAV_ITEMS = [
  { to: "/", label: "工作台", icon: LayoutDashboard, end: true },
  { to: "/projects", label: "项目", icon: FolderKanban },
  { to: "/scene", label: "场景", icon: Box },
  { to: "/settings", label: "设置", icon: Settings },
];

export function Sidebar() {
  const info = useAppInfo();

  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <LogoWordmark size={24} />
      </div>

      <nav className="sidebar__nav" aria-label="主导航">
        <span className="nav-label">工作区</span>
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              isActive ? "nav-item nav-item--active" : "nav-item"
            }
          >
            <Icon size={18} aria-hidden="true" />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar__footer">
        <div className="engine-card">
          <div className="engine-card__status">
            <span className="engine-dot" aria-hidden="true" />
            <span>引擎就绪</span>
          </div>
          <div className="engine-card__name mono">{info.engine}</div>
        </div>
        <div className="sidebar__footer-actions">
          <button className="icon-btn" type="button" title="帮助" aria-label="帮助">
            <HelpCircle size={18} />
          </button>
        </div>
      </div>
    </aside>
  );
}
