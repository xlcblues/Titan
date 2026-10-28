import { useLocation } from "react-router-dom";
import { HelpCircle } from "lucide-react";
import { StatusPill } from "./StatusPill";
import { ThemeToggle } from "./ThemeToggle";
import { useAppInfo } from "../hooks/useAppInfo";

const TITLES: Record<string, string> = {
  "/": "工作台",
  "/projects": "项目",
  "/scene": "场景",
  "/settings": "设置",
};

export function Topbar() {
  const location = useLocation();
  const info = useAppInfo();
  const title = TITLES[location.pathname] ?? "工作台";

  return (
    <header className="topbar">
      <div className="topbar__title">{title}</div>
      <div className="topbar__right">
        <StatusPill tone="success" label={`就绪 · ${info.gpuBudget}`} />
        <ThemeToggle compact />
        <button className="icon-btn" type="button" title="帮助" aria-label="帮助">
          <HelpCircle size={18} />
        </button>
      </div>
    </header>
  );
}
