import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme } from "../hooks/useTheme";

const OPTIONS = [
  { value: "light", label: "浅色", icon: Sun },
  { value: "system", label: "跟随系统", icon: Monitor },
  { value: "dark", label: "深色", icon: Moon },
] as const;

export function ThemeToggle({ compact = false }: { compact?: boolean }) {
  const { preference, setPreference } = useTheme();

  return (
    <div className="segmented theme-toggle" role="group" aria-label="主题模式">
      {OPTIONS.map(({ value, label, icon: Icon }) => (
        <button
          key={value}
          type="button"
          aria-pressed={preference === value}
          title={label}
          onClick={() => setPreference(value)}
        >
          <Icon size={16} />
          {!compact && <span>{label}</span>}
        </button>
      ))}
    </div>
  );
}
