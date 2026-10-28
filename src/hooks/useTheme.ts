import { useEffect, useState } from "react";

export type ThemePreference = "light" | "dark" | "system";

const STORAGE_KEY = "titan.theme";

function getSystemTheme(): "light" | "dark" {
  if (typeof window !== "undefined" && window.matchMedia) {
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }
  return "light";
}

function readStored(): ThemePreference {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "light" || stored === "dark" || stored === "system") {
      return stored;
    }
  } catch {
    // localStorage is unavailable in some embedded contexts.
  }
  return "system";
}

export function useTheme() {
  const [preference, setPreference] = useState<ThemePreference>(readStored);

  useEffect(() => {
    const apply = () => {
      const resolved = preference === "system" ? getSystemTheme() : preference;
      document.documentElement.setAttribute("data-theme", resolved);
    };

    apply();
    try {
      localStorage.setItem(STORAGE_KEY, preference);
    } catch {
      // Ignore persistence failures; theme still applies for this session.
    }

    if (preference === "system") {
      const query = window.matchMedia("(prefers-color-scheme: dark)");
      const onChange = () => apply();
      query.addEventListener("change", onChange);
      return () => query.removeEventListener("change", onChange);
    }
  }, [preference]);

  const resolved =
    preference === "system" ? getSystemTheme() : preference;

  return { preference, resolved, setPreference };
}
