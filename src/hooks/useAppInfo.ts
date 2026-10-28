import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

export interface AppInfo {
  name: string;
  version: string;
  engine: string;
  gpuBudget: string;
  targetPlatform: string;
}

const FALLBACK: AppInfo = {
  name: "Titan",
  version: "0.1.0",
  engine: "Facet Radiance Field",
  gpuBudget: "8GB",
  targetPlatform: "Windows",
};

export function useAppInfo() {
  const [info, setInfo] = useState<AppInfo>(FALLBACK);

  useEffect(() => {
    let mounted = true;
    invoke<AppInfo>("app_info")
      .then((value) => {
        if (mounted) setInfo(value);
      })
      .catch(() => {
        // Browser preview without the Tauri runtime uses the fallback values.
      });
    return () => {
      mounted = false;
    };
  }, []);

  return info;
}
