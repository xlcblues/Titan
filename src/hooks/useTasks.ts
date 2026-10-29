import { useCallback, useEffect, useRef, useState } from "react";
import { invoke, isTauri } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export interface TaskMetrics {
  step?: number;
  loss?: number;
  psnr?: number;
  ssim?: number;
  lpips?: number;
  vramMB?: number;
  elapsed?: number;
}

export interface TaskInfo {
  id: string;
  projectId: string;
  kind: string;
  status: string;
  stage: string;
  progress: number;
  startedAt: number | null;
  finishedAt: number | null;
  error: string | null;
  latest: TaskMetrics | null;
}

interface TaskEventPayload {
  taskId: string;
  status: string;
  stage: string;
  progress: number;
  error?: string;
  loss?: number;
  psnr?: number;
  ssim?: number;
  vramMB?: number;
  metrics?: TaskMetrics[];
}

export function useTasks(projectId: string) {
  const [task, setTask] = useState<TaskInfo | null>(null);
  const [history, setHistory] = useState<TaskMetrics[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const taskRef = useRef<TaskInfo | null>(null);

  const applyTask = useCallback((next: TaskInfo | null) => {
    taskRef.current = next;
    setTask(next);
  }, []);

  const refresh = useCallback(async () => {
    if (!isTauri()) {
      applyTask(null);
      setLoading(false);
      return;
    }
    try {
      setLoading(true);
      const result = await invoke<TaskInfo | null>("get_task", { projectId });
      applyTask(result);
      setError(null);
      if (result) {
        const metrics = await invoke<TaskMetrics[]>("get_metrics", {
          taskId: result.id,
        });
        setHistory(metrics);
      } else {
        setHistory([]);
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, [projectId, applyTask]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!isTauri()) return undefined;
    let unlisten: UnlistenFn | undefined;
    let mounted = true;
    const init = async () => {
      try {
        unlisten = await listen<TaskEventPayload>("titan://task", (event) => {
          if (!mounted) return;
          const payload = event.payload;
          const current = taskRef.current;
          if (!current) {
            void refresh();
            return;
          }
          if (payload.taskId !== current.id) return;
          if (payload.metrics?.length) {
            setHistory(payload.metrics);
          }
          const latest: TaskMetrics = payload.metrics?.length
            ? payload.metrics[payload.metrics.length - 1]
            : {
                loss: payload.loss,
                psnr: payload.psnr,
                ssim: payload.ssim,
                vramMB: payload.vramMB,
              };
          applyTask({
            ...current,
            status: payload.status,
            stage: payload.stage,
            progress: payload.progress,
            error: payload.error ?? null,
            latest,
          });
        });
      } catch {
        // Web preview does not expose task events.
      }
    };
    void init();
    return () => {
      mounted = false;
      unlisten?.();
    };
  }, [refresh, applyTask]);

  const start = useCallback(
    async (config?: Record<string, unknown>) => {
      if (!isTauri()) return null;
      const result = await invoke<TaskInfo>("start_reconstruction", {
        projectId,
        config: config ?? {},
      });
      applyTask(result);
      setHistory([]);
      return result;
    },
    [projectId, applyTask],
  );

  const pause = useCallback(async () => {
    const current = taskRef.current;
    if (!current || !isTauri()) return;
    await invoke("pause_task", { taskId: current.id });
    applyTask({ ...current, status: "paused" });
  }, [applyTask]);

  const resume = useCallback(async () => {
    const current = taskRef.current;
    if (!current || !isTauri()) return;
    await invoke("resume_task", { taskId: current.id });
    applyTask({ ...current, status: "running" });
  }, [applyTask]);

  const cancel = useCallback(async () => {
    const current = taskRef.current;
    if (!current || !isTauri()) return;
    await invoke("cancel_task", { taskId: current.id });
    applyTask({
      ...current,
      status: "cancelled",
      error: "用户取消",
      progress: 0,
    });
  }, [applyTask]);

  return {
    task,
    history,
    loading,
    error,
    refresh,
    start,
    pause,
    resume,
    cancel,
  };
}
