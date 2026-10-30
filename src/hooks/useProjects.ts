import { useCallback, useEffect, useState } from "react";
import { invoke, isTauri } from "@tauri-apps/api/core";

export interface Project {
  id: string;
  name: string;
  status: string;
  settingsJson: string;
  imageCount: number;
  createdAt: number;
  updatedAt: number;
}

export interface ImageRecord {
  id: string;
  projectId: string;
  path: string;
  filename: string | null;
  width: number | null;
  height: number | null;
  size: number | null;
  status: string;
  thumbnailPath: string | null;
  sortOrder: number;
}

const DEMO_PROJECTS: Project[] = [
  {
    id: "demo-01",
    name: "书房桌景",
    status: "done",
    settingsJson: "{}",
    imageCount: 64,
    createdAt: Date.now() - 86400000 * 3,
    updatedAt: Date.now() - 3600000,
  },
  {
    id: "demo-02",
    name: "客厅多视角",
    status: "active",
    settingsJson: "{}",
    imageCount: 120,
    createdAt: Date.now() - 86400000 * 5,
    updatedAt: Date.now() - 86400000,
  },
];

export function useProjects() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!isTauri()) {
      setProjects(DEMO_PROJECTS);
      setLoading(false);
      setError(null);
      return;
    }
    try {
      setLoading(true);
      const result = await invoke<Project[]>("list_projects");
      setProjects(result);
      setError(null);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const createProject = useCallback(
    async (name: string, settings?: Record<string, unknown>) => {
      if (!isTauri()) {
        const project: Project = {
          id: `local-${Date.now()}`,
          name,
          status: "idle",
          settingsJson: JSON.stringify(settings ?? {}),
          imageCount: 0,
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };
        setProjects((prev) => [project, ...prev]);
        return project;
      }
      const project = await invoke<Project>("create_project", { name, settings });
      setProjects((prev) => [project, ...prev]);
      return project;
    },
    [],
  );

  const deleteProject = useCallback(async (projectId: string) => {
    if (!isTauri()) {
      setProjects((prev) => prev.filter((item) => item.id !== projectId));
      return;
    }
    await invoke("delete_project", { projectId });
    setProjects((prev) => prev.filter((item) => item.id !== projectId));
  }, []);

  return { projects, loading, error, refresh, createProject, deleteProject };
}

export function useProjectImages(projectId: string) {
  const [images, setImages] = useState<ImageRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!isTauri()) {
      setImages([]);
      setLoading(false);
      setError(null);
      return;
    }
    try {
      setLoading(true);
      const result = await invoke<ImageRecord[]>("list_images", { projectId });
      setImages(result);
      setError(null);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const importImages = useCallback(
    async (paths: string[]) => {
      if (!isTauri() || paths.length === 0) {
        return [] as ImageRecord[];
      }
      const result = await invoke<ImageRecord[]>("import_images", {
        projectId,
        paths,
      });
      if (result.length > 0) {
        setImages((prev) => [...prev, ...result]);
      }
      return result;
    },
    [projectId],
  );

  return { images, loading, error, refresh, importImages };
}
