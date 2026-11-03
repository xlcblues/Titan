export type TrainingQuality = "original" | "balanced" | "compact";

export interface ReconstructionSettings {
  quality: TrainingQuality;
  maxImagePixels: number;
  steps: number;
}

const STORAGE_KEY = "titan.reconstruction.v1";
export const TRAINING_STEPS_MIN = 100;
export const TRAINING_STEPS_MAX = 10000;

const QUALITY_PRESETS: Record<TrainingQuality, ReconstructionSettings> = {
  original: {
    quality: "original",
    maxImagePixels: 2_000_000,
    steps: 1200,
  },
  balanced: {
    quality: "balanced",
    maxImagePixels: 1_000_000,
    steps: 800,
  },
  compact: {
    quality: "compact",
    maxImagePixels: 500_000,
    steps: 500,
  },
};

export const TRAINING_QUALITY_OPTIONS = [
  { value: "original", label: "原分辨率" },
  { value: "balanced", label: "均衡" },
  { value: "compact", label: "紧凑" },
] as const;

export function trainingQualityPreset(
  quality: TrainingQuality,
): ReconstructionSettings {
  return { ...QUALITY_PRESETS[quality] };
}

export function defaultReconstructionSettings(): ReconstructionSettings {
  return trainingQualityPreset("original");
}

export function readReconstructionSettings(): ReconstructionSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultReconstructionSettings();
    const parsed = JSON.parse(raw) as Partial<ReconstructionSettings>;
    const preset =
      parsed.quality && QUALITY_PRESETS[parsed.quality]
        ? QUALITY_PRESETS[parsed.quality]
        : QUALITY_PRESETS.original;
    return {
      ...preset,
      maxImagePixels:
        typeof parsed.maxImagePixels === "number" && parsed.maxImagePixels > 0
          ? parsed.maxImagePixels
          : preset.maxImagePixels,
      steps:
        typeof parsed.steps === "number" && Number.isFinite(parsed.steps)
          ? Math.min(
              TRAINING_STEPS_MAX,
              Math.max(TRAINING_STEPS_MIN, Math.round(parsed.steps)),
            )
          : preset.steps,
    };
  } catch {
    return defaultReconstructionSettings();
  }
}

export function writeReconstructionSettings(
  settings: ReconstructionSettings,
): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

export function reconstructionConfig(
  settings: ReconstructionSettings,
): Record<string, unknown> {
  return {
    train_original_resolution: true,
    train_max_image_pixels: settings.maxImagePixels,
    steps: settings.steps,
    training_progress_every: 10,
  };
}
