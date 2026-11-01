export type TrainingQuality = "original" | "balanced" | "compact";

export interface ReconstructionSettings {
  quality: TrainingQuality;
  maxImagePixels: number;
  tilePairBudget: number;
}

const STORAGE_KEY = "titan.reconstruction.v1";

const QUALITY_PRESETS: Record<TrainingQuality, ReconstructionSettings> = {
  original: {
    quality: "original",
    maxImagePixels: 2_000_000,
    tilePairBudget: 4_000_000,
  },
  balanced: {
    quality: "balanced",
    maxImagePixels: 1_000_000,
    tilePairBudget: 3_000_000,
  },
  compact: {
    quality: "compact",
    maxImagePixels: 500_000,
    tilePairBudget: 2_000_000,
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
      tilePairBudget:
        typeof parsed.tilePairBudget === "number" && parsed.tilePairBudget > 0
          ? parsed.tilePairBudget
          : preset.tilePairBudget,
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
    tile_pair_budget: settings.tilePairBudget,
    tile_max_splats: 8192,
  };
}
