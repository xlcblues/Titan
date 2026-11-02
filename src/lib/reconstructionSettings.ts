export type TrainingQuality = "original" | "balanced" | "compact";

export interface ReconstructionSettings {
  quality: TrainingQuality;
  maxImagePixels: number;
  tilePairBudget: number;
  maxSplats: number;
  steps: number;
}

const STORAGE_KEY = "titan.reconstruction.v1";
export const TRAINING_STEPS_MIN = 100;
export const TRAINING_STEPS_MAX = 10000;
export const MAX_SPLATS_MIN = 10000;
export const MAX_SPLATS_MAX = 200000;

const QUALITY_PRESETS: Record<TrainingQuality, ReconstructionSettings> = {
  original: {
    quality: "original",
    maxImagePixels: 2_000_000,
    tilePairBudget: 4_000_000,
    maxSplats: 80_000,
    steps: 1200,
  },
  balanced: {
    quality: "balanced",
    maxImagePixels: 1_000_000,
    tilePairBudget: 3_000_000,
    maxSplats: 60_000,
    steps: 800,
  },
  compact: {
    quality: "compact",
    maxImagePixels: 500_000,
    tilePairBudget: 2_000_000,
    maxSplats: 50_000,
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
      tilePairBudget:
        typeof parsed.tilePairBudget === "number" && parsed.tilePairBudget > 0
          ? parsed.tilePairBudget
          : preset.tilePairBudget,
      maxSplats:
        typeof parsed.maxSplats === "number" && Number.isFinite(parsed.maxSplats)
          ? Math.min(
              MAX_SPLATS_MAX,
              Math.max(MAX_SPLATS_MIN, Math.round(parsed.maxSplats)),
            )
          : preset.maxSplats,
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
    tile_pair_budget: settings.tilePairBudget,
    tile_max_splats: 8192,
    max_splats: settings.maxSplats,
    max_splats_absolute: settings.maxSplats,
    steps: settings.steps,
    disk_cutoff: 1.7,
    falloff_power: 1.2,
    far_prune: true,
    far_prune_after: 120,
    far_prune_every: 20,
    far_prune_opacity: 0.3,
    far_prune_max_ratio: 0.7,
    far_prune_keep_min_fraction: 0.15,
    hole_fill: true,
    hole_fill_after: 100,
    hole_fill_every: 20,
    hole_fill_add: 2800,
    hole_fill_add_scene_boost: 4000,
    hole_fill_scale_px_factor: 1.1,
    hole_fill_median_scale_factor: 0.45,
    hole_fill_scale_ceiling_factor: 1.2,
    hole_fill_opacity: 0.25,
    opacity_concentration_weight: 0.04,
    final_cleanup: true,
  };
}
