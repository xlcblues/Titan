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
    scale_limit_enabled: false,
    scale_log_clamp: 6.0,
    scale_upper_floor: 0.3,
    scale_target_cap_factor: 8.0,
    far_prune: true,
    far_prune_after: 120,
    far_prune_every: 20,
    far_prune_opacity: 0.3,
    far_prune_max_ratio: 0.7,
    far_prune_keep_min_fraction: 0.15,
    hole_fill: true,
    hole_coverage_weight: 0.5,
    hole_fill_after: 80,
    hole_fill_every: 10,
    hole_fill_add: 8000,
    hole_fill_add_scene_boost: 8000,
    hole_fill_scale_px_factor: 3.5,
    hole_protected_scale_factor: 3.0,
    hole_fill_median_scale_factor: 1.0,
    hole_fill_scale_ceiling_factor: 5.0,
    hole_fill_max_screen_fraction: 0.06,
    hole_fill_area_scale_max: 3.0,
    hole_fill_cluster_enabled: true,
    hole_fill_cluster_max_px: 8.0,
    hole_fill_cluster_min_px: 3.0,
    hole_fill_cluster_cover: 1.5,
    hole_fill_cluster_children: 3,
    hole_fill_cluster_scale_factor: 0.75,
    hole_fill_opacity: 0.5,
    hole_fill_depth_ratios: [0.9, 1.0, 1.1],
    hole_fill_depth_sweep_count: 2,
    hole_fill_depth_sweep_min: 0.7,
    hole_fill_depth_sweep_max: 1.4,
    hole_fill_depth_quantile: 0.3,
    hole_fill_depth_hint_radius: 0.14,
    hole_fill_scene_depth_hint_radius: 0.3,
    hole_fill_z_min_opacity: 0.2,
    hole_fill_low_scale_px: 2.5,
    hole_fill_depth_sweep_opacity: 0.3,
    hole_fill_refine_steps: 25,
    hole_fill_refine_resolution: 96,
    hole_fill_refine_weight: 1.5,
    hole_fill_refine_min_budget: 1000,
    hole_refine_scale_weight: 0.0,
    final_hole_refine_steps: 80,
    final_hole_refine_resolution: 192,
    final_hole_refine_weight: 1.5,
    hole_fill_bg_tol: 0.14,
    hole_fill_bad_fit_tol: 0.16,
    hole_fill_hole_dilate_px: 3,
    hole_fill_neighbor_grid: 4,
    hole_fill_neighbor_step_px: 3.0,
    hole_fill_bg_depth_factor: 0.97,
    hole_fill_targets_per_hole: 2,
    hole_fill_include_bad_fit: true,
    hole_fill_protection_cycles: 3,
    hole_fill_max_per_cycle: 80000,
    hole_split_max_per_cycle: 4000,
    hole_split_children: 8,
    hole_split_min_px: 1.2,
    hole_split_spread_px: 1.5,
    hole_recycle_opacity: 0.08,
    hole_recycle_headroom: 10000,
    hole_recycle_trigger_gap: 1000,
    final_hole_recycle_headroom: 12000,
    final_prune_opacity: 0.02,
    final_keep_ratio: 0.9,
    final_hole_fill_rounds: 3,
    final_hole_fill_cameras: 8,
    final_hole_fill_overshoot: 120000,
    final_post_fill_steps: 60,
    final_post_fill_rounds: 6,
    final_post_fill_overshoot: 200000,
    final_far_prune_opacity: 0.1,
    final_prune_hole_dilate_px: 3,
    opacity_concentration_weight: 0.04,
    final_cleanup: true,
  };
}
