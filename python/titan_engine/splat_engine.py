"""Differentiable facet splat renderer for the Titan engine (v0.1 baseline)."""

import math
import time

import numpy as np
import torch

from .tile_cuda import tile_cuda as _tile_cuda_loader
from .tile_cuda.autograd import tile_render


def _logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 1e-4, 1.0 - 1e-4)
    return np.log(value / (1.0 - value))


def _nearest_surface_scale_targets(
    positions: np.ndarray,
    surface_points: np.ndarray,
    multiplier: float = 0.8,
    floor: float = 1e-4,
) -> np.ndarray:
    """Per-point scale targets from local surface spacing."""
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    if len(positions) == 0:
        return np.zeros(0, dtype=np.float32)
    if len(surface_points) == 0:
        return np.full(len(positions), floor, dtype=np.float32)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(np.asarray(surface_points, dtype=np.float32))
        distances, _ = tree.query(positions, k=1)
    except Exception:
        distances = np.full(len(positions), 1.0, dtype=np.float32)
    distances = np.asarray(distances, dtype=np.float32).reshape(-1)
    return np.maximum(distances * max(multiplier, 1e-3), floor).astype(
        np.float32
    )


def _surface_neighbor_distances(
    positions: np.ndarray,
    surface_points: np.ndarray,
) -> np.ndarray:
    """Nearest sparse-surface distance for every facet center."""
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    if len(positions) == 0 or len(surface_points) == 0:
        return np.zeros(len(positions), dtype=np.float32)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(np.asarray(surface_points, dtype=np.float32))
        distances, _ = tree.query(positions, k=1)
        return np.asarray(distances, dtype=np.float32).reshape(-1)
    except Exception:
        return np.zeros(len(positions), dtype=np.float32)


def _surface_far_band(
    surface_points: np.ndarray,
    init_scale: float,
    band_factor: float = 1.25,
    min_dist_factor: float = 3.0,
) -> float:
    """Band beyond which a facet is treated as a floating shell/fog point."""
    surface = np.asarray(surface_points, dtype=np.float32)
    if len(surface) == 0:
        return max(float(init_scale) * min_dist_factor, 1e-4)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(surface)
        surface_dist, _ = tree.query(surface, k=2)
        spacing_p95 = float(np.quantile(surface_dist[:, 1], 0.95))
    except Exception:
        spacing_p95 = max(float(init_scale), 1e-4)
    return max(
        spacing_p95 * max(float(band_factor), 1.0),
        max(float(init_scale), 1e-4) * max(float(min_dist_factor), 1.0),
    )


def _surface_far_mask(
    positions: np.ndarray,
    surface_points: np.ndarray,
    band: float,
) -> np.ndarray:
    """Boolean mask for facets outside the sparse-surface band."""
    distances = _surface_neighbor_distances(positions, surface_points)
    return distances > max(float(band), 1e-4)


def _prune_far_surface_splats(
    model: "SplatFacetField",
    surface_points: np.ndarray,
    band: float,
    device: torch.device,
    opacity_threshold: float = 0.12,
    max_ratio: float = 0.6,
    keep_min_fraction: float = 0.15,
) -> int:
    """Drop low-opacity facets far outside the sparse-surface band.

    Soft penalties leave medium-opacity fog shells in the scene. Removing
    the faintest far facets periodically lets later densification and hole
    filling rebuild only the surface that is actually visible.
    """
    if (
        int(model.positions.shape[0]) == 0
        or len(surface_points) == 0
    ):
        return 0
    with torch.no_grad():
        far_mask = _surface_far_mask(
            model.positions.detach().cpu().numpy(),
            surface_points,
            band,
        )
        far_indices = torch.nonzero(
            torch.as_tensor(far_mask, dtype=torch.bool, device=device)
        ).reshape(-1)
        if far_indices.numel() == 0:
            return 0
        opacity = torch.sigmoid(model.opacity_logits)
        far_opacity = opacity[far_indices]
        candidates = far_opacity < max(float(opacity_threshold), 1e-3)
        if not torch.any(candidates):
            return 0
        far_count = int(far_indices.numel())
        keep_min = max(1, int(far_count * max(float(keep_min_fraction), 0.0)))
        remove_limit = max(
            0,
            int(far_count * min(max(float(max_ratio), 0.0), 0.95)),
        )
        candidate_indices = torch.nonzero(candidates).reshape(-1)
        remove_indices = candidate_indices[
            torch.argsort(far_opacity[candidate_indices])
        ][:remove_limit]
        if far_count - int(remove_indices.numel()) < keep_min:
            keep_order = torch.argsort(far_opacity, descending=True)[:keep_min]
            remove_mask = torch.ones(
                far_count, dtype=torch.bool, device=device
            )
            remove_mask[keep_order] = False
            remove_indices = torch.nonzero(remove_mask).reshape(-1)
        if remove_indices.numel() == 0:
            return 0
        keep_mask = torch.ones(
            model.positions.shape[0], dtype=torch.bool, device=device
        )
        keep_mask[far_indices[remove_indices]] = False
        protected = _model_hole_protected(model)
        keep_mask = keep_mask | protected
        keep_sorted, _ = torch.sort(torch.nonzero(keep_mask).reshape(-1))
        model.positions = torch.nn.Parameter(
            model.positions.detach()[keep_sorted]
        )
        model.log_scales = torch.nn.Parameter(
            model.log_scales.detach()[keep_sorted]
        )
        model.opacity_logits = torch.nn.Parameter(
            model.opacity_logits.detach()[keep_sorted]
        )
        model.color_logits = torch.nn.Parameter(
            model.color_logits.detach()[keep_sorted]
        )
        model.hole_protected = protected[keep_sorted]
        return int(remove_indices.numel())


def _scale_targets_for_splats(
    positions: np.ndarray,
    surface_points: np.ndarray | None,
    init_scale: float,
    multiplier: float = 0.5,
    floor_factor: float = 0.25,
    cap_factor: float = 1.8,
    far_scale_factor: float = 0.8,
) -> np.ndarray:
    """Per-splat scale priors from local surface spacing, capped tightly.

    Facets far outside the sparse-surface band get a small target instead of a
    large one, so floater blobs cannot grow into fog shells.
    """
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    count = int(positions.shape[0])
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    init_scale = max(float(init_scale), 1e-4)
    if surface_points is not None and len(surface_points) > 0:
        base = _nearest_surface_scale_targets(
            positions,
            np.asarray(surface_points, dtype=np.float32),
            multiplier=multiplier,
            floor=init_scale * floor_factor,
        )
        surface = np.asarray(surface_points, dtype=np.float32)
        try:
            from scipy.spatial import cKDTree

            surface_tree = cKDTree(surface)
            surface_dist, _ = surface_tree.query(surface, k=2)
            spacing_p95 = float(np.quantile(surface_dist[:, 1], 0.95))
        except Exception:
            spacing_p95 = float(np.median(base))
        band = max(spacing_p95 * 1.25, init_scale * 3.0)
        distances = _surface_neighbor_distances(positions, surface)
        base[distances > band] = init_scale * max(far_scale_factor, 0.3)
    else:
        base = np.full(count, max(init_scale * 0.5, 1e-4), dtype=np.float32)
    cap = max(init_scale * cap_factor, float(np.median(base)) * 1.5)
    return np.minimum(base, cap).astype(np.float32)


def _ssim_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    c1: float = 0.01,
    c2: float = 0.03,
) -> torch.Tensor:
    """Differentiable 1 - SSIM for small local image windows."""
    rendered = rendered.permute(2, 0, 1).unsqueeze(0).contiguous()
    target = target.permute(2, 0, 1).unsqueeze(0).contiguous()
    window_size = max(3, int(window_size))
    sigma = 1.5
    xs = torch.arange(window_size, dtype=torch.float32, device=rendered.device)
    gaussian = torch.exp(-((xs - window_size // 2) ** 2) / (2.0 * sigma * sigma))
    gaussian = gaussian / gaussian.sum()
    kernel = torch.outer(gaussian, gaussian)
    kernel = kernel / kernel.sum()
    weight = kernel.view(1, 1, window_size, window_size).repeat(3, 1, 1, 1)
    pad = window_size // 2

    def conv(image: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(
            image,
            weight,
            groups=3,
            padding=pad,
        )

    mean_x = conv(rendered)
    mean_y = conv(target)
    mean_xy = mean_x * mean_y
    var_x = torch.clamp(conv(rendered * rendered) - mean_x * mean_x, min=0.0)
    var_y = torch.clamp(conv(target * target) - mean_y * mean_y, min=0.0)
    cov_xy = conv(rendered * target) - mean_xy
    ssim_map = ((2.0 * mean_xy + c1) * (2.0 * cov_xy + c2)) / (
        (mean_x * mean_x + mean_y * mean_y + c1)
        * (var_x + var_y + c2)
    )
    return 1.0 - ssim_map.mean()


def _gradient_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Penalize Sobel gradient differences so edges stay crisp."""
    rendered = rendered.permute(2, 0, 1).unsqueeze(0).contiguous()
    target = target.permute(2, 0, 1).unsqueeze(0).contiguous()
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=rendered.device,
    ).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    kernel_y = kernel_x.transpose(2, 3).contiguous()
    gradient_x_r = torch.nn.functional.conv2d(
        rendered, kernel_x, groups=3, padding=1
    )
    gradient_y_r = torch.nn.functional.conv2d(
        rendered, kernel_y, groups=3, padding=1
    )
    gradient_x_t = torch.nn.functional.conv2d(
        target, kernel_x, groups=3, padding=1
    )
    gradient_y_t = torch.nn.functional.conv2d(
        target, kernel_y, groups=3, padding=1
    )
    return torch.mean(
        (gradient_x_r - gradient_x_t) ** 2
        + (gradient_y_r - gradient_y_t) ** 2
    )


def _estimate_point_scales(
    points: np.ndarray,
    global_scale: float,
    multiplier: float = 0.7,
) -> np.ndarray:
    """Per-point radii from local point spacing, capped by the global scale."""
    count = len(points)
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    if count == 1:
        return np.full(count, max(global_scale * 0.5, 1e-3), dtype=np.float32)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        distances, _ = tree.query(points, k=min(3, count))
        if distances.ndim == 1:
            neighbor = distances
        else:
            neighbor = distances[:, -1] if distances.shape[1] > 1 else distances[:, 0]
    except Exception:
        rng = np.random.default_rng(7)
        sample = points[rng.integers(0, count, size=min(4000, count))]
        diff = points[:, None, :] - sample[None, :, :]
        neighbor = np.min(
            np.sqrt(np.sum(diff * diff, axis=-1)) + 1e-8, axis=1
        )
    min_scale = max(global_scale * 0.5, 1e-4)
    max_scale = max(global_scale * 3.0, 1e-3)
    scales = np.clip(neighbor * multiplier, min_scale, max_scale)
    return scales.astype(np.float32)


class SplatFacetField(torch.nn.Module):
    """A scene made of small oriented circular facets (2D Gaussian splats).

    Each facet has a center, a log-scale radius, a logit opacity and a color
    in logit space. Rendering projects every facet into the camera and
    composites facets in depth order with per-pixel transmittance.
    """

    def __init__(
        self,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        scale: float = 0.12,
        scales: np.ndarray | None = None,
        opacity: float = 0.85,
    ) -> None:
        super().__init__()
        points = np.asarray(points, dtype=np.float32)
        self.register_parameter(
            "positions", torch.nn.Parameter(torch.as_tensor(points))
        )
        if colors is None:
            colors = np.full(
                (len(points), 3), 0.62, dtype=np.float32
            )
        colors = np.asarray(colors, dtype=np.float32)
        self.register_parameter(
            "color_logits",
            torch.nn.Parameter(
                torch.as_tensor(_logit(np.clip(colors, 0.0, 1.0)))
            ),
        )
        count = len(points)
        if scales is None:
            scales = np.full(count, max(scale, 1e-3), dtype=np.float32)
        scales = np.asarray(scales, dtype=np.float32).reshape(-1)
        if len(scales) != count:
            scales = np.full(count, max(scale, 1e-3), dtype=np.float32)
        self.register_parameter(
            "log_scales",
            torch.nn.Parameter(
                torch.as_tensor(np.log(np.clip(scales, 1e-4, None)), dtype=torch.float32)
            ),
        )
        self.register_parameter(
            "opacity_logits",
            torch.nn.Parameter(
                torch.full(
                    (count,),
                    _logit(np.array([opacity], dtype=np.float32))[0],
                    dtype=torch.float32,
                )
            ),
        )

    def forward(self):
        color = torch.sigmoid(self.color_logits)
        opacity = torch.sigmoid(self.opacity_logits)
        scale = torch.exp(self.log_scales)
        return self.positions, scale, opacity, color


def _camera_geometry(camera: dict, device: torch.device):
    c2w = np.asarray(camera["c2w"], dtype=np.float32)
    rotation = torch.as_tensor(c2w[:3, :3], dtype=torch.float32, device=device)
    origin = torch.as_tensor(c2w[:3, 3], dtype=torch.float32, device=device)
    k = camera["K"]
    width, height = camera["width"], camera["height"]
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    return rotation, origin, float(fx), float(fy), float(cx), float(cy), width, height


def _lowres_camera(camera: dict, low_res: int) -> dict:
    """Scale a camera's intrinsics down for a small full-image render."""
    source = int(camera["width"])
    factor = low_res / max(source, 1)
    k = camera["K"]
    return {
        "path": camera.get("path", ""),
        "width": low_res,
        "height": low_res,
        "K": [
            [float(k[0][0]) * factor, 0.0, float(k[0][2]) * factor],
            [0.0, float(k[1][1]) * factor, float(k[1][2]) * factor],
            [0.0, 0.0, 1.0],
        ],
        "c2w": camera["c2w"],
    }


def _aspect_lowres_camera(camera: dict, max_pixels: int = 96) -> dict:
    """Scale a camera down while preserving the source aspect ratio."""
    width = int(camera["width"])
    height = int(camera["height"])
    factor = max_pixels / max(width, height, 1)
    low_width = max(2, int(round(width * factor)))
    low_height = max(2, int(round(height * factor)))
    k = camera["K"]
    return {
        "path": camera.get("path", ""),
        "width": low_width,
        "height": low_height,
        "K": [
            [float(k[0][0]) * factor, 0.0, float(k[0][2]) * factor],
            [0.0, float(k[1][1]) * factor, float(k[1][2]) * factor],
            [0.0, 0.0, 1.0],
        ],
        "c2w": camera["c2w"],
    }


def _build_ray_coverage_points(
    points: np.ndarray,
    colors: np.ndarray | None,
    cameras: list[dict],
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Add ray-aligned points so every camera frustum has initial coverage."""
    count = max(0, int(count))
    if count == 0 or len(cameras) == 0:
        return points, colors
    origins = np.asarray(
        [
            np.asarray(camera["c2w"], dtype=np.float32)[:3, 3]
            for camera in cameras
        ],
        dtype=np.float32,
    )
    if len(points) > 0:
        distances = np.linalg.norm(
            points[:, None, :] - origins[None, :, :], axis=2
        )
        depth_scale = float(np.median(distances))
    else:
        centroid = np.mean(origins, axis=0)
        depth_scale = float(
            np.median(np.linalg.norm(origins - centroid[None, :], axis=1))
        )
    if not np.isfinite(depth_scale) or depth_scale <= 1e-4:
        depth_scale = 1.0
    per_camera = max(4, int(math.ceil(count / len(cameras))))
    grid = max(4, int(math.ceil(math.sqrt(per_camera))))
    depth_bins = np.asarray([0.8, 0.95, 1.12, 1.3], dtype=np.float32)
    new_points: list[np.ndarray] = []
    for camera in cameras:
        c2w = np.asarray(camera["c2w"], dtype=np.float32)
        origin = c2w[:3, 3]
        rotation = c2w[:3, :3]
        k = camera["K"]
        width, height = camera["width"], camera["height"]
        fx, fy, cx, cy = (
            float(k[0][0]),
            float(k[1][1]),
            float(k[0][2]),
            float(k[1][2]),
        )
        projected = None
        if len(points) > 0:
            rel = points - origin[None, :]
            camera_space = rel @ rotation
            zs = camera_space[:, 2]
            safe_z = np.maximum(zs, 1e-4)
            us = fx * camera_space[:, 0] / safe_z + cx
            vs = fy * camera_space[:, 1] / safe_z + cy
            visible = (
                (zs > 1e-3)
                & (us >= 0.0)
                & (us < width)
                & (vs >= 0.0)
                & (vs < height)
            )
            if np.any(visible):
                projected = (
                    us[visible],
                    vs[visible],
                    zs[visible],
                )
        cell_count = min(per_camera, grid * grid)
        for index in range(cell_count):
            row, col = divmod(index, grid)
            cell_w = width / grid
            cell_h = height / grid
            u = float(np.clip((col + 0.5) * cell_w + rng.uniform(-0.25, 0.25) * cell_w, 0.0, width - 1))
            v = float(np.clip((row + 0.5) * cell_h + rng.uniform(-0.25, 0.25) * cell_h, 0.0, height - 1))
            if projected is not None:
                du = projected[0] - u
                dv = projected[1] - v
                search = max(cell_w, cell_h) * 0.9
                local = du * du + dv * dv < search * search
                if np.any(local):
                    depth = float(np.median(projected[2][local])) * rng.uniform(
                        0.88, 1.14
                    )
                else:
                    nearest = int(np.argmin(du * du + dv * dv))
                    if (
                        du[nearest] * du[nearest] + dv[nearest] * dv[nearest]
                        < 6.25 * search * search
                    ):
                        depth = float(projected[2][nearest]) * rng.uniform(
                            0.88, 1.14
                        )
                    else:
                        depth = depth_scale * float(
                            depth_bins[index % len(depth_bins)]
                            * rng.uniform(0.9, 1.1)
                        )
            else:
                depth = depth_scale * float(
                    depth_bins[index % len(depth_bins)] * rng.uniform(0.9, 1.1)
                )
            cam_x = (u - cx) / fx * depth
            cam_y = (v - cy) / fy * depth
            world = origin + np.asarray([cam_x, cam_y, depth], dtype=np.float32) @ rotation.T
            new_points.append(world)
    if not new_points:
        return points, colors
    new_points_array = np.asarray(new_points, dtype=np.float32)[:count]
    merged_points = (
        points if len(points) else np.empty((0, 3), dtype=np.float32)
    )
    merged_points = np.concatenate([merged_points, new_points_array], axis=0)
    if colors is None:
        return merged_points, None
    new_colors = np.zeros((len(new_points_array), 3), dtype=np.float32)
    merged_colors = np.concatenate([colors, new_colors], axis=0)
    return merged_points, merged_colors


def _downsample_images(
    images: np.ndarray,
    low_res: int,
) -> np.ndarray:
    """Build per-view low-resolution targets for the global residual loss."""
    from PIL import Image

    low = []
    for image in images:
        source = Image.fromarray(
            np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        )
        low.append(
            np.asarray(
                source.resize((low_res, low_res), Image.LANCZOS),
                dtype=np.float32,
            )
            / 255.0
        )
    return np.stack(low, axis=0)


def _project_points(
    positions: torch.Tensor,
    rotation: torch.Tensor,
    origin: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    near: float,
):
    rel = positions - origin[None, :]
    camera_space = rel @ rotation
    z = camera_space[:, 2:3]
    x = camera_space[:, 0:1]
    y = camera_space[:, 1:2]
    visible = z > near
    safe_z = torch.clamp(z, min=1e-4)
    u = fx * x / safe_z + cx
    v = fy * y / safe_z + cy
    return camera_space, u, v, z, visible


def _accumulate_alpha(
    alphas: torch.Tensor,
    colors: torch.Tensor,
    transmittance: torch.Tensor,
    accumulated: torch.Tensor,
):
    """Depth-ordered alpha compositing with per-splat transmittance decay.

    alphas has shape (chunk, pixels) and is ordered near-to-far. Without the
    intra-chunk cumulative product, splats inside one chunk would all use the
    same transmittance, which brightens and fogs dense scenes.
    """
    survive = torch.cumprod(1.0 - alphas, dim=0)
    before = torch.ones_like(alphas)
    before[1:] = survive[:-1]
    per_splat_transmittance = transmittance[None, :] * before
    contribution = (
        alphas[:, :, None]
        * colors[:, None, :]
        * per_splat_transmittance[:, :, None]
    )
    accumulated = accumulated + contribution.sum(dim=0)
    transmittance = transmittance * survive[-1]
    return accumulated, transmittance


def _render_pixels_dense(
    model: SplatFacetField,
    camera: dict,
    pixels: torch.Tensor | None,
    width: int,
    height: int,
    background: torch.Tensor,
    near: float | None = None,
    far: float | None = None,
    scene_points: np.ndarray | None = None,
    batch_size: int = 96,
    disk_cutoff: float = 2.2,
    falloff_power: float = 1.0,
):
    device = next(model.parameters()).device
    rotation, origin, fx, fy, cx, cy, width, height = _camera_geometry(
        camera, device
    )
    positions, scale, opacity, color = model()
    if near is None or far is None:
        if scene_points is not None and len(scene_points) > 0:
            from .main import ray_bounds_from_points

            bounds = ray_bounds_from_points(
                origin.cpu().detach().numpy(),
                scene_points,
                rotation=rotation.cpu().detach().numpy(),
            )
        else:
            depths = torch.linalg.norm(
                positions - origin[None, :].to(device), dim=1
            )
            if depths.numel() == 0:
                bounds = (0.3, 7.6)
            else:
                lower = float(
                    torch.quantile(depths.detach(), 0.05).item()
                )
                upper = float(
                    torch.quantile(depths.detach(), 0.95).item()
                )
                bounds = (
                    max(lower * 0.75, 1e-4),
                    max(upper * 1.35, lower * 1.5 + 1e-4),
                )
        if near is None:
            near = bounds[0]
        if far is None:
            far = bounds[1]
    camera_space, u, v, z, visible = _project_points(
        positions,
        rotation,
        origin,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        near,
    )
    if far is not None:
        visible = visible & (z < far)

    if pixels is None:
        uu, vv = torch.meshgrid(
            torch.arange(width, dtype=torch.float32, device=device),
            torch.arange(height, dtype=torch.float32, device=device),
            indexing="xy",
        )
        pixels = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
        pixel_count = pixels.shape[0]
        output_shape = (height, width, 3)
    else:
        pixels = pixels.to(device)
        pixel_count = pixels.shape[0]
        output_shape = (pixel_count, 3)

    sigma_u = scale[:, None] * fx / torch.clamp(z, min=1e-4)
    sigma_v = scale[:, None] * fy / torch.clamp(z, min=1e-4)
    radius_u = 2.8 * sigma_u[:, 0]
    radius_v = 2.8 * sigma_v[:, 0]
    margin = 4.0
    active = (
        visible[:, 0]
        & (u[:, 0] > -radius_u - margin)
        & (u[:, 0] < width + radius_u + margin)
        & (v[:, 0] > -radius_v - margin)
        & (v[:, 0] < height + radius_v + margin)
    )
    indices = torch.nonzero(active).reshape(-1)
    if indices.numel() == 0:
        rendered = background[None, :].expand(pixel_count, -1)
        return rendered.reshape(output_shape)

    order = indices[torch.argsort(z[indices, 0], dim=0)]
    point_count = order.shape[0]
    transmittance = torch.ones(pixel_count, dtype=torch.float32, device=device)
    accumulated = torch.zeros(
        (pixel_count, 3), dtype=torch.float32, device=device
    )

    for start in range(0, point_count, batch_size):
        chunk = order[start : start + batch_size]
        du = pixels[:, 0][None, :] - u[chunk]
        dv = pixels[:, 1][None, :] - v[chunk]
        squared = (
            du * du / (2.0 * sigma_u[chunk] * sigma_u[chunk])
            + dv * dv / (2.0 * sigma_v[chunk] * sigma_v[chunk])
        )
        if falloff_power == 1.0:
            weight = torch.exp(-squared)
        else:
            weight = torch.exp(-torch.pow(squared, falloff_power))
        weight = weight * (squared < 0.5 * disk_cutoff * disk_cutoff)
        alpha = opacity[chunk, None] * weight
        alpha = torch.clamp(alpha, 0.0, 1.0)
        accumulated, transmittance = _accumulate_alpha(
            alpha, color[chunk], transmittance, accumulated
        )

    rendered = accumulated + transmittance[:, None] * background[None, :]
    return rendered.reshape(output_shape)


def _segment_log_cumsum(
    log_survive: torch.Tensor,
    segment_start: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Segment-local cumulative log survival for contiguous pixel segments.

    A global ``cumsum`` loses precision when it accumulates millions of
    splat-pixel pairs and is later unwound with ``cum[start] - cum[end]``.
    Packing each segment into a padded 2D matrix keeps the accumulated values
    small and matches a per-pixel sequential compositor to float rounding.
    """
    device = log_survive.device
    n = int(log_survive.numel())
    segment_indexes = torch.nonzero(segment_start).reshape(-1)
    segment_count = int(segment_indexes.numel())
    elem_segment = torch.cumsum(segment_start.to(torch.int64), 0) - 1
    bounds = torch.cat(
        [
            segment_indexes,
            torch.tensor([n], dtype=torch.int64, device=device),
        ],
        dim=0,
    )
    lengths = bounds[1:] - bounds[:-1]
    max_len = int(lengths.max().item()) if segment_count > 0 else 0
    local_pos = torch.arange(n, device=device) - segment_indexes[elem_segment]
    local_cum = torch.empty_like(log_survive)
    matrix_elems = 6_000_000
    segs_per_batch = max(1, matrix_elems // max_len) if max_len > 0 else 1

    for start in range(0, segment_count, segs_per_batch):
        stop = min(segment_count, start + segs_per_batch)
        seg_bounds = bounds[start : stop + 1]
        seg_count_b = stop - start
        seg_lengths_b = seg_bounds[1:] - seg_bounds[:-1]
        max_len_b = int(seg_lengths_b.max().item())
        rows = torch.arange(max_len_b, device=device)[:, None]
        cols = torch.arange(seg_count_b, device=device)[None, :]
        mask = rows < seg_lengths_b[None, :]
        src_idx = seg_bounds[:-1][None, :] + rows
        values = log_survive[src_idx.clamp(max=n - 1)]
        matrix = torch.where(
            mask,
            values,
            torch.zeros((), dtype=log_survive.dtype, device=device),
        )
        matrix_cum = torch.cumsum(matrix, dim=0)
        pair_start = int(seg_bounds[0].item())
        pair_end = int(seg_bounds[-1].item())
        row_b = local_pos[pair_start:pair_end]
        col_b = elem_segment[pair_start:pair_end] - start
        flat_idx = row_b * seg_count_b + col_b
        local_cum[pair_start:pair_end] = matrix_cum.reshape(-1)[flat_idx]

    before = torch.exp((local_cum - log_survive).clamp(max=0.0)).float()
    opaque = torch.isinf(log_survive) & (log_survive < 0)
    before = torch.where(
        opaque & segment_start,
        torch.ones_like(before),
        before,
    )
    before = torch.where(
        opaque & ~segment_start,
        torch.zeros_like(before),
        before,
    )
    segment_end = bounds[1:] - 1
    segment_total = local_cum[segment_end]
    return before, segment_end, segment_total


def _gpu_prefix_edges(
    values: torch.Tensor,
    limit: float,
    max_count: int | None = None,
    hard_split: torch.Tensor | None = None,
) -> torch.Tensor:
    """Boundaries for prefix sums that stay below ``limit`` on the GPU.

    Values larger than the limit are placed in their own span, and hard splits
    (for example oversize tiles) are isolated on both sides.
    """
    device = values.device
    n = int(values.numel())
    if n == 0:
        return torch.zeros(1, dtype=torch.int64, device=device)
    limit = max(float(limit), 1.0)
    bounded = torch.minimum(values, torch.full_like(values, limit))
    bucket = torch.div(
        torch.cumsum(bounded, dim=0),
        limit,
        rounding_mode="floor",
    ).long()
    positions = [torch.zeros(1, dtype=torch.int64, device=device)]
    changed = torch.nonzero(bucket[1:] > bucket[:-1]).reshape(-1) + 1
    if changed.numel() > 0:
        positions.append(changed)
    oversize = torch.nonzero(values >= limit).reshape(-1)
    if oversize.numel() > 0:
        positions.append(oversize + 1)
    if hard_split is not None:
        hard = torch.nonzero(hard_split).reshape(-1)
        if hard.numel() > 0:
            positions.append(hard)
            positions.append(hard + 1)
    if max_count is not None and 0 < max_count < n:
        positions.append(
            torch.arange(max_count, n, max_count, device=device)
        )
    edges = torch.unique(torch.cat(positions))
    if int(edges[0].item()) != 0:
        edges = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=device),
                edges,
            ],
            dim=0,
        )
    if int(edges[-1].item()) != n:
        edges = torch.cat(
            [
                edges,
                torch.full(
                    (1,),
                    n,
                    dtype=torch.int64,
                    device=device,
                ),
            ],
            dim=0,
        )
    return edges


def _render_pixels_tile(
    model: SplatFacetField,
    camera: dict,
    pixels: torch.Tensor | None,
    width: int,
    height: int,
    background: torch.Tensor,
    near: float | None = None,
    far: float | None = None,
    scene_points: np.ndarray | None = None,
    batch_size: int = 256,
    disk_cutoff: float = 2.2,
    falloff_power: float = 1.0,
    tile_size: int = 16,
    tile_pair_budget: int = 4_000_000,
    tile_max_splats: int = 8192,
):
    """GPU tile rasterizer that composites splat-pixel pairs in parallel.

    Splats are first expanded to the tiles they overlap. Tiles are then
    grouped into GPU chunks whose pair count fits the VRAM budget, and every
    pixel batch inside a chunk composites all of its splats with vectorized
    segment products instead of the old per-tile Python loop.
    """
    device = next(model.parameters()).device
    tile_size = max(1, int(tile_size))
    rotation, origin, fx, fy, cx, cy, width, height = _camera_geometry(
        camera, device
    )
    positions, scale, opacity, color = model()
    if near is None or far is None:
        if scene_points is not None and len(scene_points) > 0:
            from .main import ray_bounds_from_points

            bounds = ray_bounds_from_points(
                origin.cpu().detach().numpy(),
                scene_points,
                rotation=rotation.cpu().detach().numpy(),
            )
        else:
            depths = torch.linalg.norm(
                positions - origin[None, :].to(device), dim=1
            )
            if depths.numel() == 0:
                bounds = (0.3, 7.6)
            else:
                lower = float(
                    torch.quantile(depths.detach(), 0.05).item()
                )
                upper = float(
                    torch.quantile(depths.detach(), 0.95).item()
                )
                bounds = (
                    max(lower * 0.75, 1e-4),
                    max(upper * 1.35, lower * 1.5 + 1e-4),
                )
        if near is None:
            near = bounds[0]
        if far is None:
            far = bounds[1]
    camera_space, u, v, z, visible = _project_points(
        positions,
        rotation,
        origin,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        near,
    )
    if far is not None:
        visible = visible & (z < far)

    if pixels is None:
        uu, vv = torch.meshgrid(
            torch.arange(width, dtype=torch.float32, device=device),
            torch.arange(height, dtype=torch.float32, device=device),
            indexing="xy",
        )
        pixels = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
        pixel_count = pixels.shape[0]
        output_shape = (height, width, 3)
    else:
        pixels = pixels.to(device)
        pixel_count = pixels.shape[0]
        output_shape = (pixel_count, 3)

    sigma_u = scale[:, None] * fx / torch.clamp(z, min=1e-4)
    sigma_v = scale[:, None] * fy / torch.clamp(z, min=1e-4)
    radius_u = 2.8 * sigma_u[:, 0]
    radius_v = 2.8 * sigma_v[:, 0]
    margin = 4.0
    active = (
        visible[:, 0]
        & (u[:, 0] > -radius_u - margin)
        & (u[:, 0] < width + radius_u + margin)
        & (v[:, 0] > -radius_v - margin)
        & (v[:, 0] < height + radius_v + margin)
    )
    active_indices = torch.nonzero(active).reshape(-1)
    if active_indices.numel() == 0:
        rendered = background[None, :].expand(pixel_count, -1)
        return rendered.reshape(output_shape)

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    min_u = u[:, 0] - radius_u - margin
    max_u = u[:, 0] + radius_u + margin
    min_v = v[:, 0] - radius_v - margin
    max_v = v[:, 0] + radius_v + margin
    min_tile_u = torch.clamp(
        torch.floor(min_u / tile_size).long(), 0, tiles_x - 1
    )
    max_tile_u = torch.clamp(
        torch.floor(max_u / tile_size).long(), 0, tiles_x - 1
    )
    min_tile_v = torch.clamp(
        torch.floor(min_v / tile_size).long(), 0, tiles_y - 1
    )
    max_tile_v = torch.clamp(
        torch.floor(max_v / tile_size).long(), 0, tiles_y - 1
    )

    pixel_tile_u = torch.clamp(
        torch.floor(pixels[:, 0] / tile_size).long(), 0, tiles_x - 1
    )
    pixel_tile_v = torch.clamp(
        torch.floor(pixels[:, 1] / tile_size).long(), 0, tiles_y - 1
    )
    pixel_keys = pixel_tile_u + tiles_x * pixel_tile_v
    pixel_order = torch.argsort(pixel_keys)
    sorted_pixels = pixels[pixel_order]
    sorted_keys = pixel_keys[pixel_order]
    pixel_unique, pixel_counts = torch.unique_consecutive(
        sorted_keys, return_counts=True
    )
    pixel_starts = torch.cumsum(pixel_counts, 0) - pixel_counts

    a_min_u = min_tile_u[active_indices]
    a_max_u = max_tile_u[active_indices]
    a_min_v = min_tile_v[active_indices]
    a_max_v = max_tile_v[active_indices]
    span_u = a_max_u - a_min_u + 1
    span_v = a_max_v - a_min_v + 1
    products = span_u * span_v
    max_expand = max(
        1,
        min(
            4096,
            2_000_000 // max(int(active_indices.numel()), 1),
        ),
    )
    broad = products > max_expand
    broad_idx = active_indices[broad]
    small_mask = ~broad
    small_products = products[small_mask]
    small_idx = active_indices[small_mask]

    if small_idx.numel() > 0:
        small_span_u = span_u[small_mask]
        small_min_u = a_min_u[small_mask]
        small_min_v = a_min_v[small_mask]
        total_entries = int(small_products.sum().item())
        group_ids = torch.repeat_interleave(
            torch.arange(small_idx.numel(), device=device),
            small_products,
        )
        local = torch.arange(total_entries, device=device) - torch.repeat_interleave(
            torch.cumsum(small_products, 0) - small_products,
            small_products,
        )
        repeated_span = small_span_u[group_ids]
        entry_tile_u = small_min_u[group_ids] + local % repeated_span
        entry_tile_v = small_min_v[group_ids] + local // repeated_span
        entry_keys = entry_tile_u + tiles_x * entry_tile_v
        entry_splats = small_idx[group_ids]
        entry_order = torch.argsort(entry_keys)
        entry_keys = entry_keys[entry_order]
        entry_splats = entry_splats[entry_order]
    else:
        entry_keys = torch.empty(0, dtype=torch.long, device=device)
        entry_splats = torch.empty(0, dtype=torch.long, device=device)

    if broad_idx.numel() > 0:
        broad_min_u = min_tile_u[broad_idx]
        broad_max_u = max_tile_u[broad_idx]
        broad_min_v = min_tile_v[broad_idx]
        broad_max_v = max_tile_v[broad_idx]
    else:
        broad_min_u = broad_max_u = broad_min_v = broad_max_v = None

    entry_start = torch.searchsorted(entry_keys, pixel_unique)
    entry_end = torch.searchsorted(entry_keys, pixel_unique + 1)
    output = torch.zeros(
        (pixel_count, 3), dtype=torch.float32, device=device
    )

    broad_count = int(broad_idx.numel()) if broad_idx.numel() > 0 else 0
    splat_z = z[:, 0]
    tile_keys = pixel_unique
    tile_pixel_counts = pixel_counts
    tile_u = tile_keys % tiles_x
    tile_v = tile_keys // tiles_x
    small_counts = entry_end - entry_start
    broad_block = 4096

    # Count broad-splat coverage per tile in bounded GPU blocks so chunk
    # boundaries use the real pair load instead of a full B x T matrix.
    broad_counts = torch.zeros(
        tile_keys.numel(), dtype=torch.int32, device=device
    )
    if broad_count > 0:
        for start in range(0, broad_count, broad_block):
            stop = min(broad_count, start + broad_block)
            bu_min = broad_min_u[start:stop]
            bu_max = broad_max_u[start:stop]
            bv_min = broad_min_v[start:stop]
            bv_max = broad_max_v[start:stop]
            covered_u = (
                (bu_min[:, None] <= tile_u[None, :])
                & (bu_max[:, None] >= tile_u[None, :])
            )
            covered_v = (
                (bv_min[:, None] <= tile_v[None, :])
                & (bv_max[:, None] >= tile_v[None, :])
            )
            broad_counts += (covered_u & covered_v).sum(dim=0).to(
                torch.int32
            )

    # Tile pair estimates drive segment boundaries computed entirely on the
    # GPU: every segment fits the pair budget and the 4096-tile chunk limit,
    # and oversized tiles are isolated so expansion never exhausts VRAM.
    tile_entry_counts = small_counts + broad_counts
    tile_estimate = (
        tile_entry_counts.to(torch.float64)
        * tile_pixel_counts.to(torch.float64)
    )
    max_chunk_tiles = 4096
    segment_edges = _gpu_prefix_edges(
        tile_estimate,
        tile_pair_budget,
        max_count=max_chunk_tiles,
        hard_split=tile_entry_counts > tile_max_splats,
    ).tolist()

    for segment_index in range(len(segment_edges) - 1):
        t0 = segment_edges[segment_index]
        t1 = segment_edges[segment_index + 1]
        chunk_keys = tile_keys[t0:t1]
        chunk_tile_u = chunk_keys % tiles_x
        chunk_tile_v = chunk_keys // tiles_x
        p0 = int(pixel_starts[t0].item())
        p1 = int(
            pixel_starts[t0].item()
            + int(pixel_counts[t0 : t0 + (t1 - t0)].sum().item())
        )

        chunk_tile_entry_counts = entry_end[t0:t1] - entry_start[t0:t1]
        small_splat_count = int(chunk_tile_entry_counts.sum().item())
        if small_splat_count > 0:
            small_splats = entry_splats[
                int(entry_start[t0].item()) :
                int(entry_start[t0].item()) + small_splat_count
            ]
            small_splat_tiles = torch.repeat_interleave(
                torch.arange(t1 - t0, device=device),
                chunk_tile_entry_counts,
            )
        else:
            small_splats = torch.empty(0, dtype=torch.long, device=device)
            small_splat_tiles = torch.empty(0, dtype=torch.long, device=device)

        # Broad splats are expanded to (tile, splat) pairs in the same GPU
        # block scheme, replacing the old broad_count x chunk_tile matrix.
        broad_pair_splats: list[torch.Tensor] = []
        broad_pair_tiles: list[torch.Tensor] = []
        if broad_count > 0 and t1 > t0:
            for start in range(0, broad_count, broad_block):
                stop = min(broad_count, start + broad_block)
                bu_min = broad_min_u[start:stop]
                bu_max = broad_max_u[start:stop]
                bv_min = broad_min_v[start:stop]
                bv_max = broad_max_v[start:stop]
                covered_u = (
                    (bu_min[:, None] <= chunk_tile_u[None, :])
                    & (bu_max[:, None] >= chunk_tile_u[None, :])
                )
                covered_v = (
                    (bv_min[:, None] <= chunk_tile_v[None, :])
                    & (bv_max[:, None] >= chunk_tile_v[None, :])
                )
                covered_rows, covered_cols = torch.nonzero(
                    covered_u & covered_v, as_tuple=True
                )
                if covered_rows.numel() > 0:
                    broad_pair_splats.append(broad_idx[start + covered_rows])
                    broad_pair_tiles.append(covered_cols)
        if broad_pair_splats:
            broad_splats = torch.cat(broad_pair_splats, dim=0)
            broad_tiles = torch.cat(broad_pair_tiles, dim=0)
        else:
            broad_splats = torch.empty(0, dtype=torch.long, device=device)
            broad_tiles = torch.empty(0, dtype=torch.long, device=device)

        if small_splat_count > 0 and broad_splats.numel() > 0:
            chunk_splats = torch.cat([small_splats, broad_splats], dim=0)
            chunk_splat_tile = torch.cat(
                [small_splat_tiles, broad_tiles], dim=0
            )
        elif small_splat_count > 0:
            chunk_splats = small_splats
            chunk_splat_tile = small_splat_tiles
        elif broad_splats.numel() > 0:
            chunk_splats = broad_splats
            chunk_splat_tile = broad_tiles
        else:
            chunk_splats = torch.empty(0, dtype=torch.long, device=device)
            chunk_splat_tile = torch.empty(0, dtype=torch.long, device=device)

        if chunk_splats.numel() == 0 or p1 <= p0:
            output[pixel_order[p0:p1]] = background[None, :].expand(
                p1 - p0, -1
            )
            continue

        chunk_z = splat_z[chunk_splats]
        # One packed 64-bit key gives the (tile, depth) order: the low 32
        # bits preserve the positive float32 depth ordering, the high bits
        # sort by tile.
        depth_key = chunk_z.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        order_key = chunk_splat_tile.to(torch.int64) << 32 | depth_key
        order = torch.argsort(order_key, stable=True)
        ordered_splats = chunk_splats[order]
        ordered_tiles = chunk_splat_tile[order]
        chunk_u = u[ordered_splats, 0]
        chunk_v = v[ordered_splats, 0]
        chunk_sigma_u = sigma_u[ordered_splats, 0]
        chunk_sigma_v = sigma_v[ordered_splats, 0]
        chunk_opacity = opacity[ordered_splats]
        chunk_color = color[ordered_splats]

        chunk_entry_counts = (
            entry_end[t0:t1] - entry_start[t0:t1] + broad_counts[t0:t1]
        )
        entry_cum = torch.cumsum(chunk_entry_counts, 0)
        tile_pixel_cum = torch.cumsum(pixel_counts[t0:t1], 0)

        # The GPU segment boundaries already bound the real pair load, so
        # each segment is expanded as one group without a second Python pass.
        group_edges = [0, int(t1 - t0)]

        for group_index in range(len(group_edges) - 1):
            g0 = group_edges[group_index]
            g1 = group_edges[group_index + 1]
            group_splat_start = (
                int(entry_cum[g0 - 1].item()) if g0 > 0 else 0
            )
            group_splat_end = int(entry_cum[g1 - 1].item())
            group_splats = ordered_splats[group_splat_start:group_splat_end]
            group_tiles = ordered_tiles[group_splat_start:group_splat_end]
            if group_splats.numel() == 0:
                continue
            group_pixel_start = (
                int(tile_pixel_cum[g0 - 1].item()) if g0 > 0 else 0
            )
            group_pixel_end = int(tile_pixel_cum[g1 - 1].item())
            group_batch_count = group_pixel_end - group_pixel_start
            group_coords = sorted_pixels[
                p0 + group_pixel_start : p0 + group_pixel_end
            ]
            group_pixel_counts = pixel_counts[t0 + g0 : t0 + g1]
            group_tiles_local = group_tiles - g0
            num_group_tiles = g1 - g0
            tile_offsets = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int64, device=device),
                    torch.cumsum(
                        group_pixel_counts[:-1], 0
                    ),
                ],
                dim=0,
            )
            # Build pixel-major (tile, pixel, depth) pairs in one pass:
            # each pixel repeats all splats of its tile in depth order, so
            # same-pixel pairs stay contiguous without a global argsort.
            tile_splat_counts = torch.bincount(
                group_tiles_local, minlength=num_group_tiles
            )
            tile_splat_offsets = (
                torch.cumsum(tile_splat_counts, 0) - tile_splat_counts
            )
            tile_pair_counts = tile_splat_counts * group_pixel_counts
            pair_count = int(tile_pair_counts.sum().item())
            if pair_count == 0:
                continue
            pair_tile = torch.repeat_interleave(
                torch.arange(num_group_tiles, device=device),
                tile_pair_counts,
            )
            tile_pair_starts = (
                torch.cumsum(tile_pair_counts, 0) - tile_pair_counts
            )
            local_index = (
                torch.arange(pair_count, device=device)
                - torch.repeat_interleave(
                    tile_pair_starts, tile_pair_counts
                )
            )
            splat_per_tile = torch.repeat_interleave(
                tile_splat_counts, tile_pair_counts
            )
            pair_splat = (
                tile_splat_offsets[pair_tile]
                + local_index % splat_per_tile
            )
            pair_pixel = (
                tile_offsets[pair_tile]
                + local_index // splat_per_tile
            )
            group_u = chunk_u[group_splat_start:group_splat_end]
            group_v = chunk_v[group_splat_start:group_splat_end]
            group_sigma_u = chunk_sigma_u[
                group_splat_start:group_splat_end
            ]
            group_sigma_v = chunk_sigma_v[
                group_splat_start:group_splat_end
            ]
            group_opacity = chunk_opacity[
                group_splat_start:group_splat_end
            ]
            group_color = chunk_color[
                group_splat_start:group_splat_end
            ]
            pair_u = group_u[pair_splat]
            pair_v = group_v[pair_splat]
            pair_sigma_u = group_sigma_u[pair_splat]
            pair_sigma_v = group_sigma_v[pair_splat]
            pair_color = group_color[pair_splat]
            du = group_coords[:, 0][pair_pixel] - pair_u
            dv = group_coords[:, 1][pair_pixel] - pair_v
            squared = (
                du * du / (2.0 * pair_sigma_u * pair_sigma_u)
                + dv * dv / (2.0 * pair_sigma_v * pair_sigma_v)
            )
            if falloff_power == 1.0:
                weight = torch.exp(-squared)
            else:
                weight = torch.exp(-torch.pow(squared, falloff_power))
            weight = weight * (
                squared < 0.5 * disk_cutoff * disk_cutoff
            )
            alpha = torch.clamp(
                group_opacity[pair_splat] * weight, 0.0, 1.0
            )

            seg_pixel = pair_pixel
            seg_alpha = alpha
            seg_color = pair_color
            log_survive = torch.log1p(-seg_alpha).double()
            segment_start = torch.cat(
                [
                    torch.ones(1, dtype=torch.bool, device=device),
                    seg_pixel[1:] != seg_pixel[:-1],
                ],
                dim=0,
            )
            before, segment_end, segment_total = _segment_log_cumsum(
                log_survive, segment_start
            )
            contribution = seg_alpha[:, None] * before[:, None] * seg_color
            acc = torch.zeros(
                group_batch_count, 3, dtype=torch.float32, device=device
            )
            acc.index_add_(0, seg_pixel, contribution)
            transmittance = torch.ones(group_batch_count, device=device)
            transmittance[seg_pixel[segment_end]] = torch.exp(
                segment_total
            ).float()
            rendered = acc + transmittance[:, None] * background[None, :]
            output[
                pixel_order[
                    p0 + group_pixel_start : p0 + group_pixel_end
                ]
            ] = rendered

    return output.reshape(output_shape)


def _render_pixels_cuda(
    model: SplatFacetField,
    camera: dict,
    pixels: torch.Tensor | None,
    width: int,
    height: int,
    background: torch.Tensor,
    near: float | None = None,
    far: float | None = None,
    scene_points: np.ndarray | None = None,
    disk_cutoff: float = 2.2,
    falloff_power: float = 1.0,
    tile_size: int = 16,
    tile_pair_budget: int = 4_000_000,
    tile_max_splats: int = 8192,
    tile_work_budget: int = 2_000_000,
):
    """Render through the CUDA tile rasterizer with autograd support."""
    device = next(model.parameters()).device
    tile_size = max(1, int(tile_size))
    rotation, origin, fx, fy, cx, cy, width, height = _camera_geometry(
        camera, device
    )
    if near is None or far is None:
        if scene_points is not None and len(scene_points) > 0:
            from .main import ray_bounds_from_points

            bounds = ray_bounds_from_points(
                origin.cpu().detach().numpy(),
                scene_points,
                rotation=rotation.cpu().detach().numpy(),
            )
        else:
            depths = torch.linalg.norm(
                model.positions - origin[None, :], dim=1
            )
            if depths.numel() == 0:
                bounds = (0.3, 7.6)
            else:
                lower = float(
                    torch.quantile(depths.detach(), 0.05).item()
                )
                upper = float(
                    torch.quantile(depths.detach(), 0.95).item()
                )
                bounds = (
                    max(lower * 0.75, 1e-4),
                    max(upper * 1.35, lower * 1.5 + 1e-4),
                )
        if near is None:
            near = bounds[0]
        if far is None:
            far = bounds[1]
    if pixels is None:
        uu, vv = torch.meshgrid(
            torch.arange(width, dtype=torch.float32, device=device),
            torch.arange(height, dtype=torch.float32, device=device),
            indexing="xy",
        )
        pixels = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
        output_shape = (height, width, 3)
    else:
        pixels = pixels.to(device)
        output_shape = (pixels.shape[0], 3)
    rendered = tile_render(
        model.positions,
        model.log_scales,
        model.opacity_logits,
        model.color_logits,
        rotation,
        origin,
        pixels,
        background,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        tile_size=tile_size,
        near=near,
        far=far,
        disk_cutoff=disk_cutoff,
        falloff_power=falloff_power,
        tile_pair_budget=tile_pair_budget,
        tile_max_splats=tile_max_splats,
        tile_work_budget=tile_work_budget,
    )
    return rendered.reshape(output_shape)


def _render_pixels(
    model: SplatFacetField,
    camera: dict,
    pixels: torch.Tensor | None,
    width: int,
    height: int,
    background: torch.Tensor,
    near: float | None = None,
    far: float | None = None,
    scene_points: np.ndarray | None = None,
    batch_size: int = 96,
    disk_cutoff: float = 2.2,
    falloff_power: float = 1.0,
    tile_size: int = 16,
    dense_renderer: bool = False,
    auto_dense_threshold: int = 20000,
    auto_dense_pixels: int = 1_000_000,
    tile_pair_budget: int = 4_000_000,
    tile_max_splats: int = 8192,
    tile_work_budget: int = 2_000_000,
    use_cuda_tile: bool = False,
):
    if use_cuda_tile:
        try:
            if _tile_cuda_loader() is not None:
                return _render_pixels_cuda(
                    model,
                    camera,
                    pixels,
                    width,
                    height,
                    background,
                    near=near,
                    far=far,
                    scene_points=scene_points,
                    disk_cutoff=disk_cutoff,
                    falloff_power=falloff_power,
                    tile_size=tile_size,
                    tile_pair_budget=tile_pair_budget,
                    tile_max_splats=tile_max_splats,
                    tile_work_budget=tile_work_budget,
                )
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:
            pass
    splat_count = int(model.positions.numel())
    pixel_count = (
        int(pixels.shape[0]) if pixels is not None else int(width) * int(height)
    )
    use_dense = dense_renderer or (
        splat_count <= auto_dense_threshold
        and pixel_count <= auto_dense_pixels
    )
    if use_dense:
        return _render_pixels_dense(
            model,
            camera,
            pixels,
            width,
            height,
            background,
            near=near,
            far=far,
            scene_points=scene_points,
            batch_size=batch_size,
            disk_cutoff=disk_cutoff,
            falloff_power=falloff_power,
        )
    try:
        return _render_pixels_tile(
            model,
            camera,
            pixels,
            width,
            height,
            background,
            near=near,
            far=far,
            scene_points=scene_points,
            batch_size=batch_size,
            disk_cutoff=disk_cutoff,
            falloff_power=falloff_power,
            tile_size=tile_size,
            tile_pair_budget=tile_pair_budget,
            tile_max_splats=tile_max_splats,
        )
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:
        return _render_pixels_dense(
            model,
            camera,
            pixels,
            width,
            height,
            background,
            near=near,
            far=far,
            scene_points=scene_points,
            batch_size=batch_size,
            disk_cutoff=disk_cutoff,
            falloff_power=falloff_power,
        )


def render_splats_view(
    model: SplatFacetField,
    camera: dict,
    background: np.ndarray,
    scene_points: np.ndarray | None = None,
    batch_size: int = 96,
    disk_cutoff: float = 2.2,
    falloff_power: float = 1.0,
    tile_size: int = 16,
    dense_renderer: bool = False,
    auto_dense_threshold: int = 20000,
    auto_dense_pixels: int = 1_000_000,
    tile_pair_budget: int = 4_000_000,
    tile_max_splats: int = 8192,
    tile_work_budget: int = 2_000_000,
    use_cuda_tile: bool = False,
) -> np.ndarray:
    device = next(model.parameters()).device
    bg = torch.as_tensor(background, dtype=torch.float32, device=device)
    with torch.no_grad():
        image = _render_pixels(
            model,
            camera,
            None,
            camera["width"],
            camera["height"],
            bg,
            scene_points=scene_points,
            batch_size=batch_size,
            disk_cutoff=disk_cutoff,
            falloff_power=falloff_power,
            tile_size=tile_size,
            dense_renderer=dense_renderer,
            auto_dense_threshold=auto_dense_threshold,
            auto_dense_pixels=auto_dense_pixels,
            tile_pair_budget=tile_pair_budget,
            tile_max_splats=tile_max_splats,
            tile_work_budget=tile_work_budget,
            use_cuda_tile=use_cuda_tile,
        )
    return image.cpu().detach().numpy()


def _density_prior_loss(
    model: SplatFacetField,
    surface_points: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
    count: int = 1024,
    margin: float | None = None,
):
    """Keep facets inside the observed scene volume instead of drifting far."""
    if len(surface_points) == 0 or model.positions.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    position_count = model.positions.shape[0]
    position_indices = rng.integers(0, position_count, size=count)
    positions = model.positions[position_indices]
    lower = np.quantile(surface_points, 0.03, axis=0)
    upper = np.quantile(surface_points, 0.97, axis=0)
    if margin is None:
        span = float(np.mean(upper - lower))
        margin = max(span * 0.05, 1e-3)
    lower_t = torch.as_tensor(
        lower - margin, dtype=torch.float32, device=device
    )
    upper_t = torch.as_tensor(
        upper + margin, dtype=torch.float32, device=device
    )
    below = torch.clamp(lower_t[None, :] - positions, min=0.0)
    above = torch.clamp(positions - upper_t[None, :], min=0.0)
    return torch.mean(below * below + above * above)


def color_points_from_images(
    points: np.ndarray,
    cameras: list[dict],
    images: np.ndarray,
) -> np.ndarray:
    """Paint initialization facets with image samples at their projections.

    COLMAP sparse models do not always carry valid RGB, and a black initial
    cloud renders as a dark fog that is slow to recover from. Averaging the
    training images at each point's projection gives every facet a useful
    starting color without changing its geometry.
    """
    colors = np.zeros((len(points), 3), dtype=np.float32)
    counts = np.zeros(len(points), dtype=np.float32)
    for index, camera in enumerate(cameras):
        if index >= len(images):
            break
        c2w = np.asarray(camera["c2w"], dtype=np.float32)
        origin = c2w[:3, 3]
        rotation = c2w[:3, :3]
        k = camera["K"]
        width, height = camera["width"], camera["height"]
        fx, fy, cx, cy = (
            float(k[0][0]),
            float(k[1][1]),
            float(k[0][2]),
            float(k[1][2]),
        )
        relative = points - origin[None, :]
        camera_space = relative @ rotation
        z = camera_space[:, 2]
        visible = z > 0.0
        safe_z = np.maximum(z, 1e-4)
        u = fx * camera_space[:, 0] / safe_z + cx
        v = fy * camera_space[:, 1] / safe_z + cy
        visible &= (u >= 0.0) & (u < width - 1) & (v >= 0.0) & (v < height - 1)
        if not np.any(visible):
            continue
        ui = np.clip(np.round(u[visible]).astype(np.int32), 0, width - 1)
        vi = np.clip(np.round(v[visible]).astype(np.int32), 0, height - 1)
        colors[visible] += images[index][vi, ui]
        counts[visible] += 1.0
    covered = counts > 0
    if np.any(covered):
        colors[covered] /= counts[covered, None]
    return colors


def estimate_init_scale(
    points: np.ndarray,
    cameras: list[dict],
    target_px: float = 2.5,
) -> float:
    """Pick splat radii so a median point projects to roughly target_px."""
    if len(points) == 0 or len(cameras) == 0:
        return 0.05
    origins = np.asarray(
        [np.asarray(camera["c2w"], dtype=np.float32)[:3, 3] for camera in cameras]
    )
    if origins.ndim != 2 or origins.shape[0] == 0:
        return 0.05
    fxs = np.asarray(
        [float(camera["K"][0][0]) for camera in cameras], dtype=np.float32
    )
    if fxs.size == 0 or float(np.median(fxs)) <= 0:
        return 0.05
    depths = np.min(np.linalg.norm(points[:, None, :] - origins[None, :, :], axis=2), axis=1)
    median_depth = float(np.median(depths))
    median_fx = float(np.median(fxs))
    if not np.isfinite(median_depth) or median_depth <= 1e-4:
        return 0.05
    return max(median_depth * target_px / median_fx, 1e-3)


def _visible_mask(
    model: SplatFacetField,
    camera: dict,
    scene_points: np.ndarray,
    device: torch.device,
    near: float,
    far: float,
) -> torch.Tensor:
    rotation, origin, fx, fy, cx, cy, width, height = _camera_geometry(
        camera, device
    )
    positions = model.positions
    camera_space, u, v, z, visible = _project_points(
        positions,
        rotation,
        origin,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        near,
    )
    visible = visible & (z < far)
    sigma_u = torch.exp(model.log_scales)[:, None] * fx / torch.clamp(z, min=1e-4)
    sigma_v = torch.exp(model.log_scales)[:, None] * fy / torch.clamp(z, min=1e-4)
    radius_u = 2.8 * sigma_u[:, 0]
    radius_v = 2.8 * sigma_v[:, 0]
    visible = (
        visible[:, 0]
        & (u[:, 0] > -radius_u - 4.0)
        & (u[:, 0] < width + radius_u + 4.0)
        & (v[:, 0] > -radius_v - 4.0)
        & (v[:, 0] < height + radius_v + 4.0)
    )
    return visible


def _opacity_reset(
    model: SplatFacetField,
    camera: dict,
    surface_points: np.ndarray,
    device: torch.device,
    opacity: float,
    near: float,
    far: float,
    exclude_mask: torch.Tensor | None = None,
) -> None:
    """Raise only faded visible facets so haze cannot persist without shells."""
    visible = _visible_mask(
        model, camera, surface_points, device, near, far
    )
    if exclude_mask is not None:
        visible = visible & (~exclude_mask)
    if not torch.any(visible):
        return
    with torch.no_grad():
        faded = visible & (torch.sigmoid(model.opacity_logits) < opacity)
        if torch.any(faded):
            model.opacity_logits[faded] = torch.as_tensor(
                [math.log(opacity / (1.0 - opacity))],
                dtype=torch.float32,
                device=device,
            )


def _model_hole_protected(model: SplatFacetField) -> torch.Tensor:
    """Return the persistent hole-protection mask, creating it if absent."""
    mask = getattr(model, "hole_protected", None)
    count = int(model.positions.shape[0])
    if mask is None or int(mask.numel()) != count:
        mask = torch.zeros(
            count, dtype=torch.bool, device=model.positions.device
        )
        model.hole_protected = mask
    return mask


def _model_hole_protected_age(model: SplatFacetField) -> torch.Tensor:
    """Return the per-facet protection age, creating it if absent."""
    age = getattr(model, "hole_protected_age", None)
    count = int(model.positions.shape[0])
    if age is None or int(age.numel()) != count:
        age = torch.zeros(
            count, dtype=torch.int64, device=model.positions.device
        )
        model.hole_protected_age = age
    return age


def _prune_splats_to_budget(
    model: SplatFacetField,
    target_count: int,
    device: torch.device,
    min_opacity: float = 0.12,
    hole_protected_override: np.ndarray | torch.Tensor | None = None,
) -> int:
    """Drop low-opacity facets when VRAM pressure forces a smaller budget.

    Hole-protected facets are kept first so unresolved regions are not
    recycled before they have been covered. A per-camera coverage override can
    replace the persistent mask so already-covered facets become recyclable.
    """
    target_count = max(0, int(target_count))
    current = int(model.positions.shape[0])
    if target_count >= current:
        return current
    with torch.no_grad():
        opacity = torch.sigmoid(model.opacity_logits)
        persistent = _model_hole_protected(model)
        persistent_age = _model_hole_protected_age(model)
        if hole_protected_override is None:
            protected = persistent
        else:
            if isinstance(hole_protected_override, np.ndarray):
                protected = torch.as_tensor(
                    hole_protected_override,
                    dtype=torch.bool,
                    device=device,
                )
            else:
                protected = hole_protected_override.to(device=device).bool()
            if int(protected.numel()) != current:
                protected = persistent
        keep_mask = (opacity >= min_opacity) | protected
        keep = torch.nonzero(keep_mask).reshape(-1)
        if keep.numel() > target_count:
            protected_idx = torch.nonzero(protected).reshape(-1)
            if int(protected_idx.numel()) >= target_count:
                order = torch.argsort(
                    opacity[protected_idx], descending=True
                )[:target_count]
                keep = protected_idx[order]
            else:
                unprotected = torch.nonzero(
                    keep_mask & (~protected)
                ).reshape(-1)
                need = target_count - int(protected_idx.numel())
                order = torch.argsort(
                    opacity[unprotected], descending=True
                )[:need]
                keep = torch.cat([protected_idx, unprotected[order]])
        keep_sorted, _ = torch.sort(keep)
        model.positions = torch.nn.Parameter(
            model.positions.detach()[keep_sorted]
        )
        model.log_scales = torch.nn.Parameter(
            model.log_scales.detach()[keep_sorted]
        )
        model.opacity_logits = torch.nn.Parameter(
            model.opacity_logits.detach()[keep_sorted]
        )
        model.color_logits = torch.nn.Parameter(
            model.color_logits.detach()[keep_sorted]
        )
        model.hole_protected = protected[keep_sorted]
        model.hole_protected_age = persistent_age[keep_sorted]
    return int(model.positions.shape[0])


def _vram_budget_update(
    config: dict,
    device: torch.device,
    scene_cap: int,
    current_cap: int,
    tile_pair_budget: int,
    tile_work_budget: int,
    tile_max_splats: int,
) -> tuple[int, int, int, int, int, float, float]:
    """Tighten toward VRAM headroom only when memory pressure appears."""
    adaptive = bool(config.get("adaptive_budget", True))
    if not adaptive or not torch.cuda.is_available():
        return (
            current_cap,
            tile_pair_budget,
            tile_work_budget,
            tile_max_splats,
            0,
            0.0,
            0.0,
        )
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except Exception:
        return (
            current_cap,
            tile_pair_budget,
            tile_work_budget,
            tile_max_splats,
            0,
            0.0,
            0.0,
        )
    free_mb = free_bytes / (1024 * 1024)
    total_mb = total_bytes / (1024 * 1024)
    headroom_mb = min(
        float(config.get("vram_headroom_mb", 1200)),
        max(256.0, total_mb * 0.5),
    )
    usable_mb = max(0.0, free_mb - headroom_mb)
    splat_mb = max(1e-4, float(config.get("splat_mb_estimate", 0.012)))
    min_cap = min(
        scene_cap,
        max(1000, int(config.get("adaptive_min_splats", 30000))),
    )
    target_cap = int(usable_mb / splat_mb)
    target_cap = max(min_cap, min(target_cap, scene_cap))
    pressure = 0
    critical_mb = float(config.get("vram_critical_mb", 1400))
    low_mb = float(config.get("vram_low_mb", 2200))
    if free_mb < critical_mb:
        pressure = 2
    elif free_mb < low_mb:
        pressure = 1
    if pressure == 0:
        updated_cap = scene_cap
    else:
        if current_cap > 0:
            smoothed = int(current_cap * 0.7 + target_cap * 0.3)
        else:
            smoothed = target_cap
        updated_cap = max(min_cap, min(smoothed, scene_cap))
    if pressure >= 1:
        tile_pair_budget = min(
            tile_pair_budget,
            int(config.get("vram_tight_tile_pair_budget", 2_000_000)),
        )
        tile_work_budget = min(
            tile_work_budget,
            int(config.get("vram_tight_tile_work_budget", 12_000_000)),
        )
        tile_max_splats = min(
            tile_max_splats,
            int(config.get("vram_tight_tile_max_splats", 4096)),
        )
    if pressure >= 2:
        tile_pair_budget = min(
            tile_pair_budget,
            int(config.get("vram_critical_tile_pair_budget", 1_200_000)),
        )
        tile_max_splats = min(
            tile_max_splats,
            int(config.get("vram_critical_tile_max_splats", 3072)),
        )
    return (
        updated_cap,
        tile_pair_budget,
        tile_work_budget,
        tile_max_splats,
        pressure,
        free_mb,
        total_mb,
    )


def _densify_and_prune(
    model: SplatFacetField,
    camera: dict,
    surface_points: np.ndarray,
    device: torch.device,
    grad_accum: torch.Tensor,
    max_splats: int,
    prune_threshold: float,
    jitter_scale: float,
    near: float,
    far: float,
    rng: np.random.Generator,
) -> bool:
    """Clone high-gradient visible facets and prune dead ones."""
    with torch.no_grad():
        opacity = torch.sigmoid(model.opacity_logits)
        protected = _model_hole_protected(model)
        protected_age = _model_hole_protected_age(model)
        keep = (opacity >= prune_threshold) | protected
        visible = _visible_mask(
            model, camera, surface_points, device, near, far
        )
        strength = grad_accum * visible
        threshold = float(torch.quantile(strength, 0.85).item()) if torch.any(strength > 0) else 0.0
        candidates = torch.nonzero(
            (strength >= threshold) & (strength > 0) & keep
        ).reshape(-1)
        remaining = int(torch.sum(keep).item())
        budget = max_splats - remaining
        count = min(int(candidates.numel()), max(budget, 0))
        if count <= 0:
            return False
        chosen = candidates[rng.permutation(int(candidates.numel()))[:count]]
        jitter = torch.as_tensor(
            rng.normal(scale=jitter_scale, size=(count, 3)),
            dtype=torch.float32,
            device=device,
        )
        parent_scale = torch.exp(model.log_scales[chosen])
        new_positions = torch.cat(
            [model.positions.detach(), (model.positions[chosen] + jitter).detach()],
            dim=0,
        )
        new_log_scales = torch.cat(
            [
                model.log_scales.detach(),
                (torch.log(parent_scale * 0.6)).detach(),
            ],
            dim=0,
        )
        new_opacity = torch.cat(
            [
                model.opacity_logits.detach(),
                math.log(0.45 / (1.0 - 0.45))
                + 0.2 * torch.randn(count, device=device),
            ],
            dim=0,
        )
        new_colors = torch.cat(
            [
                model.color_logits.detach(),
                model.color_logits[chosen].detach(),
            ],
            dim=0,
        )
        keep_full = torch.cat(
            [
                keep,
                torch.ones(count, dtype=torch.bool, device=device),
            ]
        )
        protected_full = torch.cat(
            [
                protected,
                protected[chosen],
            ]
        )
        protected_age_full = torch.cat(
            [
                protected_age,
                protected_age[chosen],
            ]
        )
        keep_indices = torch.nonzero(keep_full).reshape(-1)
        keep_positions = new_positions[keep_indices]
        keep_scales = new_log_scales[keep_indices]
        keep_opacity = new_opacity[keep_indices]
        keep_colors = new_colors[keep_indices]
        model.positions = torch.nn.Parameter(keep_positions)
        model.log_scales = torch.nn.Parameter(keep_scales)
        model.opacity_logits = torch.nn.Parameter(keep_opacity)
        model.color_logits = torch.nn.Parameter(keep_colors)
        model.hole_protected = protected_full[keep_indices]
        model.hole_protected_age = protected_age_full[keep_indices]
        return True


def _split_large_splats(
    model: SplatFacetField,
    camera: dict,
    device: torch.device,
    max_splats: int,
    near: float,
    far: float,
    rng: np.random.Generator,
    split_scale_factor: float = 0.55,
    min_split_scale_px: float = 2.4,
    min_split_screen_fraction: float = 0.1,
    max_splits: int = 3000,
    children_per_split: int = 4,
    shrink_factor: float = 0.85,
) -> tuple[int, bool]:
    """Replace oversized visible facets with smaller copies.

    When the splat budget is exhausted the largest facets are shrunk in place
    instead, which still removes the blob-like footprint without extra VRAM.
    """
    if int(model.positions.shape[0]) == 0:
        return (0, False)
    with torch.no_grad():
        rotation, origin, fx, _, _, _, cam_width, cam_height = _camera_geometry(
            camera, device
        )
        screen_ref = max(1, min(int(cam_width), int(cam_height)))
        min_split_scale_px = max(
            float(min_split_scale_px),
            screen_ref
            * max(float(min_split_screen_fraction), 0.01)
            / 2.8,
        )
        _, _, _, z, visible = _project_points(
            model.positions,
            rotation,
            origin,
            fx,
            fx,
            0.0,
            0.0,
            1,
            1,
            near,
        )
        visible = visible[:, 0] & (z[:, 0] < far)
        opacity = torch.sigmoid(model.opacity_logits)
        visible = visible & (opacity >= 0.02)
        projected_sigma = (
            torch.exp(model.log_scales) * fx / torch.clamp(z[:, 0], min=1e-4)
        )
        candidates = torch.nonzero(
            visible & (projected_sigma > min_split_scale_px)
        ).reshape(-1)
        if candidates.numel() == 0:
            return (0, False)
        order = torch.argsort(projected_sigma[candidates], descending=True)
        count_limit = min(int(candidates.numel()), max(0, int(max_splits)))
        chosen = candidates[order[:count_limit]]
        budget = max(0, max_splats - int(model.positions.shape[0]))
        children_per_split = max(2, int(children_per_split))
        max_splits_by_budget = (
            budget // (children_per_split - 1) if children_per_split > 1 else 0
        )
        split_count = min(int(chosen.numel()), max_splits_by_budget)
        if split_count <= 0:
            shrink_targets = chosen[
                : max(1, int(model.positions.shape[0]) // 20)
            ]
            if shrink_targets.numel() > 0:
                model.log_scales[shrink_targets] = torch.log(
                    torch.clamp(
                        torch.exp(model.log_scales[shrink_targets])
                        * max(float(shrink_factor), 0.5),
                        min=1e-4,
                    )
                )
                return (0, True)
            return (0, False)
        chosen = chosen[:split_count]
        parent_scale = torch.exp(model.log_scales[chosen])
        child_scale = torch.clamp(
            parent_scale * max(float(split_scale_factor), 0.3),
            min=1e-4,
        )
        offsets = torch.tensor(
            [
                [-0.6, 0.0],
                [0.6, 0.0],
                [0.0, -0.6],
                [0.0, 0.6],
            ],
            dtype=torch.float32,
            device=device,
        )
        axis_x = rotation[:, 0]
        axis_y = rotation[:, 1]
        delta = (
            offsets[:, 0:1] * axis_x[None, :]
            + offsets[:, 1:2] * axis_y[None, :]
        )
        delta = delta[:, None, :] * child_scale[None, :, None]
        child_positions = (
            model.positions[chosen][None, :, :] + delta
        ).reshape(-1, 3)
        child_log_scales = torch.log(
            child_scale[:, None].repeat(1, children_per_split).reshape(-1)
        )
        parent_opacity = torch.sigmoid(model.opacity_logits[chosen])
        child_opacity = torch.clamp(parent_opacity * 0.55, min=0.02, max=0.85)
        child_opacity_logits = torch.log(
            child_opacity / torch.clamp(1.0 - child_opacity, min=1e-4)
        )
        child_opacity_logits = (
            child_opacity_logits[:, None].repeat(1, children_per_split).reshape(-1)
        )
        child_colors = model.color_logits[chosen].repeat_interleave(
            children_per_split, dim=0
        )
        keep_mask = torch.ones(
            model.positions.shape[0], dtype=torch.bool, device=device
        )
        keep_mask[chosen] = False
        protected_np = _model_hole_protected(model).cpu().numpy()
        protected_age_np = _model_hole_protected_age(model).cpu().numpy()
        keep_mask_np = keep_mask.cpu().numpy()
        new_protected_np = np.concatenate(
            [
                protected_np[keep_mask_np],
                protected_np[chosen.cpu().numpy()],
            ]
        )
        new_protected_age_np = np.concatenate(
            [
                protected_age_np[keep_mask_np],
                protected_age_np[chosen.cpu().numpy()],
            ]
        )
        model.positions = torch.nn.Parameter(
            torch.cat(
                [model.positions.detach()[keep_mask], child_positions.detach()],
                dim=0,
            )
        )
        model.log_scales = torch.nn.Parameter(
            torch.cat(
                [model.log_scales.detach()[keep_mask], child_log_scales.detach()],
                dim=0,
            )
        )
        model.opacity_logits = torch.nn.Parameter(
            torch.cat(
                [
                    model.opacity_logits.detach()[keep_mask],
                    child_opacity_logits.detach(),
                ],
                dim=0,
            )
        )
        model.color_logits = torch.nn.Parameter(
            torch.cat(
                [model.color_logits.detach()[keep_mask], child_colors.detach()],
                dim=0,
            )
        )
        model.hole_protected = torch.as_tensor(
            new_protected_np, dtype=torch.bool, device=device
        )
        model.hole_protected_age = torch.as_tensor(
            new_protected_age_np, dtype=torch.int64, device=device
        )
        return (split_count, True)


def _sample_hole_pixels(
    hole_mask: np.ndarray,
    hole_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick unresolved pixels with even spatial coverage.

    Holes are sampled with cell-level quotas so coverage stays spread out, and
    within a cell larger connected hole regions receive proportionally more
    samples.
    """
    ys, xs = np.nonzero(hole_mask)
    total = int(ys.size)
    hole_count = max(0, min(int(hole_count), total))
    if hole_count == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    if hole_count >= total:
        return ys.astype(np.int64), xs.astype(np.int64)
    height, width = hole_mask.shape
    cell_size = max(2, int(math.ceil(max(height, width) / 8)))
    cols = max(1, int(math.ceil(width / cell_size)))
    keys = (ys // cell_size) * cols + (xs // cell_size)
    cells, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True
    )
    quotas = np.floor(counts * hole_count / total).astype(np.int64)
    remainder = int(hole_count - int(quotas.sum()))
    if remainder > 0:
        order = np.argsort(-counts)[:remainder]
        quotas[order] += 1
    try:
        from scipy import ndimage as ndi

        label_map, _ = ndi.label(hole_mask)
        area_per_pixel = np.bincount(label_map.ravel())[
            label_map.ravel()
        ].astype(np.float64)
    except Exception:
        area_per_pixel = None
    selected_y: list[int] = []
    selected_x: list[int] = []
    for cell_index, _ in enumerate(cells):
        quota = int(quotas[cell_index])
        if quota <= 0:
            continue
        positions = np.nonzero(inverse == cell_index)[0]
        if quota >= positions.size:
            picked = positions
        else:
            if area_per_pixel is not None:
                weights = area_per_pixel[
                    ys[positions] * width + xs[positions]
                ]
                weights = np.nan_to_num(weights, nan=0.0)
                if float(weights.sum()) > 0:
                    picked = positions[
                        rng.choice(
                            positions.size,
                            size=quota,
                            replace=False,
                            p=weights / weights.sum(),
                        )
                    ]
                else:
                    picked = positions[
                        rng.choice(positions.size, size=quota, replace=False)
                    ]
            else:
                picked = positions[
                    rng.choice(positions.size, size=quota, replace=False)
                ]
        selected_y.extend(ys[picked].tolist())
        selected_x.extend(xs[picked].tolist())
    order = rng.permutation(len(selected_y))
    return (
        np.asarray(selected_y, dtype=np.int64)[order],
        np.asarray(selected_x, dtype=np.int64)[order],
    )


def _letterbox_ignore_mask(
    target_low: np.ndarray,
    black_tol: float = 0.045,
) -> np.ndarray:
    """Return pixels that are black letterbox fill and should be ignored.

    Image loading pads source frames to a common canvas with black bars. Those
    bars are never part of the scene, but they differ from the render
    background strongly enough to be mistaken for holes. Only black regions
    touching the image border are treated as letterbox so a genuinely black
    object in the scene is still considered content.
    """
    target_low = np.asarray(target_low, dtype=np.float32)
    black = np.mean(target_low, axis=-1) < float(black_tol)
    if not np.any(black):
        return np.zeros_like(black)
    try:
        from scipy import ndimage as ndi

        labels, _ = ndi.label(black)
        border_labels = np.unique(
            np.concatenate(
                [
                    labels[0],
                    labels[-1],
                    labels[:, 0],
                    labels[:, -1],
                ]
            )
        )
        border_labels = border_labels[border_labels > 0]
        if border_labels.size == 0:
            return np.zeros_like(black)
        return np.isin(labels, border_labels)
    except Exception:
        return black


def _count_lowres_holes(
    model: "SplatFacetField",
    camera: dict,
    target_image: np.ndarray,
    background: np.ndarray,
    near: float,
    far: float,
    device: torch.device,
    low_res: int = 192,
    bg_tol: float = 0.14,
    bad_fit_tol: float = 0.16,
    strict_bg: bool = False,
    has_target_tol: float = 0.06,
    disk_cutoff: float = 1.7,
    falloff_power: float = 1.0,
    tile_pair_budget: int = 4_000_000,
    tile_max_splats: int = 8192,
    tile_work_budget: int = 2_000_000,
    use_cuda_tile: bool = True,
    include_bad_fit: bool = False,
) -> tuple[np.ndarray, int]:
    """Render one camera at low resolution and report unresolved pixels."""
    from PIL import Image

    with torch.no_grad():
        low_camera = _aspect_lowres_camera(camera, max_pixels=low_res)
        low_width = int(low_camera["width"])
        low_height = int(low_camera["height"])
        bg_tensor = torch.as_tensor(
            background, dtype=torch.float32, device=device
        )
        rendered = _render_pixels(
            model,
            low_camera,
            None,
            low_width,
            low_height,
            bg_tensor,
            near=near,
            far=far,
            batch_size=256,
            disk_cutoff=disk_cutoff,
            falloff_power=falloff_power,
            tile_size=16,
            dense_renderer=False,
            tile_pair_budget=tile_pair_budget,
            tile_max_splats=tile_max_splats,
            tile_work_budget=tile_work_budget,
            use_cuda_tile=use_cuda_tile,
        )
        render_low = (
            rendered.detach().cpu().numpy().reshape(
                low_height, low_width, 3
            )
        )
        target_low = np.asarray(
            Image.fromarray(
                np.clip(target_image * 255.0, 0.0, 255.0).astype(np.uint8)
            ).resize((low_width, low_height), Image.LANCZOS),
            dtype=np.float32,
        ) / 255.0
        bg = np.asarray(background, dtype=np.float32)
        near_bg = np.mean(np.abs(render_low - bg), axis=-1) < float(bg_tol)
        has_target = (
            np.mean(np.abs(target_low - bg), axis=-1) > float(has_target_tol)
        ) & (~_letterbox_ignore_mask(target_low))
        bad_fit = (
            np.mean(np.abs(render_low - target_low), axis=-1)
            > float(bad_fit_tol)
        )
        if strict_bg or not include_bad_fit:
            hole_mask = near_bg & has_target
        else:
            hole_mask = (near_bg & has_target) | (bad_fit & has_target)
        return hole_mask, int(hole_mask.sum())


def _hole_coverage_protected_mask(
    model: "SplatFacetField",
    cameras: list[dict],
    images: list[np.ndarray],
    surface_points: np.ndarray,
    device: torch.device,
    low_res: int = 192,
    dilate_px: int = 3,
    bg_tol: float = 0.14,
    bad_fit_tol: float = 0.16,
    has_target_tol: float = 0.06,
    disk_cutoff: float = 1.7,
    falloff_power: float = 1.0,
    tile_pair_budget: int = 4_000_000,
    tile_max_splats: int = 8192,
    tile_work_budget: int = 2_000_000,
    use_cuda_tile: bool = True,
    include_bad_fit: bool = True,
) -> np.ndarray:
    """Mark facets that project into unresolved hole pixels so pruning keeps them."""
    count = int(model.positions.shape[0])
    protected = np.zeros(count, dtype=bool)
    if count == 0 or not cameras or len(images) < len(cameras):
        return protected
    from .main import SCENE_BACKGROUND, ray_bounds_from_points

    for cam_idx, camera in enumerate(cameras):
        if cam_idx >= len(images):
            break
        c2w = np.asarray(camera["c2w"], dtype=np.float32)
        near, far = (
            ray_bounds_from_points(
                c2w[:3, 3],
                surface_points,
                rotation=c2w[:3, :3],
            )
            if surface_points is not None and len(surface_points)
            else (0.3, 7.6)
        )
        hole_mask, hole_count = _count_lowres_holes(
            model,
            camera,
            images[cam_idx],
            SCENE_BACKGROUND,
            near,
            far,
            device,
            low_res=low_res,
            bg_tol=bg_tol,
            bad_fit_tol=bad_fit_tol,
            has_target_tol=has_target_tol,
            disk_cutoff=disk_cutoff,
            falloff_power=falloff_power,
            tile_pair_budget=tile_pair_budget,
            tile_max_splats=tile_max_splats,
            tile_work_budget=tile_work_budget,
            use_cuda_tile=use_cuda_tile,
            strict_bg=not include_bad_fit,
        )
        if hole_count == 0:
            continue
        low_camera = _aspect_lowres_camera(camera, max_pixels=low_res)
        low_width = int(low_camera["width"])
        low_height = int(low_camera["height"])
        if hole_mask.shape != (low_height, low_width):
            continue
        dilate = max(1, int(dilate_px))
        padded = np.pad(hole_mask, dilate, mode="edge")
        dilated = np.zeros_like(hole_mask)
        for dy in range(-dilate, dilate + 1):
            for dx in range(-dilate, dilate + 1):
                dilated |= padded[
                    dilate + dy : dilate + dy + low_height,
                    dilate + dx : dilate + dx + low_width,
                ]
        rotation, origin, fx, fy, cx, cy, _, _ = _camera_geometry(
            low_camera, device
        )
        positions = model.positions.detach()
        rel = positions - origin[None, :]
        camera_space = rel @ rotation
        z = camera_space[:, 2:3]
        safe_z = torch.clamp(z, min=1e-4)
        u = fx * camera_space[:, 0:1] / safe_z + cx
        v = fy * camera_space[:, 1:2] / safe_z + cy
        visible = (
            (z[:, 0] > near)
            & (z[:, 0] < far)
            & (u[:, 0] >= 0.0)
            & (u[:, 0] < low_width)
            & (v[:, 0] >= 0.0)
            & (v[:, 0] < low_height)
        )
        if not torch.any(visible):
            continue
        ui_np = np.floor(u[:, 0].cpu().numpy() + 0.5).astype(np.int32)
        vi_np = np.floor(v[:, 0].cpu().numpy() + 0.5).astype(np.int32)
        pixel_ok = (
            (ui_np >= 0)
            & (ui_np < low_width)
            & (vi_np >= 0)
            & (vi_np < low_height)
        )
        vis_np = visible.cpu().numpy() & pixel_ok
        in_hole = dilated[vi_np[vis_np], ui_np[vis_np]]
        indices = np.nonzero(vis_np)[0]
        protected[indices[in_hole]] = True
    return protected


def _refresh_hole_protection(
    model: SplatFacetField,
    cameras: list[dict],
    images: list[np.ndarray],
    surface_points: np.ndarray,
    device: torch.device,
    camera_indices: list[int] | None = None,
    low_res: int = 192,
    dilate_px: int = 4,
    bg_tol: float = 0.14,
    bad_fit_tol: float = 0.16,
    has_target_tol: float = 0.06,
    disk_cutoff: float = 1.7,
    falloff_power: float = 1.0,
    tile_pair_budget: int = 4_000_000,
    tile_max_splats: int = 8192,
    tile_work_budget: int = 2_000_000,
    use_cuda_tile: bool = True,
    include_bad_fit: bool = True,
) -> int:
    """Rebuild the persistent hole mask from current unresolved coverage.

    Splats that have already filled their hole are no longer protected, so
    pruning and scale boosting stop wasting budget on covered regions.
    """
    count = int(model.positions.shape[0])
    if count == 0 or not cameras or len(images) < len(cameras):
        return 0
    if camera_indices is None:
        indices = list(range(len(cameras)))
    else:
        indices = [
            int(idx)
            for idx in camera_indices
            if 0 <= int(idx) < len(cameras) and int(idx) < len(images)
        ]
    if not indices:
        return 0
    protected = np.zeros(count, dtype=bool)
    for cam_idx in indices:
        coverage = _hole_coverage_protected_mask(
            model,
            [cameras[cam_idx]],
            [images[cam_idx]],
            surface_points,
            device,
            low_res=low_res,
            dilate_px=dilate_px,
            bg_tol=bg_tol,
            bad_fit_tol=bad_fit_tol,
            has_target_tol=has_target_tol,
            disk_cutoff=disk_cutoff,
            falloff_power=falloff_power,
            tile_pair_budget=tile_pair_budget,
            tile_max_splats=tile_max_splats,
            tile_work_budget=tile_work_budget,
            use_cuda_tile=use_cuda_tile,
            include_bad_fit=include_bad_fit,
        )
        protected |= coverage
    age = _model_hole_protected_age(model)
    if int(age.numel()) != count:
        age = _model_hole_protected_age(model)
    protected = protected | (age.cpu().numpy() > 0)
    new_age = np.maximum(age.cpu().numpy() - 1, 0).astype(np.int64)
    model.hole_protected = torch.as_tensor(
        protected, dtype=torch.bool, device=device
    )
    model.hole_protected_age = torch.as_tensor(
        new_age, dtype=torch.int64, device=device
    )
    return int(protected.sum())


def _inpaint_depth_map(
    base_depth: np.ndarray,
    fallback: float,
    max_radius: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaN depth pixels from the nearest valid pixel.

    Returns the filled map and the distance each pixel had to travel. Pixels
    farther than ``max_radius`` fall back to ``fallback`` so a single distant
    surface edge is not stretched across an entire hole.
    """
    base_depth = np.asarray(base_depth, dtype=np.float32)
    filled = np.full(
        base_depth.shape, float(fallback), dtype=np.float32
    )
    distances = np.full(
        base_depth.shape, np.inf, dtype=np.float32
    )
    valid = np.isfinite(base_depth)
    if not np.any(valid):
        return filled, distances
    if np.all(valid):
        return base_depth.copy(), np.zeros_like(distances)
    try:
        from scipy import ndimage as ndi

        dist, indices = ndi.distance_transform_edt(
            ~valid, return_distances=True, return_indices=True
        )
        nearest = base_depth[indices[0], indices[1]]
        filled = np.where(
            np.isfinite(nearest), nearest, float(fallback)
        )
        distances = dist.astype(np.float32)
        if max_radius is not None and max_radius >= 0:
            too_far = distances > float(max_radius)
            filled[too_far] = float(fallback)
            distances[too_far] = np.inf
    except Exception:
        pass
    return filled, distances


def _facet_probe_model(
    positions: torch.Tensor,
    log_scales: torch.Tensor,
    opacity_logits: torch.Tensor,
    color_logits: torch.Tensor,
) -> SplatFacetField:
    """Build a temporary render-only model from precomputed facet tensors."""
    probe = SplatFacetField(
        positions.detach().cpu().numpy(),
        colors=torch.sigmoid(color_logits.detach()).cpu().numpy(),
        scales=torch.exp(log_scales.detach()).cpu().numpy(),
        opacity=0.5,
    ).to(positions.device)
    probe.positions = torch.nn.Parameter(positions.clone())
    probe.log_scales = torch.nn.Parameter(log_scales.clone())
    probe.opacity_logits = torch.nn.Parameter(opacity_logits.clone())
    probe.color_logits = torch.nn.Parameter(color_logits.clone())
    return probe


def _cluster_hole_seeds(
    hole_mask: np.ndarray,
    max_seeds: int,
    min_sigma_px: float = 3.0,
    max_sigma_px: float = 8.0,
    cover_factor: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cover a hole mask with a few large screen-space facet seeds.

    The distance-to-boundary transform gives the radius of the largest disk
    that fits at each pixel. Seeds are dropped greedily at the deepest
    uncovered pixels, so one facet can fill a large connected hole instead of
    many tiny fragments near its edges.
    """
    hole_mask = np.asarray(hole_mask, dtype=bool)
    if hole_mask.ndim != 2 or int(hole_mask.sum()) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )
    max_seeds = max(0, int(max_seeds))
    min_sigma_px = max(1.0, float(min_sigma_px))
    max_sigma_px = max(min_sigma_px, float(max_sigma_px))
    cover_factor = max(1.0, float(cover_factor))
    ys_all, xs_all = np.nonzero(hole_mask)
    try:
        from scipy import ndimage as ndi

        dist = ndi.distance_transform_edt(~hole_mask)
    except Exception:
        dist = np.ones(hole_mask.shape, dtype=np.float32)
    depths = dist[ys_all, xs_all]
    order = np.argsort(-depths, kind="stable")
    covered = np.zeros(hole_mask.shape, dtype=bool)
    seeds_y: list[int] = []
    seeds_x: list[int] = []
    seeds_sigma: list[float] = []
    for idx in order:
        if len(seeds_y) >= max_seeds:
            break
        py = int(ys_all[idx])
        px = int(xs_all[idx])
        if covered[py, px]:
            continue
        sigma = float(
            np.clip(depths[idx] * 0.9, min_sigma_px, max_sigma_px)
        )
        seeds_y.append(py)
        seeds_x.append(px)
        seeds_sigma.append(sigma)
        radius = sigma * cover_factor
        r_int = max(1, int(math.ceil(radius)))
        y0 = max(0, py - r_int)
        y1 = min(hole_mask.shape[0], py + r_int + 1)
        x0 = max(0, px - r_int)
        x1 = min(hole_mask.shape[1], px + r_int + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        within = (
            (yy - py) ** 2 + (xx - px) ** 2
        ) <= radius * radius
        covered[y0:y1, x0:x1] |= within
    return (
        np.asarray(seeds_y, dtype=np.int64),
        np.asarray(seeds_x, dtype=np.int64),
        np.asarray(seeds_sigma, dtype=np.float32),
    )


def _hole_fill_splats(
    model: SplatFacetField,
    camera: dict,
    target_image: np.ndarray,
    background: np.ndarray,
    near: float,
    far: float,
    device: torch.device,
    rng: np.random.Generator,
    max_add: int,
    low_res: int = 96,
    scale_floor: float = 1e-3,
    scale_ceiling: float = 0.4,
    tile_pair_budget: int = 4_000_000,
    tile_max_splats: int = 8192,
    tile_work_budget: int = 2_000_000,
    use_cuda_tile: bool = True,
    scale_px_factor: float = 9.0,
    median_scale_factor: float = 1.0,
    scale_ceiling_factor: float = 5.0,
    max_scale_screen_fraction: float = 0.06,
    opacity_center: float = 0.35,
    depth_ratios: tuple[float, ...] = (0.9, 1.0, 1.1),
    depth_quantile: float = 0.3,
    low_scale_px: float = 2.5,
    depth_sweep_count: int = 1,
    depth_sweep_opacity: float = 0.25,
    disk_cutoff: float = 1.8,
    falloff_power: float = 1.0,
    scene_points: np.ndarray | None = None,
    scene_depth_ratios: tuple[float, ...] = (0.9, 1.1),
    median_depth_ratios: tuple[float, ...] = (0.6, 1.0),
    bg_tol: float = 0.14,
    bad_fit_tol: float = 0.16,
    hole_dilate_px: int = 3,
    neighbor_grid: int = 4,
    neighbor_step_px: float = 3.0,
    bg_depth_factor: float = 0.97,
    targets_per_hole: int = 1,
    depth_hint_radius_factor: float = 0.14,
    scene_depth_hint_radius_factor: float = 0.3,
    z_min_opacity: float = 0.2,
    include_bad_fit: bool = True,
    depth_sweep_min: float = 0.7,
    depth_sweep_max: float = 1.4,
    protection_cycles: int = 3,
    area_scale_max: float = 3.0,
    cluster_fill: bool = True,
    cluster_max_sigma_px: float = 8.0,
    cluster_min_sigma_px: float = 3.0,
    cluster_cover_factor: float = 1.5,
    cluster_cover_tol: float = 0.3,
    cluster_children: int = 3,
    cluster_scale_factor: float = 0.75,
) -> tuple[bool, int, np.ndarray | None]:
    """Backproject unresolved pixels into new facets along multiple depths.

    Local surface continuation depth is filled from the nearest valid facet
    and sparse COLMAP point, then combined with log-spaced ray probes so a
    hole can be covered even when the direct depth is unreliable.
    """
    from PIL import Image

    max_add = max(0, int(max_add))
    if model.positions.numel() == 0:
        return (False, 0, None)
    depth_ratios = tuple(
        float(value) for value in (depth_ratios or (1.0,))
    )
    depth_per_hole = max(1, len(depth_ratios))
    depth_sweep_count = max(0, int(depth_sweep_count))
    depth_candidate_count = depth_per_hole + depth_sweep_count
    with torch.no_grad():
        low_camera = _aspect_lowres_camera(camera, max_pixels=low_res)
        low_width = int(low_camera["width"])
        low_height = int(low_camera["height"])
        full_k = np.asarray(camera["K"], dtype=np.float32)
        fx_full = float(full_k[0][0])
        fy_full = float(full_k[1][1])
        full_width = int(camera["width"])
        full_height = int(camera["height"])
        bg_tensor = torch.as_tensor(
            background, dtype=torch.float32, device=device
        )
        rendered = _render_pixels(
            model,
            low_camera,
            None,
            low_width,
            low_height,
            bg_tensor,
            near=near,
            far=far,
            batch_size=256,
            disk_cutoff=disk_cutoff,
            falloff_power=falloff_power,
            tile_size=16,
            dense_renderer=False,
            tile_pair_budget=tile_pair_budget,
            tile_max_splats=tile_max_splats,
            tile_work_budget=tile_work_budget,
            use_cuda_tile=use_cuda_tile,
        )
        render_low = (
            rendered.detach().cpu().numpy().reshape(low_height, low_width, 3)
        )
        target_low = np.asarray(
            Image.fromarray(
                np.clip(target_image * 255.0, 0.0, 255.0).astype(np.uint8)
            ).resize((low_width, low_height), Image.LANCZOS),
            dtype=np.float32,
        ) / 255.0
        bg = np.asarray(background, dtype=np.float32)
        near_bg = np.mean(np.abs(render_low - bg), axis=-1) < float(bg_tol)
        has_target = (
            np.mean(np.abs(target_low - bg), axis=-1) > 0.06
        ) & (~_letterbox_ignore_mask(target_low))
        bad_fit = (
            np.mean(np.abs(render_low - target_low), axis=-1)
            > float(bad_fit_tol)
        )
        strict_hole = near_bg & has_target
        dilate_px = max(0, int(hole_dilate_px))
        padded = np.pad(strict_hole, dilate_px, mode="edge")
        dilated = np.zeros_like(strict_hole)
        for dy in range(-dilate_px, dilate_px + 1):
            for dx in range(-dilate_px, dilate_px + 1):
                dilated |= padded[
                    dilate_px + dy : dilate_px + dy + low_height,
                    dilate_px + dx : dilate_px + dx + low_width,
                ]
        strict_hole = dilated
        if include_bad_fit:
            bad_fit_hole = bad_fit & has_target & (~strict_hole)
        else:
            bad_fit_hole = np.zeros_like(strict_hole)
        hole_mask = strict_hole | bad_fit_hole
        strict_count = int(strict_hole.sum())
        bad_count = int(bad_fit_hole.sum())
        raw_hole_count = strict_count + bad_count
        area_scale_max = max(1.0, float(area_scale_max))
        strict_label: np.ndarray | None = None
        strict_area: np.ndarray | None = None
        bad_label: np.ndarray | None = None
        bad_area: np.ndarray | None = None
        if area_scale_max > 1.0:
            try:
                from scipy import ndimage as ndi

                strict_label, _ = ndi.label(strict_hole)
                strict_area = np.bincount(strict_label.ravel())
                bad_label, _ = ndi.label(bad_fit_hole)
                bad_area = np.bincount(bad_label.ravel())
            except Exception:
                strict_label = None
                strict_area = None
                bad_label = None
                bad_area = None
        if raw_hole_count == 0:
            return (False, 0, hole_mask)
        grid = max(1, int(neighbor_grid))
        half = grid // 2
        offsets = [
            (
                dx * float(neighbor_step_px),
                dy * float(neighbor_step_px),
            )
            for dy in range(-half, half + 1)
            for dx in range(-half, half + 1)
        ]
        targets_per_hole = max(
            1,
            min(int(targets_per_hole), max(2, len(offsets) * 2)),
        )
        facets_per_hole = max(1, targets_per_hole)
        cluster_children = (
            max(1, int(cluster_children)) if bool(cluster_fill) else 1
        )
        if cluster_children > 1:
            budget_holes = max(1, max_add // cluster_children)
            facets_per_hole = cluster_children
        else:
            budget_holes = max_add // facets_per_hole
        strict_fill = min(strict_count, budget_holes)
        ys = np.empty(0, dtype=np.int64)
        xs = np.empty(0, dtype=np.int64)
        per_hole_scale_px = np.empty(0, dtype=np.float32)
        if strict_fill > 0 and bool(cluster_fill):
            seeds_y, seeds_x, seeds_sigma = _cluster_hole_seeds(
                strict_hole,
                strict_fill,
                min_sigma_px=float(cluster_min_sigma_px),
                max_sigma_px=float(cluster_max_sigma_px),
                cover_factor=float(cluster_cover_factor),
            )
            if seeds_y.size > 0:
                ys = seeds_y
                xs = seeds_x
                per_hole_scale_px = seeds_sigma.astype(np.float32)
                strict_fill = int(ys.size)
                targets_per_hole = min(
                    cluster_children,
                    max(1, max_add // max(1, int(ys.size))),
                )
                facets_per_hole = max(1, targets_per_hole)
                budget_holes = max(1, max_add // facets_per_hole)
                cluster_targets = np.ones(int(ys.size), dtype=bool)
            else:
                cluster_targets = np.zeros(int(ys.size), dtype=bool)
        else:
            cluster_targets = np.zeros(int(ys.size), dtype=bool)
        if int(ys.size) == 0 and strict_fill > 0:
            ys, xs = _sample_hole_pixels(
                strict_hole, strict_fill, rng
            )
            per_hole_scale_px = np.full(
                int(ys.size), float(low_scale_px), dtype=np.float32
            )
        bad_fill = min(
            bad_count,
            max(
                0,
                max_add
                - int(ys.size) * max(1, targets_per_hole),
            ),
        )
        if bad_fill > 0:
            bad_ys, bad_xs = _sample_hole_pixels(
                bad_fit_hole, bad_fill, rng
            )
            cluster_targets = np.concatenate(
                [
                    cluster_targets,
                    np.zeros(int(bad_ys.size), dtype=bool),
                ]
            )
            ys = np.concatenate([ys, bad_ys])
            xs = np.concatenate([xs, bad_xs])
            per_hole_scale_px = np.concatenate(
                [
                    per_hole_scale_px,
                    np.full(
                        int(bad_ys.size),
                        float(low_scale_px),
                        dtype=np.float32,
                    ),
                ]
            )
        pixel_area = np.ones(int(ys.size), dtype=np.float32)
        if (
            not bool(cluster_fill)
            and strict_area is not None
            and strict_fill > 0
        ):
            pixel_area[:strict_fill] = strict_area[
                strict_label[ys[:strict_fill], xs[:strict_fill]]
            ]
        if bad_area is not None and bad_fill > 0:
            pixel_area[strict_fill:] = bad_area[
                bad_label[bad_ys, bad_xs]
            ]
        if ys.size == 0:
            return (False, raw_hole_count, hole_mask)
        order = rng.permutation(int(ys.size))
        ys = ys[order]
        xs = xs[order]
        cluster_targets = cluster_targets[order]
        pixel_area = pixel_area[order]
        per_hole_scale_px = per_hole_scale_px[order]
        hole_count = int(ys.size)
        target_ys: list[int] = []
        target_xs: list[int] = []
        target_counts: list[int] = []
        for hole_index in range(hole_count):
            py = int(ys[hole_index])
            px = int(xs[hole_index])
            if bool(cluster_targets[hole_index]):
                disk_radius = max(
                    1.0,
                    float(per_hole_scale_px[hole_index])
                    * float(cluster_cover_factor),
                )
                grid_half = 2
                cluster_step = disk_radius / max(1, grid_half)
                local_offsets = [
                    (dx * cluster_step, dy * cluster_step)
                    for dy in range(-grid_half, grid_half + 1)
                    for dx in range(-grid_half, grid_half + 1)
                ]
            else:
                local_offsets = offsets
            local: list[tuple[int, int]] = []
            for ox, oy in local_offsets:
                ty = int(round(py + oy))
                tx = int(round(px + ox))
                if (
                    0 <= ty < low_height
                    and 0 <= tx < low_width
                    and hole_mask[ty, tx]
                ):
                    local.append((ty, tx))
            if not local:
                local.append((py, px))
            want = (
                targets_per_hole
                if bool(cluster_targets[hole_index])
                else 1
            )
            chosen: list[tuple[int, int]]
            if len(local) > want:
                center = (py, px)
                rest = [pixel for pixel in local if pixel != center]
                spread = rng.choice(
                    len(rest),
                    size=want - 1,
                    replace=False,
                )
                chosen = [center] + [
                    rest[int(entry)] for entry in spread.tolist()
                ]
            else:
                chosen = local[:want]
            while len(chosen) < want:
                chosen.append((py, px))
            for ty, tx in chosen:
                target_ys.append(ty)
                target_xs.append(tx)
            target_counts.append(want)
        target_ys_np = np.asarray(target_ys, dtype=np.int64)
        target_xs_np = np.asarray(target_xs, dtype=np.int64)
        target_counts_np = np.asarray(target_counts, dtype=np.int64)
        total_targets = int(target_ys_np.size)
        if total_targets == 0:
            return (False, raw_hole_count, hole_mask)
        target_area_px = np.repeat(
            pixel_area.astype(np.float32), target_counts_np
        )
        target_scale_px = np.repeat(
            per_hole_scale_px.astype(np.float32), target_counts_np
        )
        cluster_target_repeated = np.repeat(
            cluster_targets, target_counts_np
        )
        target_scale_px = target_scale_px * np.where(
            cluster_target_repeated,
            float(cluster_scale_factor),
            1.0,
        )

        c2w = np.asarray(camera["c2w"], dtype=np.float32)
        origin = c2w[:3, 3]
        rotation = c2w[:3, :3]
        k = low_camera["K"]
        fx_lr = float(k[0][0])
        fy_lr = float(k[1][1])
        cx_lr = float(k[0][2])
        cy_lr = float(k[1][2])

        origin_t = torch.as_tensor(origin, dtype=torch.float32, device=device)
        rotation_t = torch.as_tensor(
            rotation, dtype=torch.float32, device=device
        )
        model_camera = (model.positions.detach() - origin_t[None, :]) @ rotation_t
        model_z = model_camera[:, 2]
        safe_model_z = torch.clamp(model_z, min=1e-4)
        model_u = fx_lr * model_camera[:, 0] / safe_model_z + cx_lr
        model_v = fy_lr * model_camera[:, 1] / safe_model_z + cy_lr
        z_ok = model_z[(model_z > near) & (model_z < far)]
        z_median = (
            float(z_ok.median().item())
            if z_ok.numel() > 0
            else float(math.sqrt(max(near, 1e-4) * max(far, 1e-4)))
        )
        opacity_np = torch.sigmoid(
            model.opacity_logits.detach()
        ).cpu().numpy()
        mz_np = model_z.detach().cpu().numpy()
        mu_np = model_u.detach().cpu().numpy()
        mv_np = model_v.detach().cpu().numpy()
        ui_int = np.floor(mu_np + 0.5).astype(np.int32)
        vi_int = np.floor(mv_np + 0.5).astype(np.int32)
        ui_int = np.clip(ui_int, 0, low_width - 1)
        vi_int = np.clip(vi_int, 0, low_height - 1)
        z_min_opacity = max(0.0, min(float(z_min_opacity), 1.0))
        in_bounds = (
            (ui_int >= 0)
            & (ui_int < low_width)
            & (vi_int >= 0)
            & (vi_int < low_height)
            & (mz_np > near)
            & (mz_np < far)
            & (opacity_np >= z_min_opacity)
        )
        z_map_np = np.full((low_height, low_width), np.nan, dtype=np.float32)
        valid_indices = np.nonzero(in_bounds)[0]
        if valid_indices.size > 0:
            np.fmin.at(
                z_map_np,
                (vi_int[valid_indices], ui_int[valid_indices]),
                mz_np[valid_indices],
            )
        scene_z_map = np.full(
            (low_height, low_width), np.nan, dtype=np.float32
        )
        if scene_points is not None and len(scene_points) > 0:
            scene_pts = np.asarray(scene_points, dtype=np.float32).reshape(-1, 3)
            scene_cam = (scene_pts - origin[None, :]) @ rotation
            scene_z_all = scene_cam[:, 2]
            safe_scene_z = np.clip(scene_z_all, 1e-4, None)
            scene_u_all = fx_lr * scene_cam[:, 0] / safe_scene_z + cx_lr
            scene_v_all = fy_lr * scene_cam[:, 1] / safe_scene_z + cy_lr
            scene_ok = (
                (scene_u_all >= 0.0)
                & (scene_u_all < low_width)
                & (scene_v_all >= 0.0)
                & (scene_v_all < low_height)
                & (scene_z_all > near)
                & (scene_z_all < far)
            )
            scene_valid = np.nonzero(scene_ok)[0]
            if scene_valid.size > 0:
                scene_ui = np.floor(
                    scene_u_all[scene_valid] + 0.5
                ).astype(np.int32)
                scene_vi = np.floor(
                    scene_v_all[scene_valid] + 0.5
                ).astype(np.int32)
                scene_ui = np.clip(scene_ui, 0, low_width - 1)
                scene_vi = np.clip(scene_vi, 0, low_height - 1)
                np.fmin.at(
                    scene_z_map,
                    (scene_vi, scene_ui),
                    scene_z_all[scene_valid],
                )
        model_hint_radius = max(
            8.0, low_width * max(float(depth_hint_radius_factor), 0.0)
        )
        scene_hint_radius = max(
            12.0,
            low_width * max(float(scene_depth_hint_radius_factor), 0.0),
        )
        model_filled, model_dist = _inpaint_depth_map(
            z_map_np, z_median, model_hint_radius
        )
        scene_filled, scene_dist = _inpaint_depth_map(
            scene_z_map, z_median, scene_hint_radius
        )
        model_hint = model_filled[target_ys_np, target_xs_np]
        model_hint_dist = model_dist[target_ys_np, target_xs_np]
        scene_hint = scene_filled[target_ys_np, target_xs_np]
        scene_hint_dist = scene_dist[target_ys_np, target_xs_np]
        base_depths = np.where(
            model_hint_dist <= model_hint_radius,
            model_hint,
            np.where(
                scene_hint_dist <= scene_hint_radius,
                scene_hint,
                z_median,
            ),
        ).astype(np.float32)
        border = np.concatenate(
            [
                target_low[0],
                target_low[-1],
                target_low[:, 0],
                target_low[:, -1],
            ],
            axis=0,
        )
        border_color = np.median(border, axis=0)
        bg_like = (
            np.mean(
                np.abs(
                    target_low[target_ys_np, target_xs_np]
                    - border_color[None, :]
                ),
                axis=-1,
            )
            < 0.10
        )
        if np.any(bg_like):
            bg_depth = max(
                near * 1.1,
                min(far * float(bg_depth_factor), far * 0.98),
            )
            base_depths[bg_like] = bg_depth
        local_ratios = np.asarray(depth_ratios, dtype=np.float32)
        scene_ratios = np.asarray(
            scene_depth_ratios or (1.0,), dtype=np.float32
        )
        median_ratios = np.asarray(
            median_depth_ratios or (1.0,), dtype=np.float32
        )
        scene_anchor = np.where(
            scene_hint_dist <= scene_hint_radius,
            scene_hint,
            base_depths,
        ).astype(np.float32)
        median_anchor = np.full(
            base_depths.shape, float(z_median), dtype=np.float32
        )
        if depth_sweep_count > 0:
            sweep_min = max(
                0.2, min(float(depth_sweep_min), float(depth_sweep_max))
            )
            sweep_max = max(float(depth_sweep_max), sweep_min)
            sweep_ratios = np.geomspace(
                sweep_min, sweep_max, int(depth_sweep_count)
            ).astype(np.float32)
        else:
            sweep_ratios = np.empty(0, dtype=np.float32)
        z_columns: list[np.ndarray] = [
            base_depths[:, None] * local_ratios[None, :],
            scene_anchor[:, None] * scene_ratios[None, :],
            median_anchor[:, None] * median_ratios[None, :],
        ]
        if sweep_ratios.size > 0:
            z_columns.append(
                base_depths[:, None] * sweep_ratios[None, :]
            )
        z_candidates = np.concatenate(z_columns, axis=1).astype(
            np.float32
        )
        candidate_count = int(z_candidates.shape[1])
        z_candidates = np.clip(
            z_candidates, near * 1.02, far * 0.98
        ).astype(np.float32)
        z_values = z_candidates.reshape(-1)
        u_values = np.repeat(
            (target_xs_np + 0.5).astype(np.float32),
            candidate_count,
        )
        v_values = np.repeat(
            (target_ys_np + 0.5).astype(np.float32),
            candidate_count,
        )
        cam_x = (u_values - cx_lr) / fx_lr * z_values
        cam_y = (v_values - cy_lr) / fy_lr * z_values
        new_positions = origin[None, :] + np.stack(
            [cam_x, cam_y, z_values], axis=-1
        ) @ rotation.T
        scales = torch.exp(model.log_scales.detach())
        median_scale = float(scales.median().item())
        pixel_scale = z_values / max(fx_full, 1e-4)
        low_pixel_scale = z_values / max(fx_lr, 1e-4)
        per_candidate_scale_px = np.repeat(
            target_scale_px, candidate_count
        )
        cluster_candidate = np.repeat(
            cluster_target_repeated.astype(np.int8), candidate_count
        )
        screen_ref = max(1, max(full_width, full_height))
        screen_scale_cap = (
            z_values
            * max(float(max_scale_screen_fraction), 0.02)
            * screen_ref
            / max(fx_full, fy_full)
        )
        cluster_scale_cap = (
            low_pixel_scale
            * np.maximum(per_candidate_scale_px, 0.5)
            * 2.0
        )
        screen_scale_cap = np.maximum(
            screen_scale_cap,
            cluster_scale_cap * cluster_candidate,
        )
        area_ref = max(
            1.0, float(low_width * low_height) * 0.04
        )
        area_factor = (
            1.0
            + (area_scale_max - 1.0)
            * np.clip(target_area_px / area_ref, 0.0, 1.0)
        )
        area_factor_full = np.repeat(area_factor, candidate_count)
        base_scale = np.maximum(
            np.maximum(
                pixel_scale * max(float(scale_px_factor), 0.5),
                low_pixel_scale
                * np.maximum(per_candidate_scale_px, 0.5),
            ),
            median_scale * max(float(median_scale_factor), 0.2),
        )
        new_scale = np.clip(
            base_scale * area_factor_full,
            float(scale_floor),
            np.minimum(
                max(
                    float(scale_ceiling),
                    median_scale * max(float(scale_ceiling_factor), 1.0),
                ),
                screen_scale_cap,
            ),
        )
        new_colors = np.repeat(
            target_low[target_ys_np, target_xs_np],
            candidate_count,
            axis=0,
        )
        local_count = int(local_ratios.size)
        scene_count = int(scene_ratios.size)
        median_count = int(median_ratios.size)
        opacity_matrix = np.clip(
            float(opacity_center)
            + 0.12 * rng.uniform(
                size=(total_targets, candidate_count)
            ),
            0.15,
            0.85,
        )
        if depth_sweep_count > 0:
            sweep_start = local_count + scene_count + median_count
            opacity_matrix[:, sweep_start:] = np.clip(
                float(depth_sweep_opacity)
                + 0.08 * rng.uniform(
                    size=(total_targets, int(sweep_ratios.size))
                ),
                0.15,
                0.7,
            )
        new_positions = new_positions.reshape(
            total_targets, candidate_count, 3
        )
        new_scale = new_scale.reshape(total_targets, candidate_count)
        new_colors = new_colors.reshape(
            total_targets, candidate_count, 3
        )
        new_opacity = opacity_matrix.reshape(
            total_targets, candidate_count
        )
        z_candidates = z_candidates.reshape(
            total_targets, candidate_count
        )
        if depth_candidate_count > 1:
            # Trial-render each depth group alone and keep the depth that
            # really covers the unresolved pixel. This avoids committing
            # facets whose world-space depth is behind the visible surface
            # or so far away that they project to almost nothing.
            existing_z = np.where(
                bad_fit_hole[target_ys_np, target_xs_np],
                z_map_np[target_ys_np, target_xs_np],
                np.nan,
            )
            front_ok = np.isnan(existing_z)[:, None] | (
                z_candidates < existing_z[:, None] * 0.98
            )
            low_sigma_px = (
                new_scale
                * fx_lr
                / np.clip(z_candidates, 1e-4, None)
            )
            max_sigma_px = max(
                2.0,
                low_width
                * max(float(max_scale_screen_fraction), 0.02)
                * 1.5,
            )
            min_sigma_px = np.maximum(
                0.8,
                target_scale_px[:, None] * 0.45,
            )
            sigma_ok = (
                low_sigma_px >= min_sigma_px
            ) & (low_sigma_px <= max_sigma_px)
            front_ok = front_ok & sigma_ok
            coverage_score = np.zeros(
                (total_targets, candidate_count), dtype=np.float32
            )
            color_err = np.zeros(
                (total_targets, candidate_count), dtype=np.float32
            )
            cover_tol = max(0.02, float(bg_tol) * 0.2)
            cluster_cover_radius = np.zeros(total_targets, dtype=np.int32)
            if cluster_target_repeated.any():
                cluster_cover_radius[
                    cluster_target_repeated
                ] = np.clip(
                    np.ceil(
                        target_scale_px[cluster_target_repeated]
                        * float(cluster_cover_factor)
                    ),
                    2,
                    max(2, max(low_width, low_height) // 3),
                ).astype(np.int32)
            for group in range(candidate_count):
                pos_g = torch.as_tensor(
                    new_positions[:, group, :],
                    dtype=torch.float32,
                    device=device,
                )
                scale_g = torch.as_tensor(
                    np.log(
                        np.clip(
                            new_scale[:, group], 1e-4, None
                        )
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                opacity_g = torch.as_tensor(
                    _logit(new_opacity[:, group]),
                    dtype=torch.float32,
                    device=device,
                )
                color_g = torch.as_tensor(
                    _logit(new_colors[:, group]),
                    dtype=torch.float32,
                    device=device,
                )
                probe = _facet_probe_model(
                    pos_g, scale_g, opacity_g, color_g
                )
                rendered_g = _render_pixels(
                    probe,
                    low_camera,
                    None,
                    low_width,
                    low_height,
                    bg_tensor,
                    near=near,
                    far=far,
                    batch_size=256,
                    disk_cutoff=disk_cutoff,
                    falloff_power=falloff_power,
                    tile_size=16,
                    dense_renderer=False,
                    tile_pair_budget=tile_pair_budget,
                    tile_max_splats=tile_max_splats,
                    tile_work_budget=tile_work_budget,
                    use_cuda_tile=use_cuda_tile,
                )
                rendered_np = (
                    rendered_g.detach().cpu().numpy().reshape(
                        low_height, low_width, 3
                    )
                )
                diff = np.mean(np.abs(rendered_np - bg), axis=-1)
                err = np.mean(
                    np.abs(rendered_np - target_low), axis=-1
                )
                try:
                    from scipy import ndimage as ndi

                    if cluster_target_repeated.any():
                        coverage_smooth = ndi.uniform_filter(
                            diff, size=5
                        )
                    else:
                        coverage_smooth = ndi.uniform_filter(
                            diff, size=3
                        )
                except Exception:
                    coverage_smooth = diff
                center_coverage = coverage_smooth[
                    target_ys_np, target_xs_np
                ]
                if cluster_target_repeated.any():
                    disk_sums = np.zeros(total_targets, dtype=np.float32)
                    disk_counts = np.zeros(total_targets, dtype=np.int64)
                    for radius in np.unique(
                        cluster_cover_radius[
                            cluster_target_repeated
                        ]
                    ):
                        rr = int(radius)
                        dy, dx = np.ogrid[-rr : rr + 1, -rr : rr + 1]
                        disk = (dy * dy + dx * dx) <= rr * rr
                        sel = np.nonzero(
                            (cluster_target_repeated)
                            & (cluster_cover_radius == radius)
                        )[0]
                        rows = np.clip(
                            target_ys_np[sel, None]
                            + np.nonzero(disk)[0][None, :],
                            0,
                            low_height - 1,
                        )
                        cols = np.clip(
                            target_xs_np[sel, None]
                            + np.nonzero(disk)[1][None, :],
                            0,
                            low_width - 1,
                        )
                        vals = coverage_smooth[rows, cols]
                        disk_sums[sel] = vals.sum(axis=1)
                        disk_counts[sel] = vals.shape[1]
                    disk_frac = np.divide(
                        disk_sums,
                        np.maximum(disk_counts, 1),
                        out=np.zeros_like(disk_sums),
                        where=disk_counts > 0,
                    )
                    coverage_score[:, group] = np.where(
                        cluster_target_repeated,
                        disk_frac,
                        center_coverage,
                    )
                else:
                    coverage_score[:, group] = center_coverage
                color_err[:, group] = err[
                    target_ys_np, target_xs_np
                ]
                del probe
            selected = np.zeros(total_targets, dtype=np.int64)
            covered = coverage_score > (
                np.where(
                    cluster_target_repeated,
                    float(cluster_cover_tol),
                    cover_tol,
                )[:, None]
            )
            for target in range(total_targets):
                valid = np.nonzero(front_ok[target])[0]
                if valid.size == 0:
                    valid = np.arange(candidate_count)
                covered_ok = valid[
                    covered[target][valid]
                ]
                if covered_ok.size > 0:
                    order = np.lexsort(
                        (
                            color_err[target, covered_ok],
                            -coverage_score[target, covered_ok],
                            np.abs(
                                low_sigma_px[
                                    target, covered_ok
                                ]
                                - np.maximum(
                                    target_scale_px[target], 1.0
                                )
                            ),
                        )
                    )
                    selected[target] = covered_ok[order[0]]
                else:
                    order = np.lexsort(
                        (
                            color_err[target, valid],
                            -low_sigma_px[target, valid],
                        )
                    )
                    selected[target] = valid[order[0]]
            flat = selected + np.arange(total_targets) * candidate_count
            new_positions = new_positions.reshape(-1, 3)[flat]
            new_scale = new_scale.reshape(-1)[flat]
            new_colors = new_colors.reshape(-1, 3)[flat]
            new_opacity = new_opacity.reshape(-1)[flat]
        else:
            new_positions = new_positions.reshape(-1, 3)
            new_scale = new_scale.reshape(-1)
            new_colors = new_colors.reshape(-1, 3)
            new_opacity = new_opacity.reshape(-1)
        position_t = torch.as_tensor(
            new_positions, dtype=torch.float32, device=device
        )
        color_t = torch.as_tensor(
            _logit(new_colors), dtype=torch.float32, device=device
        )
        opacity_t = torch.as_tensor(
            _logit(new_opacity), dtype=torch.float32, device=device
        )
        scale_t = torch.as_tensor(
            np.log(np.clip(new_scale, 1e-4, None)),
            dtype=torch.float32,
            device=device,
        )
        old_protected = _model_hole_protected(model)
        old_protected_age = _model_hole_protected_age(model)
        model.positions = torch.nn.Parameter(
            torch.cat([model.positions.detach(), position_t], dim=0)
        )
        model.log_scales = torch.nn.Parameter(
            torch.cat([model.log_scales.detach(), scale_t], dim=0)
        )
        model.opacity_logits = torch.nn.Parameter(
            torch.cat([model.opacity_logits.detach(), opacity_t], dim=0)
        )
        model.color_logits = torch.nn.Parameter(
            torch.cat([model.color_logits.detach(), color_t], dim=0)
        )
        model.hole_protected = torch.cat(
            [
                old_protected,
                torch.ones(
                    int(position_t.shape[0]),
                    dtype=torch.bool,
                    device=device,
                ),
            ],
            dim=0,
        )
        model.hole_protected_age = torch.cat(
            [
                old_protected_age,
                torch.full(
                    (int(position_t.shape[0]),),
                    max(1, int(protection_cycles)),
                    dtype=torch.int64,
                    device=device,
                ),
            ],
            dim=0,
        )
        return (True, raw_hole_count, hole_mask)


def _split_splats_into_holes(
    model: SplatFacetField,
    camera: dict,
    device: torch.device,
    near: float,
    far: float,
    hole_mask: np.ndarray | None,
    low_res: int = 192,
    max_splits: int = 2000,
    max_splats: int = 150000,
    split_scale_factor: float = 0.5,
    opacity_floor: float = 0.3,
    children_per_split: int = 8,
    min_split_px: float = 1.2,
    spread_px: float = 1.5,
    target_image: np.ndarray | None = None,
    protection_cycles: int = 3,
) -> tuple[int, bool]:
    """Split facets near unresolved pixels into smaller children.

    Children are placed at the nearest hole pixel instead of only around the
    parent, so coverage can grow inside a hole even when the total splat count
    is already at its cap.
    """
    if (
        hole_mask is None
        or model.positions.numel() == 0
        or int(np.count_nonzero(hole_mask)) == 0
    ):
        return (0, False)
    with torch.no_grad():
        low_camera = _aspect_lowres_camera(camera, max_pixels=low_res)
        low_width = int(low_camera["width"])
        low_height = int(low_camera["height"])
        full_k = np.asarray(camera["K"], dtype=np.float32)
        fx_full = float(full_k[0][0])
        fy_full = float(full_k[1][1])
        full_width = int(camera["width"])
        full_height = int(camera["height"])
        if hole_mask.shape != (low_height, low_width):
            return (0, False)
        target_low: np.ndarray | None = None
        if target_image is not None:
            from PIL import Image

            target_low = np.asarray(
                Image.fromarray(
                    np.clip(target_image * 255.0, 0.0, 255.0).astype(
                        np.uint8
                    )
                ).resize((low_width, low_height), Image.LANCZOS),
                dtype=np.float32,
            ) / 255.0
        rotation, origin, fx, fy, cx, cy, _, _ = _camera_geometry(
            low_camera, device
        )
        positions = model.positions.detach()
        rel = positions - origin[None, :]
        camera_space = rel @ rotation
        z = camera_space[:, 2:3]
        safe_z = torch.clamp(z, min=1e-4)
        u = fx * camera_space[:, 0:1] / safe_z + cx
        v = fy * camera_space[:, 1:2] / safe_z + cy
        opacity = torch.sigmoid(model.opacity_logits)
        visible = (
            (z[:, 0] > near)
            & (z[:, 0] < far)
            & (u[:, 0] >= 0.0)
            & (u[:, 0] < low_width)
            & (v[:, 0] >= 0.0)
            & (v[:, 0] < low_height)
            & (opacity >= 0.02)
        )
        if not torch.any(visible):
            return (0, False)
        ui_np = np.floor(u[:, 0].cpu().numpy() + 0.5).astype(np.int32)
        vi_np = np.floor(v[:, 0].cpu().numpy() + 0.5).astype(np.int32)
        z_np = z[:, 0].cpu().numpy()
        pixel_ok = (
            (ui_np >= 0)
            & (ui_np < low_width)
            & (vi_np >= 0)
            & (vi_np < low_height)
        )
        pad = 2
        padded = np.pad(hole_mask, pad, mode="edge")
        dilated = np.zeros_like(hole_mask)
        for dy in range(-pad, pad + 1):
            for dx in range(-pad, pad + 1):
                dilated |= padded[
                    pad + dy : pad + dy + low_height,
                    pad + dx : pad + dx + low_width,
                ]
        visible_np = visible.cpu().numpy() & pixel_ok
        candidate_mask = np.zeros(model.positions.shape[0], dtype=bool)
        candidate_mask[visible_np] = dilated[
            vi_np[visible_np], ui_np[visible_np]
        ]
        candidate_idx = np.nonzero(candidate_mask)[0]
        if candidate_idx.size == 0:
            return (0, False)
        sigma_np = (
            torch.exp(model.log_scales)
            * fx_full
            / torch.clamp(z[:, 0], min=1e-4)
        )
        sigma_np = sigma_np[candidate_idx].cpu().numpy()
        cand_px = ui_np[candidate_idx]
        cand_py = vi_np[candidate_idx]
        cand_inside = hole_mask[cand_py, cand_px]
        try:
            from scipy import ndimage as ndi

            hole_dist = ndi.distance_transform_edt(~hole_mask)
            label_map, _ = ndi.label(hole_mask)
            label_area = np.bincount(label_map.ravel())
            use_labels = True
        except Exception:
            hole_dist = None
            label_map = None
            label_area = None
            use_labels = False
        cand_dist = (
            hole_dist[cand_py, cand_px]
            if hole_dist is not None
            else np.zeros(candidate_idx.size, dtype=np.float32)
        )
        max_splats = max(1, int(max_splats))
        children = max(2, int(children_per_split))
        current_count = int(model.positions.shape[0])
        headroom = max(0, max_splats - current_count)
        budget_by_headroom = headroom // children
        max_splits = max(1, int(max_splits))
        if budget_by_headroom > 0:
            max_splits = min(max_splits, budget_by_headroom)
        cell_size = max(
            2, int(math.ceil(max(low_width, low_height) / 8))
        )
        rows = max(1, int(math.ceil(low_height / cell_size)))
        cols = max(1, int(math.ceil(low_width / cell_size)))
        cell_keys = (cand_py // cell_size) * cols + (cand_px // cell_size)
        cells, inverse = np.unique(cell_keys, return_inverse=True)
        hole_ys, hole_xs = np.nonzero(hole_mask)
        hole_cell_keys = (hole_ys // cell_size) * cols + (
            hole_xs // cell_size
        )
        total_cells = rows * cols
        cell_hole_counts = np.bincount(
            hole_cell_keys, minlength=total_cells
        )[cells]
        total_holes = max(1, int(np.count_nonzero(hole_mask)))
        quotas = np.floor(
            cell_hole_counts * max_splits / total_holes
        ).astype(np.int64)
        remainder = max_splits - int(quotas.sum())
        if remainder > 0:
            cell_order = np.argsort(-cell_hole_counts)[:remainder]
            quotas[cell_order] += 1
        chosen_pool: list[int] = []
        for cell_index in range(len(cells)):
            quota = int(quotas[cell_index])
            if quota <= 0:
                continue
            cand_positions = np.nonzero(inverse == cell_index)[0]
            if quota >= cand_positions.size:
                chosen_pool.extend(cand_positions.tolist())
                continue
            inside = cand_inside[cand_positions]
            if use_labels:
                labels = label_map[
                    cand_py[cand_positions], cand_px[cand_positions]
                ]
                area_scores = np.asarray(
                    [
                        label_area[int(label)] if label > 0 else 0.0
                        for label in labels
                    ],
                    dtype=np.float32,
                )
                order = np.lexsort(
                    (
                        cand_dist[cand_positions],
                        -sigma_np[cand_positions],
                        -area_scores,
                        (~inside).astype(np.int8),
                    )
                )
            else:
                order = np.lexsort(
                    (
                        cand_dist[cand_positions],
                        -sigma_np[cand_positions],
                        (~inside).astype(np.int8),
                    )
                )
            chosen_pool.extend(
                cand_positions[order[:quota]].tolist()
            )
        if not chosen_pool:
            order = np.argsort(-sigma_np)[:max_splits]
            chosen_pool = order.tolist()
        chosen_pool = np.asarray(chosen_pool, dtype=np.int64)

        # Index hole pixels by cell so each parent can gather several distinct
        # targets inside the hole instead of stacking all children at one pixel.
        hole_cells: dict[int, list[tuple[int, int]]] = {}
        for h_y, h_x in zip(hole_ys.tolist(), hole_xs.tolist()):
            key = (int(h_y) // cell_size) * cols + (int(h_x) // cell_size)
            hole_cells.setdefault(key, []).append((int(h_y), int(h_x)))
        target_us: list[list[int]] = []
        target_vs: list[list[int]] = []
        kept_chosen: list[int] = []
        low_px_ratio = low_width / max(full_width, 1)
        for rel_index in chosen_pool:
            index = int(candidate_idx[rel_index])
            px = int(cand_px[rel_index])
            py = int(cand_py[rel_index])
            if not (0 <= px < low_width and 0 <= py < low_height):
                continue
            local_hole: list[tuple[int, int]] = []
            if hole_mask[py, px]:
                local_hole.append((py, px))
            search_radius = max(
                3,
                min(
                    int(
                        math.ceil(
                            float(sigma_np[rel_index]) * low_px_ratio * 1.5
                        )
                    ),
                    max(4, max(low_width, low_height) // 5),
                ),
            )
            for radius_factor in (1, 2, 4, 8):
                window = search_radius * radius_factor
                collected: list[tuple[int, int]] = []
                center_cell_y = py // cell_size
                center_cell_x = px // cell_size
                cell_radius = max(
                    1, int(math.ceil(window / cell_size)) + 1
                )
                for cy in range(
                    center_cell_y - cell_radius,
                    center_cell_y + cell_radius + 1,
                ):
                    if cy < 0:
                        continue
                    for cx in range(
                        center_cell_x - cell_radius,
                        center_cell_x + cell_radius + 1,
                    ):
                        if cx < 0:
                            continue
                        for yy, xx in hole_cells.get(
                            cy * cols + cx, ()
                        ):
                            if (
                                abs(yy - py) <= window
                                and abs(xx - px) <= window
                            ):
                                collected.append((yy, xx))
                if not collected:
                    continue
                unique = list(dict.fromkeys(collected))
                local_hole = unique
                if len(unique) >= children:
                    break
            if not local_hole:
                nearest: tuple[int, int] | None = None
                best_dist = pad * pad * 2 + 1
                for dy in range(-pad, pad + 1):
                    yy = py + dy
                    if yy < 0 or yy >= low_height:
                        continue
                    for dx in range(-pad, pad + 1):
                        xx = px + dx
                        if xx < 0 or xx >= low_width:
                            continue
                        if hole_mask[yy, xx]:
                            dist = dx * dx + dy * dy
                            if dist < best_dist:
                                best_dist = dist
                                nearest = (yy, xx)
                if nearest is None:
                    continue
                local_hole = [nearest]
            combined = [(py, px)] + [
                pixel for pixel in local_hole if pixel != (py, px)
            ]
            dists = np.asarray(
                [
                    (yy - py) ** 2 + (xx - px) ** 2
                    for yy, xx in combined
                ],
                dtype=np.float64,
            )
            order = np.argsort(dists)
            if len(order) >= children:
                picks = np.unique(
                    np.linspace(0, len(order) - 1, children)
                    .round()
                    .astype(np.int64)
                ).tolist()
                while len(picks) < children:
                    picks.append(picks[-1])
            else:
                picks = list(range(len(order)))
                while len(picks) < children:
                    picks.append(picks[-1])
            target_us.append(
                [combined[int(entry)][1] for entry in picks]
            )
            target_vs.append(
                [combined[int(entry)][0] for entry in picks]
            )
            kept_chosen.append(index)
        if not kept_chosen:
            return (0, False)
        chosen_idx = np.asarray(kept_chosen, dtype=np.int64)
        parent_z = z_np[chosen_idx]
        parent_scale = (
            torch.exp(model.log_scales[chosen_idx]).cpu().numpy()
        )
        parent_color = model.color_logits[chosen_idx].detach()
        parent_opacity = (
            torch.sigmoid(model.opacity_logits[chosen_idx]).cpu().numpy()
        )
        target_u = np.asarray(target_us, dtype=np.float32) + 0.5
        target_v = np.asarray(target_vs, dtype=np.float32) + 0.5
        target_cam_x = (
            (target_u - cx) / fx * parent_z[:, None]
        )
        target_cam_y = (
            (target_v - cy) / fy * parent_z[:, None]
        )
        child_scale = np.clip(
            parent_scale * max(float(split_scale_factor), 0.25),
            1e-4,
            parent_scale * 0.95,
        )
        min_px = max(float(min_split_px), 0.3)
        child_scale = np.clip(
            child_scale,
            parent_z / max(fx, 1e-4) * min_px,
            parent_z / max(fx, 1e-4) * max(min_px * 2.0, 1.2),
        )
        offsets = np.zeros((children, 2), dtype=np.float32)
        spread = max(float(spread_px), 0.2)
        if children >= 1:
            offsets[0] = [0.0, 0.0]
        if children >= 2:
            offsets[1] = [spread, 0.0]
        if children >= 3:
            offsets[2] = [-spread, 0.0]
        if children >= 4:
            offsets[3] = [0.0, spread]
        if children >= 5:
            offsets[4] = [-spread, -spread]
        if children >= 6:
            offsets[5] = [spread, -spread]
        if children >= 7:
            offsets[6] = [-spread, spread]
        if children >= 8:
            offsets[7] = [spread, spread]
        cam_offsets = (
            offsets[:, None, :]
            * (parent_z[None, :, None] / max(fx, 1e-4))
            * 0.35
        )
        parent_z_2d = np.repeat(parent_z[:, None], children, axis=1)
        base_cam = np.stack(
            [target_cam_x, target_cam_y, parent_z_2d], axis=-1
        )
        child_cam = base_cam.transpose(1, 0, 2)
        child_cam[..., 0] += cam_offsets[..., 0]
        child_cam[..., 1] += cam_offsets[..., 1]
        rotation_np = rotation.cpu().numpy()
        origin_np = origin.cpu().numpy()
        child_positions = (
            child_cam @ rotation_np.T + origin_np[None, None, :]
        ).transpose(1, 0, 2).reshape(-1, 3)
        child_scales = np.repeat(child_scale, children, axis=0)
        if target_low is not None:
            target_vs_np = np.asarray(target_vs, dtype=np.int32)
            target_us_np = np.asarray(target_us, dtype=np.int32)
            child_rgb = target_low[
                target_vs_np, target_us_np
            ].reshape(-1, 3)
            child_colors = torch.as_tensor(
                _logit(child_rgb),
                dtype=torch.float32,
                device=device,
            )
        else:
            child_colors = parent_color.repeat_interleave(children, dim=0)
        child_opacity = np.clip(
            parent_opacity * 0.85,
            float(opacity_floor),
            0.9,
        )
        child_opacity_logits = torch.as_tensor(
            _logit(np.repeat(child_opacity, children)),
            dtype=torch.float32,
            device=device,
        )

        keep_mask = np.ones(model.positions.shape[0], dtype=bool)
        keep_t = torch.as_tensor(keep_mask, dtype=torch.bool, device=device)
        old_keep_indices = np.nonzero(keep_mask)[0]
        new_positions = torch.cat(
            [
                positions[keep_t],
                torch.as_tensor(
                    child_positions, dtype=torch.float32, device=device
                ),
            ],
            dim=0,
        )
        new_scales = torch.cat(
            [
                model.log_scales.detach()[keep_t],
                torch.as_tensor(
                    np.log(child_scales), dtype=torch.float32, device=device
                ),
            ],
            dim=0,
        )
        new_opacities = torch.cat(
            [
                model.opacity_logits.detach()[keep_t],
                child_opacity_logits,
            ],
            dim=0,
        )
        new_colors = torch.cat(
            [
                model.color_logits.detach()[keep_t],
                child_colors,
            ],
            dim=0,
        )
        protected_old_np = _model_hole_protected(model).cpu().numpy()
        protected_age_old_np = _model_hole_protected_age(model).cpu().numpy()
        new_protected_np = np.concatenate(
            [
                protected_old_np[keep_mask],
                np.ones(int(child_positions.shape[0]), dtype=bool),
            ]
        )
        child_protection_age = np.full(
            int(child_positions.shape[0]),
            max(1, int(protection_cycles)),
            dtype=np.int64,
        )
        new_protected_age_np = np.concatenate(
            [
                protected_age_old_np[keep_mask],
                child_protection_age,
            ]
        )
        remove_mask = np.zeros(new_positions.shape[0], dtype=bool)
        excess = int(new_positions.shape[0]) - max(0, int(max_splats))
        if excess > 0:
            new_opacity_np = torch.sigmoid(
                torch.cat(
                    [
                        model.opacity_logits.detach()[keep_t],
                        child_opacity_logits,
                    ]
                )
            ).cpu().numpy()
            removal_order = np.argsort(new_opacity_np, kind="stable")
            protected_flags = new_protected_np[removal_order]
            removal_order = np.concatenate(
                [
                    removal_order[~protected_flags],
                    removal_order[protected_flags],
                ]
            )
            remove_mask[removal_order[:excess]] = True
        remove_t = torch.as_tensor(
            ~remove_mask, dtype=torch.bool, device=device
        )
        model.positions = torch.nn.Parameter(new_positions[remove_t])
        model.log_scales = torch.nn.Parameter(new_scales[remove_t])
        model.opacity_logits = torch.nn.Parameter(
            new_opacities[remove_t]
        )
        model.color_logits = torch.nn.Parameter(new_colors[remove_t])
        model.hole_protected = torch.as_tensor(
            new_protected_np[~remove_mask],
            dtype=torch.bool,
            device=device,
        )
        model.hole_protected_age = torch.as_tensor(
            new_protected_age_np[~remove_mask],
            dtype=torch.int64,
            device=device,
        )
        return (int(chosen_idx.size), True)


def train_tfrf_splat(
    images: np.ndarray,
    cameras: list[dict],
    config: dict,
    project_dir,
    task_id: str,
    sparse_points: list[list[float]] | None = None,
) -> tuple[dict, list[dict]]:
    from .main import (
        SCENE_BACKGROUND,
        compute_metrics,
        emit_status,
        emit_training_status,
        log,
        ray_bounds_from_points,
        scene_scale_from_cameras,
        status_after,
    )

    train_count = int(config.get("train_views", 16))
    steps = int(config.get("steps", 1200))
    patch = int(config.get("patch_size", 1024))
    eval_every = int(config.get("eval_every", 100))
    training_progress_every = max(
        1,
        int(config.get("training_progress_every", config.get("progress_every", 1))),
    )
    learning_rate = float(config.get("learning_rate", 0.02))
    render_batch = int(config.get("render_batch_size", 256))
    render_tile = max(1, int(config.get("render_tile_size", 16)))
    dense_renderer = bool(config.get("dense_renderer", False))
    auto_dense_threshold = int(config.get("auto_dense_threshold", 20000))
    auto_dense_pixels = int(config.get("auto_dense_pixels", 1_000_000))
    tile_pair_budget = int(config.get("tile_pair_budget", 4_000_000))
    tile_work_budget = int(config.get("tile_work_budget", 2_000_000))
    tile_max_splats = int(config.get("tile_max_splats", 8192))
    base_tile_pair_budget = tile_pair_budget
    base_tile_work_budget = tile_work_budget
    base_tile_max_splats = tile_max_splats
    disk_cutoff = float(config.get("disk_cutoff", 1.8))
    falloff_power = float(config.get("falloff_power", 1.0))
    train_original = bool(config.get("train_original_resolution", False))
    train_max_pixels = int(config.get("train_max_image_pixels", 2_000_000))
    use_cuda_tile = bool(config.get("use_cuda_tile", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_scale = scene_scale_from_cameras(cameras)
    train_cameras = cameras[:train_count]
    use_image_colors = bool(config.get("use_image_colors", True))

    if sparse_points:
        points = np.asarray(sparse_points, dtype=np.float32)[:, :3]
        colors = np.asarray(sparse_points, dtype=np.float32)[:, 3:6] / 255.0
        if use_image_colors and colors.size > 0 and np.mean(colors) < 0.05:
            image_colors = color_points_from_images(points, cameras, images)
            covered = int(np.sum(np.any(image_colors > 0, axis=1)))
            log(
                f"[init] 稀疏点颜色缺失，从 {min(len(cameras), len(images))} 张"
                f"图片投影采样颜色，{covered}/{len(points)} 点获得颜色"
            )
            colors = image_colors
    else:
        points = np.asarray(
            [
                [math.cos(angle), math.sin(angle), 0.0]
                for angle in np.linspace(0.0, math.tau, 2400)
            ],
            dtype=np.float32,
        )
        colors = None
    max_splats = int(config.get("max_splats", 12000))
    adaptive_budget = bool(config.get("adaptive_budget", True))
    budget_check_every = max(1, int(config.get("budget_check_every", 20)))
    current_cap = max_splats
    scene_cap = max_splats
    max_splats_absolute = max(
        max_splats, int(config.get("max_splats_absolute", 150000))
    )
    scene_cap_boost = max(1000, int(config.get("scene_cap_boost", 10000)))
    ray_cover_count = min(
        int(config.get("ray_cover_points", 6000)),
        max(0, max_splats - len(points)),
    )
    if ray_cover_count > 0:
        init_rng = np.random.default_rng(11)
        points, colors = _build_ray_coverage_points(
            points, colors, train_cameras, ray_cover_count, init_rng
        )
        if use_image_colors:
            image_colors = color_points_from_images(
                points, train_cameras, images[: len(train_cameras)]
            )
            covered = np.any(image_colors > 0, axis=1)
            if colors is None:
                colors = np.full_like(image_colors, 0.55)
            colors[covered] = image_colors[covered]
            log(
                f"[init] 射线补点 {ray_cover_count}，"
                f"{int(np.sum(covered))} 个点从训练图采样颜色"
            )
        log(f"[init] 初始棱面 {len(points)} 个（含射线补点）")
    init_scale = float(config.get("init_scale", 0.0))
    if init_scale <= 0:
        init_scale = estimate_init_scale(
            points,
            cameras,
            target_px=float(config.get("init_target_px", 1.3)),
        )
        log(f"[init] 自适应棱面半径 {init_scale:.4f}")
    point_scales = _estimate_point_scales(
        points,
        init_scale,
        multiplier=float(config.get("scale_neighbor_factor", 1.0)),
    )
    model = SplatFacetField(
        points,
        colors,
        scale=init_scale,
        scales=point_scales,
        opacity=float(config.get("init_opacity", 0.45)),
    ).to(device)

    def build_optimizer() -> torch.optim.Adam:
        return torch.optim.Adam(
            [
                {"params": [model.positions], "lr": learning_rate * 0.6},
                {"params": [model.color_logits], "lr": learning_rate},
                {"params": [model.log_scales], "lr": learning_rate * 0.6},
                {"params": [model.opacity_logits], "lr": learning_rate * 0.6},
            ],
        )

    optimizer = build_optimizer()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=learning_rate * 0.1
    )

    eval_cameras = cameras[train_count:]
    if not eval_cameras:
        eval_cameras = cameras[-1:]
    camera_choices = np.random.default_rng(3).integers(
        0, len(train_cameras), size=steps
    )
    surface_points = (
        np.asarray(sparse_points, dtype=np.float32)[:, :3] if sparse_points else None
    )
    surface_far_prior_weight = float(
        config.get("surface_far_prior_weight", 0.02)
    )
    surface_far_opacity_cap = float(
        config.get("surface_far_opacity_cap", 0.18)
    )
    surface_far_opacity_clamp = float(
        config.get("surface_far_opacity_clamp", 0.0)
    )
    surface_far_prior_every = max(
        1, int(config.get("surface_far_prior_every", 25))
    )
    far_prune_enabled = bool(config.get("far_prune", False))
    far_prune_after = int(config.get("far_prune_after", 160))
    far_prune_every = max(1, int(config.get("far_prune_every", 40)))
    far_prune_opacity = float(config.get("far_prune_opacity", 0.15))
    far_prune_max_ratio = float(config.get("far_prune_max_ratio", 0.6))
    far_prune_keep_min_fraction = float(
        config.get("far_prune_keep_min_fraction", 0.2)
    )
    surface_far_band = None
    far_surface_mask = None
    far_mask_active = False
    if surface_points is not None and len(surface_points) > 0:
        surface_far_band = _surface_far_band(
            surface_points,
            init_scale,
            band_factor=float(config.get("surface_far_band_factor", 1.25)),
            min_dist_factor=float(config.get("surface_far_min_dist_factor", 3.0)),
        )
        if surface_far_prior_weight > 0:
            far_surface_mask = torch.as_tensor(
                _surface_far_mask(points, surface_points, surface_far_band),
                dtype=torch.bool,
                device=device,
            )
            far_mask_active = bool(far_surface_mask.any().item())
    scale_target_cap_factor = float(config.get("scale_target_cap_factor", 8.0))
    scale_targets = torch.as_tensor(
        _scale_targets_for_splats(
            points,
            surface_points,
            init_scale,
            far_scale_factor=float(config.get("far_scale_factor", 0.8)),
            cap_factor=scale_target_cap_factor,
        ),
        dtype=torch.float32,
        device=device,
    )
    hole_scale_boost = float(config.get("hole_protected_scale_factor", 3.0))

    def apply_hole_scale_boost(targets: torch.Tensor) -> torch.Tensor:
        if hole_scale_boost <= 1.0:
            return targets
        protected = _model_hole_protected(model)
        if not bool(protected.any().item()):
            return targets
        return targets * (
            1.0 + (hole_scale_boost - 1.0) * protected.float()
        )

    scale_targets = apply_hole_scale_boost(scale_targets)

    def rebuild_scale_targets() -> None:
        nonlocal scale_targets, far_surface_mask, far_mask_active
        positions_np = model.positions.detach().cpu().numpy()
        scale_targets = torch.as_tensor(
            _scale_targets_for_splats(
                positions_np,
                surface_points,
                init_scale,
                far_scale_factor=float(config.get("far_scale_factor", 0.8)),
                cap_factor=scale_target_cap_factor,
            ),
            dtype=torch.float32,
            device=device,
        )
        scale_targets = apply_hole_scale_boost(scale_targets)
        if surface_far_band is not None:
            far_surface_mask = torch.as_tensor(
                _surface_far_mask(
                    positions_np, surface_points, surface_far_band
                ),
                dtype=torch.bool,
                device=device,
            )
            far_mask_active = bool(far_surface_mask.any().item())

    background = SCENE_BACKGROUND
    bg_tensor = torch.as_tensor(background, dtype=torch.float32, device=device)
    global_res = int(config.get("global_residual_resolution", 16))
    global_res_weight = float(config.get("global_residual_weight", 0.6))
    images_low = (
        _downsample_images(images, global_res) if global_res > 0 else None
    )
    if images_low is not None:
        images_low_t = torch.as_tensor(
            images_low, dtype=torch.float32, device=device
        )
    else:
        images_low_t = None
    cached_pixel_grid: torch.Tensor | None = None
    cached_grid_shape: tuple[int, int] | None = None

    def full_pixel_grid(grid_width: int, grid_height: int) -> torch.Tensor:
        nonlocal cached_pixel_grid, cached_grid_shape
        if (
            cached_pixel_grid is None
            or cached_grid_shape != (grid_width, grid_height)
        ):
            uu, vv = torch.meshgrid(
                torch.arange(grid_width, dtype=torch.float32, device=device),
                torch.arange(grid_height, dtype=torch.float32, device=device),
                indexing="xy",
            )
            cached_pixel_grid = torch.stack(
                [uu.reshape(-1), vv.reshape(-1)], dim=1
            )
            cached_grid_shape = (grid_width, grid_height)
        return cached_pixel_grid

    def refine_hole_steps(
        refine_camera: dict,
        refine_image: np.ndarray,
        refine_near: float,
        refine_far: float,
        refine_steps: int,
        refine_low_res: int,
        refine_weight: float,
    ) -> None:
        """Run a short low-res pass so freshly placed hole splats settle."""
        if refine_steps <= 0:
            return
        from PIL import Image

        low_camera = _aspect_lowres_camera(
            refine_camera, max_pixels=refine_low_res
        )
        low_width = int(low_camera["width"])
        low_height = int(low_camera["height"])
        target_low = np.asarray(
            Image.fromarray(
                np.clip(refine_image * 255.0, 0.0, 255.0).astype(np.uint8)
            ).resize((low_width, low_height), Image.LANCZOS),
            dtype=np.float32,
        ) / 255.0
        target_t = torch.as_tensor(
            target_low.reshape(-1, 3), dtype=torch.float32, device=device
        )
        letterbox_t = torch.as_tensor(
            _letterbox_ignore_mask(target_low), device=device
        )
        pixels = full_pixel_grid(low_width, low_height)
        opt = build_optimizer()
        refine_bg_tol = float(config.get("hole_bg_tol", 0.14))
        refine_target_tol = float(config.get("hole_target_tol", 0.06))
        refine_bad_fit_tol = float(
            config.get("hole_fill_bad_fit_tol", 0.16)
        )
        refine_include_bad_fit = bool(
            config.get("hole_fill_include_bad_fit", True)
        )
        refine_scale_weight = float(
            config.get("hole_refine_scale_weight", 0.0)
        )
        for _ in range(refine_steps):
            opt.zero_grad()
            rendered = _render_pixels(
                model,
                low_camera,
                pixels,
                low_width,
                low_height,
                bg_tensor,
                near=refine_near,
                far=refine_far,
                batch_size=render_batch,
                disk_cutoff=disk_cutoff,
                falloff_power=falloff_power,
                tile_size=render_tile,
                dense_renderer=dense_renderer,
                tile_pair_budget=tile_pair_budget,
                tile_max_splats=tile_max_splats,
                tile_work_budget=tile_work_budget,
                use_cuda_tile=use_cuda_tile,
            )
            target_img = target_t.reshape(low_height, low_width, 3)
            rendered_img = rendered.reshape(low_height, low_width, 3)
            target_content = (
                torch.mean(torch.abs(target_img - bg_tensor), dim=-1)
                > refine_target_tol
            ) & (~letterbox_t)
            render_bg = (
                torch.mean(torch.abs(rendered_img - bg_tensor), dim=-1)
                < refine_bg_tol
            )
            bad_fit = (
                torch.mean(torch.abs(rendered_img - target_img), dim=-1)
                > refine_bad_fit_tol
            )
            hole_mask = target_content & render_bg
            if refine_include_bad_fit:
                hole_mask = hole_mask | (target_content & bad_fit)
            if bool(hole_mask.any()):
                loss = refine_weight * torch.mean(
                    ((target_img - rendered_img) ** 2)
                    * hole_mask.unsqueeze(-1)
                )
            else:
                loss = torch.mean((rendered - target_t) ** 2)
            protected_mask = _model_hole_protected(model)
            if (
                refine_scale_weight > 0
                and bool(protected_mask.any().item())
                and scale_targets.numel() == model.log_scales.numel()
            ):
                loss = loss + refine_scale_weight * torch.mean(
                    (
                        model.log_scales[protected_mask]
                        - torch.log(
                            torch.clamp(
                                scale_targets[protected_mask], min=1e-4
                            )
                        )
                    )
                    ** 2
                )
            loss.backward()
            if (
                protected_mask.numel() == model.positions.numel()
                and not bool(protected_mask.all().item())
            ):
                unprotected = ~protected_mask
                for param in (
                    model.positions,
                    model.color_logits,
                    model.log_scales,
                    model.opacity_logits,
                ):
                    if param.grad is not None:
                        param.grad[unprotected] = 0.0
            opt.step()
            with torch.no_grad():
                model.log_scales.clamp_(
                    math.log(1e-4), math.log(scale_upper)
                )

    log(
        f"[train] TFRF-facets 开始，{len(train_cameras)} 训练视角，"
        f"{len(points)} 棱面，{steps} 步，设备 {device}，"
        f"场景尺度 {scene_scale:.4f}，原分辨率 {train_original}，"
        f"最大像素 {train_max_pixels}"
    )
    emit_training_status(project_dir, task_id, 0, steps)
    if use_cuda_tile:
        log("[train] 正在编译/加载 CUDA 渲染内核，首次约需 1-2 分钟")
        _tile_cuda_loader()
    history: list[dict] = []
    start_wall = time.time()
    densify_every = max(1, int(config.get("densify_every", 50)))
    densify_after = int(config.get("densify_after", 80))
    prune_opacity = float(config.get("prune_opacity", 0.04))
    reset_after = int(config.get("opacity_reset_after", 140))
    reset_every = int(config.get("opacity_reset_every", 100))
    reset_opacity = float(config.get("opacity_reset_value", 0.7))
    scale_limit_enabled = bool(config.get("scale_limit_enabled", False))
    if scale_limit_enabled:
        scale_upper = max(
            init_scale * float(config.get("scale_log_clamp", 6.0)),
            float(config.get("scale_upper_floor", 0.2)),
        )
    else:
        scale_upper = float("inf")
    ssim_loss_weight = float(config.get("ssim_loss_weight", 0.2))
    ssim_window_size = int(config.get("ssim_window_size", 11))
    edge_loss_weight = float(config.get("edge_loss_weight", 0.0))
    hole_coverage_weight = float(config.get("hole_coverage_weight", 0.25))
    projected_scale_prior_weight = float(
        config.get("projected_scale_prior_weight", 0.015)
    )
    projected_scale_target_px = float(
        config.get("projected_scale_target_px", 1.6)
    )
    split_large_enabled = bool(config.get("split_large_splats", True))
    split_after = int(config.get("split_after", 200))
    split_every = max(1, int(config.get("split_every", 50)))
    split_scale_factor = float(config.get("split_scale_factor", 0.55))
    split_min_scale_px = float(config.get("split_min_projected_px", 2.4))
    split_min_screen_fraction = float(
        config.get("split_min_screen_fraction", 0.1)
    )
    split_max_per_cycle = int(config.get("split_max_per_cycle", 3000))
    split_children = int(config.get("split_children", 4))
    split_shrink_factor = float(config.get("split_shrink_factor", 0.85))
    densify_enabled = bool(config.get("densify", True))
    hole_fill_enabled = bool(config.get("hole_fill", True))
    hole_fill_after = int(config.get("hole_fill_after", 100))
    hole_fill_every = max(1, int(config.get("hole_fill_every", 80)))
    hole_fill_add = int(config.get("hole_fill_add", 500))
    hole_fill_low_res = int(config.get("hole_fill_resolution", 96))
    hole_fill_refine_steps = max(
        0, int(config.get("hole_fill_refine_steps", 25))
    )
    hole_fill_refine_resolution = max(
        1, int(config.get("hole_fill_refine_resolution", 96))
    )
    hole_fill_refine_weight = float(
        config.get("hole_fill_refine_weight", 1.5)
    )
    hole_fill_refine_min_budget = max(
        0, int(config.get("hole_fill_refine_min_budget", 1000))
    )
    hole_fill_add_scene_boost = int(
        config.get("hole_fill_add_scene_boost", 2500)
    )
    scene_demand_hole_min = max(
        0, int(config.get("scene_demand_hole_min", 200))
    )
    prune_low_opacity = bool(config.get("prune_low_opacity", True))
    prune_opacity_every = max(1, int(config.get("prune_opacity_every", 80)))
    prune_opacity_threshold = float(
        config.get("prune_opacity_threshold", 0.03)
    )
    prune_opacity_keep_ratio = float(
        config.get("prune_opacity_keep_ratio", 0.75)
    )
    grad_accum = torch.zeros(len(points), dtype=torch.float32, device=device)
    (
        current_cap,
        tile_pair_budget,
        tile_work_budget,
        tile_max_splats,
        vram_pressure,
        vram_free_mb,
        vram_total_mb,
    ) = _vram_budget_update(
        config,
        device,
        scene_cap,
        current_cap,
        base_tile_pair_budget,
        base_tile_work_budget,
        base_tile_max_splats,
    )
    if adaptive_budget and vram_total_mb > 0:
        log(
            f"[vram] 显存 {vram_free_mb:.0f}/{vram_total_mb:.0f} MiB，"
            f"初始棱面上限 {current_cap}"
        )
    last_raw_holes = 0

    def recent_train_camera_indices(
        step: int, window: int, limit: int
    ) -> list[int]:
        recent = camera_choices[max(0, step - window) : step + 1]
        ordered = list(dict.fromkeys(int(idx) for idx in recent.tolist()))
        if len(ordered) <= limit:
            return ordered
        return ordered[-limit:]

    def refresh_hole_protection(
        camera_indices: list[int] | None = None,
    ) -> int:
        if not hole_fill_enabled:
            return 0
        return _refresh_hole_protection(
            model,
            cameras,
            images,
            surface_points,
            device,
            camera_indices=camera_indices,
            low_res=hole_fill_low_res,
            dilate_px=int(config.get("hole_fill_hole_dilate_px", 2)),
            bg_tol=float(config.get("hole_fill_bg_tol", 0.14)),
            bad_fit_tol=float(config.get("hole_fill_bad_fit_tol", 0.16)),
            has_target_tol=float(config.get("hole_target_tol", 0.06)),
            disk_cutoff=disk_cutoff,
            falloff_power=falloff_power,
            tile_pair_budget=tile_pair_budget,
            tile_max_splats=tile_max_splats,
            tile_work_budget=tile_work_budget,
            use_cuda_tile=use_cuda_tile,
        )

    def accumulate_gradients(current: torch.Tensor | None) -> None:
        if current is None:
            return
        if current.dim() > 1:
            current = current.norm(dim=1)
        if current.shape[0] != grad_accum.shape[0]:
            grad_accum.resize_(current.shape[0])
        grad_accum.mul_(0.95).add_(current)

    for step in range(steps):
        step_start = time.time()
        camera = train_cameras[int(camera_choices[step])]
        c2w = np.asarray(camera["c2w"], dtype=np.float32)
        width, height = camera["width"], camera["height"]
        k = camera["K"]
        fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
        near, far = ray_bounds_from_points(
            c2w[:3, 3], surface_points, rotation=c2w[:3, :3]
        ) if surface_points is not None and len(surface_points) else (0.3, 7.6)
        rng = np.random.default_rng(step)
        if train_original:
            pixels = full_pixel_grid(width, height)
        else:
            ys = rng.integers(0, height, size=patch)
            xs = rng.integers(0, width, size=patch)
            pixels = torch.as_tensor(
                np.stack([xs, ys], axis=-1), dtype=torch.float32, device=device
            )
        rendered = _render_pixels(
            model,
            camera,
            pixels,
            width,
            height,
            bg_tensor,
            near=near,
            far=far,
            batch_size=render_batch,
            disk_cutoff=disk_cutoff,
            falloff_power=falloff_power,
            tile_size=render_tile,
            dense_renderer=dense_renderer,
            auto_dense_threshold=auto_dense_threshold,
            auto_dense_pixels=auto_dense_pixels,
            tile_pair_budget=tile_pair_budget,
            tile_max_splats=tile_max_splats,
            tile_work_budget=tile_work_budget,
            use_cuda_tile=use_cuda_tile,
        )
        if train_original:
            target = torch.as_tensor(
                images[int(camera_choices[step])],
                dtype=torch.float32,
                device=device,
            ).reshape(-1, 3)
        else:
            target = torch.as_tensor(
                images[int(camera_choices[step])][ys, xs],
                dtype=torch.float32,
                device=device,
            )
        loss = torch.mean((rendered - target) ** 2)
        if train_original and ssim_loss_weight > 0:
            ssim_term = _ssim_loss(
                rendered.reshape(height, width, 3),
                target.reshape(height, width, 3),
                window_size=ssim_window_size,
            )
            loss = loss + ssim_loss_weight * ssim_term
        if train_original and hole_coverage_weight > 0:
            rendered_img = rendered.reshape(height, width, 3)
            target_img = target.reshape(height, width, 3)
            target_content = (
                torch.mean(torch.abs(target_img - bg_tensor), dim=-1)
                > float(config.get("hole_target_tol", 0.06))
            )
            render_bg = (
                torch.mean(torch.abs(rendered_img - bg_tensor), dim=-1)
                < float(config.get("hole_bg_tol", 0.14))
            )
            hole_mask = target_content & render_bg
            if bool(hole_mask.any()):
                hole_loss = torch.mean(
                    ((target_img - rendered_img) ** 2)
                    * hole_mask.unsqueeze(-1)
                )
                loss = loss + hole_coverage_weight * hole_loss
        if train_original and edge_loss_weight > 0:
            loss = loss + edge_loss_weight * _gradient_loss(
                rendered.reshape(height, width, 3),
                target.reshape(height, width, 3),
            )
        if surface_points is not None:
            prior_rng = np.random.default_rng(step + 50_000)
            prior_weight = float(config.get("surface_prior_weight", 0.015))
            loss = loss + prior_weight * _density_prior_loss(
                model, surface_points, device, prior_rng
            )
        if (
            surface_far_prior_weight > 0
            and far_mask_active
            and far_surface_mask is not None
        ):
            far_opac = torch.sigmoid(
                model.opacity_logits[far_surface_mask]
            )
            loss = loss + surface_far_prior_weight * torch.mean(
                torch.clamp(far_opac - surface_far_opacity_cap, min=0.0) ** 2
            )
        scale_loss = float(config.get("scale_prior_weight", 0.01)) * torch.mean(
            (model.log_scales - torch.log(torch.clamp(scale_targets, min=1e-4)))
            ** 2
        )
        loss = loss + scale_loss
        if projected_scale_prior_weight > 0 and model.positions.numel() > 0:
            origin_t = torch.as_tensor(
                c2w[:3, 3], dtype=torch.float32, device=device
            )
            rotation_t = torch.as_tensor(
                c2w[:3, :3], dtype=torch.float32, device=device
            )
            cam_z = ((model.positions - origin_t[None, :]) @ rotation_t)[:, 2]
            front = cam_z > max(near * 0.8, 1e-4)
            if torch.any(front):
                safe_z = torch.clamp(cam_z[front], min=max(near * 0.8, 1e-4))
                target_scale = torch.clamp(
                    safe_z * projected_scale_target_px / fx,
                    min=max(init_scale * 0.25, 1e-4),
                    max=max(scale_upper * 0.9, init_scale),
                )
                loss = loss + projected_scale_prior_weight * torch.mean(
                    (model.log_scales[front] - torch.log(target_scale)) ** 2
                )
        if float(config.get("opacity_concentration_weight", 0.01)) > 0:
            opac = torch.sigmoid(model.opacity_logits)
            loss = loss + float(config.get("opacity_concentration_weight", 0.01)) * torch.mean(
                opac * (1.0 - opac)
            )
        if images_low_t is not None and not train_original:
            low_camera = _lowres_camera(camera, global_res)
            low_pixels = torch.meshgrid(
                torch.arange(global_res, dtype=torch.float32, device=device),
                torch.arange(global_res, dtype=torch.float32, device=device),
                indexing="xy",
            )
            low_pixels = torch.stack(
                [low_pixels[0].reshape(-1), low_pixels[1].reshape(-1)],
                dim=1,
            )
            low_rendered = _render_pixels(
                model,
                low_camera,
                low_pixels,
                global_res,
                global_res,
                bg_tensor,
                near=near,
                far=far,
                batch_size=render_batch,
                disk_cutoff=disk_cutoff,
                falloff_power=falloff_power,
                tile_size=render_tile,
                dense_renderer=dense_renderer,
                auto_dense_threshold=auto_dense_threshold,
                auto_dense_pixels=auto_dense_pixels,
                tile_pair_budget=tile_pair_budget,
                tile_max_splats=tile_max_splats,
                tile_work_budget=tile_work_budget,
                use_cuda_tile=use_cuda_tile,
            )
            low_target = images_low_t[
                int(camera_choices[step])
            ].reshape(-1, 3)
            loss = loss + global_res_weight * torch.mean(
                (low_rendered - low_target) ** 2
            )
        if float(config.get("background_residual_weight", 0.0)) > 0:
            bg_count = min(int(config.get("background_residual_pixels", 256)), patch)
            bg_rng = np.random.default_rng(step + 101)
            bg_ys = bg_rng.integers(0, height, size=bg_count)
            bg_xs = bg_rng.integers(0, width, size=bg_count)
            bg_pixels = torch.as_tensor(
                np.stack([bg_xs, bg_ys], axis=-1),
                dtype=torch.float32,
                device=device,
            )
            bg_rendered = _render_pixels(
                model,
                camera,
                bg_pixels,
                width,
                height,
                bg_tensor,
                near=near,
                far=far,
                batch_size=render_batch,
                disk_cutoff=disk_cutoff,
                falloff_power=falloff_power,
                tile_size=render_tile,
                dense_renderer=dense_renderer,
                auto_dense_threshold=auto_dense_threshold,
                auto_dense_pixels=auto_dense_pixels,
                tile_pair_budget=tile_pair_budget,
                tile_max_splats=tile_max_splats,
                tile_work_budget=tile_work_budget,
                use_cuda_tile=use_cuda_tile,
            )
            bg_target = torch.as_tensor(
                images[int(camera_choices[step])][bg_ys, bg_xs],
                dtype=torch.float32,
                device=device,
            )
            loss = loss + float(config.get("background_residual_weight", 0.15)) * torch.mean(
                (bg_rendered - bg_target) ** 2
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        step_grad = (
            model.positions.grad.abs() if model.positions.grad is not None else None
        )
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            model.log_scales.clamp_(
                math.log(1e-4), math.log(scale_upper)
            )
            if (
                surface_far_prior_weight > 0
                and surface_far_band is not None
                and step % surface_far_prior_every == 0
            ):
                positions_np = model.positions.detach().cpu().numpy()
                far_surface_mask = torch.as_tensor(
                    _surface_far_mask(
                        positions_np, surface_points, surface_far_band
                    ),
                    dtype=torch.bool,
                    device=device,
                )
                far_mask_active = bool(far_surface_mask.any().item())
            if (
                surface_far_opacity_clamp > 0
                and far_mask_active
                and far_surface_mask is not None
            ):
                far_opac = torch.sigmoid(
                    model.opacity_logits[far_surface_mask]
                )
                over_cap = far_opac > surface_far_opacity_clamp
                if torch.any(over_cap):
                    cap_logit = math.log(
                        surface_far_opacity_clamp
                        / max(1.0 - surface_far_opacity_clamp, 1e-4)
                    )
                    model.opacity_logits[far_surface_mask][over_cap] = (
                        cap_logit
                    )
            if step >= reset_after and step % reset_every == 0:
                _opacity_reset(
                    model,
                    camera,
                    surface_points,
                    device,
                    reset_opacity,
                    near,
                    far,
                    exclude_mask=(
                        far_surface_mask
                        if (
                            surface_far_prior_weight > 0
                            or surface_far_opacity_clamp > 0
                        )
                        else None
                    ),
                )
                log(
                    f"[train] step {step} 重置可见棱面透明度到 {reset_opacity:.2f}"
                )
            if adaptive_budget and step % budget_check_every == 0:
                (
                    current_cap,
                    tile_pair_budget,
                    tile_work_budget,
                    tile_max_splats,
                    vram_pressure,
                    vram_free_mb,
                    vram_total_mb,
                ) = _vram_budget_update(
                    config,
                    device,
                    scene_cap,
                    current_cap,
                    base_tile_pair_budget,
                    base_tile_work_budget,
                    base_tile_max_splats,
                )
                pruned = False
                if (
                    vram_pressure >= 1
                    and int(model.positions.shape[0]) > current_cap
                ):
                    before = int(model.positions.shape[0])
                    refresh_hole_protection(
                        recent_train_camera_indices(
                            step,
                            hole_fill_every * 2,
                            max(
                                1,
                                int(config.get("hole_refresh_cameras", 8)),
                            ),
                        )
                    )
                    _prune_splats_to_budget(
                        model,
                        current_cap,
                        device,
                        min_opacity=float(
                            config.get("vram_prune_opacity", 0.12)
                        ),
                    )
                    pruned = int(model.positions.shape[0]) < before
                if pruned:
                    optimizer = build_optimizer()
                    grad_accum = torch.zeros(
                        len(model.positions), dtype=torch.float32, device=device
                    )
                    rebuild_scale_targets()
                    log(
                        f"[vram] step {step} 压力 {vram_pressure}，"
                        f"剪枝到 {len(model.positions)} 棱面"
                    )
                elif vram_pressure > 0:
                    log(
                        f"[vram] step {step} 压力 {vram_pressure}，"
                        f"空闲 {vram_free_mb:.0f}/{vram_total_mb:.0f} MiB，"
                        f"上限 {current_cap}，tile 预算 {tile_pair_budget}"
                    )
            if (
                densify_enabled
                and step >= densify_after
                and step % densify_every == 0
            ):
                accumulate_gradients(step_grad)
                densified = _densify_and_prune(
                    model,
                    camera,
                    surface_points,
                    device,
                    grad_accum,
                    current_cap,
                    prune_opacity,
                    jitter_scale=max(
                        float(np.median(point_scales)) * 0.18,
                        1e-4,
                    ),
                    near=near,
                    far=far,
                    rng=np.random.default_rng(step),
                )
                if densified:
                    optimizer = build_optimizer()
                    grad_accum = torch.zeros(
                        len(model.positions), dtype=torch.float32, device=device
                    )
                    rebuild_scale_targets()
                    log(
                        f"[train] 密化后 {len(model.positions)} 棱面，"
                        f"透明度 {float(torch.sigmoid(model.opacity_logits).mean()):.3f}"
                    )
                    if (
                        vram_pressure == 0
                        and int(model.positions.shape[0]) >= current_cap
                        and scene_cap < max_splats_absolute
                    ):
                        scene_cap = min(
                            max_splats_absolute,
                            scene_cap + scene_cap_boost,
                        )
                        current_cap = scene_cap
                        log(
                            f"[vram] step {step} 密化达到上限，"
                            f"棱面上限提升到 {scene_cap}"
                        )
            if (
                prune_low_opacity
                and step > 0
                and step % prune_opacity_every == 0
                and int(model.positions.shape[0]) > 1000
            ):
                opac = torch.sigmoid(model.opacity_logits)
                protected = _model_hole_protected(model)
                keep = (opac >= prune_opacity_threshold) | protected
                keep_count = int(keep.sum().item())
                min_keep = int(
                    model.positions.shape[0] * min(prune_opacity_keep_ratio, 0.95)
                )
                if keep_count < min_keep:
                    unprotected = torch.nonzero(~keep).reshape(-1)
                    if unprotected.numel() > 0:
                        need = int(min_keep - keep_count)
                        order = torch.argsort(
                            opac[unprotected], descending=True
                        )[:need]
                        keep[unprotected[order]] = True
                    keep_count = int(keep.sum().item())
                if keep_count < int(model.positions.shape[0]):
                    keep_sorted, _ = torch.sort(
                        torch.nonzero(keep).reshape(-1)
                    )
                    model.positions = torch.nn.Parameter(
                        model.positions.detach()[keep_sorted]
                    )
                    model.log_scales = torch.nn.Parameter(
                        model.log_scales.detach()[keep_sorted]
                    )
                    model.opacity_logits = torch.nn.Parameter(
                        model.opacity_logits.detach()[keep_sorted]
                    )
                    model.color_logits = torch.nn.Parameter(
                        model.color_logits.detach()[keep_sorted]
                    )
                    model.hole_protected = protected[keep_sorted]
                    optimizer = build_optimizer()
                    grad_accum = torch.zeros(
                        len(model.positions), dtype=torch.float32, device=device
                    )
                    rebuild_scale_targets()
                    log(
                        f"[train] step {step} 清除低透明度雾点 "
                        f"{keep_count} 个棱面"
                    )
            if (
                far_prune_enabled
                and surface_far_band is not None
                and step >= far_prune_after
                and step % far_prune_every == 0
            ):
                removed = _prune_far_surface_splats(
                    model,
                    surface_points,
                    surface_far_band,
                    device,
                    opacity_threshold=far_prune_opacity,
                    max_ratio=far_prune_max_ratio,
                    keep_min_fraction=far_prune_keep_min_fraction,
                )
                if removed > 0:
                    optimizer = build_optimizer()
                    grad_accum = torch.zeros(
                        len(model.positions), dtype=torch.float32, device=device
                    )
                    rebuild_scale_targets()
                    log(
                        f"[train] step {step} 剪除远端雾面 "
                        f"{removed} 个棱面，当前 {len(model.positions)}"
                    )
            if (
                split_large_enabled
                and step >= split_after
                and step % split_every == 0
            ):
                split_count, split_changed = _split_large_splats(
                    model,
                    camera,
                    device,
                    current_cap,
                    near,
                    far,
                    np.random.default_rng(step + 7),
                    split_scale_factor=split_scale_factor,
                    min_split_scale_px=split_min_scale_px,
                    min_split_screen_fraction=split_min_screen_fraction,
                    max_splits=split_max_per_cycle,
                    children_per_split=split_children,
                    shrink_factor=split_shrink_factor,
                )
                if split_changed:
                    optimizer = build_optimizer()
                    grad_accum = torch.zeros(
                        len(model.positions), dtype=torch.float32, device=device
                    )
                    rebuild_scale_targets()
                    log(
                        f"[train] step {step} 拆分/收缩大棱面 "
                        f"({split_count} 个拆分)，当前 {len(model.positions)} 棱面"
                    )
            if (
                hole_fill_enabled
                and step >= hole_fill_after
                and step % hole_fill_every == 0
                and step % densify_every != 0
            ):
                pruned_for_holes = False
                hole_split_changed = False
                if (
                    int(model.positions.shape[0])
                    >= current_cap
                    - int(config.get("hole_recycle_trigger_gap", 1000))
                ):
                    before = int(model.positions.shape[0])
                    recycle_opacity = float(
                        config.get("hole_recycle_opacity", 0.08)
                    )
                    recycle_target = max(
                        int(
                            current_cap
                            - max(
                                int(
                                    config.get(
                                        "hole_recycle_headroom", 5000
                                    )
                                ),
                                200,
                            )
                        ),
                        1000,
                    )
                    refresh_hole_protection(
                        recent_train_camera_indices(
                            step,
                            hole_fill_every * 2,
                            max(
                                1,
                                int(config.get("hole_refresh_cameras", 8)),
                            ),
                        )
                    )
                    _prune_splats_to_budget(
                        model,
                        recycle_target,
                        device,
                        min_opacity=recycle_opacity,
                    )
                    pruned_for_holes = (
                        int(model.positions.shape[0]) < before
                    )
                    if pruned_for_holes:
                        optimizer = build_optimizer()
                        grad_accum = torch.zeros(
                            len(model.positions),
                            dtype=torch.float32,
                            device=device,
                        )
                        rebuild_scale_targets()
                        log(
                            f"[train] step {step} 空洞回收低透明度棱面 "
                            f"{before - len(model.positions)} 个，"
                            f"当前 {len(model.positions)}"
                        )
                scene_hole_boost = (
                    hole_fill_add_scene_boost
                    if vram_pressure == 0
                    and last_raw_holes >= scene_demand_hole_min
                    else 0
                )
                headroom = max(
                    0, current_cap - int(model.positions.shape[0])
                )
                depth_ratios = tuple(
                    float(value)
                    for value in config.get(
                        "hole_fill_depth_ratios",
                        [0.85, 1.0, 1.15],
                    )
                )
                depth_sweep_count = max(
                    0, int(config.get("hole_fill_depth_sweep_count", 1))
                )
                demand_budget = (
                    min(
                        int(config.get("hole_fill_max_per_cycle", 40000)),
                        max(0, last_raw_holes)
                        * max(
                            1, len(depth_ratios) + depth_sweep_count
                        ),
                    )
                    if last_raw_holes > 0
                    else 0
                )
                hole_budget = min(
                    max(hole_fill_add + scene_hole_boost, demand_budget),
                    headroom,
                )
                filled, raw_holes, hole_mask = _hole_fill_splats(
                    model,
                    camera,
                    images[int(camera_choices[step])],
                    background,
                    near,
                    far,
                    device,
                    rng,
                    hole_budget,
                    low_res=hole_fill_low_res,
                    scale_ceiling=scale_upper,
                    tile_pair_budget=tile_pair_budget,
                    tile_max_splats=tile_max_splats,
                    tile_work_budget=tile_work_budget,
                    use_cuda_tile=use_cuda_tile,
                    scale_px_factor=float(
                        config.get("hole_fill_scale_px_factor", 9.0)
                    ),
                    median_scale_factor=float(
                        config.get("hole_fill_median_scale_factor", 1.4)
                    ),
                    scale_ceiling_factor=float(
                        config.get("hole_fill_scale_ceiling_factor", 10.0)
                    ),
                    max_scale_screen_fraction=float(
                        config.get("hole_fill_max_screen_fraction", 0.12)
                    ),
                    opacity_center=float(
                        config.get("hole_fill_opacity", 0.5)
                    ),
                    depth_ratios=depth_ratios,
                    depth_quantile=float(
                        config.get("hole_fill_depth_quantile", 0.3)
                    ),
                    low_scale_px=float(
                        config.get("hole_fill_low_scale_px", 6.0)
                    ),
                    depth_sweep_count=depth_sweep_count,
                    depth_sweep_opacity=float(
                        config.get(
                            "hole_fill_depth_sweep_opacity", 0.35
                        )
                    ),
                    disk_cutoff=disk_cutoff,
                    falloff_power=falloff_power,
                    scene_points=surface_points,
                    bg_tol=float(
                        config.get("hole_fill_bg_tol", 0.14)
                    ),
                    bad_fit_tol=float(
                        config.get("hole_fill_bad_fit_tol", 0.16)
                    ),
                    hole_dilate_px=int(
                        config.get("hole_fill_hole_dilate_px", 2)
                    ),
                    neighbor_grid=int(
                        config.get("hole_fill_neighbor_grid", 1)
                    ),
                    neighbor_step_px=float(
                        config.get("hole_fill_neighbor_step_px", 2.0)
                    ),
                    bg_depth_factor=float(
                        config.get("hole_fill_bg_depth_factor", 0.97)
                    ),
                    depth_hint_radius_factor=float(
                        config.get("hole_fill_depth_hint_radius", 0.14)
                    ),
                    scene_depth_hint_radius_factor=float(
                        config.get(
                            "hole_fill_scene_depth_hint_radius", 0.3
                        )
                    ),
                    z_min_opacity=float(
                        config.get("hole_fill_z_min_opacity", 0.2)
                    ),
                    targets_per_hole=int(
                            config.get("hole_fill_targets_per_hole", 1)
                    ),
                    include_bad_fit=bool(
                        config.get("hole_fill_include_bad_fit", True)
                    ),
                    depth_sweep_min=float(
                        config.get("hole_fill_depth_sweep_min", 0.7)
                    ),
                    depth_sweep_max=float(
                        config.get("hole_fill_depth_sweep_max", 1.4)
                    ),
                    protection_cycles=int(
                        config.get("hole_fill_protection_cycles", 3)
                    ),
                    area_scale_max=float(
                        config.get("hole_fill_area_scale_max", 3.0)
                    ),
                    cluster_fill=bool(
                        config.get("hole_fill_cluster_enabled", True)
                    ),
                    cluster_max_sigma_px=float(
                        config.get("hole_fill_cluster_max_px", 8.0)
                    ),
                    cluster_min_sigma_px=float(
                        config.get("hole_fill_cluster_min_px", 3.0)
                    ),
                    cluster_cover_factor=float(
                        config.get("hole_fill_cluster_cover", 1.5)
                    ),
                    cluster_children=int(
                        config.get("hole_fill_cluster_children", 2)
                    ),
                    cluster_scale_factor=float(
                        config.get(
                            "hole_fill_cluster_scale_factor", 0.75
                        )
                    ),
                )
                if filled:
                    optimizer = build_optimizer()
                    grad_accum = torch.zeros(
                        len(model.positions), dtype=torch.float32, device=device
                    )
                    rebuild_scale_targets()
                    log(
                        f"[train] step {step} 补洞后 {len(model.positions)} 棱面"
                    )
                if (
                    vram_pressure == 0
                    and raw_holes >= scene_demand_hole_min
                    and scene_cap < max_splats_absolute
                ):
                    scene_cap = min(
                        max_splats_absolute,
                        scene_cap + scene_cap_boost,
                    )
                    current_cap = scene_cap
                    log(
                        f"[vram] step {step} 场景仍有 {raw_holes} 个空洞像素，"
                        f"棱面上限提升到 {scene_cap}"
                    )
                if (
                    raw_holes >= scene_demand_hole_min
                    and hole_mask is not None
                    and int(model.positions.shape[0]) >= current_cap - 1000
                ):
                    hole_split_count, hole_split_changed = (
                        _split_splats_into_holes(
                            model,
                            camera,
                            device,
                            near,
                            far,
                            hole_mask,
                            low_res=hole_fill_low_res,
                            max_splits=int(
                                config.get(
                                    "hole_split_max_per_cycle", 4000
                                )
                            ),
                            max_splats=current_cap,
                            split_scale_factor=float(
                                config.get(
                                    "hole_split_scale_factor", 0.5
                                )
                            ),
                            opacity_floor=float(
                                config.get("hole_split_opacity", 0.3)
                            ),
                            children_per_split=int(
                                config.get("hole_split_children", 8)
                            ),
                            min_split_px=float(
                                config.get("hole_split_min_px", 1.2)
                            ),
                            spread_px=float(
                                config.get("hole_split_spread_px", 1.5)
                            ),
                            target_image=images[int(camera_choices[step])],
                            protection_cycles=int(
                                config.get(
                                    "hole_fill_protection_cycles", 3
                                )
                            ),
                        )
                    )
                    if hole_split_changed:
                        optimizer = build_optimizer()
                        grad_accum = torch.zeros(
                            len(model.positions),
                            dtype=torch.float32,
                            device=device,
                        )
                        rebuild_scale_targets()
                        log(
                            f"[train] step {step} 空洞拆入 "
                            f"{hole_split_count} 组棱面，"
                            f"当前 {len(model.positions)} 棱面"
                        )
                if (
                    (filled or hole_split_changed)
                    and hole_fill_refine_steps > 0
                    and hole_budget >= hole_fill_refine_min_budget
                ):
                    torch.set_grad_enabled(True)
                    refine_hole_steps(
                        camera,
                        images[int(camera_choices[step])],
                        near,
                        far,
                        hole_fill_refine_steps,
                        hole_fill_refine_resolution,
                        hole_fill_refine_weight,
                    )
                    log(
                        f"[train] step {step} 补洞微调 "
                        f"{hole_fill_refine_steps} 步"
                    )
                if filled or hole_split_changed:
                    refresh_hole_protection(
                        recent_train_camera_indices(
                            step,
                            hole_fill_every * 2,
                            max(
                                1,
                                int(config.get("hole_refresh_cameras", 8)),
                            ),
                        )
                    )
                last_raw_holes = raw_holes

        log(f"[train] step {step + 1}/{steps} 用时 {time.time() - step_start:.2f}s")
        if step % training_progress_every == 0:
            emit_training_status(
                project_dir,
                task_id,
                step,
                steps,
                float(loss.detach().cpu()),
            )
        if step % eval_every == 0 or step == steps - 1:
            eval_index = (step // eval_every) % len(eval_cameras)
            eval_camera = eval_cameras[eval_index]
            with torch.no_grad():
                rendered_view = render_splats_view(
                    model,
                    eval_camera,
                    background,
                    scene_points=surface_points,
                    batch_size=render_batch,
                    disk_cutoff=disk_cutoff,
                    falloff_power=falloff_power,
                    tile_size=render_tile,
                    dense_renderer=dense_renderer,
                    auto_dense_threshold=auto_dense_threshold,
                    auto_dense_pixels=auto_dense_pixels,
                    tile_pair_budget=tile_pair_budget,
                    tile_max_splats=tile_max_splats,
                    tile_work_budget=tile_work_budget,
                    use_cuda_tile=use_cuda_tile,
                )
            target_view = images[len(cameras) - len(eval_cameras) + eval_index]
            metrics = compute_metrics(rendered_view, target_view)
            vram_mb = 0
            if torch.cuda.is_available():
                try:
                    vram_mb = int(
                        torch.cuda.memory_reserved(device) / (1024 * 1024)
                    )
                except Exception:
                    vram_mb = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            metrics.update(
                {
                    "step": step,
                    "totalSteps": steps,
                    "vramMB": vram_mb,
                    "vramFreeMB": int(vram_free_mb),
                    "vramPressure": vram_pressure,
                    "splatLimit": current_cap,
                    "splatCount": int(model.positions.shape[0]),
                    "elapsed": round(time.time() - start_wall, 2),
                }
            )
            history.append(metrics)
            log(
                f"[train] step {step} loss {metrics['loss']:.5f} "
                f"psnr {metrics['psnr']:.2f}"
            )
            emit_status(
                project_dir,
                {
                    "taskId": task_id,
                    "status": "running",
                    "stage": "training",
                    "progress": round(0.12 + 0.78 * ((step + 1) / steps), 4),
                    "step": step,
                    "totalSteps": steps,
                    "loss": metrics["loss"],
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "vramMB": vram_mb,
                    "vramFreeMB": int(vram_free_mb),
                    "vramPressure": vram_pressure,
                    "splatLimit": current_cap,
                    "splatCount": int(model.positions.shape[0]),
                    "metrics": history,
                },
            )
        status_after(project_dir, task_id)

    if bool(config.get("final_cleanup", False)) and steps > 0:
        final_depth_ratios = tuple(
            float(value)
            for value in config.get(
                "hole_fill_depth_ratios",
                [0.85, 1.0, 1.15],
            )
        )
        final_rounds = max(1, int(config.get("final_hole_fill_rounds", 3)))
        final_hole_refine_steps = max(
            0, int(config.get("final_hole_refine_steps", 40))
        )
        final_hole_refine_resolution = max(
            1, int(config.get("final_hole_refine_resolution", 96))
        )
        final_hole_refine_weight = float(
            config.get("final_hole_refine_weight", 1.5)
        )
        final_camera_count = max(
            1,
            min(
                int(config.get("final_hole_fill_cameras", 8)),
                len(cameras),
            ),
        )
        if len(cameras) <= final_camera_count:
            final_camera_idx = list(range(len(cameras)))
        else:
            final_camera_idx = np.linspace(
                0, len(cameras) - 1, final_camera_count
            ).astype(int).tolist()
        final_hole_fill_overshoot = max(
            0,
            int(config.get("final_hole_fill_overshoot", 60000)),
        )
        final_fill_cap = min(
            max_splats_absolute + final_hole_fill_overshoot,
            current_cap + final_hole_fill_overshoot,
        )
        final_raw_holes = 0
        final_hole_mask = None
        for final_round in range(final_rounds):
            round_changed = False
            round_remaining_holes: list[int] = []
            for cam_idx in final_camera_idx:
                final_camera = cameras[cam_idx]
                final_c2w = np.asarray(
                    final_camera["c2w"], dtype=np.float32
                )
                final_near, final_far = (
                    ray_bounds_from_points(
                        final_c2w[:3, 3],
                        surface_points,
                        rotation=final_c2w[:3, :3],
                    )
                    if surface_points is not None
                    and len(surface_points)
                    else (0.3, 7.6)
                )
                final_pruned = False
                if (
                    int(model.positions.shape[0])
                    >= final_fill_cap
                    - int(config.get("hole_recycle_trigger_gap", 1000))
                ):
                    final_before = int(model.positions.shape[0])
                    refresh_hole_protection(list(final_camera_idx))
                    _prune_splats_to_budget(
                        model,
                        max(
                            int(
                                final_fill_cap
                                - max(
                                    int(
                                        config.get(
                                            "final_hole_recycle_headroom",
                                            12000,
                                        )
                                    ),
                                    200,
                                )
                            ),
                            1000,
                        ),
                        device,
                        min_opacity=float(
                            config.get("hole_recycle_opacity", 0.08)
                        ),
                    )
                    final_pruned = (
                        int(model.positions.shape[0]) < final_before
                    )
                    if final_pruned:
                        rebuild_scale_targets()
                final_fill_budget = min(
                    int(config.get("hole_fill_max_per_cycle", 40000)),
                    max(
                        0,
                        final_fill_cap - int(model.positions.shape[0]),
                    ),
                )
                final_filled, final_raw_holes, final_hole_mask = (
                    _hole_fill_splats(
                        model,
                        final_camera,
                        images[cam_idx],
                        background,
                        final_near,
                        final_far,
                        device,
                        rng,
                        final_fill_budget,
                        low_res=hole_fill_low_res,
                        scale_ceiling=scale_upper,
                        tile_pair_budget=tile_pair_budget,
                        tile_max_splats=tile_max_splats,
                        tile_work_budget=tile_work_budget,
                        use_cuda_tile=use_cuda_tile,
                        scale_px_factor=float(
                            config.get("hole_fill_scale_px_factor", 9.0)
                        ),
                        median_scale_factor=float(
                            config.get(
                                "hole_fill_median_scale_factor", 1.4
                            )
                        ),
                        scale_ceiling_factor=float(
                            config.get(
                                "hole_fill_scale_ceiling_factor", 10.0
                            )
                        ),
                        max_scale_screen_fraction=float(
                            config.get(
                                "hole_fill_max_screen_fraction", 0.12
                            )
                        ),
                        opacity_center=float(
                            config.get("hole_fill_opacity", 0.5)
                        ),
                        depth_ratios=final_depth_ratios,
                        depth_quantile=float(
                            config.get("hole_fill_depth_quantile", 0.3)
                        ),
                        low_scale_px=float(
                            config.get("hole_fill_low_scale_px", 6.0)
                        ),
                        depth_sweep_count=max(
                            0,
                            int(
                                config.get(
                                    "hole_fill_depth_sweep_count", 1
                                )
                            ),
                        ),
                        depth_sweep_opacity=float(
                            config.get(
                                "hole_fill_depth_sweep_opacity", 0.35
                            )
                        ),
                        disk_cutoff=disk_cutoff,
                        falloff_power=falloff_power,
                        scene_points=surface_points,
                        bg_tol=float(
                            config.get("hole_fill_bg_tol", 0.14)
                        ),
                        bad_fit_tol=float(
                            config.get("hole_fill_bad_fit_tol", 0.16)
                        ),
                        hole_dilate_px=int(
                            config.get("hole_fill_hole_dilate_px", 2)
                        ),
                        neighbor_grid=int(
                            config.get("hole_fill_neighbor_grid", 1)
                        ),
                        neighbor_step_px=float(
                            config.get(
                                "hole_fill_neighbor_step_px", 2.0
                            )
                        ),
                        bg_depth_factor=float(
                            config.get("hole_fill_bg_depth_factor", 0.97)
                        ),
                        depth_hint_radius_factor=float(
                            config.get("hole_fill_depth_hint_radius", 0.14)
                        ),
                        scene_depth_hint_radius_factor=float(
                            config.get(
                                "hole_fill_scene_depth_hint_radius", 0.3
                            )
                        ),
                        z_min_opacity=float(
                            config.get("hole_fill_z_min_opacity", 0.2)
                        ),
                        targets_per_hole=int(
                            config.get("hole_fill_targets_per_hole", 1)
                        ),
                        include_bad_fit=bool(
                            config.get("hole_fill_include_bad_fit", True)
                        ),
                        depth_sweep_min=float(
                            config.get("hole_fill_depth_sweep_min", 0.7)
                        ),
                        depth_sweep_max=float(
                            config.get("hole_fill_depth_sweep_max", 1.4)
                        ),
                        protection_cycles=int(
                            config.get(
                                "hole_fill_protection_cycles", 3
                            )
                        ),
                        area_scale_max=float(
                            config.get("hole_fill_area_scale_max", 3.0)
                        ),
                        cluster_fill=bool(
                            config.get("hole_fill_cluster_enabled", True)
                        ),
                        cluster_max_sigma_px=float(
                            config.get("hole_fill_cluster_max_px", 8.0)
                        ),
                        cluster_min_sigma_px=float(
                            config.get("hole_fill_cluster_min_px", 3.0)
                        ),
                        cluster_cover_factor=float(
                            config.get("hole_fill_cluster_cover", 1.5)
                        ),
                        cluster_children=int(
                            config.get("hole_fill_cluster_children", 2)
                        ),
                        cluster_scale_factor=float(
                            config.get(
                                "hole_fill_cluster_scale_factor", 0.75
                            )
                        ),
                    )
                )
                final_split_changed = False
                final_split_count = 0
                if (
                    final_raw_holes >= scene_demand_hole_min
                    and final_hole_mask is not None
                    and int(model.positions.shape[0])
                    >= final_fill_cap - 1000
                ):
                    final_split_count, final_split_changed = (
                        _split_splats_into_holes(
                            model,
                            final_camera,
                            device,
                            final_near,
                            final_far,
                            final_hole_mask,
                            low_res=hole_fill_low_res,
                            max_splits=int(
                                config.get(
                                    "hole_split_max_per_cycle", 4000
                                )
                            ),
                            max_splats=final_fill_cap,
                            split_scale_factor=float(
                                config.get(
                                    "hole_split_scale_factor", 0.5
                                )
                            ),
                            opacity_floor=float(
                                config.get("hole_split_opacity", 0.3)
                            ),
                            children_per_split=int(
                                config.get("hole_split_children", 8)
                            ),
                            min_split_px=float(
                                config.get("hole_split_min_px", 1.2)
                            ),
                            spread_px=float(
                                config.get("hole_split_spread_px", 1.5)
                            ),
                            target_image=images[cam_idx],
                            protection_cycles=int(
                                config.get(
                                    "hole_fill_protection_cycles", 3
                                )
                            ),
                        )
                    )
                if final_filled or final_split_changed:
                    round_changed = True
                    if final_hole_refine_steps > 0:
                        rebuild_scale_targets()
                        refine_hole_steps(
                            final_camera,
                            images[cam_idx],
                            final_near,
                            final_far,
                            final_hole_refine_steps,
                            final_hole_refine_resolution,
                            final_hole_refine_weight,
                        )
                    refresh_hole_protection(list(final_camera_idx))
                    final_raw_holes = _count_lowres_holes(
                        model,
                        final_camera,
                        images[cam_idx],
                        background,
                        final_near,
                        final_far,
                        device,
                        low_res=hole_fill_low_res,
                        bg_tol=float(
                            config.get("hole_fill_bg_tol", 0.14)
                        ),
                        bad_fit_tol=float(
                            config.get("hole_fill_bad_fit_tol", 0.16)
                        ),
                        disk_cutoff=disk_cutoff,
                        falloff_power=falloff_power,
                        tile_pair_budget=tile_pair_budget,
                        tile_max_splats=tile_max_splats,
                        tile_work_budget=tile_work_budget,
                        use_cuda_tile=use_cuda_tile,
                        strict_bg=True,
                        include_bad_fit=True,
                    )[1]
                    log(
                        f"[train] 最终清理相机 {cam_idx} "
                        f"第 {final_round + 1} 轮："
                        f"{final_raw_holes} 个空洞像素，"
                        f"拆入 {final_split_count} 组，"
                        f"当前 {len(model.positions)} 棱面"
                    )
                elif final_pruned:
                    round_changed = True
                    log(
                        f"[train] 最终回收相机 {cam_idx} "
                        f"低透明度棱面后仍有 "
                        f"{final_raw_holes} 个空洞像素"
                    )
                round_remaining_holes.append(int(final_raw_holes))
            if not round_changed:
                break
            if all(
                count < scene_demand_hole_min
                for count in round_remaining_holes
            ):
                break
        if final_raw_holes > 0:
            log(
                f"[train] 最终清理后剩余 {final_raw_holes} 个空洞像素，"
                f"当前 {len(model.positions)} 棱面"
            )
        if int(model.positions.shape[0]) > final_fill_cap:
            refresh_hole_protection(list(final_camera_idx))
            _prune_splats_to_budget(
                model,
                final_fill_cap,
                device,
                min_opacity=float(
                    config.get("hole_recycle_opacity", 0.08)
                ),
            )
            rebuild_scale_targets()
            log(
                f"[train] 最终清理后按补洞上限回收至 "
                f"{len(model.positions)} 棱面"
            )

    if bool(config.get("final_cleanup", False)):
        with torch.no_grad():
            opacity = torch.sigmoid(model.opacity_logits)
            keep = opacity >= float(config.get("final_prune_opacity", 0.02))
            if surface_far_band is not None:
                positions_np = model.positions.detach().cpu().numpy()
                final_far_mask = torch.as_tensor(
                    _surface_far_mask(
                        positions_np, surface_points, surface_far_band
                    ),
                    dtype=torch.bool,
                    device=device,
                )
                keep = keep & (
                    ~(
                        final_far_mask
                        & (
                            opacity
                            < float(config.get("final_far_prune_opacity", 0.1))
                        )
                    )
                )
            protected_np = _hole_coverage_protected_mask(
                model,
                cameras,
                images,
                surface_points,
                device,
                low_res=hole_fill_low_res,
                dilate_px=int(
                    config.get("final_prune_hole_dilate_px", 3)
                ),
                bg_tol=float(config.get("hole_fill_bg_tol", 0.14)),
                bad_fit_tol=float(
                    config.get("hole_fill_bad_fit_tol", 0.16)
                ),
                disk_cutoff=disk_cutoff,
                falloff_power=falloff_power,
                tile_pair_budget=tile_pair_budget,
                tile_max_splats=tile_max_splats,
                tile_work_budget=tile_work_budget,
                use_cuda_tile=use_cuda_tile,
            )
            protected = torch.as_tensor(
                protected_np, dtype=torch.bool, device=device
            )
            protected = protected | _model_hole_protected(model)
            keep = keep | protected
            min_keep = int(
                model.positions.shape[0]
                * min(float(config.get("final_keep_ratio", 0.90)), 0.95)
            )
            if int(keep.sum().item()) < min_keep:
                unprotected = torch.nonzero(~protected).reshape(-1)
                if unprotected.numel() > 0:
                    need = int(min_keep - int(keep.sum().item()))
                    order = torch.argsort(
                        opacity[unprotected], descending=True
                    )
                    keep[unprotected[order[:need]]] = True
            if int(keep.sum().item()) < int(model.positions.shape[0]):
                keep_sorted, _ = torch.sort(
                    torch.nonzero(keep).reshape(-1)
                )
                model.positions = torch.nn.Parameter(
                    model.positions.detach()[keep_sorted]
                )
                model.log_scales = torch.nn.Parameter(
                    model.log_scales.detach()[keep_sorted]
                )
                model.opacity_logits = torch.nn.Parameter(
                    model.opacity_logits.detach()[keep_sorted]
                )
                model.color_logits = torch.nn.Parameter(
                    model.color_logits.detach()[keep_sorted]
                )
                model.hole_protected = protected[keep_sorted]
                log(
                    f"[train] 最终清理后 {len(model.positions)} 棱面"
                )

    if bool(config.get("final_cleanup", False)) and steps > 0:
        diag_camera_count = max(
            1, min(int(config.get("final_hole_fill_cameras", 8)), len(cameras))
        )
        if len(cameras) <= diag_camera_count:
            diag_camera_idx = list(range(len(cameras)))
        else:
            diag_camera_idx = np.linspace(
                0, len(cameras) - 1, diag_camera_count
            ).astype(int).tolist()
        diag_holes: list[int] = []
        for diag_cam_idx in diag_camera_idx:
            diag_camera = cameras[diag_cam_idx]
            diag_c2w = np.asarray(diag_camera["c2w"], dtype=np.float32)
            diag_near, diag_far = (
                ray_bounds_from_points(
                    diag_c2w[:3, 3],
                    surface_points,
                    rotation=diag_c2w[:3, :3],
                )
                if surface_points is not None and len(surface_points)
                else (0.3, 7.6)
            )
            diag_holes.append(
                _count_lowres_holes(
                    model,
                    diag_camera,
                    images[diag_cam_idx],
                    background,
                    diag_near,
                    diag_far,
                    device,
                    low_res=hole_fill_low_res,
                    bg_tol=float(config.get("hole_fill_bg_tol", 0.14)),
                    bad_fit_tol=float(
                        config.get("hole_fill_bad_fit_tol", 0.16)
                    ),
                    disk_cutoff=disk_cutoff,
                    falloff_power=falloff_power,
                    tile_pair_budget=tile_pair_budget,
                    tile_max_splats=tile_max_splats,
                    tile_work_budget=tile_work_budget,
                    use_cuda_tile=use_cuda_tile,
                    strict_bg=True,
                    include_bad_fit=True,
                )[1]
            )
        log(
            f"[train] 最终剪枝后空洞诊断 "
            f"{list(zip(diag_camera_idx, diag_holes))}"
        )

    if (
        bool(config.get("final_cleanup", False))
        and steps > 0
        and len(cameras) > 0
    ):
        post_fill_steps = max(
            0, int(config.get("final_post_fill_steps", 60))
        )
        post_fill_overshoot = max(
            0, int(config.get("final_post_fill_overshoot", 200000))
        )
        post_fill_rounds = max(
            1, int(config.get("final_post_fill_rounds", 6))
        )
        post_fill_refine_resolution = max(
            1, int(config.get("final_hole_refine_resolution", 192))
        )
        post_fill_hole_min = max(
            0, int(config.get("scene_demand_hole_min", 200))
        )
        post_fill_budget = max(
            0,
            int(max_splats_absolute)
            + post_fill_overshoot
            - int(model.positions.shape[0]),
        )
        if (
            post_fill_steps > 0
            and post_fill_budget > 0
            and post_fill_rounds > 0
        ):
            post_depth_ratios = tuple(
                float(value)
                for value in config.get(
                    "hole_fill_depth_ratios",
                    [0.85, 1.0, 1.15],
                )
            )
            post_camera_count = max(
                1,
                min(
                    int(config.get("final_hole_fill_cameras", 8)),
                    len(cameras),
                ),
            )
            if len(cameras) <= post_camera_count:
                post_camera_idx = list(range(len(cameras)))
            else:
                post_camera_idx = np.linspace(
                    0, len(cameras) - 1, post_camera_count
                ).astype(int).tolist()
            post_filled_total = 0
            post_holes_total = 0
            remaining_budget = post_fill_budget
            for post_round in range(post_fill_rounds):
                if remaining_budget <= 0:
                    break
                round_filled = 0
                round_holes: list[int] = []
                cameras_left = len(post_camera_idx)
                for cam_idx in post_camera_idx:
                    if remaining_budget <= 0:
                        break
                    post_camera = cameras[cam_idx]
                    post_c2w = np.asarray(
                        post_camera["c2w"], dtype=np.float32
                    )
                    post_near, post_far = (
                        ray_bounds_from_points(
                            post_c2w[:3, 3],
                            surface_points,
                            rotation=post_c2w[:3, :3],
                        )
                        if surface_points is not None
                        and len(surface_points)
                        else (0.3, 7.6)
                    )
                    share = remaining_budget // cameras_left
                    if share <= 0:
                        break
                    post_budget = min(
                        remaining_budget,
                        share,
                        int(config.get("hole_fill_max_per_cycle", 40000)),
                    )
                    post_before = int(model.positions.shape[0])
                    post_filled, post_holes, _ = _hole_fill_splats(
                        model,
                        post_camera,
                        images[cam_idx],
                        background,
                        post_near,
                        post_far,
                        device,
                        rng,
                        post_budget,
                        low_res=hole_fill_low_res,
                        scale_ceiling=scale_upper,
                        tile_pair_budget=tile_pair_budget,
                        tile_max_splats=tile_max_splats,
                        tile_work_budget=tile_work_budget,
                        use_cuda_tile=use_cuda_tile,
                        scale_px_factor=float(
                            config.get("hole_fill_scale_px_factor", 9.0)
                        ),
                        median_scale_factor=float(
                            config.get(
                                "hole_fill_median_scale_factor", 1.4
                            )
                        ),
                        scale_ceiling_factor=float(
                            config.get(
                                "hole_fill_scale_ceiling_factor", 10.0
                            )
                        ),
                        max_scale_screen_fraction=float(
                            config.get(
                                "hole_fill_max_screen_fraction", 0.12
                            )
                        ),
                        opacity_center=float(
                            config.get("hole_fill_opacity", 0.5)
                        ),
                        depth_ratios=post_depth_ratios,
                        depth_quantile=float(
                            config.get("hole_fill_depth_quantile", 0.3)
                        ),
                        low_scale_px=float(
                            config.get("hole_fill_low_scale_px", 6.0)
                        ),
                        depth_sweep_count=max(
                            0,
                            int(
                                config.get(
                                    "hole_fill_depth_sweep_count", 1
                                )
                            ),
                        ),
                        depth_sweep_opacity=float(
                            config.get(
                                "hole_fill_depth_sweep_opacity", 0.35
                            )
                        ),
                        disk_cutoff=disk_cutoff,
                        falloff_power=falloff_power,
                        scene_points=surface_points,
                        bg_tol=float(
                            config.get("hole_fill_bg_tol", 0.14)
                        ),
                        bad_fit_tol=float(
                            config.get("hole_fill_bad_fit_tol", 0.16)
                        ),
                        hole_dilate_px=int(
                            config.get("hole_fill_hole_dilate_px", 2)
                        ),
                        neighbor_grid=int(
                            config.get("hole_fill_neighbor_grid", 1)
                        ),
                        neighbor_step_px=float(
                            config.get(
                                "hole_fill_neighbor_step_px", 2.0
                            )
                        ),
                        bg_depth_factor=float(
                            config.get("hole_fill_bg_depth_factor", 0.97)
                        ),
                        depth_hint_radius_factor=float(
                            config.get("hole_fill_depth_hint_radius", 0.14)
                        ),
                        scene_depth_hint_radius_factor=float(
                            config.get(
                                "hole_fill_scene_depth_hint_radius", 0.3
                            )
                        ),
                        z_min_opacity=float(
                            config.get("hole_fill_z_min_opacity", 0.2)
                        ),
                        targets_per_hole=int(
                            config.get("hole_fill_targets_per_hole", 1)
                        ),
                        include_bad_fit=bool(
                            config.get("hole_fill_include_bad_fit", True)
                        ),
                        depth_sweep_min=float(
                            config.get("hole_fill_depth_sweep_min", 0.7)
                        ),
                        depth_sweep_max=float(
                            config.get("hole_fill_depth_sweep_max", 1.4)
                        ),
                        protection_cycles=int(
                            config.get(
                                "hole_fill_protection_cycles", 3
                            )
                        ),
                        area_scale_max=float(
                            config.get("hole_fill_area_scale_max", 3.0)
                        ),
                        cluster_fill=bool(
                            config.get("hole_fill_cluster_enabled", True)
                        ),
                        cluster_max_sigma_px=float(
                            config.get("hole_fill_cluster_max_px", 8.0)
                        ),
                        cluster_min_sigma_px=float(
                            config.get("hole_fill_cluster_min_px", 3.0)
                        ),
                        cluster_cover_factor=float(
                            config.get("hole_fill_cluster_cover", 1.5)
                        ),
                        cluster_children=int(
                            config.get("hole_fill_cluster_children", 2)
                        ),
                        cluster_scale_factor=float(
                            config.get(
                                "hole_fill_cluster_scale_factor", 0.75
                            )
                        ),
                    )
                    remaining_budget = max(
                        0,
                        remaining_budget
                        - (
                            int(model.positions.shape[0]) - post_before
                        ),
                    )
                    if post_filled:
                        post_filled_total += 1
                        round_filled += 1
                        torch.set_grad_enabled(True)
                        rebuild_scale_targets()
                        refine_hole_steps(
                            post_camera,
                            images[cam_idx],
                            post_near,
                            post_far,
                            post_fill_steps,
                            post_fill_refine_resolution,
                            float(
                                config.get(
                                    "final_hole_refine_weight", 1.5
                                )
                            ),
                        )
                        post_holes = _count_lowres_holes(
                            model,
                            post_camera,
                            images[cam_idx],
                            background,
                            post_near,
                            post_far,
                            device,
                            low_res=hole_fill_low_res,
                            bg_tol=float(
                                config.get("hole_fill_bg_tol", 0.14)
                            ),
                            bad_fit_tol=float(
                                config.get(
                                    "hole_fill_bad_fit_tol", 0.16
                                )
                            ),
                            disk_cutoff=disk_cutoff,
                            falloff_power=falloff_power,
                            tile_pair_budget=tile_pair_budget,
                            tile_max_splats=tile_max_splats,
                            tile_work_budget=tile_work_budget,
                            use_cuda_tile=use_cuda_tile,
                            strict_bg=True,
                            include_bad_fit=True,
                        )[1]
                    post_holes_total += int(post_holes)
                    round_holes.append(int(post_holes))
                    cameras_left = max(1, cameras_left - 1)
                if round_filled == 0:
                    break
                if all(
                    count < post_fill_hole_min for count in round_holes
                ):
                    break
            if post_filled_total > 0:
                log(
                    f"[train] 最终剪枝后补洞 {post_filled_total} "
                    f"个相机，剩余 {post_holes_total} 个空洞像素，"
                    f"当前 {len(model.positions)} 棱面"
                )

    return {"model": model, "history": history, "engine": "facets"}, history
