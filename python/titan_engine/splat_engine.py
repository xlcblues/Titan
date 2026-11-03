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


def _prune_splats_to_budget(
    model: SplatFacetField,
    target_count: int,
    device: torch.device,
    min_opacity: float = 0.12,
) -> int:
    """Drop low-opacity facets when VRAM pressure forces a smaller budget."""
    target_count = max(0, int(target_count))
    current = int(model.positions.shape[0])
    if target_count >= current:
        return current
    with torch.no_grad():
        opacity = torch.sigmoid(model.opacity_logits)
        keep = torch.nonzero(opacity >= min_opacity).reshape(-1)
        if keep.numel() > target_count:
            order = torch.argsort(opacity[keep], descending=True)[:target_count]
            keep = keep[order]
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
        keep = opacity >= prune_threshold
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
        keep_indices = torch.nonzero(keep_full).reshape(-1)
        keep_positions = new_positions[keep_indices]
        keep_scales = new_log_scales[keep_indices]
        keep_opacity = new_opacity[keep_indices]
        keep_colors = new_colors[keep_indices]
        model.positions = torch.nn.Parameter(keep_positions)
        model.log_scales = torch.nn.Parameter(keep_scales)
        model.opacity_logits = torch.nn.Parameter(keep_opacity)
        model.color_logits = torch.nn.Parameter(keep_colors)
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
        rotation, origin, fx, _, _, _, _, _ = _camera_geometry(camera, device)
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
        return (split_count, True)


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
    scale_px_factor: float = 1.2,
    median_scale_factor: float = 0.55,
    scale_ceiling_factor: float = 1.4,
    opacity_center: float = 0.3,
    depth_ratios: tuple[float, ...] = (0.90, 1.00, 1.10),
    disk_cutoff: float = 1.8,
    falloff_power: float = 1.0,
) -> tuple[bool, int]:
    """Backproject unresolved pixels into new facets along multiple depths."""
    from PIL import Image

    max_add = max(0, int(max_add))
    if max_add == 0 or model.positions.numel() == 0:
        return (False, 0)
    depth_ratios = tuple(
        float(value) for value in (depth_ratios or (1.0,))
    )
    depth_per_hole = max(1, len(depth_ratios))
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
            rendered.detach().cpu().numpy().reshape(low_height, low_width, 3)
        )
        target_low = np.asarray(
            Image.fromarray(
                np.clip(target_image * 255.0, 0.0, 255.0).astype(np.uint8)
            ).resize((low_width, low_height), Image.LANCZOS),
            dtype=np.float32,
        ) / 255.0
        bg = np.asarray(background, dtype=np.float32)
        near_bg = np.mean(np.abs(render_low - bg), axis=-1) < 0.08
        has_target = np.mean(np.abs(target_low - bg), axis=-1) > 0.06
        bad_fit = np.mean(np.abs(render_low - target_low), axis=-1) > 0.22
        hole_mask = near_bg & has_target
        padded = np.pad(hole_mask, 1, mode="edge")
        dilated = np.zeros_like(hole_mask)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                dilated |= padded[
                    1 + dy : 1 + dy + low_height,
                    1 + dx : 1 + dx + low_width,
                ]
        hole_mask = dilated | (bad_fit & has_target & ~near_bg)
        hole_count = int(hole_mask.sum())
        raw_hole_count = hole_count
        if hole_count == 0:
            return (False, 0)
        hole_count = min(hole_count, max_add // depth_per_hole)
        if hole_count == 0:
            return (False, raw_hole_count)
        ys, xs = np.nonzero(hole_mask)
        chosen = rng.choice(len(ys), size=hole_count, replace=False)
        ys = ys[chosen].astype(np.int64)
        xs = xs[chosen].astype(np.int64)

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
        mz_np = model_z.detach().cpu().numpy()
        mu_np = model_u.detach().cpu().numpy()
        mv_np = model_v.detach().cpu().numpy()
        search_radius = max(2.0, low_width * 0.015)
        base_depths = np.empty(hole_count, dtype=np.float32)
        for hole_index in range(hole_count):
            du = mu_np - xs[hole_index]
            dv = mv_np - ys[hole_index]
            dist2 = du * du + dv * dv
            local = (dist2 < search_radius * search_radius)
            local &= mz_np > near
            local &= mz_np < far
            candidate = np.nonzero(local)[0]
            if candidate.size:
                nearest = candidate[int(np.argmin(dist2[candidate]))]
                base_depths[hole_index] = float(mz_np[nearest])
            else:
                base_depths[hole_index] = z_median
        depth_ratio_values = np.tile(
            np.asarray(depth_ratios, dtype=np.float32),
            hole_count,
        )
        z_values = np.clip(
            np.repeat(base_depths, depth_per_hole) * depth_ratio_values,
            near * 1.02,
            far * 0.98,
        ).astype(np.float32)
        u_values = np.repeat((xs + 0.5).astype(np.float32), depth_per_hole)
        v_values = np.repeat((ys + 0.5).astype(np.float32), depth_per_hole)
        cam_x = (u_values - cx_lr) / fx_lr * z_values
        cam_y = (v_values - cy_lr) / fy_lr * z_values
        new_positions = origin[None, :] + np.stack(
            [cam_x, cam_y, z_values], axis=-1
        ) @ rotation.T
        scales = torch.exp(model.log_scales.detach())
        median_scale = float(scales.median().item())
        pixel_scale = z_values / max(fx_lr, 1e-4)
        new_scale = np.clip(
            np.maximum(
                pixel_scale * max(float(scale_px_factor), 0.5),
                median_scale * max(float(median_scale_factor), 0.2),
            ),
            float(scale_floor),
            min(
                float(scale_ceiling),
                median_scale * max(float(scale_ceiling_factor), 1.0),
            ),
        )
        new_colors = np.repeat(
            target_low[ys, xs],
            depth_per_hole,
            axis=0,
        )
        new_opacity = np.clip(
            float(opacity_center) + 0.1 * rng.uniform(size=len(z_values)),
            0.1,
            0.45,
        )
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
        return (True, raw_hole_count)


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
        int(config.get("training_progress_every", config.get("progress_every", 10))),
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
    scale_targets = torch.as_tensor(
        _scale_targets_for_splats(
            points,
            surface_points,
            init_scale,
            far_scale_factor=float(config.get("far_scale_factor", 0.8)),
        ),
        dtype=torch.float32,
        device=device,
    )

    def rebuild_scale_targets() -> None:
        nonlocal scale_targets, far_surface_mask, far_mask_active
        positions_np = model.positions.detach().cpu().numpy()
        scale_targets = torch.as_tensor(
            _scale_targets_for_splats(
                positions_np,
                surface_points,
                init_scale,
                far_scale_factor=float(config.get("far_scale_factor", 0.8)),
            ),
            dtype=torch.float32,
            device=device,
        )
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
    scale_upper = max(
        init_scale * float(config.get("scale_log_clamp", 4.0)),
        float(config.get("scale_upper_floor", 0.2)),
    )
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
    split_max_per_cycle = int(config.get("split_max_per_cycle", 3000))
    split_children = int(config.get("split_children", 4))
    split_shrink_factor = float(config.get("split_shrink_factor", 0.85))
    densify_enabled = bool(config.get("densify", True))
    hole_fill_enabled = bool(config.get("hole_fill", True))
    hole_fill_after = int(config.get("hole_fill_after", 100))
    hole_fill_every = max(1, int(config.get("hole_fill_every", 80)))
    hole_fill_add = int(config.get("hole_fill_add", 500))
    hole_fill_low_res = int(config.get("hole_fill_resolution", 96))
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
                < float(config.get("hole_bg_tol", 0.10))
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
                keep = opac >= prune_opacity_threshold
                keep_count = int(keep.sum().item())
                min_keep = int(
                    model.positions.shape[0] * min(prune_opacity_keep_ratio, 0.95)
                )
                if keep_count < min_keep:
                    order = torch.argsort(opac, descending=True)[:min_keep]
                    keep = torch.zeros_like(keep)
                    keep[order] = True
                    keep_count = min_keep
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
                scene_hole_boost = (
                    hole_fill_add_scene_boost
                    if vram_pressure == 0
                    and last_raw_holes >= scene_demand_hole_min
                    else 0
                )
                hole_budget = min(
                    hole_fill_add + scene_hole_boost,
                    max(0, current_cap - int(model.positions.shape[0])),
                )
                filled = False
                raw_holes = 0
                if hole_budget > 0:
                    filled, raw_holes = _hole_fill_splats(
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
                            config.get("hole_fill_scale_px_factor", 1.2)
                        ),
                        median_scale_factor=float(
                            config.get("hole_fill_median_scale_factor", 0.55)
                        ),
                        scale_ceiling_factor=float(
                            config.get("hole_fill_scale_ceiling_factor", 1.4)
                        ),
                        opacity_center=float(
                            config.get("hole_fill_opacity", 0.3)
                        ),
                        depth_ratios=tuple(
                            float(value)
                            for value in config.get(
                                "hole_fill_depth_ratios",
                                [0.97, 1.00, 1.03],
                            )
                        ),
                        disk_cutoff=disk_cutoff,
                        falloff_power=falloff_power,
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
                last_raw_holes = raw_holes

        log(f"[train] step {step + 1}/{steps} 用时 {time.time() - step_start:.2f}s")
        if step % training_progress_every == 0:
            emit_training_status(project_dir, task_id, step, steps)
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

    if bool(config.get("final_cleanup", False)):
        with torch.no_grad():
            opacity = torch.sigmoid(model.opacity_logits)
            keep = opacity >= float(config.get("final_prune_opacity", 0.08))
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
                            < float(config.get("final_far_prune_opacity", 0.25))
                        )
                    )
                )
            min_keep = int(
                model.positions.shape[0]
                * min(float(config.get("final_keep_ratio", 0.60)), 0.95)
            )
            if int(keep.sum().item()) < min_keep:
                order = torch.argsort(opacity, descending=True)[:min_keep]
                keep = torch.zeros_like(keep)
                keep[order] = True
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
                log(
                    f"[train] 最终清理后 {len(model.positions)} 棱面"
                )

    return {"model": model, "history": history, "engine": "facets"}, history
