"""Differentiable facet splat renderer for the Titan engine (v0.1 baseline)."""

import math
import time

import numpy as np
import torch


def _logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 1e-4, 1.0 - 1e-4)
    return np.log(value / (1.0 - value))


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
    scales = np.clip(neighbor * multiplier, 1e-4, max(global_scale, 1e-3))
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
                origin.cpu().detach().numpy(), scene_points
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
        contribution = alpha[:, :, None] * color[chunk][:, None, :] * transmittance[None, :, None]
        accumulated = accumulated + contribution.sum(dim=0)
        transmittance = transmittance * torch.prod(1.0 - alpha, dim=0)

    rendered = accumulated + transmittance[:, None] * background[None, :]
    return rendered.reshape(output_shape)


def render_splats_view(
    model: SplatFacetField,
    camera: dict,
    background: np.ndarray,
    scene_points: np.ndarray | None = None,
    batch_size: int = 96,
    disk_cutoff: float = 2.2,
    falloff_power: float = 1.0,
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
        )
    return image.cpu().detach().numpy()


def _density_prior_loss(
    model: SplatFacetField,
    surface_points: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
    count: int = 1024,
):
    """Keep facets near the observed surface instead of drifting into haze."""
    if len(surface_points) == 0 or model.positions.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    position_count = model.positions.shape[0]
    position_indices = rng.integers(0, position_count, size=count)
    positions = model.positions[position_indices]
    surface_count = min(len(surface_points), 2048)
    surface = surface_points[
        rng.integers(0, len(surface_points), size=surface_count)
    ]
    target = torch.as_tensor(surface, dtype=torch.float32, device=device)
    diff = positions[:, None, :] - target[None, :, :]
    distance = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-8)
    nearest = torch.min(distance, dim=1).values
    return torch.mean(nearest * nearest)


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
) -> None:
    """Re-raise opacity for visible facets so haze cannot persist."""
    visible = _visible_mask(
        model, camera, surface_points, device, near, far
    )
    if not torch.any(visible):
        return
    with torch.no_grad():
        model.opacity_logits[visible] = torch.as_tensor(
            [math.log(opacity / (1.0 - opacity))],
            dtype=torch.float32,
            device=device,
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
        threshold = float(torch.quantile(strength, 0.72).item()) if torch.any(strength > 0) else 0.0
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
                (torch.log(parent_scale * 0.65)).detach(),
            ],
            dim=0,
        )
        new_opacity = torch.cat(
            [
                model.opacity_logits.detach(),
                math.log(0.9 / (1.0 - 0.9))
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
    progress_every = int(config.get("progress_every", 10))
    learning_rate = float(config.get("learning_rate", 0.02))
    render_batch = int(config.get("render_batch_size", 256))
    disk_cutoff = float(config.get("disk_cutoff", 1.8))
    falloff_power = float(config.get("falloff_power", 1.0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_scale = scene_scale_from_cameras(cameras)

    if sparse_points:
        points = np.asarray(sparse_points, dtype=np.float32)[:, :3]
        colors = np.asarray(sparse_points, dtype=np.float32)[:, 3:6] / 255.0
        use_image_colors = bool(config.get("use_image_colors", True))
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
        opacity=float(config.get("init_opacity", 0.85)),
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

    train_cameras = cameras[:train_count]
    eval_cameras = cameras[train_count:]
    if not eval_cameras:
        eval_cameras = cameras[-1:]
    camera_choices = np.random.default_rng(3).integers(
        0, len(train_cameras), size=steps
    )
    surface_points = (
        np.asarray(sparse_points, dtype=np.float32)[:, :3] if sparse_points else None
    )
    background = SCENE_BACKGROUND
    bg_tensor = torch.as_tensor(background, dtype=torch.float32, device=device)
    global_res = int(config.get("global_residual_resolution", 16))
    global_res_weight = float(config.get("global_residual_weight", 0.6))
    images_low = (
        _downsample_images(images, global_res) if global_res > 0 else None
    )

    log(
        f"[train] TFRF-facets 开始，{len(train_cameras)} 训练视角，"
        f"{len(points)} 棱面，{steps} 步，设备 {device}，"
        f"场景尺度 {scene_scale:.4f}"
    )
    history: list[dict] = []
    start_wall = time.time()
    densify_every = int(config.get("densify_every", 50))
    densify_after = int(config.get("densify_after", 80))
    max_splats = int(config.get("max_splats", 12000))
    prune_opacity = float(config.get("prune_opacity", 0.04))
    reset_after = int(config.get("opacity_reset_after", 140))
    reset_every = int(config.get("opacity_reset_every", 100))
    reset_opacity = float(config.get("opacity_reset_value", 0.92))
    scale_upper = max(
        init_scale * float(config.get("scale_log_clamp", 2.5)), 0.2
    )
    densify_enabled = bool(config.get("densify", True))
    grad_accum = torch.zeros(len(points), dtype=torch.float32, device=device)

    def accumulate_gradients(current: torch.Tensor | None) -> None:
        if current is None:
            return
        if current.dim() > 1:
            current = current.norm(dim=1)
        if current.shape[0] != grad_accum.shape[0]:
            grad_accum.resize_(current.shape[0])
        grad_accum.mul_(0.95).add_(current)

    for step in range(steps):
        camera = train_cameras[int(camera_choices[step])]
        c2w = np.asarray(camera["c2w"], dtype=np.float32)
        width, height = camera["width"], camera["height"]
        k = camera["K"]
        fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
        near, far = ray_bounds_from_points(
            c2w[:3, 3], surface_points
        ) if surface_points is not None and len(surface_points) else (0.3, 7.6)
        rng = np.random.default_rng(step)
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
        )
        target = torch.as_tensor(
            images[int(camera_choices[step])][ys, xs],
            dtype=torch.float32,
            device=device,
        )
        loss = torch.mean((rendered - target) ** 2)
        if surface_points is not None:
            prior_rng = np.random.default_rng(step + 50_000)
            prior_weight = float(config.get("surface_prior_weight", 0.05))
            loss = loss + prior_weight * _density_prior_loss(
                model, surface_points, device, prior_rng
            )
        scale_target_count = min(
            model.positions.shape[0], point_scales.shape[0]
        )
        scale_loss = float(config.get("scale_prior_weight", 0.03)) * torch.mean(
            (
                model.log_scales[:scale_target_count]
                - torch.log(
                    torch.as_tensor(
                        point_scales[:scale_target_count],
                        dtype=torch.float32,
                        device=device,
                    )
                )
            )
            ** 2
        )
        loss = loss + scale_loss
        if float(config.get("opacity_concentration_weight", 0.01)) > 0:
            opac = torch.sigmoid(model.opacity_logits)
            loss = loss + float(config.get("opacity_concentration_weight", 0.01)) * torch.mean(
                opac * (1.0 - opac)
            )
        if images_low is not None:
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
            )
            low_target = torch.as_tensor(
                images_low[int(camera_choices[step])].reshape(-1, 3),
                dtype=torch.float32,
                device=device,
            )
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
            if step >= reset_after and step % reset_every == 0:
                _opacity_reset(
                    model,
                    camera,
                    surface_points,
                    device,
                    reset_opacity,
                    near,
                    far,
                )
                log(
                    f"[train] step {step} 重置可见棱面透明度到 {reset_opacity:.2f}"
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
                    max_splats,
                    prune_opacity,
                    jitter_scale=max(
                        float(np.median(point_scales)) * 0.25,
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
                    log(
                        f"[train] 密化后 {len(model.positions)} 棱面，"
                        f"透明度 {float(torch.sigmoid(model.opacity_logits).mean()):.3f}"
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
                    "progress": round((step + 1) / steps, 4),
                    "step": step,
                    "loss": metrics["loss"],
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "vramMB": vram_mb,
                    "metrics": history,
                },
            )
        elif step % progress_every == 0:
            emit_training_status(
                project_dir,
                task_id,
                step,
                steps,
                float(loss.detach().cpu()),
            )
        status_after(project_dir, task_id)

    return {"model": model, "history": history, "engine": "facets"}, history
