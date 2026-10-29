"""Differentiable facet splat renderer for the Titan engine (v0.1 baseline)."""

import math
import time

import numpy as np
import torch


def _logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 1e-4, 1.0 - 1e-4)
    return np.log(value / (1.0 - value))


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
        self.register_parameter(
            "log_scales",
            torch.nn.Parameter(
                torch.full((count,), math.log(max(scale, 1e-3)), dtype=torch.float32)
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
    near: float = 0.3,
    batch_size: int = 96,
    disk_cutoff: float = 2.2,
):
    device = next(model.parameters()).device
    rotation, origin, fx, fy, cx, cy, width, height = _camera_geometry(
        camera, device
    )
    positions, scale, opacity, color = model()
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
        weight = torch.exp(-squared)
        weight = weight * (squared < 0.5 * disk_cutoff * disk_cutoff)
        alpha = opacity[chunk, None] * weight
        alpha = torch.clamp(alpha, 0.0, 1.0)
        keep = (1.0 - alpha) + 1e-8
        before = torch.cat(
            [
                torch.ones(
                    (1, pixel_count), dtype=torch.float32, device=device
                ),
                torch.cumprod(keep, dim=0)[:-1],
            ],
            dim=0,
        )
        t_before = transmittance[None, :] * before
        contribution = (
            alpha[:, :, None]
            * color[chunk][:, None, :]
            * t_before[:, :, None]
        )
        accumulated = accumulated + contribution.sum(dim=0)
        transmittance = transmittance * torch.prod(keep, dim=0)

    rendered = accumulated + transmittance[:, None] * background[None, :]
    return rendered.reshape(output_shape)


def render_splats_view(
    model: SplatFacetField,
    camera: dict,
    background: np.ndarray,
    batch_size: int = 96,
) -> np.ndarray:
    device = next(model.parameters()).device
    bg = torch.as_tensor(background, dtype=torch.float32, device=device)
    image = _render_pixels(
        model,
        camera,
        None,
        camera["width"],
        camera["height"],
        bg,
        batch_size=batch_size,
    )
    return image.cpu().detach().numpy()


def _density_prior_loss(
    model: SplatFacetField,
    surface_points: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
    count: int = 1024,
):
    positions = model.positions
    position_indices = rng.integers(0, len(positions), size=count)
    positions = positions[position_indices]
    surface = surface_points[
        rng.integers(0, len(surface_points), size=count)
    ]
    target = torch.as_tensor(surface, dtype=torch.float32, device=device)
    return torch.mean((positions - target) ** 2)


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
        log,
        status_after,
    )

    train_count = int(config.get("train_views", 16))
    steps = int(config.get("steps", 600))
    patch = int(config.get("patch_size", 2048))
    eval_every = int(config.get("eval_every", 50))
    learning_rate = float(config.get("learning_rate", 0.02))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if sparse_points:
        points = np.asarray(sparse_points, dtype=np.float32)[:, :3]
        colors = np.asarray(sparse_points, dtype=np.float32)[:, 3:6] / 255.0
    else:
        points = np.asarray(
            [
                [math.cos(angle), math.sin(angle), 0.0]
                for angle in np.linspace(0.0, math.tau, 2400)
            ],
            dtype=np.float32,
        )
        colors = None
    model = SplatFacetField(
        points,
        colors,
        scale=float(config.get("init_scale", 0.12)),
        opacity=float(config.get("init_opacity", 0.85)),
    ).to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": [model.positions], "lr": learning_rate * 0.6},
            {"params": [model.color_logits], "lr": learning_rate},
            {"params": [model.log_scales], "lr": learning_rate * 0.6},
            {"params": [model.opacity_logits], "lr": learning_rate * 0.6},
        ],
    )
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

    log(
        f"[train] TFRF-facets 开始，{len(train_cameras)} 训练视角，"
        f"{len(points)} 棱面，{steps} 步，设备 {device}"
    )
    history: list[dict] = []
    start_wall = time.time()

    for step in range(steps):
        camera = train_cameras[int(camera_choices[step])]
        c2w = np.asarray(camera["c2w"], dtype=np.float32)
        width, height = camera["width"], camera["height"]
        k = camera["K"]
        fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
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
            batch_size=96,
        )
        target = torch.as_tensor(
            images[int(camera_choices[step])][ys, xs],
            dtype=torch.float32,
            device=device,
        )
        loss = torch.mean((rendered - target) ** 2)
        if surface_points is not None:
            prior_rng = np.random.default_rng(step + 50_000)
            prior_weight = float(config.get("surface_prior_weight", 0.0))
            loss = loss + prior_weight * _density_prior_loss(
                model, surface_points, device, prior_rng
            )
        scale_loss = 0.02 * torch.mean(model.log_scales**2)
        loss = loss + scale_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

        if step % eval_every == 0 or step == steps - 1:
            eval_index = (step // eval_every) % len(eval_cameras)
            eval_camera = eval_cameras[eval_index]
            with torch.no_grad():
                rendered_view = render_splats_view(model, eval_camera, background)
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
            metrics.update(
                {
                    "step": step,
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
        status_after(project_dir, task_id)

    model.cpu()
    return {"model": model, "history": history, "engine": "facets"}, history
