"""PyTorch training path for the Titan Facet Radiance Field."""

import math
import time

import numpy as np
import torch
import torch.nn.functional as F


def _encode_position(points, levels: int = 5):
    values = [points]
    for level in range(levels):
        values.append(torch.sin((2.0**level) * math.pi * points))
        values.append(torch.cos((2.0**level) * math.pi * points))
    return torch.cat(values, dim=-1)


def _encode_direction(directions, levels: int = 5):
    values = [directions]
    for level in range(levels):
        values.append(torch.sin((2.0**level) * math.pi * directions))
        values.append(torch.cos((2.0**level) * math.pi * directions))
    return torch.cat(values, dim=-1)


class TorchFacetField(torch.nn.Module):
    def __init__(self, feature_dim: int = 66, hidden: int = 128) -> None:
        super().__init__()
        self.feature = torch.nn.Parameter(
            torch.randn(feature_dim, hidden) * 0.08
        )
        self.bias_hidden = torch.nn.Parameter(torch.zeros(hidden))
        self.linear_out = torch.nn.Parameter(
            torch.randn(hidden, 4) * 0.06
        )
        self.bias_out = torch.nn.Parameter(
            torch.tensor([-2.44, -2.31, -2.20, -3.0], dtype=torch.float32)
        )

    def forward(self, points, directions):
        encoded = torch.cat(
            [_encode_position(points), _encode_direction(directions)], dim=-1
        )
        hidden = torch.tanh(encoded @ self.feature + self.bias_hidden)
        out = hidden @ self.linear_out + self.bias_out
        color = torch.sigmoid(out[..., :3])
        density = torch.sigmoid(out[..., 3:4])
        return color, density


def render_rays(
    model,
    origin: np.ndarray,
    direction: np.ndarray,
    near: float = 2.4,
    far: float = 7.6,
    samples: int = 48,
    background: np.ndarray | None = None,
):
    device = next(model.parameters()).device
    direction = torch.as_tensor(direction, dtype=torch.float32, device=device)
    origin = torch.as_tensor(origin, dtype=torch.float32, device=device)
    depth = torch.linspace(
        near, far, samples, dtype=torch.float32, device=device
    )
    points = origin[None, None, :] + depth[:, None, None] * direction[None, :, :]
    directions = direction[None, :, :].expand(points.shape)
    color, density = model(points, directions)
    alpha = 1.0 - torch.exp(-0.24 * density)
    transmittance = torch.cat(
        [
            torch.ones_like(alpha[:1]),
            torch.cumprod(1.0 - alpha + 1e-8, dim=0)[:-1],
        ],
        dim=0,
    )
    weights = alpha * transmittance
    rendered = torch.sum(weights * color, dim=0)
    if background is not None:
        bg = torch.as_tensor(background, dtype=torch.float32, device=device)
        rendered = rendered + (1.0 - torch.sum(weights, dim=0)) * bg
    return rendered


def render_view(model, camera: dict, background: np.ndarray):
    device = next(model.parameters()).device
    c2w = np.asarray(camera["c2w"], dtype=np.float32)
    k = camera["K"]
    width, height = camera["width"], camera["height"]
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    origin = c2w[:3, 3]
    rotation = c2w[:3, :3]
    ys, xs = np.mgrid[0:height, 0:width]
    local = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)], axis=-1)
    directions = local @ rotation.T
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    rendered = render_rays(
        model,
        origin,
        directions.reshape(-1, 3),
        background=background,
    )
    return rendered.reshape(height, width, 3)


def _density_prior_loss(model, surface_points, device, rng, count: int = 2048):
    directions = torch.zeros((count * 2, 3), dtype=torch.float32, device=device)
    surface = surface_points[
        rng.integers(0, len(surface_points), size=count)
    ]
    surface = surface + rng.normal(0.0, 0.05, size=(count, 3))
    empty = rng.uniform(
        [-3.2, -2.2, -3.2], [3.2, 2.2, 3.2], size=(count, 3)
    )
    points = torch.as_tensor(
        np.concatenate([surface, empty], axis=0),
        dtype=torch.float32,
        device=device,
    )
    _, density = model(points, directions)
    target = torch.cat(
        [
            torch.full((count,), 0.9, dtype=torch.float32, device=device),
            torch.full((count,), 0.03, dtype=torch.float32, device=device),
        ],
        dim=0,
    )
    return F.mse_loss(density[..., 0], target)


def train_tfrf_torch(
    images: np.ndarray,
    cameras: list[dict],
    config: dict,
    project_dir,
    task_id: str,
    sparse_points: list[list[float]] | None = None,
) -> tuple[dict, list[dict]]:
    from .main import (
        SCENE_BACKGROUND,
        FacetRadianceField,
        compute_metrics,
        emit_status,
        log,
        status_after,
    )

    import torch

    train_count = int(config.get("train_views", 16))
    steps = int(config.get("steps", 400))
    patch = int(config.get("patch_size", 512))
    eval_every = int(config.get("eval_every", 60))
    sample_count = int(config.get("samples", 32))
    learning_rate = float(config.get("learning_rate", 0.004))
    full_image = bool(config.get("full_image", False))
    prior_weight = float(config.get("density_prior_weight", 0.6))
    surface_points = (
        np.asarray(sparse_points, dtype=np.float32)[:, :3]
        if sparse_points
        else None
    )

    train_cameras = cameras[:train_count]
    eval_cameras = cameras[train_count:]
    if not eval_cameras:
        eval_cameras = cameras[-1:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TorchFacetField().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    background = SCENE_BACKGROUND
    camera_choices = np.random.default_rng(3).integers(
        0, len(train_cameras), size=steps
    )

    log(
        f"[train] TFRF-torch 开始，{len(train_cameras)} 训练视角，"
        f"{steps} 步，设备 {device}"
    )
    history: list[dict] = []
    start_wall = time.time()

    for step in range(steps):
        camera = train_cameras[int(camera_choices[step])]
        c2w = np.asarray(camera["c2w"], dtype=np.float32)
        width, height = camera["width"], camera["height"]
        origin = c2w[:3, 3]
        rotation = c2w[:3, :3]
        k = camera["K"]
        fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]

        if full_image:
            ys = np.repeat(np.arange(height), width)
            xs = np.tile(np.arange(width), height)
        else:
            rng = np.random.default_rng(step)
            ys = rng.integers(0, height, size=patch)
            xs = rng.integers(0, width, size=patch)
        local = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)], axis=-1)
        directions = local @ rotation.T
        directions = directions / np.linalg.norm(
            directions, axis=-1, keepdims=True
        )

        rendered = render_rays(
            model,
            origin,
            directions,
            samples=sample_count,
            background=background,
        )
        target = torch.as_tensor(
            images[int(camera_choices[step])][ys, xs],
            dtype=torch.float32,
            device=device,
        )
        loss = torch.mean((rendered - target) ** 2)
        if surface_points is not None and prior_weight > 0:
            rng_prior = np.random.default_rng(step + 100_000)
            loss = loss + prior_weight * _density_prior_loss(
                model, surface_points, device, rng_prior
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_every == 0 or step == steps - 1:
            eval_index = (step // eval_every) % len(eval_cameras)
            eval_camera = eval_cameras[eval_index]
            with torch.no_grad():
                rendered_view = render_view(model, eval_camera, background)
            target_view = images[len(cameras) - len(eval_cameras) + eval_index]
            metrics = compute_metrics(
                rendered_view.cpu().numpy(), target_view
            )
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

    model_np = FacetRadianceField()
    model_np.feature = model.feature.detach().cpu().numpy()
    model_np.bias_hidden = model.bias_hidden.detach().cpu().numpy()
    model_np.linear_out = model.linear_out.detach().cpu().numpy()
    model_np.bias_out = model.bias_out.detach().cpu().numpy()
    return {"model": model_np, "history": history}, history
