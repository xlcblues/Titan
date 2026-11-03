"""Titan Coherence Field (TCF) engine.

TCF is a self-designed implicit scene representation. It stores a
multi-resolution coherence lattice and answers appearance queries with a
small shared readout. Rendering is a coherence voting process: every
sample along a ray casts an appearance vote, and the votes are weighted by
how coherent the local field is and whether the ray has already been
carried to that point.
"""

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn


def log(message: str) -> None:
    print(message, flush=True)


def _read_control(project_dir: Path) -> str | None:
    control = project_dir / "control.json"
    try:
        return json.loads(control.read_text(encoding="utf-8")).get("command")
    except (OSError, ValueError, KeyError):
        return None


def _device_peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.max_memory_reserved(0) / (1024 * 1024)
    except Exception:
        return 0.0


def _device_reserved_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.memory_reserved(0) / (1024 * 1024)
    except Exception:
        return 0.0


class CoherentLattice(nn.Module):
    """Multi-resolution hash lattice used as the scene's memory."""

    def __init__(
        self,
        levels: int = 8,
        base_resolution: int = 32,
        scale: float = 1.6,
        max_resolution: int = 512,
        entries_per_level: int = 131072,
        feature_dim: int = 4,
        seed: int = 11,
    ) -> None:
        super().__init__()
        self.levels = max(2, int(levels))
        self.base_resolution = int(base_resolution)
        self.scale = float(scale)
        self.max_resolution = int(max_resolution)
        self.entries_per_level = int(entries_per_level)
        self.feature_dim = int(feature_dim)

        growth = self.scale ** np.arange(self.levels)
        resolutions = np.minimum(
            self.max_resolution,
            np.floor(self.base_resolution * growth).astype(np.int64),
        )
        resolutions = np.maximum(resolutions, 2)
        self.register_buffer("resolutions", torch.from_numpy(resolutions).long())
        self.tables = nn.ModuleList(
            [
                nn.Embedding(self.entries_per_level, self.feature_dim)
                for _ in range(self.levels)
            ]
        )
        generator = torch.Generator().manual_seed(seed)
        for table in self.tables:
            nn.init.normal_(table.weight, mean=0.0, std=0.08, generator=generator)

    def params(self) -> list[nn.Parameter]:
        return [table.weight for table in self.tables]


class CoherenceReadout(nn.Module):
    """Shared tiny readout that turns lattice context into a vote."""

    def __init__(
        self,
        feature_dim: int = 4,
        levels: int = 8,
        position_levels: int = 4,
        direction_levels: int = 2,
        hidden: int = 64,
        seed: int = 12,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.levels = int(levels)
        self.position_levels = int(position_levels)
        self.direction_levels = int(direction_levels)
        position_in = 3 * (1 + 2 * self.position_levels)
        direction_in = 3 * (1 + 2 * self.direction_levels)
        in_dim = self.feature_dim * self.levels + position_in + direction_in
        self.linear_in = nn.Linear(in_dim, hidden, bias=True)
        self.linear_hidden = nn.Linear(hidden, hidden, bias=True)
        self.linear_out = nn.Linear(hidden, 5, bias=True)
        generator = torch.Generator().manual_seed(seed)
        nn.init.xavier_uniform_(self.linear_in.weight, generator=generator)
        nn.init.xavier_uniform_(self.linear_hidden.weight, generator=generator)
        nn.init.xavier_uniform_(self.linear_out.weight, generator=generator)
        nn.init.zeros_(self.linear_in.bias)
        nn.init.zeros_(self.linear_hidden.bias)
        nn.init.zeros_(self.linear_out.bias)

    def forward(
        self,
        context: torch.Tensor,
        positions: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        pos = self._encode(positions, self.position_levels)
        direction = self._encode(directions, self.direction_levels)
        hidden = torch.relu(
            self.linear_in(torch.cat([context, pos, direction], dim=-1))
        )
        hidden = torch.relu(self.linear_hidden(hidden))
        return self.linear_out(hidden)

    @staticmethod
    def _encode(values: torch.Tensor, levels: int) -> torch.Tensor:
        encoded = [values]
        for level in range(levels):
            freq = (2.0**level) * math.pi
            encoded.append(torch.sin(values * freq))
            encoded.append(torch.cos(values * freq))
        return torch.cat(encoded, dim=-1)


class TitanCoherenceField(nn.Module):
    """The complete TCF scene model."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        levels = int(config.get("lattice_levels", 8))
        self.lattice = CoherentLattice(
            levels=levels,
            base_resolution=int(config.get("lattice_base_resolution", 32)),
            scale=float(config.get("lattice_scale", 1.6)),
            max_resolution=int(config.get("lattice_max_resolution", 512)),
            entries_per_level=int(config.get("lattice_entries", 131072)),
            feature_dim=int(config.get("lattice_feature_dim", 4)),
        )
        self.readout = CoherenceReadout(
            feature_dim=int(config.get("lattice_feature_dim", 4)),
            levels=levels,
            position_levels=int(config.get("position_levels", 4)),
            direction_levels=int(config.get("direction_levels", 2)),
            hidden=int(config.get("readout_hidden", 64)),
        )
        self.background = nn.Parameter(
            torch.tensor([0.08, 0.09, 0.10], dtype=torch.float32)
        )

    def lattice_mb(self) -> float:
        total = sum(p.numel() for p in self.lattice.params())
        return total * 4 / (1024 * 1024)

    def hash_feature(self, points: torch.Tensor) -> torch.Tensor:
        """Trilinear lookup over every lattice level."""
        if points.numel() == 0:
            return torch.empty(
                (0, self.lattice.levels * self.lattice.feature_dim),
                dtype=points.dtype,
                device=points.device,
            )
        levels = self.lattice.levels
        feature_dim = self.lattice.feature_dim
        points = points.reshape(-1, 3)
        resolution = self.lattice.resolutions
        scaled = points[None, :, :] * resolution[:, None, None].to(points.dtype)
        base = torch.floor(scaled - 0.5).long()
        fraction = scaled - 0.5 - base.to(points.dtype)
        lo = torch.remainder(base, resolution[:, None, None].long())
        hi = torch.remainder(lo + 1, resolution[:, None, None].long())

        level_index = torch.arange(
            levels, device=points.device, dtype=torch.long
        ).reshape(levels, 1)
        x, y, z = lo[..., 0], lo[..., 1], lo[..., 2]
        xh, yh, zh = hi[..., 0], hi[..., 1], hi[..., 2]
        table_size = torch.tensor(
            self.lattice.entries_per_level,
            device=points.device,
            dtype=torch.long,
        )
        index000 = self._hash_index(level_index, x, y, z, table_size)
        index001 = self._hash_index(level_index, x, y, zh, table_size)
        index010 = self._hash_index(level_index, x, yh, z, table_size)
        index011 = self._hash_index(level_index, x, yh, zh, table_size)
        index100 = self._hash_index(level_index, xh, y, z, table_size)
        index101 = self._hash_index(level_index, xh, y, zh, table_size)
        index110 = self._hash_index(level_index, xh, yh, z, table_size)
        index111 = self._hash_index(level_index, xh, yh, zh, table_size)

        table_weights = torch.stack([table.weight for table in self.lattice.tables])
        f000 = table_weights[level_index, index000]
        f001 = table_weights[level_index, index001]
        f010 = table_weights[level_index, index010]
        f011 = table_weights[level_index, index011]
        f100 = table_weights[level_index, index100]
        f101 = table_weights[level_index, index101]
        f110 = table_weights[level_index, index110]
        f111 = table_weights[level_index, index111]

        xd = fraction[..., 0:1]
        yd = fraction[..., 1:2]
        zd = fraction[..., 2:3]
        c00 = f000 * (1 - xd) + f100 * xd
        c01 = f001 * (1 - xd) + f101 * xd
        c10 = f010 * (1 - xd) + f110 * xd
        c11 = f011 * (1 - xd) + f111 * xd
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        result = c0 * (1 - zd) + c1 * zd
        result = result.permute(1, 0, 2).reshape(-1, levels * feature_dim)
        return result

    @staticmethod
    def _hash_index(
        level: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        table_size: torch.Tensor,
    ) -> torch.Tensor:
        hashed = torch.bitwise_xor(
            torch.bitwise_xor(
                torch.remainder(x * 73856093 + level * 2654435761, 2**31 - 1),
                y * 19349663,
            ),
            z * 83492791,
        )
        return torch.remainder(hashed, table_size)

    def readout_logits(
        self,
        points: torch.Tensor,
        directions: torch.Tensor,
        chunk: int = 262144,
    ) -> torch.Tensor:
        context = self.hash_feature(points)
        if context.shape[0] <= chunk:
            return self.readout(context, points, directions)
        outputs = []
        for start in range(0, context.shape[0], chunk):
            stop = min(start + chunk, context.shape[0])
            outputs.append(
                self.readout(
                    context[start:stop],
                    points[start:stop],
                    directions[start:stop],
                )
            )
        return torch.cat(outputs, dim=0)


def _voting_weights(
    coherence: torch.Tensor,
    gates: torch.Tensor,
) -> torch.Tensor:
    """Compute ray sample vote weights from coherence and carry gates."""
    carry = torch.cumsum(gates, dim=1) - gates
    votes = torch.softmax(coherence - carry, dim=1)
    return votes


def _sample_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
    near: float,
    far: float,
    samples: int,
    fine_inverse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inverse_near = 1.0 / max(float(near), 1e-4)
    inverse_far = 1.0 / max(float(far), 1e-3)
    if fine_inverse is None:
        inverse_depth = torch.linspace(
            inverse_far,
            inverse_near,
            samples,
            device=origins.device,
            dtype=origins.dtype,
        )
        inverse_depth = inverse_depth.unsqueeze(0).expand(
            origins.shape[0], samples
        )
    else:
        spread = max((inverse_far - inverse_near) * 0.5, 1e-3)
        offsets = torch.linspace(
            -1.0,
            1.0,
            samples,
            device=origins.device,
            dtype=origins.dtype,
        )
        inverse_depth = fine_inverse + offsets[None, :] * spread
        inverse_depth = torch.clamp(
            inverse_depth,
            min=inverse_far * 0.98,
            max=inverse_near * 1.02,
        )
    positions = (
        origins.unsqueeze(1) + directions.unsqueeze(1) / inverse_depth.unsqueeze(-1)
    )
    return positions, inverse_depth


def render_rays_chunk(
    model: TitanCoherenceField,
    origins: torch.Tensor,
    directions: torch.Tensor,
    near: float,
    far: float,
    coarse_samples: int = 32,
    fine_samples: int = 16,
    readout_chunk: int = 131072,
    use_amp: bool = True,
) -> torch.Tensor:
    """Render a batch of rays with coherence voting.

    A coarse pass locates the coherent region, then a fine pass refines the
    vote around it. Both passes share the same lattice and readout.
    """
    device = origins.device
    with torch.autocast("cuda", enabled=use_amp and device.type == "cuda"):
        coarse_positions, inverse_depths = _sample_rays(
            origins, directions, near, far, coarse_samples
        )
        coarse_logits = model.readout_logits(
            coarse_positions.reshape(-1, 3),
            directions.unsqueeze(1).expand_as(coarse_positions).reshape(-1, 3),
            chunk=readout_chunk,
        )
        coarse_logits = coarse_logits.reshape(
            origins.shape[0], coarse_samples, 5
        )
        coarse_color = torch.sigmoid(coarse_logits[..., 0:3])
        coarse_coherence = coarse_logits[..., 3:4]
        coarse_gates = torch.sigmoid(coarse_logits[..., 4:5])
        coarse_weights = _voting_weights(coarse_coherence, coarse_gates)
        expected_inverse = torch.sum(
            coarse_weights * inverse_depths.unsqueeze(-1), dim=1
        )

        fine_positions, _ = _sample_rays(
            origins,
            directions,
            near,
            far,
            fine_samples,
            fine_inverse=expected_inverse,
        )
        fine_logits = model.readout_logits(
            fine_positions.reshape(-1, 3),
            directions.unsqueeze(1).expand_as(fine_positions).reshape(-1, 3),
            chunk=readout_chunk,
        )
        fine_logits = fine_logits.reshape(origins.shape[0], fine_samples, 5)
        fine_color = torch.sigmoid(fine_logits[..., 0:3])
        fine_coherence = fine_logits[..., 3:4]
        fine_gates = torch.sigmoid(fine_logits[..., 4:5])
        fine_weights = _voting_weights(fine_coherence, fine_gates)

        color = torch.sum(fine_weights * fine_color, dim=1)
        visibility = fine_weights.sum(dim=1).clamp(max=1.0)
        residual = torch.clamp(1.0 - visibility, min=0.0)
        background = model.background.to(color.dtype)
        rendered = color + residual * background
    return rendered


def render_rays(
    model: TitanCoherenceField,
    origins: torch.Tensor,
    directions: torch.Tensor,
    near: float,
    far: float,
    ray_batch: int = 4096,
    coarse_samples: int = 32,
    fine_samples: int = 16,
    use_amp: bool = True,
) -> torch.Tensor:
    """Render arbitrary ray counts in memory-bounded batches."""
    outputs = []
    for start in range(0, origins.shape[0], ray_batch):
        stop = min(start + ray_batch, origins.shape[0])
        outputs.append(
            render_rays_chunk(
                model,
                origins[start:stop],
                directions[start:stop],
                near,
                far,
                coarse_samples=coarse_samples,
                fine_samples=fine_samples,
                use_amp=use_amp,
            )
        )
    return torch.cat(outputs, dim=0)


def _camera_rays_from_pose(
    camera: dict,
    device: torch.device,
    ys: torch.Tensor | None = None,
    xs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    c2w = np.asarray(camera["c2w"], dtype=np.float32)
    k = camera["K"]
    width, height = camera["width"], camera["height"]
    fx, fy, cx, cy = float(k[0][0]), float(k[1][1]), float(k[0][2]), float(k[1][2])
    if ys is None:
        ys_flat = np.arange(height)
        xs_flat = np.arange(width)
    else:
        ys_flat = ys
        xs_flat = xs
    if ys is None:
        grid_y, grid_x = np.meshgrid(ys_flat, xs_flat, indexing="ij")
        local = np.stack(
            [
                (grid_x - cx) / fx,
                (grid_y - cy) / fy,
                np.ones_like(grid_x),
            ],
            axis=-1,
        )
        directions = local @ np.asarray(c2w[:3, :3], dtype=np.float32).T
        flat = directions.reshape(-1, 3)
    else:
        local = np.stack(
            [(xs_flat - cx) / fx, (ys_flat - cy) / fy, np.ones_like(xs_flat)],
            axis=-1,
        )
        flat = local @ np.asarray(c2w[:3, :3], dtype=np.float32).T
    flat = flat / np.linalg.norm(flat, axis=-1, keepdims=True)
    origins = torch.tensor(
        np.asarray(c2w[:3, 3], dtype=np.float32),
        device=device,
        dtype=torch.float32,
    ).expand(flat.shape[0], 3)
    directions = torch.tensor(flat, device=device, dtype=torch.float32)
    return origins, directions


def _scene_transform(
    cameras: list[dict],
    scene_points: np.ndarray | None = None,
) -> tuple[torch.Tensor, float]:
    positions = np.asarray(
        [np.asarray(camera["c2w"], dtype=np.float32)[:3, 3] for camera in cameras]
    )
    anchors = positions
    if scene_points is not None and len(scene_points):
        points = np.asarray(scene_points, dtype=np.float32).reshape(-1, 3)
        if len(points):
            anchors = np.concatenate([positions, points], axis=0)
    if len(anchors) == 0:
        return (
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            1.0,
        )
    shift = np.median(anchors, axis=0)
    radius = float(np.median(np.linalg.norm(anchors - shift, axis=1)))
    if not np.isfinite(radius) or radius <= 1e-6:
        radius = 1.0
    return torch.tensor(shift, dtype=torch.float32), radius


def _ray_bounds(
    origins: torch.Tensor,
    scene_points: np.ndarray,
    radius: float,
    scale: float,
    shift: torch.Tensor,
) -> tuple[float, float]:
    if len(scene_points) == 0:
        return 0.3, 7.6
    points = np.asarray(scene_points, dtype=np.float32)[:, :3]
    points = (points - shift.cpu().numpy()) * scale
    distances = np.linalg.norm(points - origins.cpu().numpy()[0], axis=1)
    distances = distances[distances > 1e-4]
    if distances.size == 0:
        return 0.3, 7.6
    lower = float(np.percentile(distances, 5.0))
    upper = float(np.percentile(distances, 95.0))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= 1e-6:
        return 0.3, 7.6
    return max(lower * 0.6, 0.05), upper * 1.5


def train_tcf(
    images: np.ndarray,
    cameras: list[dict],
    config: dict,
    project_dir: Path,
    task_id: str,
    device_name: str = "cuda",
    sparse_points: list[list[float]] | None = None,
    status_cb: Callable[[dict], None] | None = None,
) -> tuple[dict, list[dict]]:
    """Train a TCF scene model with per-step progress reporting."""
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log(f"[tcf] 使用 CUDA：{torch.cuda.get_device_name(0)}")
    else:
        log("[tcf] 使用 CPU 训练（未检测到 CUDA）")

    model = TitanCoherenceField(config).to(device)
    scene_points = (
        np.asarray(sparse_points, dtype=np.float32)[:, :3]
        if sparse_points is not None and len(sparse_points)
        else np.empty((0, 3), dtype=np.float32)
    )
    shift, radius = _scene_transform(cameras, scene_points)
    shift = shift.to(device)
    scale = 1.0 / max(float(radius), 1e-6)
    log(
        f"[tcf] 场景归一化 shift {shift.cpu().numpy().tolist()} "
        f"radius {radius:.4f} scale {scale:.6f}"
    )

    train_count = max(1, int(config.get("train_views", 16)))
    train_cameras = cameras[:train_count]
    if not train_cameras:
        train_cameras = cameras[:1]
    eval_cameras = cameras[train_count:]
    if not eval_cameras:
        eval_cameras = cameras[-1:]

    steps = max(1, int(config.get("steps", 1200)))
    half_step = max(0, steps // 2)
    patch = max(64, int(config.get("patch_size", 1024)))
    coarse_samples = int(config.get("coarse_samples", 32))
    fine_samples = int(config.get("fine_samples", 16))
    ray_batch = int(config.get("ray_batch_size", 4096))
    readout_chunk = int(config.get("readout_chunk", 131072))
    progress_every = max(1, int(config.get("training_progress_every", 10)))
    learning_rate = float(config.get("learning_rate", 0.01))
    lr_end = float(config.get("learning_rate_end", 0.002))
    use_amp = bool(config.get("mixed_precision", True))
    background_weight = float(config.get("background_weight", 0.02))

    train_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(train_parameters, lr=learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr_end
    )
    rng = np.random.default_rng(2026)
    history: list[dict] = []
    start_wall = time.time()

    log(
        f"[tcf] 开始：{len(train_cameras)} 训练视角，{steps} 步，"
        f"lattice {model.lattice_mb():.1f} MB，patch {patch} px"
    )
    if status_cb:
        status_cb(
            {
                "taskId": task_id,
                "status": "running",
                "stage": "training",
                "progress": 0.12,
                "step": 0,
                "totalSteps": steps,
                "loss": None,
                "vramMB": _device_reserved_mb(),
                "latticeMB": model.lattice_mb(),
            }
        )

    for step in range(steps):
        command = _read_control(project_dir)
        if command == "cancel":
            log("[tcf] 收到取消指令，安全停止训练")
            if status_cb:
                status_cb(
                    {
                        "taskId": task_id,
                        "status": "cancelled",
                        "stage": "cancel",
                        "progress": 0.0,
                        "step": step,
                    }
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model.to("cpu")
            return {
                "cancelled": True,
                "model": model,
                "shift": shift.cpu(),
                "scale": scale,
                "radius": radius,
                "engine": "tcf",
                "history": history,
            }, history
        if command == "pause":
            log("[tcf] 训练已暂停")
            if status_cb:
                status_cb(
                    {
                        "taskId": task_id,
                        "status": "paused",
                        "stage": "training",
                        "progress": round(0.12 + 0.78 * ((step + 1) / steps), 4),
                        "step": step,
                        "totalSteps": steps,
                        "loss": None,
                    }
                )
            while _read_control(project_dir) == "pause":
                time.sleep(0.4)
            log("[tcf] 训练已恢复")
            if status_cb:
                status_cb(
                    {
                        "taskId": task_id,
                        "status": "running",
                        "stage": "training",
                        "progress": round(0.12 + 0.78 * ((step + 1) / steps), 4),
                        "step": step,
                        "totalSteps": steps,
                        "loss": None,
                        "vramMB": _device_reserved_mb(),
                        "latticeMB": model.lattice_mb(),
                    }
                )
        camera_index = int(rng.integers(0, len(train_cameras)))
        camera = train_cameras[camera_index]
        image = images[camera_index] if camera_index < len(images) else images[0]
        height, width = int(camera["height"]), int(camera["width"])
        ys = rng.integers(0, height, size=patch)
        xs = rng.integers(0, width, size=patch)
        origins, directions = _camera_rays_from_pose(camera, device, ys, xs)
        origins_scaled = (origins - shift) * scale
        near, far = _ray_bounds(origins_scaled, scene_points, radius, scale, shift)
        target = torch.tensor(
            image[ys, xs].astype(np.float32), device=device, dtype=torch.float32
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", enabled=use_amp and device.type == "cuda"
        ):
            rendered = render_rays_chunk(
                model,
                origins_scaled,
                directions,
                near,
                far,
                coarse_samples=coarse_samples,
                fine_samples=fine_samples,
                readout_chunk=readout_chunk,
                use_amp=use_amp,
            )
            color_loss = torch.mean((rendered - target) ** 2)
            bg_penalty = torch.mean(torch.relu(-model.background)) * background_weight
            loss = color_loss + bg_penalty

        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_parameters, 1.0)
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.detach().cpu())
        if step % progress_every == 0 or step == steps - 1:
            payload = {
                "taskId": task_id,
                "status": "running",
                "stage": "training",
                "progress": round(0.12 + 0.78 * ((step + 1) / steps), 4),
                "step": step,
                "totalSteps": steps,
                "loss": loss_value,
                "vramMB": _device_reserved_mb(),
                "latticeMB": model.lattice_mb(),
            }
            if status_cb:
                status_cb(payload)

        if step == half_step or step == steps - 1:
            eval_index = 0 if step == half_step else len(eval_cameras) - 1
            eval_camera = eval_cameras[eval_index]
            target_index = len(cameras) - len(eval_cameras) + eval_index
            if target_index < len(images):
                target_view = images[target_index]
            else:
                target_view = image
            rendered_view = render_tcf_view(
                model,
                eval_camera,
                shift,
                scale,
                scene_points,
                radius,
                ray_batch=ray_batch,
                coarse_samples=coarse_samples,
                fine_samples=fine_samples,
                use_amp=use_amp,
            )
            mse = float(np.mean((rendered_view - target_view) ** 2))
            from .metrics import psnr as compute_psnr, ssim as compute_ssim

            metrics = {
                "step": step,
                "totalSteps": steps,
                "loss": mse,
                "psnr": compute_psnr(rendered_view, target_view),
                "ssim": compute_ssim(rendered_view, target_view),
                "vramMB": int(_device_reserved_mb()),
                "latticeMB": model.lattice_mb(),
                "elapsed": round(time.time() - start_wall, 2),
            }
            history.append(metrics)
            log(
                f"[tcf] step {step} loss {mse:.5f} "
                f"psnr {metrics['psnr']:.2f} ssim {metrics['ssim']:.3f}"
            )
            if status_cb:
                status_cb(
                    {
                        **payload,
                        "psnr": metrics["psnr"],
                        "ssim": metrics["ssim"],
                        "vramMB": int(_device_reserved_mb()),
                        "metrics": history,
                    }
                )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.to("cpu")
    return {
        "model": model,
        "shift": shift.cpu(),
        "scale": scale,
        "radius": radius,
        "engine": "tcf",
        "history": history,
        "latticeMB": model.lattice_mb(),
        "peakVramMB": _device_peak_mb(),
    }, history


def render_tcf_view(
    model: TitanCoherenceField,
    camera: dict,
    shift: torch.Tensor,
    scale: float,
    scene_points: np.ndarray,
    radius: float,
    ray_batch: int = 4096,
    coarse_samples: int = 32,
    fine_samples: int = 16,
    use_amp: bool = True,
) -> np.ndarray:
    """Render a full view with a memory-bounded ray loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    shift = shift.to(device)
    origins, directions = _camera_rays_from_pose(camera, device)
    origins_scaled = (origins - shift) * scale
    near, far = _ray_bounds(origins_scaled, scene_points, radius, scale, shift)
    with torch.no_grad():
        rendered = render_rays(
            model,
            origins_scaled,
            directions,
            near,
            far,
            ray_batch=ray_batch,
            coarse_samples=coarse_samples,
            fine_samples=fine_samples,
            use_amp=use_amp,
        )
    image = rendered.detach().cpu().numpy()
    height, width = int(camera["height"]), int(camera["width"])
    return image.reshape(height, width, 3)
