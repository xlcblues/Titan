"""Titan TFRF v0.1 reconstruction engine (subprocess sidecar)."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Anaconda ships OpenMP with several packages; keep the duplicate runtime
# warning from aborting the sidecar process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from PIL import Image

from . import __version__
from .metrics import psnr as compute_psnr, ssim as compute_ssim
from .synthetic import generate_scene, sphere_color_array


SCENE_BACKGROUND = np.array([0.08, 0.09, 0.10], dtype=np.float32)


def log(message: str) -> None:
    print(message, flush=True)


def emit_status(project_dir: Path, payload: dict) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "status.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def write_control(project_dir: Path, command: str) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "control.json").write_text(
        json.dumps({"command": command, "at": int(time.time() * 1000)}),
        encoding="utf-8",
    )


def read_control(project_dir: Path) -> str | None:
    control = project_dir / "control.json"
    try:
        return json.loads(control.read_text(encoding="utf-8")).get("command")
    except (OSError, ValueError, KeyError):
        return None


def encode_position(points: np.ndarray, levels: int) -> np.ndarray:
    values = [points]
    for level in range(levels):
        values.append(np.sin((2.0**level) * math.pi * points))
        values.append(np.cos((2.0**level) * math.pi * points))
    return np.concatenate(values, axis=-1)


def encode_direction(directions: np.ndarray, levels: int = 5) -> np.ndarray:
    values = [directions]
    for level in range(levels):
        values.append(np.sin((2.0**level) * math.pi * directions))
        values.append(np.cos((2.0**level) * math.pi * directions))
    return np.concatenate(values, axis=-1)


def load_pose_data(project_dir: Path, config: dict) -> dict:
    pose_file = project_dir / "pose.json"
    if not pose_file.exists():
        inputs_file = project_dir / "inputs.json"
        if inputs_file.exists():
            try:
                inputs = json.loads(inputs_file.read_text(encoding="utf-8"))
                image_paths = [
                    str(Path(item)) for item in inputs.get("images", [])
                    if Path(item).exists()
                ]
            except (OSError, ValueError):
                image_paths = []
            if image_paths:
                count = len(image_paths)
                size = int(config.get("resolution", 256))
                log(
                    f"[data] 未检测到位姿，为 {count} 张图片生成假设环绕位姿（低精度降级）"
                )
                cameras = generate_orbit_pose(project_dir, image_paths, count, size)
                images = [str(camera["path"]) for camera in cameras]
                return {
                    "images": images,
                    "cameras": cameras,
                    "points": [],
                }
        count = int(config.get("synthetic_views", 24))
        size = int(config.get("resolution", 256))
        log(f"[data] 未检测到位姿，生成合成测试场景 {count} 视角 x {size}")
        generate_scene(project_dir, count=count, size=size)
    payload = json.loads(pose_file.read_text(encoding="utf-8"))
    images = []
    cameras = []
    for camera in payload["cameras"]:
        image_path = project_dir / camera["path"]
        if not image_path.exists():
            continue
        camera["path"] = str(image_path)
        images.append(str(image_path))
        cameras.append(camera)
    return {"images": images, "cameras": cameras, "points": payload.get("points", [])}


def generate_orbit_pose(
    project_dir: Path,
    image_paths: list[str],
    count: int,
    size: int,
) -> list[dict]:
    cameras = []
    for index in range(count):
        angle = (index / count) * math.pi * 1.9
        position = np.array(
            [
                5.4 * math.cos(angle),
                1.15 + 0.6 * math.sin(angle * 3.0),
                5.4 * math.sin(angle),
            ],
            dtype=np.float64,
        )
        forward = -position / np.linalg.norm(position)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        c2w = np.eye(4)
        c2w[:3, :3] = np.stack([right, up, forward], axis=1)
        c2w[:3, 3] = position
        source = image_paths[index % len(image_paths)]
        try:
            with Image.open(source) as source_image:
                image_width, image_height = source_image.size
        except OSError:
            image_width, image_height = size, size
        focal = max(image_width, image_height) * 1.15
        k = [
            [focal, 0.0, image_width / 2.0],
            [0.0, focal, image_height / 2.0],
            [0.0, 0.0, 1.0],
        ]
        cameras.append(
            {
                "path": source,
                "width": image_width,
                "height": image_height,
                "K": k,
                "c2w": c2w.tolist(),
            }
        )
    return cameras


def load_images(paths: list[str]) -> np.ndarray:
    first = np.asarray(Image.open(paths[0]).convert("RGB"), dtype=np.float32) / 255.0
    images = np.empty((len(paths), first.shape[0], first.shape[1], 3), dtype=np.float32)
    images[0] = first
    for index, path in enumerate(paths[1:], start=1):
        images[index] = (
            np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        )
    return images


def load_and_prepare_images(
    paths: list[str],
    cameras: list[dict],
    config: dict,
) -> np.ndarray:
    """Load images and normalize them to the training resolution.

    Real captures often have mixed sizes; the M4 baseline resizes every
    input to a square training resolution and rebuilds a matching pinhole
    camera so the splat renderer and metrics share one consistent grid.
    """
    resolution = int(config.get("resolution", 256))
    target = (resolution, resolution)
    images = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            if image.size != target:
                image = image.resize(target, Image.LANCZOS)
            images.append(np.asarray(image, dtype=np.float32) / 255.0)
    focal = resolution * 1.15
    for camera in cameras:
        camera["width"] = resolution
        camera["height"] = resolution
        camera["K"] = [
            [focal, 0.0, resolution / 2.0],
            [0.0, focal, resolution / 2.0],
            [0.0, 0.0, 1.0],
        ]
    return np.stack(images, axis=0)


def build_sparse_points(cameras: list[dict], count: int = 2400) -> list[list[float]]:
    rng = np.random.default_rng(7)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = 2.0 + 0.08 * rng.normal(size=(count, 1))
    points = directions * radii
    colors = (sphere_color_array(points) * 255.0).astype(np.uint8)
    return [
        [float(x), float(y), float(z), int(r), int(g), int(b)]
        for (x, y, z), (r, g, b) in zip(points, colors)
    ]


class FacetRadianceField:
    def __init__(self, feature_dim: int = 66, hidden: int = 128) -> None:
        self.hidden = hidden
        # Points are rendered as soft faceted occupancy; the MLP predicts
        # a radiance color for each query point.
        self.feature = np.random.default_rng(11).normal(0.0, 0.08, size=(feature_dim, hidden))
        self.bias_hidden = np.zeros(hidden, dtype=np.float32)
        self.linear_out = np.random.default_rng(12).normal(0.0, 0.06, size=(hidden, 4))
        # Initialize color to the synthetic background so early steps are
        # near the constant background, then the field learns the object.
        self.bias_out = np.array([-2.44, -2.31, -2.20, -3.0], dtype=np.float32)

    def forward(
        self, points: np.ndarray, directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        encoded = np.concatenate(
            [encode_position(points, levels=5), encode_direction(directions)],
            axis=-1,
        )
        hidden = np.tanh(encoded @ self.feature + self.bias_hidden)
        out = hidden @ self.linear_out + self.bias_out
        color = 1.0 / (1.0 + np.exp(-out[..., :3]))
        density = 1.0 / (1.0 + np.exp(-out[..., 3:4]))
        return color, density


def render_rays(
    model: FacetRadianceField,
    origin: np.ndarray,
    direction: np.ndarray,
    near: float = 2.4,
    far: float = 7.6,
    samples: int = 48,
    background: np.ndarray | None = None,
) -> np.ndarray:
    depth = np.linspace(near, far, samples)
    points = origin[None, None, :] + depth[:, None, None] * direction[None, :, :]
    directions = np.broadcast_to(direction[None, :, :], points.shape)
    color, density = model.forward(points, directions)
    alpha = 1.0 - np.exp(-0.24 * density)
    transmittance = np.concatenate(
        [np.ones_like(alpha[:1]), np.cumprod(1.0 - alpha + 1e-8, axis=0)[:-1]],
        axis=0,
    )
    weights = alpha * transmittance
    rendered = np.sum(weights * color, axis=0)
    if background is not None:
        rendered = rendered + (1.0 - np.sum(weights, axis=0)) * background
    return rendered


def render_view(
    model: FacetRadianceField,
    camera: dict,
    background: np.ndarray,
) -> np.ndarray:
    if hasattr(model, "positions") and hasattr(model, "color_logits"):
        from .splat_engine import render_splats_view

        return render_splats_view(model, camera, background)
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
    flat_dirs = directions.reshape(-1, 3)
    rendered = render_rays(model, origin, flat_dirs, background=background)
    return rendered.reshape(height, width, 3)


def compute_metrics(
    rendered: np.ndarray,
    ground_truth: np.ndarray,
) -> dict:
    rendered = np.clip(rendered, 0.0, 1.0)
    ground_truth = np.clip(ground_truth, 0.0, 1.0)
    mse = float(np.mean((rendered - ground_truth) ** 2))
    return {
        "loss": mse,
        "psnr": compute_psnr(rendered, ground_truth),
        "ssim": compute_ssim(rendered, ground_truth),
    }


def prepare_device(config: dict) -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "numpy"
    if torch.cuda.is_available():
        try:
            properties = torch.cuda.get_device_properties(0)
            total_mb = int(properties.total_memory / (1024 * 1024))
            log(f"[device] CUDA {properties.name} {total_mb} MiB")
            return True, "torch"
        except Exception:
            return False, "numpy"
    log("[device] CUDA 不可用，使用 CPU/torch 引擎")
    return False, "torch"


def status_after(project_dir: Path, task_id: str) -> None:
    command = read_control(project_dir)
    if command == "cancel":
        emit_status(
            project_dir,
            {"taskId": task_id, "status": "cancelled", "stage": "cancel", "progress": 0.0},
        )
        sys.exit(0)
    if command == "pause":
        emit_status(
            project_dir,
            {"taskId": task_id, "status": "paused", "stage": "training", "progress": 0.0},
        )
        while read_control(project_dir) == "pause":
            time.sleep(0.4)


def orbit_camera(angle: float, size: int) -> dict:
    position = np.array(
        [5.4 * math.cos(angle), 1.35, 5.4 * math.sin(angle)], dtype=np.float64
    )
    forward = -position / np.linalg.norm(position)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([right, up, forward], axis=1)
    c2w[:3, 3] = position
    focal = size * 1.15
    return {
        "width": size,
        "height": size,
        "K": [[focal, 0.0, size / 2.0], [0.0, focal, size / 2.0], [0.0, 0.0, 1.0]],
        "c2w": c2w.tolist(),
    }


def train_tfrf(
    images: np.ndarray,
    cameras: list[dict],
    config: dict,
    project_dir: Path,
    task_id: str,
    device_name: str = "numpy",
    sparse_points: list[list[float]] | None = None,
) -> tuple[dict, list[dict]]:
    engine = config.get("engine", "facets")
    if engine == "facets":
        from .splat_engine import train_tfrf_splat

        return train_tfrf_splat(
            images,
            cameras,
            config,
            project_dir,
            task_id,
            sparse_points=sparse_points,
        )
    if device_name == "torch":
        from .torch_engine import train_tfrf_torch

        return train_tfrf_torch(
            images,
            cameras,
            config,
            project_dir,
            task_id,
            sparse_points=sparse_points,
        )
    train_count = int(config.get("train_views", 16))
    steps = int(config.get("steps", 400))
    patch = int(config.get("patch_size", 512))
    eval_every = int(config.get("eval_every", 60))
    sample_count = int(config.get("samples", 32))

    train_cameras = cameras[:train_count]
    eval_cameras = cameras[train_count:]
    if not eval_cameras:
        eval_cameras = cameras[-1:]
    background = SCENE_BACKGROUND
    model = FacetRadianceField()
    learning_rate = float(config.get("learning_rate", 0.003))
    adam = {
        "feature": {
            "m": np.zeros_like(model.feature),
            "v": np.zeros_like(model.feature),
        },
        "bias_hidden": {
            "m": np.zeros_like(model.bias_hidden),
            "v": np.zeros_like(model.bias_hidden),
        },
        "linear_out": {
            "m": np.zeros_like(model.linear_out),
            "v": np.zeros_like(model.linear_out),
        },
        "bias_out": {
            "m": np.zeros_like(model.bias_out),
            "v": np.zeros_like(model.bias_out),
        },
    }
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    grad_clip = 1.0
    camera_choices = np.random.default_rng(3).integers(
        0, len(train_cameras), size=steps
    )

    log(f"[train] TFRF v0.1 开始，{len(train_cameras)} 训练视角，{steps} 步")
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

        rng = np.random.default_rng(step)
        ys = rng.integers(0, height, size=patch)
        xs = rng.integers(0, width, size=patch)
        local = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)], axis=-1)
        directions = local @ rotation.T
        directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

        rendered = render_rays(
            model, origin, directions, samples=sample_count, background=background
        )
        target = images[int(camera_choices[step])][ys, xs]
        grad = 2.0 * (rendered - target)
        loss = float(np.mean((rendered - target) ** 2))

        depth = np.linspace(2.4, 7.6, sample_count)
        points = origin[None, None, :] + depth[:, None, None] * directions[None, :, :]
        dirs = np.broadcast_to(directions[None, :, :], points.shape)
        encoded = np.concatenate(
            [encode_position(points, levels=5), encode_direction(dirs)],
            axis=-1,
        )
        hidden = np.tanh(encoded @ model.feature + model.bias_hidden)
        out = hidden @ model.linear_out + model.bias_out
        color = 1.0 / (1.0 + np.exp(-out[..., :3]))
        density = 1.0 / (1.0 + np.exp(-out[..., 3:4]))
        alpha = 1.0 - np.exp(-0.24 * density)
        transmittance = np.concatenate(
            [np.ones_like(alpha[:1]), np.cumprod(1.0 - alpha + 1e-8, axis=0)[:-1]],
            axis=0,
        )
        weights = alpha * transmittance

        color_grad = weights * color * (1.0 - color) * grad[None, ...]
        density_contribution = np.sum(
            (color - rendered) * grad[None, ...], axis=-1, keepdims=True
        )
        density_grad = (
            density
            * (1.0 - density)
            * 0.24
            * weights
            * (1.0 - alpha)
            * density_contribution
        )
        out_grad = np.concatenate([color_grad, density_grad], axis=-1)

        hidden_grad = out_grad @ model.linear_out.T
        feature_grad = np.einsum("sni,snh->ih", encoded, hidden_grad)
        linear_grad = np.einsum("snh,sno->ho", hidden, out_grad)
        hidden_bias_grad = np.sum(hidden_grad, axis=(0, 1))
        out_bias_grad = np.sum(out_grad, axis=(0, 1))

        grads = {
            "feature": feature_grad,
            "bias_hidden": hidden_bias_grad,
            "linear_out": linear_grad,
            "bias_out": out_bias_grad,
        }
        norm_squared = sum(float(np.sum(g * g)) for g in grads.values())
        norm = math.sqrt(norm_squared)
        scale = 1.0 if norm < grad_clip else grad_clip / norm
        t = step + 1
        for name, gradient in grads.items():
            gradient = gradient * scale
            state = adam[name]
            state["m"] = beta1 * state["m"] + (1.0 - beta1) * gradient
            state["v"] = beta2 * state["v"] + (1.0 - beta2) * gradient * gradient
            m_hat = state["m"] / (1.0 - beta1**t)
            v_hat = state["v"] / (1.0 - beta2**t)
            delta = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            if name == "feature":
                model.feature = model.feature - delta
            elif name == "bias_hidden":
                model.bias_hidden = model.bias_hidden - delta
            elif name == "linear_out":
                model.linear_out = model.linear_out - delta
            else:
                model.bias_out = model.bias_out - delta

        if step % eval_every == 0 or step == steps - 1:
            eval_index = (step // eval_every) % len(eval_cameras)
            eval_camera = eval_cameras[eval_index]
            rendered_view = render_view(model, eval_camera, background)
            target_view = images[len(cameras) - len(eval_cameras) + eval_index]
            metrics = compute_metrics(rendered_view, target_view)
            metrics.update(
                {
                    "step": step,
                    "vramMB": 0,
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
                    "vramMB": metrics["vramMB"],
                    "metrics": history,
                },
            )
        status_after(project_dir, task_id)

    return {"model": model, "history": history}, history


def write_render(
    project_dir: Path,
    name: str,
    image: np.ndarray,
) -> str:
    render_dir = project_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    target = render_dir / name
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)).save(target)
    return f"renders/{name}"


def write_scene_output(
    project_dir: Path,
    task_id: str,
    cameras: list[dict],
    sparse_points: list[list[float]],
    final_metrics: dict,
    config: dict,
    history: list[dict],
    render_paths: list[dict],
    novel_path: str,
    engine: str = "facets",
) -> None:
    scene = {
        "version": "0.1.0",
        "engine": engine,
        "taskId": task_id,
        "projectDir": str(project_dir),
        "points": sparse_points,
        "cameras": [
            {
                "id": index,
                "path": camera.get("path", ""),
                "position": [float(v) for v in np.asarray(camera["c2w"])[:3, 3]],
                "forward": [float(v) for v in np.asarray(camera["c2w"])[:3, 2]],
                "width": camera["width"],
                "height": camera["height"],
            }
            for index, camera in enumerate(cameras)
        ],
        "renders": render_paths,
        "novelView": novel_path,
        "metrics": final_metrics,
        "history": history,
        "config": config,
    }
    (project_dir / "scene.json").write_text(
        json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "taskId": task_id,
        "status": "done",
        "stage": "done",
        "progress": 1.0,
        "metrics": final_metrics,
        "history": history,
        "sceneReady": True,
    }
    (project_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    emit_status(project_dir, summary)


def run(project_dir: Path, task_id: str, config: dict) -> int:
    project_dir = project_dir.resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    emit_status(
        project_dir,
        {"taskId": task_id, "status": "running", "stage": "data", "progress": 0.02},
    )

    data = load_pose_data(project_dir, config)
    if not data["images"]:
        raise RuntimeError("项目没有可用图片或位姿")
    images = load_and_prepare_images(data["images"], data["cameras"], config)
    log(f"[data] 图片 {len(data['images'])} 张，尺寸 {images.shape[1]}x{images.shape[2]}")

    _, device_name = prepare_device(config)
    sparse_points = data["points"] or build_sparse_points(data["cameras"])
    log(f"[data] 稀疏点云 {len(sparse_points)} 点，设备 {device_name}")

    emit_status(
        project_dir,
        {
            "taskId": task_id,
            "status": "running",
            "stage": "training",
            "progress": 0.12,
        },
    )
    trained_result, history = train_tfrf(
        images,
        data["cameras"],
        config,
        project_dir,
        task_id=task_id,
        device_name=device_name,
        sparse_points=sparse_points,
    )
    trained = trained_result["model"]

    emit_status(
        project_dir,
        {
            "taskId": task_id,
            "status": "running",
            "stage": "render",
            "progress": 0.92,
        },
    )
    background = SCENE_BACKGROUND
    render_paths: list[dict] = []
    for index in (0, 2, 4):
        if index >= len(data["cameras"]):
            break
        rendered = render_view(trained, data["cameras"][index], background)
        path = write_render(project_dir, f"render_train_{index:03d}.png", rendered)
        render_paths.append(
            {"name": f"render_train_{index:03d}.png", "path": path, "view": "train"}
        )

    train_count = int(config.get("train_views", 16))
    heldout = data["cameras"][train_count:]
    if not heldout:
        heldout = data["cameras"][-1:]
    heldout_scores = []
    for index, camera in enumerate(heldout):
        rendered = render_view(trained, camera, background)
        path = write_render(project_dir, f"render_eval_{index:03d}.png", rendered)
        render_paths.append(
            {"name": f"render_eval_{index:03d}.png", "path": path, "view": "eval"}
        )
        target_index = len(data["cameras"]) - len(heldout) + index
        target = images[target_index]
        heldout_scores.append(compute_metrics(rendered, target))

    novel = orbit_camera(0.7, int(config.get("resolution", 256)))
    rendered = render_view(trained, novel, background)
    novel_path = write_render(project_dir, "novel_orbit.png", rendered)

    final_metrics = {
        "psnr": float(np.mean([item["psnr"] for item in heldout_scores])),
        "ssim": float(np.mean([item["ssim"] for item in heldout_scores])),
        "loss": float(np.mean([item["loss"] for item in heldout_scores])),
        "vramMB": int(history[-1]["vramMB"]) if history else 0,
        "views": len(heldout_scores),
    }
    engine = trained_result.get("engine", "facets") if isinstance(trained_result, dict) else "facets"
    write_scene_output(
        project_dir,
        task_id,
        data["cameras"],
        sparse_points,
        final_metrics,
        config,
        history,
        render_paths,
        novel_path,
        engine=engine,
    )
    log("[done] 端到端重建完成")
    return 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Titan TFRF reconstruction engine")
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--config-json", default="{}")
    args = parser.parse_args()
    try:
        config = json.loads(args.config_json)
    except ValueError:
        config = {}
    os.chdir(Path(__file__).resolve().parent.parent)
    try:
        exit_code = run(Path(args.project_dir), args.task_id, config)
        sys.exit(exit_code)
    except Exception as error:
        emit_status(
            Path(args.project_dir),
            {"taskId": args.task_id, "status": "failed", "stage": "error", "error": str(error)},
        )
        import traceback

        traceback.print_exc()
        log(f"[error] {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
