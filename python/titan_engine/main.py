"""Titan Coherence Field (TCF) reconstruction engine (subprocess sidecar).

TCF is a self-designed implicit scene representation: a multi-resolution
coherence lattice plus a small shared readout network. Rendering casts
coherence votes along rays instead of emitting discrete primitives.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable

# Anaconda ships OpenMP with several packages; keep the duplicate runtime
# warning from aborting the sidecar process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image

from . import __version__
from .colmap_io import (
    find_poses_bounds,
    find_sparse_dir,
    load_colmap_sparse,
    load_llff_poses,
)
from .colmap_sfm import run_colmap_sfm
from .metrics import psnr as compute_psnr, ssim as compute_ssim
from .synthetic import generate_scene, sphere_color_array
from .tcf_engine import extract_dense_points, render_tcf_view, train_tcf


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


def load_pose_data(
    project_dir: Path,
    config: dict,
    status_cb: Callable[[float, str], None] | None = None,
) -> dict:
    pose_file = project_dir / "pose.json"
    cached: dict | None = None
    if pose_file.exists():
        try:
            cached = json.loads(pose_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = None
    if config.get("reload_pose", False):
        cached = None

    inputs_file = project_dir / "inputs.json"
    image_paths: list[str] = []
    if inputs_file.exists():
        try:
            inputs = json.loads(inputs_file.read_text(encoding="utf-8"))
            image_paths = [
                str(Path(item)) for item in inputs.get("images", [])
                if Path(item).exists()
            ]
        except (OSError, ValueError):
            image_paths = []

    def save_and_return(payload: dict, source: str, message: str) -> dict:
        cameras: list[dict] = []
        for camera in payload["cameras"]:
            raw_path = Path(camera["path"])
            image_path = raw_path if raw_path.is_absolute() else project_dir / raw_path
            camera["path"] = str(image_path)
            cameras.append(camera)
        payload["cameras"] = cameras
        payload["source"] = source
        pose_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log(message)
        return {
            "images": [str(camera["path"]) for camera in cameras],
            "cameras": cameras,
            "points": payload.get("points", []),
            "source": source,
        }

    if not image_paths:
        count = int(config.get("synthetic_views", 24))
        size = int(config.get("resolution", 192))
        log(f"[data] 未检测到图片，生成合成测试场景 {count} 视角 x {size}")
        generate_scene(project_dir, count=count, size=size)
        cached = None
    else:
        pose_source = str(config.get("pose_source", "auto")).lower()
        want_colmap = pose_source in {"auto", "colmap"}

        if want_colmap:
            log(f"[data] 开始用 COLMAP 估计 {len(image_paths)} 张图片位姿")
            colmap_data = run_colmap_sfm(
                project_dir,
                image_paths,
                config,
                status_cb=status_cb,
            )
            if colmap_data and colmap_data["cameras"]:
                if status_cb:
                    status_cb(
                        0.1,
                        f"完成，{len(colmap_data['cameras'])} 张图片注册",
                    )
                return save_and_return(
                    colmap_data,
                    "colmap",
                    f"[data] COLMAP 位姿估计完成："
                    f"{len(colmap_data['cameras'])} 张图片，"
                    f"{len(colmap_data['points'])} 个稀疏点",
                )
            log("[data] COLMAP 未产出可用位姿，尝试已有位姿文件")

        if (
            cached
            and cached.get("cameras")
            and len(cached["cameras"]) == len(image_paths)
        ):
            source = str(cached.get("source", "orbit"))
            return save_and_return(
                cached,
                source,
                f"[data] 使用已有位姿缓存（{source}）",
            )

        sparse_dir = find_sparse_dir(image_paths)
        if sparse_dir:
            point_limit = int(config.get("sparse_points_limit", 50000))
            colmap_data = load_colmap_sparse(
                sparse_dir, image_paths, point_limit=point_limit
            )
            if colmap_data and colmap_data["cameras"]:
                return save_and_return(
                    colmap_data,
                    "colmap",
                    f"[data] 检测到 COLMAP 真实位姿，"
                    f"{len(colmap_data['cameras'])} 张图片，"
                    f"{len(colmap_data['points'])} 个稀疏点",
                )

        poses_file = find_poses_bounds(image_paths)
        if poses_file:
            llff_data = load_llff_poses(poses_file, image_paths)
            if llff_data and llff_data["cameras"]:
                return save_and_return(
                    llff_data,
                    "llff",
                    f"[data] 检测到 LLFF 真实位姿，"
                    f"{len(llff_data['cameras'])} 张图片",
                )

        count = len(image_paths)
        size = int(config.get("resolution", 192))
        cameras = generate_orbit_pose(project_dir, image_paths, count, size)
        payload = {
            "cameras": cameras,
            "points": [],
        }
        return save_and_return(
            payload,
            "orbit",
            f"[data] 未检测到位姿，为 {count} 张图片生成假设环绕位姿（低精度降级）",
        )

    payload = json.loads(pose_file.read_text(encoding="utf-8"))
    images = []
    cameras = []
    for camera in payload["cameras"]:
        raw_path = Path(camera["path"])
        image_path = raw_path if raw_path.is_absolute() else project_dir / raw_path
        if not image_path.exists():
            continue
        camera["path"] = str(image_path)
        images.append(str(image_path))
        cameras.append(camera)
    return {
        "images": images,
        "cameras": cameras,
        "points": payload.get("points", []),
        "source": payload.get("source", "orbit"),
    }


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
    """Load images and normalize them to the training grid.

    In original-resolution mode the common canvas is derived from the largest
    source frame and is only scaled down when it exceeds
    ``train_max_image_pixels``, so training sees near-native detail instead of
    the fixed low-resolution square. Otherwise images are letterboxed into the
    ``resolution`` square as before. Camera intrinsics are scaled by the same
    factor and shifted by the letterbox offset in both modes.
    """
    resolution = int(config.get("resolution", 192))
    original = bool(config.get("train_original_resolution", False))
    max_pixels = int(config.get("train_max_image_pixels", 2_000_000))

    actual_sizes: list[tuple[int, int]] = []
    for path in paths:
        with Image.open(path) as source:
            source_width, source_height = source.size
            actual_sizes.append((source_width, source_height))

    if original:
        max_width = max(size[0] for size in actual_sizes)
        max_height = max(size[1] for size in actual_sizes)
        pixel_scale = math.sqrt(max_pixels / max(max_width * max_height, 1))
        if pixel_scale >= 1.0:
            target_width, target_height = max_width, max_height
        else:
            target_width = max(1, int(round(max_width * pixel_scale)))
            target_height = max(1, int(round(max_height * pixel_scale)))
    else:
        target_width, target_height = resolution, resolution

    images = []
    for path, (source_width, source_height) in zip(paths, actual_sizes):
        with Image.open(path) as source:
            image = source.convert("RGB")
            scale = min(
                target_width / source_width,
                target_height / source_height,
            )
            scaled_size = (
                max(1, round(source_width * scale)),
                max(1, round(source_height * scale)),
            )
            scaled = image.resize(scaled_size, Image.LANCZOS)
            canvas = Image.new(
                "RGB", (target_width, target_height), (0, 0, 0)
            )
            offset_x = (target_width - scaled_size[0]) // 2
            offset_y = (target_height - scaled_size[1]) // 2
            canvas.paste(scaled, (offset_x, offset_y))
            images.append(np.asarray(canvas, dtype=np.float32) / 255.0)

    for index, camera in enumerate(cameras):
        if index >= len(paths):
            break
        actual_width, actual_height = actual_sizes[index]
        source_width = max(1, int(camera.get("width", actual_width)))
        source_height = max(1, int(camera.get("height", actual_height)))
        source_k = camera.get("K")
        if source_k is None or not source_k:
            focal_source = max(actual_width, actual_height) * 1.15
            source_k = [
                [focal_source, 0.0, actual_width / 2.0],
                [0.0, focal_source, actual_height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        if (
            abs(source_width - actual_width) > 1
            or abs(source_height - actual_height) > 1
        ):
            scale_x = actual_width / source_width
            scale_y = actual_height / source_height
            source_k = [
                [float(source_k[0][0]) * scale_x, 0.0, float(source_k[0][2]) * scale_x],
                [0.0, float(source_k[1][1]) * scale_y, float(source_k[1][2]) * scale_y],
                [0.0, 0.0, 1.0],
            ]
        scale = min(target_width / actual_width, target_height / actual_height)
        scaled_width = max(1, round(actual_width * scale))
        scaled_height = max(1, round(actual_height * scale))
        offset_x = (target_width - scaled_width) / 2.0
        offset_y = (target_height - scaled_height) / 2.0
        fx = float(source_k[0][0]) * scale
        fy = float(source_k[1][1]) * scale
        cx = float(source_k[0][2]) * scale + offset_x
        cy = float(source_k[1][2]) * scale + offset_y
        camera["width"] = target_width
        camera["height"] = target_height
        camera["K"] = [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ]
        camera["source_width"] = actual_width
        camera["source_height"] = actual_height
        camera["source_K"] = source_k
    return np.stack(images, axis=0)


def build_sparse_points(cameras: list[dict], count: int = 2400) -> list[list[float]]:
    """Build an initialization cloud that follows the observed camera rays.

    The cloud is only used for near/far estimation; TCF itself is implicit
    and does not require an explicit primitive initialization.
    """
    count = max(int(count), 64)
    rng = np.random.default_rng(7)
    cameras_np = np.asarray(
        [np.asarray(camera["c2w"], dtype=np.float32) for camera in cameras]
    )
    if cameras_np.ndim != 3 or cameras_np.shape[0] < 2:
        directions = rng.normal(size=(count, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = 2.0 + 0.08 * rng.normal(size=(count, 1))
        points = directions * radii
        colors = (sphere_color_array(points) * 255.0).astype(np.uint8)
        return [
            [float(x), float(y), float(z), int(r), int(g), int(b)]
            for (x, y, z), (r, g, b) in zip(points, colors)
        ]

    positions = cameras_np[:, :3, 3]
    forwards = cameras_np[:, :3, 2]
    centroid = np.mean(positions, axis=0)
    typical_depth = float(
        np.median(
            np.linalg.norm(positions - centroid[None, :], axis=1)
        )
    )
    if typical_depth <= 1e-6:
        typical_depth = 1.0

    def triangulate_pair(index_a: int, index_b: int) -> np.ndarray:
        origin_a, forward_a = positions[index_a], forwards[index_a]
        origin_b, forward_b = positions[index_b], forwards[index_b]
        origin_b -= origin_a
        dot_aa = float(np.dot(forward_a, forward_a))
        dot_bb = float(np.dot(forward_b, forward_b))
        dot_ab = float(np.dot(forward_a, forward_b))
        denom = dot_aa * dot_bb - dot_ab * dot_ab
        if abs(denom) < 1e-8:
            return np.empty((0, 3), dtype=np.float32)
        origin_b_dot_a = float(np.dot(origin_b, forward_a))
        origin_b_dot_b = float(np.dot(origin_b, forward_b))
        t_a = (dot_bb * origin_b_dot_a - dot_ab * origin_b_dot_b) / denom
        t_b = (dot_ab * origin_b_dot_a - dot_aa * origin_b_dot_b) / denom
        if t_a < 0.05 or t_b < 0.05:
            return np.empty((0, 3), dtype=np.float32)
        point = origin_a + forward_a * t_a
        return np.asarray([point], dtype=np.float32)

    triangulated: list[np.ndarray] = []
    cameras_count = len(cameras_np)
    step = max(1, cameras_count // 40)
    pair_count = 0
    for index_a in range(0, cameras_count, step):
        index_b = (index_a + cameras_count // 2) % cameras_count
        if index_b == index_a:
            continue
        point = triangulate_pair(index_a, index_b)
        if point.size == 0:
            continue
        depth_a = float(
            np.dot(point[0] - positions[index_a], forwards[index_a])
        )
        depth_b = float(
            np.dot(point[0] - positions[index_b], forwards[index_b])
        )
        lateral_a = float(
            np.linalg.norm(
                point[0] - positions[index_a] - forwards[index_a] * depth_a
            )
        )
        lateral_b = float(
            np.linalg.norm(
                point[0] - positions[index_b] - forwards[index_b] * depth_b
            )
        )
        if depth_a < 0.05 or depth_b < 0.05:
            continue
        if lateral_a > typical_depth * 0.6 or lateral_b > typical_depth * 0.6:
            continue
        if max(depth_a, depth_b) > typical_depth * 4.0:
            continue
        triangulated.append(point)
        pair_count += 1
        if pair_count >= count // 2:
            break

    if triangulated:
        anchors = np.concatenate(triangulated, axis=0)
        radius = float(np.median(np.linalg.norm(anchors - centroid, axis=1)))
        if radius <= 1e-6:
            radius = typical_depth * 0.4
        spread = max(radius * 0.12, typical_depth * 0.02)
        indices = rng.integers(0, len(anchors), size=count)
        points = anchors[indices] + rng.normal(scale=spread, size=(count, 3))
    else:
        shells: list[np.ndarray] = []
        per_camera = max(4, count // max(cameras_count, 1))
        for index in range(cameras_count):
            depth = typical_depth * rng.uniform(0.55, 1.15, size=per_camera)
            shells.append(
                positions[index][None, :] + forwards[index][None, :] * depth[:, None]
            )
        points = np.concatenate(shells, axis=0)[:count]

    colors = (sphere_color_array(points) * 255.0).astype(np.uint8)
    return [
        [float(x), float(y), float(z), int(r), int(g), int(b)]
        for (x, y, z), (r, g, b) in zip(points, colors)
    ]


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


def orbit_camera_around_scene(
    angle: float,
    size: int,
    scene_points: np.ndarray | None = None,
    cameras: list[dict] | None = None,
) -> dict:
    """Build a novel view that fits the observed scene instead of a fixed orbit."""
    center = np.zeros(3, dtype=np.float64)
    radius = 5.4
    if scene_points is not None and len(scene_points):
        points = np.asarray(scene_points, dtype=np.float64).reshape(-1, 3)
        center = np.median(points, axis=0)
        lo = np.quantile(points, 0.1, axis=0)
        hi = np.quantile(points, 0.9, axis=0)
        radius = max(float(np.linalg.norm(hi - lo)) * 0.65, 0.25)
    if cameras:
        origins = np.asarray(
            [
                np.asarray(camera["c2w"], dtype=np.float64)[:3, 3]
                for camera in cameras
            ],
            dtype=np.float64,
        ).reshape(-1, 3)
        camera_dist = np.linalg.norm(origins - center[None, :], axis=1)
        radius = max(
            radius,
            float(np.quantile(camera_dist, 0.9)) * 1.4,
            float(np.quantile(camera_dist, 0.5)) * 1.8,
        )
    elevation = 0.16
    position = center + np.asarray(
        [
            math.cos(angle) * math.cos(elevation),
            math.sin(elevation),
            math.sin(angle) * math.cos(elevation),
        ],
        dtype=np.float64,
    ) * radius
    forward = center - position
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
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
    pose_source: str,
    cameras: list[dict],
    sparse_points: list[list[float]],
    dense_points: list[float],
    final_metrics: dict,
    config: dict,
    history: list[dict],
    render_paths: list[dict],
    novel_path: str,
    engine: str = "tcf",
) -> None:
    scene = {
        "version": "0.2.0",
        "engine": engine,
        "taskId": task_id,
        "projectDir": str(project_dir),
        "poseSource": pose_source,
        "points": sparse_points,
        "densePoints": dense_points,
        "densePointCount": len(dense_points) // 6,
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
        json.dumps(scene, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
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
    write_control(project_dir, "run")
    emit_status(
        project_dir,
        {"taskId": task_id, "status": "running", "stage": "data", "progress": 0.02},
    )

    data = load_pose_data(
        project_dir,
        config,
        status_cb=lambda progress, message: emit_status(
            project_dir,
            {
                "taskId": task_id,
                "status": "running",
                "stage": "sfm",
                "progress": progress,
                "message": f"COLMAP {message}",
            },
        ),
    )
    if not data["images"]:
        raise RuntimeError("项目没有可用图片或位姿")
    images = load_and_prepare_images(data["images"], data["cameras"], config)
    train_resolution_note = (
        "（原分辨率）" if config.get("train_original_resolution") else ""
    )
    log(
        f"[data] 图片 {len(data['images'])} 张，"
        f"训练尺寸 {images.shape[1]}x{images.shape[2]}{train_resolution_note}"
    )

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    sparse_points = data["points"] or build_sparse_points(data["cameras"])
    log(f"[data] 稀疏点云 {len(sparse_points)} 点，设备 {device_name}")

    trained_result, history = train_tcf(
        images,
        data["cameras"],
        config,
        project_dir,
        task_id=task_id,
        device_name=device_name,
        sparse_points=sparse_points,
        status_cb=lambda payload: emit_status(project_dir, payload),
    )
    if trained_result.get("cancelled"):
        return 0
    trained = trained_result["model"]

    scene_points_np = (
        np.asarray(sparse_points, dtype=np.float32)[:, :3]
        if sparse_points
        else np.empty((0, 3), dtype=np.float32)
    )
    shift = trained_result["shift"]
    scale = float(trained_result["scale"])
    radius = float(trained_result["radius"])
    ray_batch = int(config.get("ray_batch_size", 4096))
    coarse_samples = int(config.get("coarse_samples", 32))
    fine_samples = int(config.get("fine_samples", 16))
    use_amp = bool(config.get("mixed_precision", True))

    emit_status(
        project_dir,
        {
            "taskId": task_id,
            "status": "running",
            "stage": "extract",
            "progress": 0.88,
            "message": "正在提取三维稠密点云",
        },
    )
    dense_points = extract_dense_points(
        trained,
        data["cameras"],
        shift,
        scale,
        scene_points_np,
        radius,
        torch.device(device_name),
        stride=int(config.get("dense_extract_stride", 4)),
        ray_batch=ray_batch,
        coarse_samples=coarse_samples,
        fine_samples=fine_samples,
        visibility_threshold=float(
            config.get("dense_visibility_threshold", 0.2)
        ),
        max_points=int(config.get("dense_extract_max_points", 200000)),
        use_amp=use_amp,
    )
    log(f"[extract] 稠密点云 {len(dense_points) // 6} 点")

    emit_status(
        project_dir,
        {
            "taskId": task_id,
            "status": "running",
            "stage": "render",
            "progress": 0.92,
        },
    )
    render_paths: list[dict] = []

    train_count = max(1, int(config.get("train_views", 16)))
    train_cameras = data["cameras"][:train_count]
    if not train_cameras:
        train_cameras = data["cameras"][:1]
    for camera_index, camera in enumerate(train_cameras):
        rendered = render_tcf_view(
            trained,
            camera,
            shift,
            scale,
            scene_points_np,
            radius,
            ray_batch=ray_batch,
            coarse_samples=coarse_samples,
            fine_samples=fine_samples,
            use_amp=use_amp,
        )
        path = write_render(project_dir, f"render_train_{camera_index:03d}.png", rendered)
        render_paths.append(
            {"name": f"render_train_{camera_index:03d}.png", "path": path, "view": "train"}
        )

    heldout = data["cameras"][train_count:]
    if not heldout:
        heldout = data["cameras"][-1:]
    heldout_scores: list[dict] = []
    for index, camera in enumerate(heldout):
        rendered = render_tcf_view(
            trained,
            camera,
            shift,
            scale,
            scene_points_np,
            radius,
            ray_batch=ray_batch,
            coarse_samples=coarse_samples,
            fine_samples=fine_samples,
            use_amp=use_amp,
        )
        path = write_render(project_dir, f"render_eval_{index:03d}.png", rendered)
        render_paths.append(
            {"name": f"render_eval_{index:03d}.png", "path": path, "view": "eval"}
        )
        target_index = len(data["cameras"]) - len(heldout) + index
        target = images[target_index]
        metric_rendered = rendered
        if metric_rendered.shape[:2] != target.shape[:2]:
            metric_rendered = (
                np.asarray(
                    Image.fromarray(
                        (np.clip(metric_rendered, 0.0, 1.0) * 255.0).astype(np.uint8)
                    ).resize((target.shape[1], target.shape[0]), Image.LANCZOS),
                    dtype=np.float32,
                )
                / 255.0
            )
        heldout_scores.append(compute_metrics(metric_rendered, target))

    novel = orbit_camera_around_scene(
        0.7,
        int(config.get("resolution", 192)),
        scene_points=scene_points_np,
        cameras=data["cameras"],
    )
    rendered = render_tcf_view(
        trained,
        novel,
        shift,
        scale,
        scene_points_np,
        radius,
        ray_batch=ray_batch,
        coarse_samples=coarse_samples,
        fine_samples=fine_samples,
        use_amp=use_amp,
    )
    novel_path = write_render(project_dir, "novel_orbit.png", rendered)

    final_metrics = {
        "psnr": float(np.mean([item["psnr"] for item in heldout_scores])),
        "ssim": float(np.mean([item["ssim"] for item in heldout_scores])),
        "loss": float(np.mean([item["loss"] for item in heldout_scores])),
        "vramMB": int(history[-1]["vramMB"]) if history else 0,
        "latticeMB": float(trained_result.get("latticeMB", 0.0)),
        "peakVramMB": int(trained_result.get("peakVramMB", 0)),
        "views": len(heldout_scores),
    }
    engine = trained_result.get("engine", "tcf")
    write_scene_output(
        project_dir,
        task_id,
        str(data.get("source", "orbit")),
        data["cameras"],
        sparse_points,
        dense_points,
        final_metrics,
        config,
        history,
        render_paths,
        novel_path,
        engine=engine,
    )
    log("[done] TCF 端到端重建完成")
    return 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Titan Coherence Field engine")
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
        sys.exit(1)


if __name__ == "__main__":
    main()
