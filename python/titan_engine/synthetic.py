import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image


def _hash_noise(x: float, y: float, z: float) -> float:
    value = math.sin(x * 12.9898 + y * 78.233 + z * 37.719) * 43758.5453
    return value - math.floor(value)


def _hash_noise_array(cells: np.ndarray) -> np.ndarray:
    x = cells[..., 0]
    y = cells[..., 1]
    z = cells[..., 2]
    value = np.sin(x * 12.9898 + y * 78.233 + z * 37.719) * 43758.5453
    return value - np.floor(value)


def sphere_color(direction: np.ndarray) -> np.ndarray:
    x, y, z = direction
    base = np.array([0.58, 0.66, 0.70])
    gradient = np.array([0.22, 0.60, 0.55])
    blend = 0.35 + 0.3 * math.sin(x * 5.0) * math.cos(y * 4.0) * math.sin(z * 6.0)
    color = base + gradient * (blend - 0.35)
    cell = np.floor(np.array([x, y, z]) * 18.0 + 512.0)
    h = _hash_noise(float(cell[0]), float(cell[1]), float(cell[2]))
    if h < 0.06:
        color = np.array([0.92, 0.58, 0.30])
    elif h < 0.12:
        color = np.array([0.28, 0.22, 0.32])
    shading = 0.75 + 0.45 * max(0.0, float(np.dot(direction, np.array([0.3, 1.0, 0.5]))))
    return np.clip(color * shading, 0.0, 1.0)


def sphere_color_array(direction: np.ndarray) -> np.ndarray:
    x = direction[..., 0]
    y = direction[..., 1]
    z = direction[..., 2]
    base = np.array([0.58, 0.66, 0.70])
    gradient = np.array([0.22, 0.60, 0.55])
    blend = 0.35 + 0.3 * np.sin(x * 5.0) * np.cos(y * 4.0) * np.sin(z * 6.0)
    color = base + gradient * (blend - 0.35)[..., None]
    cells = np.floor(np.stack([x, y, z], axis=-1) * 18.0 + 512.0)
    noise = _hash_noise_array(cells)
    color = np.where(noise[..., None] < 0.06, np.array([0.92, 0.58, 0.30]), color)
    color = np.where(noise[..., None] < 0.12, np.array([0.28, 0.22, 0.32]), color)
    shading = 0.75 + 0.45 * np.maximum(
        0.0, x * 0.3 + y * 1.0 + z * 0.5
    )[..., None]
    return np.clip(color * shading, 0.0, 1.0)


def render_sphere(size: int, camera: dict) -> np.ndarray:
    width, height = camera["width"], camera["height"]
    k = camera["K"]
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    c2w = np.array(camera["c2w"])
    origin = c2w[:3, 3]
    R = c2w[:3, :3]
    image = np.zeros((height, width, 3), dtype=np.float32)
    sphere_center = np.array([0.0, 0.0, 0.0])
    radius = 2.0
    for y in range(height):
        for x in range(width):
            ray_local = np.array(
                [(x - cx) / fx, (y - cy) / fy, 1.0], dtype=np.float64
            )
            ray_world = R @ ray_local
            ray_world /= np.linalg.norm(ray_world)
            oc = origin - sphere_center
            b = float(np.dot(oc, ray_world))
            c = float(np.dot(oc, oc)) - radius * radius
            disc = b * b - c
            if disc <= 0:
                image[y, x] = np.array([0.08, 0.09, 0.10])
                continue
            t = -b - math.sqrt(disc)
            hit = origin + t * ray_world
            normal = (hit - sphere_center) / radius
            image[y, x] = sphere_color(normal)
    return image


def render_sphere_array(size: int, camera: dict) -> np.ndarray:
    width, height = camera["width"], camera["height"]
    k = camera["K"]
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    c2w = np.array(camera["c2w"])
    origin = c2w[:3, 3]
    rotation = c2w[:3, :3]
    ys, xs = np.mgrid[0:height, 0:width]
    ray_local = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)], axis=-1)
    ray_world = ray_local @ rotation.T
    ray_world = ray_world / np.linalg.norm(ray_world, axis=-1, keepdims=True)
    oc = origin - np.array([0.0, 0.0, 0.0])
    b = np.sum(oc * ray_world, axis=-1)
    c = np.sum(oc * oc, axis=-1) - 4.0
    c = np.broadcast_to(c[..., None], b.shape)
    disc = b * b - c
    hit_mask = disc > 0
    t = -b - np.sqrt(np.maximum(disc, 0.0))
    hit = origin[None, None, :] + t[..., None] * ray_world
    normal = hit / 2.0
    color = sphere_color_array(normal)
    background = np.array([0.08, 0.09, 0.10])
    image = np.where(hit_mask[..., None], color, background)
    return image.astype(np.float32)


def generate_scene(
    out_dir: Path,
    count: int = 24,
    size: int = 256,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    focal = size * 1.15
    k = [
        [focal, 0.0, size / 2.0],
        [0.0, focal, size / 2.0],
        [0.0, 0.0, 1.0],
    ]
    cameras = []
    images = []
    for index in range(count):
        angle = (index / count) * math.pi * 1.9
        position = np.array(
            [5.4 * math.cos(angle), 1.15 + 0.6 * math.sin(angle * 3.0), 5.4 * math.sin(angle)],
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
        camera = {
            "path": f"img_{index:03d}.png",
            "width": size,
            "height": size,
            "K": k,
            "c2w": c2w.tolist(),
        }
        cameras.append(camera)
        rgb = render_sphere_array(size, camera)
        image_path = out_dir / camera["path"]
        Image.fromarray((rgb * 255.0).clip(0, 255).astype(np.uint8)).save(image_path)
        images.append(str(image_path))

    rng = np.random.default_rng(7)
    points = []
    for _ in range(2400):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        point = direction * (2.0 + 0.08 * rng.normal())
        color = sphere_color(direction)
        points.append(
            [float(point[0]), float(point[1]), float(point[2]), *[int(v * 255) for v in color]]
        )

    payload = {"cameras": cameras, "points": points}
    (out_dir / "pose.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return {"images": images, "pose_file": str(out_dir / "pose.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic TFRF test scene")
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    result = generate_scene(Path(args.out), count=args.count, size=args.size)
    print(f"images={len(result['images'])} pose={result['pose_file']}")


if __name__ == "__main__":
    main()
