"""Temporary memory/time baseline for tile rendering (not committed)."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
import torch

from titan_engine.splat_engine import SplatFacetField, _render_pixels


def make_scene(width: int, height: int, count: int = 30_000):
    torch.manual_seed(5)
    rng = np.random.default_rng(9)
    points = rng.uniform(-0.6, 0.6, size=(count, 3)).astype(np.float32)
    points[:, 2] += 3.0
    colors = rng.uniform(0.2, 0.8, size=(count, 3)).astype(np.float32)
    scales = rng.uniform(0.01, 0.05, size=(count,)).astype(np.float32)
    model = SplatFacetField(points, colors=colors, scales=scales, opacity=0.6)
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    camera = {
        "width": width,
        "height": height,
        "K": [
            [width * 1.2, 0.0, width / 2.0],
            [0.0, width * 1.2, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        "c2w": np.eye(4).tolist(),
    }
    background = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32).to(device)
    return model, camera, background, device


def run(label: str, width: int, height: int, grad: bool, budget: int):
    model, camera, background, device = make_scene(width, height)
    target = torch.rand_like(background).repeat(
        width * height, 1
    )
    target = torch.randn(height * width, 3, device=device) * 0.1
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    start = time.perf_counter()
    rendered = _render_pixels(
        model,
        camera,
        None,
        width,
        height,
        background,
        tile_size=16,
        auto_dense_threshold=0,
        tile_pair_budget=budget,
        tile_max_splats=8192,
    )
    if grad:
        loss = torch.mean((rendered - target) ** 2)
        loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if torch.cuda.is_available()
        else 0.0
    )
    reserved = (
        torch.cuda.memory_reserved(device) / (1024 * 1024)
        if torch.cuda.is_available()
        else 0.0
    )
    print(
        f"{label} {width}x{height} grad={grad} budget={budget}: "
        f"{elapsed * 1000:.1f} ms, peak {peak:.0f} MiB, reserved {reserved:.0f} MiB"
    )


if __name__ == "__main__":
    run("patch", 1920, 1080, True, 4_000_000)
    run("patch", 1920, 1080, True, 2_000_000)
    run("full", 1920, 1080, True, 4_000_000)
    run("full", 1920, 1080, True, 2_000_000)
