"""Regression check for the tile rasterizer against the dense reference."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
import torch

from titan_engine.splat_engine import (
    SplatFacetField,
    _render_pixels,
    _render_pixels_dense,
    _render_pixels_tile,
)


def build_model(n=600, seed=7):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-0.5, 0.5, size=(n, 3)).astype(np.float32)
    points[:, 2] += 3.0
    colors = rng.uniform(0.2, 0.8, size=(n, 3)).astype(np.float32)
    scales = rng.uniform(0.02, 0.10, size=(n,)).astype(np.float32)
    return SplatFacetField(points, colors=colors, scales=scales, opacity=0.7)


def build_camera(width=32, height=32):
    fx = fy = float(width) * 1.2
    cx = width / 2.0
    cy = height / 2.0
    c2w = np.eye(4, dtype=np.float64)
    return {
        "width": width,
        "height": height,
        "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "c2w": c2w.tolist(),
    }


def max_diff(a, b):
    return float((a - b).abs().max().item())


def main():
    torch.manual_seed(3)
    model = build_model()
    camera = build_camera()
    background = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32)
    uu, vv = torch.meshgrid(
        torch.arange(32, dtype=torch.float32),
        torch.arange(32, dtype=torch.float32),
        indexing="xy",
    )
    full = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
    rng = np.random.default_rng(11)
    random_pixels = torch.as_tensor(
        rng.uniform(0, 31, size=(64, 2)), dtype=torch.float32
    )

    dense_random = None
    for label, pixels in (("random", random_pixels), ("full", full)):
        dense = _render_pixels_dense(
            model, camera, pixels, camera["width"], camera["height"], background
        )
        if label == "random":
            dense_random = dense
        for tile_size in (32, 16, 8):
            tile = _render_pixels_tile(
                model,
                camera,
                pixels,
                camera["width"],
                camera["height"],
                background,
                tile_size=tile_size,
            )
            diff = max_diff(dense, tile)
            assert diff < 1e-5, f"{label} tile={tile_size} max diff {diff}"

    auto = _render_pixels(
        model,
        camera,
        random_pixels,
        camera["width"],
        camera["height"],
        background,
    )
    assert max_diff(auto, dense_random) < 1e-5

    tiled = _render_pixels(
        model,
        camera,
        random_pixels,
        camera["width"],
        camera["height"],
        background,
        auto_dense_threshold=0,
    )
    tile_ref = _render_pixels_tile(
        model,
        camera,
        random_pixels,
        camera["width"],
        camera["height"],
        background,
        tile_size=16,
    )
    assert max_diff(tiled, tile_ref) < 1e-5

    single = SplatFacetField(
        np.array([[0.0, 0.0, 3.0]], dtype=np.float32),
        colors=np.array([[0.6, 0.4, 0.2]], dtype=np.float32),
        scales=np.array([0.05], dtype=np.float32),
        opacity=0.7,
    )
    dense_single = _render_pixels_dense(
        single, camera, full, 32, 32, background
    )
    tile_single = _render_pixels_tile(
        single, camera, full, 32, 32, background, tile_size=32
    )
    assert max_diff(dense_single, tile_single) < 1e-5

    print("renderer regression ok")


if __name__ == "__main__":
    main()
