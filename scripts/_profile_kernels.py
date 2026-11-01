"""Temporary torch.profiler breakdown for tile rendering (not committed)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
import torch

from titan_engine.splat_engine import SplatFacetField, _render_pixels


def main():
    torch.manual_seed(5)
    rng = np.random.default_rng(9)
    count = 30_000
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
    width, height = 1920, 1080
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
    with torch.no_grad():
        for _ in range(2):
            _render_pixels(
                model,
                camera,
                None,
                width,
                height,
                background,
                tile_size=16,
                auto_dense_threshold=0,
                tile_pair_budget=4_000_000,
                tile_max_splats=8192,
            )
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            _render_pixels(
                model,
                camera,
                None,
                width,
                height,
                background,
                tile_size=16,
                auto_dense_threshold=0,
                tile_pair_budget=4_000_000,
                tile_max_splats=8192,
            )
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))


if __name__ == "__main__":
    main()
