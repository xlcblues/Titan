"""Temporary phase profiler for the tile rasterizer (not committed)."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import inspect

import numpy as np
import torch

import titan_engine.splat_engine as se


def _instrument():
    source = inspect.getsource(se._render_pixels_tile)
    markers = [
        ("project", "tile_u = tile_keys % tiles_x"),
        ("broad_counts", "tile_entry_counts = small_counts + broad_counts"),
        ("segments", "for segment_index in range(len(segment_edges) - 1):"),
        ("pairs", "if chunk_splats.numel() == 0 or p1 <= p0:"),
        ("sort", "chunk_z = splat_z[chunk_splats]"),
        ("group_loop", "for group_index in range(len(group_edges) - 1):"),
        ("pair_expand", "pair_u = group_u[pair_splat]"),
        ("composite", "before, segment_end, segment_total = _segment_log_cumsum("),
        ("write", "return output.reshape(output_shape)"),
    ]
    first_stmt = True
    lines = source.splitlines()
    out = []
    phase_idx = 0
    for line in lines:
        stripped = line.strip()
        if first_stmt and stripped:
            if not (stripped.startswith('"""') or stripped.startswith("'''")):
                indent = len(line) - len(line.lstrip())
                out.append(
                    f'{" " * indent}_t = {{}}; _t0 = time.perf_counter()'
                )
                first_stmt = False
            elif stripped.endswith('"""') or stripped.endswith("'''"):
                first_stmt = False
        marker_text = markers[phase_idx][1]
        indent = len(line) - len(line.lstrip())
        if phase_idx < len(markers) and marker_text in line:
            name = markers[phase_idx][0]
            phase_idx += 1
            out.append(
                f'{" " * indent}torch.cuda.synchronize(); _t[name] = time.perf_counter() - _t0; _t0 = time.perf_counter()'
            )
        out.append(line)
    ns = dict(se.__dict__)
    exec(compile("\n".join(out), "<instrumented>", "exec"), ns)
    return ns["_render_pixels_tile"]


def main():
    torch.manual_seed(5)
    rng = np.random.default_rng(9)
    count = 30_000
    points = rng.uniform(-0.6, 0.6, size=(count, 3)).astype(np.float32)
    points[:, 2] += 3.0
    colors = rng.uniform(0.2, 0.8, size=(count, 3)).astype(np.float32)
    scales = rng.uniform(0.01, 0.05, size=(count,)).astype(np.float32)
    model = se.SplatFacetField(points, colors=colors, scales=scales, opacity=0.6)
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
    instrumented = _instrument()
    se._render_pixels_tile = instrumented
    t = {}
    t0 = time.perf_counter()
    with torch.no_grad():
        image = se._render_pixels(
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
    total = time.perf_counter() - t0
    print(f"total {total * 1000:.1f} ms, image mean {image.mean().item():.4f}")
    for name, dt in t.items():
        print(f"{name}: {dt * 1000:.1f} ms ({dt / total * 100:.1f}%)")


if __name__ == "__main__":
    main()
