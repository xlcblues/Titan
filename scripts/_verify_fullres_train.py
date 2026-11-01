"""Short end-to-end verification for full-resolution CUDA tile training."""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
import torch

from titan_engine.main import (
    load_and_prepare_images,
    load_pose_data,
    render_view,
    write_render,
)
from titan_engine.splat_engine import train_tfrf_splat


def main() -> None:
    project = Path(r"D:\code\Titan\.tmp_titan_horns")
    out_dir = project / "renders"
    config = {
        "pose_source": "cached",
        "train_original_resolution": True,
        "train_max_image_pixels": 2_000_000,
        "train_views": 16,
        "steps": 8,
        "eval_every": 100000,
        "progress_every": 1,
        "densify": True,
        "densify_after": 2,
        "densify_every": 2,
        "opacity_reset_after": 100000,
        "max_splats": 50000,
        "patch_size": 1024,
        "tile_pair_budget": 4_000_000,
        "tile_work_budget": 32_000_000,
        "tile_max_splats": 8192,
        "render_tile_size": 16,
        "use_cuda_tile": True,
        "disk_cutoff": 1.8,
        "falloff_power": 1.0,
    }
    data = load_pose_data(project, config)
    images = load_and_prepare_images(data["images"], data["cameras"], config)
    print(
        f"[verify] images {images.shape} cameras {len(data['cameras'])} "
        f"points {len(data['points'])}"
    )
    before = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    result, history = train_tfrf_splat(
        images,
        data["cameras"],
        config,
        project,
        task_id="verify_fullres",
        sparse_points=data["points"],
    )
    elapsed = time.time() - started
    model = result["model"]
    reserved_mb = torch.cuda.memory_reserved() // (1024 * 1024)
    peak_alloc_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
    print(
        f"[verify] train {config['steps']} steps in {elapsed:.2f}s "
        f"(~{elapsed / max(config['steps'], 1):.3f}s/step), "
        f"reserved {reserved_mb} MiB, peak {peak_alloc_mb} MiB, "
        f"splats {len(model.positions)}"
    )
    torch.cuda.reset_peak_memory_stats()
    camera = data["cameras"][0]
    rendered = render_view(
        model,
        camera,
        np.array([0.08, 0.09, 0.10], dtype=np.float32),
        scene_points=(
            np.asarray(data["points"], dtype=np.float32)[:, :3]
            if data["points"]
            else None
        ),
        tile_pair_budget=config["tile_pair_budget"],
        tile_max_splats=config["tile_max_splats"],
        tile_work_budget=config["tile_work_budget"],
        use_cuda_tile=True,
    )
    render_peak_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
    path = write_render(project, "verify_fullres.png", rendered)
    print(
        f"[verify] rendered {rendered.shape} mean {float(rendered.mean()):.4f} "
        f"std {float(rendered.std()):.4f} render_peak {render_peak_mb} MiB -> {path}"
    )
    print("[verify] ok")


if __name__ == "__main__":
    main()
