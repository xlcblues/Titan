"""End-to-end check: COLMAP pose estimation -> full-res training -> render."""

import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np

from titan_engine.main import (
    load_and_prepare_images,
    load_pose_data,
    render_view,
    write_render,
)
from titan_engine.splat_engine import train_tfrf_splat


def main() -> None:
    source_images = json.loads(
        Path(r"D:\code\Titan\.tmp_titan_horns\inputs.json").read_text(
            encoding="utf-8"
        )
    )["images"]
    project = Path(r"D:\code\Titan\.tmp_titan_pipeline_e2e")
    if project.exists():
        shutil.rmtree(project)
    project.mkdir(parents=True)
    image_dir = project / "images"
    image_dir.mkdir()
    selected = source_images[:10]
    paths = []
    for index, source in enumerate(selected):
        target = image_dir / f"cam_{index:02d}.png"
        shutil.copy2(source, target)
        paths.append(str(target))
    (project / "inputs.json").write_text(
        json.dumps({"images": paths}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    config = {
        "pose_source": "auto",
        "colmap_path": r"F:\colmap\colmap-x64-windows-cuda\bin\colmap.exe",
        "sfm_min_images": 10,
        "sfm_min_registered": 10,
        "sfm_max_image_size": 1600,
        "sfm_matcher": "sequential",
        "train_original_resolution": True,
        "train_max_image_pixels": 2_000_000,
        "train_views": 8,
        "steps": 3,
        "eval_every": 100000,
        "progress_every": 1,
        "densify": False,
        "opacity_reset_after": 100000,
        "tile_pair_budget": 4_000_000,
        "tile_work_budget": 32_000_000,
        "tile_max_splats": 8192,
        "render_tile_size": 16,
        "use_cuda_tile": True,
        "disk_cutoff": 1.8,
        "falloff_power": 1.0,
    }
    started = time.time()
    data = load_pose_data(
        project,
        config,
        status_cb=lambda progress, message: print(
            f"[e2e] colmap {progress:.2f} {message}", flush=True
        ),
    )
    print(
        f"[e2e] pose source {data['source']}, {len(data['cameras'])} cameras, "
        f"{len(data['points'])} points ({time.time() - started:.1f}s)"
    )
    if data["source"] != "colmap":
        raise AssertionError("expected COLMAP pose source")

    images = load_and_prepare_images(data["images"], data["cameras"], config)
    print(f"[e2e] images {images.shape}")
    result, _ = train_tfrf_splat(
        images,
        data["cameras"],
        config,
        project,
        task_id="e2e",
        sparse_points=data["points"],
    )
    model = result["model"]
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
    path = write_render(project, "render_e2e.png", rendered)
    print(
        f"[e2e] rendered {rendered.shape} mean {float(rendered.mean()):.4f} "
        f"-> {path}"
    )
    print("[e2e] ok")


if __name__ == "__main__":
    main()
