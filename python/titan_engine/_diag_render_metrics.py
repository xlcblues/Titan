"""One-off diagnostic: compare horns renders against the aligned targets."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from titan_engine.main import compute_metrics, load_and_prepare_images


def load_render(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Titan render diagnostics")
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--task-id", default="")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    pose = json.loads((project_dir / "pose.json").read_text(encoding="utf-8"))
    cameras = pose["cameras"]
    paths = [camera["path"] for camera in cameras]
    images = load_and_prepare_images(paths, cameras, {"resolution": 192})
    scene = json.loads((project_dir / "scene.json").read_text(encoding="utf-8"))
    train_count = int(scene.get("config", {}).get("train_views", 24))
    print(f"task {args.task_id or scene.get('taskId')} train_views {train_count}")

    print("== training renders ==")
    for index in (0, 2, 4):
        render = load_render(project_dir / "renders" / f"render_train_{index:03d}.png")
        metrics = compute_metrics(render, images[index])
        print(f"train {index}: {metrics}")

    print("== heldout render stats ==")
    scores = []
    for index in range(len(cameras) - train_count):
        render = load_render(
            project_dir / "renders" / f"render_eval_{index:03d}.png"
        )
        target = images[train_count + index]
        metrics = compute_metrics(render, target)
        metrics["view"] = index
        metrics["render_mean"] = float(render.mean())
        metrics["render_std"] = float(render.std())
        metrics["target_mean"] = float(target.mean())
        metrics["target_std"] = float(target.std())
        scores.append(metrics)
    sorted_scores = sorted(scores, key=lambda item: item["psnr"])
    print(f"count {len(scores)} mean psnr {np.mean([s['psnr'] for s in scores]):.3f}")
    print("worst 6:")
    for item in sorted_scores[:6]:
        print(item)
    print("best 6:")
    for item in sorted_scores[-6:]:
        print(item)


if __name__ == "__main__":
    main()
