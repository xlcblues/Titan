"""Run COLMAP SfM on a set of photos and convert the result to Titan poses."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from .colmap_io import load_colmap_sparse, read_images_bin, read_points3d_bin


StatusCallback = Callable[[float, str], None]


def find_colmap_executable(config: dict | None = None) -> str | None:
    config = config or {}
    configured = str(config.get("colmap_path", "")).strip()
    if configured:
        candidate = Path(configured)
        if candidate.is_file():
            return str(candidate)
        if candidate.is_dir():
            exe = candidate / "colmap.exe"
            if exe.is_file():
                return str(exe)

    for variable in ("COLAMP_EXECUTABLE", "COLMAP_EXECUTABLE"):
        value = os.environ.get(variable, "").strip()
        if value:
            candidate = Path(value)
            if candidate.is_file():
                return str(candidate)
            if candidate.is_dir():
                exe = candidate / "colmap.exe"
                if exe.is_file():
                    return str(exe)

    found = shutil.which("colmap")
    if found:
        return found

    roots = [
        Path.home() / "colmap" / "colmap-x64-windows-cuda" / "bin" / "colmap.exe",
        Path.home() / "colmap" / "bin" / "colmap.exe",
        Path("F:/colmap/colmap-x64-windows-cuda/bin/colmap.exe"),
        Path("C:/colmap/colmap-x64-windows-cuda/bin/colmap.exe"),
        Path("D:/colmap/colmap-x64-windows-cuda/bin/colmap.exe"),
    ]
    for root in roots:
        if root.is_file():
            return str(root)
    return None


def _tail(path: Path, lines: int = 80) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(raw.splitlines()[-lines:])


def _run(
    colmap_exe: str,
    command: str,
    args: list[str],
    log_path: Path,
    timeout_minutes: int = 60,
) -> int:
    cmd = [colmap_exe, command, "--log_level", "2"] + args
    creation_flags = 0x08000000 if sys.platform == "win32" else 0
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {command} {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        try:
            result = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
                timeout=timeout_minutes * 60,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log.write(f"\n=== {command} timed out after {timeout_minutes} minutes ===\n")
            return 124
        except OSError as error:
            log.write(f"\n=== failed to start {command}: {error} ===\n")
            return 127
    return result.returncode


def _best_model_dir(sparse_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for child in sorted(sparse_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "images.bin").is_file() and (child / "cameras.bin").is_file():
            candidates.append(child)
    if not candidates:
        return None

    def score(model_dir: Path) -> tuple[int, int]:
        try:
            image_count = len(read_images_bin(model_dir / "images.bin"))
        except Exception:
            image_count = 0
        points_file = model_dir / "points3D.bin"
        point_count = 0
        if points_file.is_file():
            try:
                point_count = len(read_points3d_bin(points_file))
            except Exception:
                point_count = 0
        return image_count, point_count

    return max(candidates, key=score)


def _copy_images(workspace: Path, image_paths: list[str]) -> tuple[list[Path], dict[str, str]]:
    """Copy photos into COLMAP's image folder with unique names.

    Returns the copied paths and a map from COLMAP image name to the original
    absolute image path.
    """
    images_dir = workspace / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    name_to_path: dict[str, str] = {}
    seen_hashes: dict[int, Path] = {}
    for index, source in enumerate(image_paths):
        source_path = Path(source)
        suffix = source_path.suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            suffix = ".jpg"
        unique = False
        candidate = None
        while not unique:
            candidate = images_dir / f"{index:05d}_{source_path.stem}{suffix}"
            if not candidate.exists():
                unique = True
        try:
            shutil.copy2(source_path, candidate)
        except OSError as error:
            raise RuntimeError(f"复制图片 {source_path} 到 COLMAP 工作目录失败：{error}") from error
        copied.append(candidate)
        name_to_path[candidate.name] = str(source_path)
    return copied, name_to_path


def run_colmap_sfm(
    project_dir: Path,
    image_paths: list[str],
    config: dict,
    status_cb: StatusCallback | None = None,
) -> dict | None:
    """Run COLMAP SfM and return Titan camera/point data.

    Returns None when SfM could not produce a usable reconstruction (for
    example because COLMAP is not installed or the image set is too small).
    """
    colmap_exe = find_colmap_executable(config)
    if not colmap_exe:
        print("[sfm] 未找到 COLMAP，跳过自动位姿估计", flush=True)
        return None

    if len(image_paths) < int(config.get("sfm_min_images", 4)):
        print("[sfm] 图片少于 4 张，不足以运行 COLMAP", flush=True)
        return None

    workspace = project_dir / "colmap" / "workspace"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    log_path = project_dir / "colmap.log"
    log_path.write_text(f"COLMAP: {colmap_exe}\n", encoding="utf-8")

    copied, name_to_path = _copy_images(workspace, image_paths)
    images_dir = workspace / "images"
    database_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"

    use_gpu = 1 if str(config.get("sfm_use_gpu", True)).lower() not in {"0", "false"} else 0
    max_image_size = int(config.get("sfm_max_image_size", 1600))
    matcher = str(config.get("sfm_matcher", "sequential")).lower()
    if matcher not in {"sequential", "exhaustive"}:
        matcher = "sequential"

    if status_cb:
        status_cb(0.03, "特征提取")
    common_extract = [
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--FeatureExtraction.use_gpu", str(use_gpu),
        "--FeatureExtraction.max_image_size", str(max_image_size),
        "--SiftExtraction.estimate_affine_shape", "0",
        "--SiftExtraction.domain_size_pooling", "0",
    ]
    if _run(colmap_exe, "feature_extractor", common_extract, log_path) != 0:
        print("[sfm] 特征提取失败，详情见 colmap.log", flush=True)
        return None
    if status_cb:
        status_cb(0.05, "特征匹配")

    def run_matcher(mode: str) -> bool:
        if mode == "exhaustive":
            return (
                _run(
                    colmap_exe,
                    "exhaustive_matcher",
                    [
                        "--database_path", str(database_path),
                        "--FeatureMatching.use_gpu", str(use_gpu),
                        "--FeatureMatching.max_num_matches", "12000",
                        "--TwoViewGeometry.max_error", "3",
                        "--TwoViewGeometry.min_num_inliers", "8",
                    ],
                    log_path,
                )
                == 0
            )
        return (
            _run(
                colmap_exe,
                "sequential_matcher",
                [
                    "--database_path", str(database_path),
                    "--SequentialMatching.overlap", "20",
                    "--SequentialMatching.loop_detection", "0",
                    "--FeatureMatching.use_gpu", str(use_gpu),
                    "--FeatureMatching.max_num_matches", "12000",
                    "--TwoViewGeometry.max_error", "3",
                    "--TwoViewGeometry.min_num_inliers", "8",
                ],
                log_path,
            )
            == 0
        )

    matchers: list[str] = []
    if matcher == "exhaustive":
        matchers = ["exhaustive"]
    else:
        matchers = ["sequential", "exhaustive"]

    mapping_attempt = 0
    for matcher_name in matchers:
        if not run_matcher(matcher_name):
            print(f"[sfm] {matcher_name} 匹配失败，尝试下一策略", flush=True)
            continue
        mapping_attempt += 1
        shutil.rmtree(sparse_dir, ignore_errors=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        if status_cb:
            status_cb(0.07, f"稀疏重建（{matcher_name}）")
        mapping_ok = (
            _run(
                colmap_exe,
                "mapper",
                [
                    "--database_path", str(database_path),
                    "--image_path", str(images_dir),
                    "--output_path", str(sparse_dir),
                    "--Mapper.multiple_models", "0",
                    "--Mapper.ba_refine_focal_length", "1",
                    "--Mapper.ba_refine_extra_params", "1",
                    "--Mapper.ba_global_max_num_iterations", "200",
                    "--Mapper.ba_global_function_tolerance", "1e-6",
                    "--Mapper.min_num_matches", "8",
                    "--Mapper.max_num_models", "1",
                    "--Mapper.init_min_num_inliers", "8",
                    "--Mapper.init_min_tri_angle", "1.5",
                    "--Mapper.init_max_forward_motion", "1.0",
                    "--Mapper.filter_max_reproj_error", "4",
                    "--Mapper.filter_min_tri_angle", "1.0",
                ],
                log_path,
                timeout_minutes=int(config.get("sfm_mapper_timeout_minutes", 60)),
            )
            == 0
        )
        if mapping_ok:
            model_dir = _best_model_dir(sparse_dir)
            if model_dir:
                try:
                    data = load_colmap_sparse(
                        model_dir,
                        image_paths,
                        point_limit=int(config.get("sparse_points_limit", 50000)),
                        path_map=name_to_path,
                    )
                except Exception as error:
                    print(f"[sfm] 读取 COLMAP 结果失败：{error}", flush=True)
                    data = None
                if data and data["cameras"]:
                    registered = len(data["cameras"])
                    min_registered = max(3, int(config.get("sfm_min_registered", 4)))
                    if registered >= min_registered:
                        data["source"] = "colmap"
                        data["sfmModel"] = str(model_dir)
                        data["matcher"] = matcher_name
                        print(
                            f"[sfm] COLMAP 完成：{registered} 张图片，"
                            f"{len(data['points'])} 个稀疏点（{matcher_name}）",
                            flush=True,
                        )
                        return data
                    print(
                        f"[sfm] 稀疏重建只注册 {registered} 张图片，低于阈值 "
                        f"{min_registered}，尝试下一策略",
                        flush=True,
                    )
                    continue
        if matcher_name == matchers[-1]:
            print("[sfm] 所有匹配策略均未产出可用稀疏模型", flush=True)

    return None

