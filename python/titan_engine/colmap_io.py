"""Read COLMAP sparse reconstruction binaries into Titan camera/point data."""

from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np


# Camera model id -> (param_count, shared_focal). COLMAP writes one focal
# length for the SIMPLE_*/RADIAL family and separate fx/fy for PINHOLE/OPENCV.
_CAMERA_MODELS: dict[int, tuple[int, bool]] = {
    0: (3, True),   # SIMPLE_PINHOLE: f, cx, cy
    1: (4, False),  # PINHOLE: fx, fy, cx, cy
    2: (4, True),   # SIMPLE_RADIAL: f, cx, cy, k
    3: (5, True),   # RADIAL: f, cx, cy, k1, k2
    4: (8, False),  # OPENCV
    5: (8, False),  # OPENCV_FISHEYE
    6: (12, False), # FULL_OPENCV
    7: (5, False),  # FOV: fx, fy, cx, cy, omega
    8: (4, True),   # SIMPLE_RADIAL_FISHEYE: f, cx, cy, k
    9: (5, True),   # RADIAL_FISHEYE: f, cx, cy, k1, k2
    10: (12, False), # THIN_PRISM_FISHEYE
}


def _read_u64(stream: io.BytesIO) -> int:
    return struct.unpack("<Q", stream.read(8))[0]


def _read_i32(stream: io.BytesIO) -> int:
    return struct.unpack("<i", stream.read(4))[0]


def _read_u32(stream: io.BytesIO) -> int:
    return struct.unpack("<I", stream.read(4))[0]


def _read_string(stream: io.BytesIO) -> str:
    chunks: list[bytes] = []
    while True:
        byte = stream.read(1)
        if not byte or byte == b"\x00":
            break
        chunks.append(byte)
    return b"".join(chunks).decode("utf-8", errors="replace")


def read_cameras_bin(path: Path) -> dict[int, dict]:
    """Read cameras.bin and return a camera-id indexed dict."""
    with path.open("rb") as file:
        stream = io.BytesIO(file.read())
    count = _read_u64(stream)
    cameras: dict[int, dict] = {}
    for _ in range(count):
        camera_id = _read_u32(stream)
        model_id = _read_u32(stream)
        width = _read_u64(stream)
        height = _read_u64(stream)
        if model_id not in _CAMERA_MODELS:
            raise ValueError(f"unsupported COLMAP camera model id {model_id}")
        param_count, shared_focal = _CAMERA_MODELS[model_id]
        params = struct.unpack(f"<{param_count}d", stream.read(8 * param_count))
        if shared_focal:
            fx = fy = float(params[0])
            cx, cy = float(params[1]), float(params[2])
        else:
            fx, fy = float(params[0]), float(params[1])
            cx, cy = float(params[2]), float(params[3])
        cameras[camera_id] = {
            "model_id": model_id,
            "width": width,
            "height": height,
            "K": [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
        }
    return cameras


def read_images_bin(path: Path) -> list[dict]:
    """Read images.bin and return image records in file order."""
    with path.open("rb") as file:
        stream = io.BytesIO(file.read())
    count = _read_u64(stream)
    images: list[dict] = []
    for _ in range(count):
        image_id = _read_u32(stream)
        qvec = struct.unpack("<4d", stream.read(32))
        tvec = struct.unpack("<3d", stream.read(24))
        camera_id = _read_u32(stream)
        name = _read_string(stream)
        num_points2d = _read_u64(stream)
        if num_points2d:
            stream.read(16 * num_points2d)
        if num_points2d:
            stream.read(8 * num_points2d)
        images.append(
            {
                "id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }
        )
    return images


def read_points3d_bin(path: Path) -> np.ndarray:
    """Read points3D.bin and return an Nx7 [xyz, rgb, track] array."""
    with path.open("rb") as file:
        stream = io.BytesIO(file.read())
    count = _read_u64(stream)
    rows: list[np.ndarray] = []
    for _ in range(count):
        stream.read(8)  # point3D id
        xyz = np.asarray(struct.unpack("<3d", stream.read(24)), dtype=np.float64)
        rgb = np.asarray(struct.unpack("<3B", stream.read(3)), dtype=np.uint8)
        stream.read(8)  # reprojection error
        track_length = _read_u64(stream)
        if track_length:
            stream.read(8 * track_length)
        rows.append(
            np.concatenate(
                [xyz, rgb.astype(np.float64), np.asarray([track_length])]
            )
        )
    if not rows:
        return np.empty((0, 7), dtype=np.float64)
    return np.stack(rows, axis=0)


def qvec_to_rotmat(qvec: tuple[float, float, float, float]) -> np.ndarray:
    """Convert COLMAP quaternion [w, x, y, z] to a world rotation matrix."""
    w, x, y, z = qvec
    norm_sq = w * w + x * x + y * y + z * z
    if norm_sq <= 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / norm_sq
    return np.asarray(
        [
            [
                1.0 - s * (y * y + z * z),
                s * (x * y - z * w),
                s * (x * z + y * w),
            ],
            [
                s * (x * y + z * w),
                1.0 - s * (x * x + z * z),
                s * (y * z - x * w),
            ],
            [
                s * (x * z - y * w),
                s * (y * z + x * w),
                1.0 - s * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def camera_to_world(qvec: tuple[float, float, float, float], tvec: tuple[float, float, float]) -> np.ndarray:
    r_wc = qvec_to_rotmat(qvec)
    t_wc = -r_wc.T @ np.asarray(tvec, dtype=np.float64)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = r_wc.T
    c2w[:3, 3] = t_wc
    return c2w


def find_sparse_dir(image_paths: list[str]) -> Path | None:
    """Locate the nearest COLMAP sparse/0 directory for an image set."""
    candidates: list[Path] = []
    for image_path in image_paths:
        parent = Path(image_path).resolve().parent
        for base in (parent, parent.parent, parent.parent.parent):
            for candidate in (
                base / "sparse" / "0",
                base / "sparse",
                base / "colmap" / "sparse" / "0",
            ):
                if candidate.is_dir() and (candidate / "cameras.bin").is_file():
                    candidates.append(candidate)
    if not candidates:
        return None
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique[0]


def load_colmap_sparse(
    sparse_dir: Path,
    image_paths: list[str],
    point_limit: int = 2400,
) -> dict | None:
    """Load COLMAP poses and sparse points aligned to the input images."""
    sparse_dir = Path(sparse_dir)
    cameras_bin = sparse_dir / "cameras.bin"
    images_bin = sparse_dir / "images.bin"
    points_bin = sparse_dir / "points3D.bin"
    if not (cameras_bin.is_file() and images_bin.is_file()):
        return None
    cameras = read_cameras_bin(cameras_bin)
    image_records = read_images_bin(images_bin)
    by_name: dict[str, dict] = {}
    for record in image_records:
        name = Path(record["name"]).stem.lower()
        by_name.setdefault(name, record)

    matched: list[dict] = []
    for image_path in image_paths:
        record = by_name.get(Path(image_path).stem.lower())
        if record is None or record["camera_id"] not in cameras:
            continue
        camera = cameras[record["camera_id"]]
        matched.append(
            {
                "path": str(image_path),
                "width": int(camera["width"]),
                "height": int(camera["height"]),
                "K": [row[:] for row in camera["K"]],
                "c2w": camera_to_world(record["qvec"], record["tvec"]).tolist(),
            }
        )
    if not matched:
        return None

    points = np.empty((0, 7), dtype=np.float64)
    if points_bin.is_file():
        points = read_points3d_bin(points_bin)
    if len(points) > max(point_limit, 64):
        if points.shape[1] >= 7 and np.any(points[:, 6] > 0):
            order = np.argsort(-points[:, 6], kind="stable")
        else:
            rng = np.random.default_rng(7)
            order = rng.permutation(len(points))
        points = points[order[:point_limit]]
    point_list = [
        [
            float(x), float(y), float(z),
            int(r), int(g), int(b),
        ]
        for x, y, z, r, g, b, _track in points.tolist()
    ]
    return {
        "cameras": matched,
        "points": point_list,
        "source": "colmap",
    }


def find_poses_bounds(image_paths: list[str]) -> Path | None:
    """Locate an LLFF poses_bounds.npy beside the image set."""
    for image_path in image_paths:
        parent = Path(image_path).resolve().parent
        for base in (parent, parent.parent, parent.parent.parent):
            candidate = base / "poses_bounds.npy"
            if candidate.is_file():
                return candidate
    return None


def load_llff_poses(
    poses_path: Path,
    image_paths: list[str],
) -> dict | None:
    """Load LLFF c2w poses and convert them to Titan's forward-z convention."""
    poses = np.load(poses_path)
    if poses.ndim == 3 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.ndim != 2 or poses.shape[1] < 17:
        return None
    paths = sorted(image_paths)
    if len(paths) < 2:
        return None
    count = min(len(poses), len(paths))
    cameras: list[dict] = []
    for index in range(count):
        row = poses[index]
        block = np.asarray(row[:15], dtype=np.float64).reshape(3, 5)
        height, width, focal = (
            float(block[0, 4]),
            float(block[1, 4]),
            float(block[2, 4]),
        )
        if width <= 1 or height <= 1 or focal <= 0:
            continue
        c2w = np.eye(4, dtype=np.float64)
        # The raw LLFF block keeps [down, right, back, t, hwf] columns.
        # Reorder to Titan's right/down/forward camera axes.
        c2w[:3, :3] = np.stack(
            [block[:, 1], block[:, 0], -block[:, 2]], axis=1
        )
        c2w[:3, 3] = block[:, 3]
        cameras.append(
            {
                "path": paths[index],
                "width": int(width),
                "height": int(height),
                "K": [
                    [focal, 0.0, width / 2.0],
                    [0.0, focal, height / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                "c2w": c2w.tolist(),
            }
        )
    if not cameras:
        return None
    return {
        "cameras": cameras,
        "points": [],
        "source": "llff",
    }
