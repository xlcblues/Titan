"""One-off diagnostic: verify COLMAP point tracks reproduce with c2w/K."""

import struct
from pathlib import Path

import numpy as np

from titan_engine.colmap_io import camera_to_world, read_cameras_bin


def _read_u64(stream):
    return struct.unpack("<Q", stream.read(8))[0]


def _read_u32(stream):
    return struct.unpack("<I", stream.read(4))[0]


def _read_string(stream):
    chunks = []
    while True:
        byte = stream.read(1)
        if not byte or byte == b"\x00":
            break
        chunks.append(byte)
    return b"".join(chunks).decode("utf-8", errors="replace")


def main() -> None:
    sparse = Path(r"D:\data\nerf_llff_data\nerf_llff_data\horns\sparse\0")
    cameras = read_cameras_bin(sparse / "cameras.bin")

    image_records = {}
    with (sparse / "images.bin").open("rb") as stream:
        count = _read_u64(stream)
        for _ in range(count):
            image_id = _read_u32(stream)
            qvec = struct.unpack("<4d", stream.read(32))
            tvec = struct.unpack("<3d", stream.read(24))
            camera_id = _read_u32(stream)
            _name = _read_string(stream)
            num_points = _read_u64(stream)
            observations = []
            for _ in range(num_points):
                x, y = struct.unpack("<2d", stream.read(16))
                point_id = struct.unpack("<q", stream.read(8))[0]
                observations.append((x, y, point_id))
            image_records[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "observations": observations,
            }

    points = {}
    with (sparse / "points3D.bin").open("rb") as stream:
        count = _read_u64(stream)
        for _ in range(count):
            point_id = struct.unpack("<q", stream.read(8))[0]
            xyz = np.asarray(struct.unpack("<3d", stream.read(24)), dtype=np.float64)
            _rgb = struct.unpack("<3B", stream.read(3))
            _error = struct.unpack("<d", stream.read(8))[0]
            track_length = _read_u64(stream)
            if track_length:
                stream.read(8 * track_length)
            points[point_id] = xyz

    errors = []
    checked = 0
    for point_id, xyz in points.items():
        track = []
        for record in image_records.values():
            camera = cameras.get(record["camera_id"])
            if camera is None:
                continue
            for x, y, observed_id in record["observations"]:
                if observed_id == point_id:
                    track.append((x, y, record["qvec"], record["tvec"], camera))
        if len(track) < 2:
            continue
        checked += 1
        for x, y, qvec, tvec, camera in track[:4]:
            c2w = camera_to_world(qvec, tvec)
            k = camera["K"]
            rel = xyz - np.asarray(c2w)[:3, 3]
            camera_space = rel @ np.asarray(c2w)[:3, :3]
            z = float(camera_space[2])
            if z <= 1e-4:
                continue
            u = k[0][0] * camera_space[0] / z + k[0][2]
            v = k[1][1] * camera_space[1] / z + k[1][2]
            errors.append(float(np.hypot(u - x, v - y)))
        if checked >= 400:
            break

    print(f"points checked {checked}, reprojection mean {np.mean(errors):.3f} "
          f"median {np.median(errors):.3f} p95 {np.percentile(errors, 95):.3f}")


if __name__ == "__main__":
    main()
