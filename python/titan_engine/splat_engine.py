"""Differentiable facet splat renderer for the Titan engine (v0.1 baseline)."""

import math
import time

import numpy as np
import torch

from .tile_cuda import tile_cuda as _tile_cuda_loader
from .tile_cuda.autograd import tile_render


def _logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 1e-4, 1.0 - 1e-4)
    return np.log(value / (1.0 - value))


def _estimate_point_scales(
    points: np.ndarray,
    global_scale: float,
    multiplier: float = 0.7,
) -> np.ndarray:
    """Per-point radii from local point spacing, capped by the global scale."""
    count = len(points)
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    if count == 1:
        return np.full(count, max(global_scale * 0.5, 1e-3), dtype=np.float32)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        distances, _ = tree.query(points, k=min(3, count))
        if distances.ndim == 1:
            neighbor = distances
        else:
            neighbor = distances[:, -1] if distances.shape[1] > 1 else distances[:, 0]
    except Exception:
        rng = np.random.default_rng(7)
        sample = points[rng.integers(0, count, size=min(4000, count))]
        diff = points[:, None, :] - sample[None, :, :]
        neighbor = np.min(
            np.sqrt(np.sum(diff * diff, axis=-1)) + 1e-8, axis=1
        )
    scales = np.clip(neighbor * multiplier, 1e-4, max(global_scale, 1e-3))
    return scales.astype(np.float32)


class SplatFacetField(torch.nn.Module):
    """A scene made of small oriented circular facets (2D Gaussian splats).

    Each facet has a center, a log-scale radius, a logit opacity and a color
    in logit space. Rendering projects every facet into the camera and
    composites facets in depth order with per-pixel transmittance.
    """

    def __init__(
        self,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        scale: float = 0.12,
        scales: np.ndarray | None = None,
        opacity: float = 0.85,
    ) -> None:
        super().__init__()
        points = np.asarray(points, dtype=np.float32)
        self.register_parameter(
            "positions", torch.nn.Parameter(torch.as_tensor(points))
        )
        if colors is None:
            colors = np.full(
                (len(points), 3), 0.62, dtype=np.float32
            )
        colors = np.asarray(colors, dtype=np.float32)
        self.register_parameter(
            "color_logits",
            torch.nn.Parameter(
                torch.as_tensor(_logit(np.clip(colors, 0.0, 1.0)))
            ),
        )
        count = len(points)
        if scales is None:
            scales = np.full(count, max(scale, 1e-3), dtype=np.float32)
        scales = np.asarray(scales, dtype=np.float32).reshape(-1)
        if len(scales) != count:
            scales = np.full(count, max(scale, 1e-3), dtype=np.float32)
        self.register_parameter(
            "log_scales",
            torch.nn.Parameter(
                torch.as_tensor(np.log(np.clip(scales, 1e-4, None)), dtype=torch.float32)
            ),
        )
        self.register_parameter(
            "opacity_logits",
            torch.nn.Parameter(
                torch.full(
                    (count,),
                    _logit(np.array([opacity], dtype=np.float32))[0],
                    dtype=torch.float32,
                )
            ),
        )

    def forward(self):
        color = torch.sigmoid(self.color_logits)
        opacity = torch.sigmoid(self.opacity_logits)
        scale = torch.exp(self.log_scales)
        return self.positions, scale, opacity, color


def _camera_geometry(camera: dict, device: torch.device):
    c2w = np.asarray(camera["c2w"], dtype=np.float32)
    rotation = torch.as_tensor(c2w[:3, :3], dtype=torch.float32, device=device)
    origin = torch.as_tensor(c2w[:3, 3], dtype=torch.float32, device=device)
    k = camera["K"]
    width, height = camera["width"], camera["height"]
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    return rotation, origin, float(fx), float(fy), float(cx), float(cy), width, height


def _lowres_camera(camera: dict, low_res: int) -> dict:
    """Scale a camera's intrinsics down for a small full-image render."""
    source = int(camera["width"])
    factor = low_res / max(source, 1)
    k = camera["K"]
    return {
        "path": camera.get("path", ""),
        "width": low_res,
        "height": low_res,
        "K": [
            [float(k[0][0]) * factor, 0.0, float(k[0][2]) * factor],
            [0.0, float(k[1][1]) * factor, float(k[1][2]) * factor],
            [0.0, 0.0, 1.0],
        ],
        "c2w": camera["c2w"],
    }


def _downsample_images(
    images: np.ndarray,
    low_res: int,
) -> np.ndarray:
    """Build per-view low-resolution targets for the global residual loss."""
    from PIL import Image

    low = []
    for image in images:
        source = Image.fromarray(
            np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        )
        low.append(
            np.asarray(
                source.resize((low_res, low_res), Image.LANCZOS),
                dtype=np.float32,
            )
            / 255.0
        )
    return np.stack(low, axis=0)


def _project_points(
    positions: torch.Tensor,
    rotation: torch.Tensor,
    origin: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    near: float,
):
    rel = positions - origin[None, :]
    camera_space = rel @ rotation
    z = camera_space[:, 2:3]
    x = camera_space[:, 0:1]
    y = camera_space[:, 1:2]
    visible = z > near
    safe_z = torch.clamp(z, min=1e-4)
    u = fx * x / safe_z + cx
    v = fy * y / safe_z + cy
    return camera_space, u, v, z, visible


def _accumulate_alpha(
    alphas: torch.Tensor,
    colors: torch.Tensor,
    transmittance: torch.Tensor,
    accumulated: torch.Tensor,
):
    """Depth-ordered alpha compositing with per-splat transmittance decay.

    alphas has shape (chunk, pixels) and is ordered near-to-far. Without the
    intra-chunk cumulative product, splats inside one chunk would all use the
    same transmittance, which brightens and fogs dense scenes.
    """
    survive = torch.cumprod(1.0 - alphas, dim=0)
    before = torch.ones_like(alphas)
    before[1:] = survive[:-1]
    per_splat_transmittance = transmittance[None, :] * before
    contribution = (
        alphas[:, :, None]
        * colors[:, None, :]
        * per_splat_transmittance[:, :, None]
    )
    accumulated = accumulated + contribution.sum(dim=0)
    transmittance = transmittance * survive[-1]
    return accumulated, transmittance


def _render_pixels_dense(
    model: SplatFacetField,
    camera: dict,
    pixels: torch.Tensor | None,
    width: int,
    height: int,
    background: torch.Tensor,
    near: float | None = None,
    far: float | None = None,
    scene_points: np.ndarray | None = None,
    batch_size: int = 96,
    disk_cutoff: float = 2.2,
    falloff_power: float = 1.0,
):
    device = next(model.parameters()).device
    rotation, origin, fx, fy, cx, cy, width, height = _camera_geometry(
        camera, device
    )
    positions, scale, opacity, color = model()
    if near is None or far is None:
        if scene_points is not None and len(scene_points) > 0:
            from .main import ray_bounds_from_points

            bounds = ray_bounds_from_points(
                origin.cpu().detach().numpy(), scene_points
            )
        else:
            depths = torch.linalg.norm(
                positions - origin[None, :].to(device), dim=1
            )
            if depths.numel() == 0:
                bounds = (0.3, 7.6)
            else:
                lower = float(
                    torch.quantile(depths.detach(), 0.05).item()
                )
                upper = float(
                    torch.quantile(depths.detach(), 0.95).item()
                )
                bounds = (
                    max(lower * 0.75, 1e-4),
                    max(upper * 1.35, lower * 1.5 + 1e-4),
                )
        if near is None:
            near = bounds[0]
        if far is None:
            far = bounds[1]
    camera_space, u, v, z, visible = _project_points(
        positions,
        rotation,
        origin,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        near,
    )
    if far is not None:
        visible = visible & (z < far)

    if pixels is None:
        uu, vv = torch.meshgrid(
            torch.arange(width, dtype=torch.float32, device=device),
            torch.arange(height, dtype=torch.float32, device=device),
            indexing="xy",
        )
        pixels = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
        pixel_count = pixels.shape[0]
        output_shape = (height, width, 3)
    else:
        pixels = pixels.to(device)
        pixel_count = pixels.shape[0]
        output_shape = (pixel_count, 3)

    sigma_u = scale[:, None] * fx / torch.clamp(z, min=1e-4)
    sigma_v = scale[:, None] * fy / torch.clamp(z, min=1e-4)
    radius_u = 2.8 * sigma_u[:, 0]
    radius_v = 2.8 * sigma_v[:, 0]
    margin = 4.0
    active = (
        visible[:, 0]
        & (u[:, 0] > -radius_u - margin)
        & (u[:, 0] < width + radius_u + margin)
        & (v[:, 0] > -radius_v - margin)
        & (v[:, 0] < height + radius_v + margin)
    )
    indices = torch.nonzero(active).reshape(-1)
    if indices.numel() == 0:
        rendered = background[None, :].expand(pixel_count, -1)
        return rendered.reshape(output_shape)

    order = indices[torch.argsort(z[indices, 0], dim=0)]
    point_count = order.shape[0]
    transmittance = torch.ones(pixel_count, dtype=torch.float32, device=device)
    accumulated = torch.zeros(
        (pixel_count, 3), dtype=torch.float32, device=device
    )

    for start in range(0, point_count, batch_size):
        chunk = order[start : start + batch_size]
        du = pixels[:, 0][None, :] - u[chunk]
        dv = pixels[:, 1][None, :] - v[chunk]
        squared = (
            du * du / (2.0 * sigma_u[chunk] * sigma_u[chunk])
            + dv * dv / (2.0 * sigma_v[chunk] * sigma_v[chunk])
        )
        if falloff_power == 1.0:
            weight = torch.exp(-squared)
        else:
            weight = torch.exp(-torch.pow(squared, falloff_power))
        weight = weight * (squared < 0.5 * disk_cutoff * disk_cutoff)
        alpha = opacity[chunk, None] * weight
        alpha = torch.clamp(alpha, 0.0, 1.0)
        accumulated, transmittance = _accumulate_alpha(
            alpha, color[chunk], transmittance, accumulated
        )

    rendered = accumulated + transmittance[:, None] * background[None, :]
    return rendered.reshape(output_shape)


def _segment_log_cumsum(
    log_survive: torch.Tensor,
    segment_start: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Segment-local cumulative log survival for contiguous pixel segments.

    A global ``cumsum`` loses precision when it accumulates millions of
    splat-pixel pairs and is later unwound with ``cum[start] - cum[end]``.
    Packing each segment into a padded 2D matrix keeps the accumulated values
    small and matches a per-pixel sequential compositor to float rounding.
    """
    device = log_survive.device
    n = int(log_survive.numel())
    segment_indexes = torch.nonzero(segment_start).reshape(-1)
    segment_count = int(segment_indexes.numel())
    elem_segment = torch.cumsum(segment_start.to(torch.int64), 0) - 1
    bounds = torch.cat(
        [
            segment_indexes,
            torch.tensor([n], dtype=torch.int64, device=device),
        ],
        dim=0,
    )
    lengths = bounds[1:] - bounds[:-1]
    max_len = int(lengths.max().item()) if segment_count > 0 else 0
    local_pos = torch.arange(n, device=device) - segment_indexes[elem_segment]
    local_cum = torch.empty_like(log_survive)
    matrix_elems = 6_000_000
    segs_per_batch = max(1, matrix_elems // max_len) if max_len > 0 else 1

    for start in range(0, segment_count, segs_per_batch):
        stop = min(segment_count, start + segs_per_batch)
        seg_bounds = bounds[start : stop + 1]
        seg_count_b = stop - start
        seg_lengths_b = seg_bounds[1:] - seg_bounds[:-1]
        max_len_b = int(seg_lengths_b.max().item())
        rows = torch.arange(max_len_b, device=device)[:, None]
        cols = torch.arange(seg_count_b, device=device)[None, :]
        mask = rows < seg_lengths_b[None, :]
        src_idx = seg_bounds[:-1][None, :] + rows
        values = log_survive[src_idx.clamp(max=n - 1)]
        matrix = torch.where(
            mask,
            values,
            torch.zeros((), dtype=log_survive.dtype, device=device),
        )
        matrix_cum = torch.cumsum(matrix, dim=0)
        pair_start = int(seg_bounds[0].item())
        pair_end = int(seg_bounds[-1].item())
        row_b = local_pos[pair_start:pair_end]
        col_b = elem_segment[pair_start:pair_end] - start
        flat_idx = row_b * seg_count_b + col_b
        local_cum[pair_start:pair_end] = matrix_cum.reshape(-1)[flat_idx]

    before = torch.exp((local_cum - log_survive).clamp(max=0.0)).float()
    opaque = torch.isinf(log_survive) & (log_survive < 0)
    before = torch.where(
        opaque & segment_start,
        torch.ones_like(before),
        before,
    )
    before = torch.where(
        opaque & ~segment_start,
        torch.zeros_like(before),
        before,
    )
    segment_end = bounds[1:] - 1
    segment_total = local_cum[segment_end]
    return before, segment_end, segment_total


def _gpu_prefix_edges(
    values: torch.Tensor,
    limit: float,
    max_count: int | None = None,
    hard_split: torch.Tensor | None = None,
) -> torch.Tensor:
    """Boundaries for prefix sums that stay below ``limit`` on the GPU.

    Values larger than the limit are placed in their own span, and hard splits
    (for example oversize tiles) are isolated on both sides.
    """
    device = values.device
    n = int(values.numel())
    if n == 0:
        return torch.zeros(1, dtype=torch.int64, device=device)
    limit = max(float(limit), 1.0)
    bounded = torch.minimum(values, torch.full_like(values, limit))
    bucket = torch.div(
        torch.cumsum(bounded, dim=0),
        limit,
        rounding_mode="floor",
    ).long()
    positions = [torch.zeros(1, dtype=torch.int64, device=device)]
    changed = torch.nonzero(bucket[1:] > bucket[:-1]).reshape(-1) + 1
    if changed.numel() > 0:
        positions.append(changed)
    oversize = torch.nonzero(values >= limit).reshape(-1)
    if oversize.numel() > 0:
        positions.append(oversize + 1)
    if hard_split is not None:
        hard = torch.nonzero(hard_split).reshape(-1)
        if hard.numel() > 0:
            positions.append(hard)
            positions.append(hard + 1)
    if max_count is not None and 0 < max_count < n:
        positions.append(
            torch.arange(max_count, n, max_count, device=device)
        )
    edges = torch.unique(torch.cat(positions))
    if int(edges[0].item()) != 0:
        edges = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=device),
                edges,
            ],
            dim=0,
        )
    if int(edges[-1].item()) != n:
        edges = torch.cat(
            [
                edges,
                torch.full(
                    (1,),
                    n,
                    dtype=torch.int64,
                    device=device,
                ),
            ],
            dim=0,
        )
    return edges


def _render_pixels_tile(
    model: SplatFacetField,
    camera: dict,
    pixels: torch.Tensor | None,
    width: int,
    height: int,
    background: torch.Tensor,
    near: float | None = None,
    far: float | None = None,
    scene_points: np.ndarray | None = None,
    batch_size: int = 256,
    disk_cutoff: float = 2.2,
    falloff_power: float = 1.0,
    tile_size: int = 16,
    tile_pair_budget: int = 4_000_000,
    tile_max_splats: int = 8192,
):
    """GPU tile rasterizer that composites splat-pixel pairs in parallel.

    Splats are first expanded to the tiles they overlap. Tiles are then
    grouped into GPU chunks whose pair count fits the VRAM budget, and every
    pixel batch inside a chunk composites all of its splats with vectorized
    segment products instead of the old per-tile Python loop.
    """
    device = next(model.parameters()).device
    tile_size = max(1, int(tile_size))
    rotation, origin, fx, fy, cx, cy, width, height = _camera_geometry(
        camera, device
    )
    positions, scale, opacity, color = model()
    if near is None or far is None:
        if scene_points is not None and len(scene_points) > 0:
            from .main import ray_bounds_from_points

            bounds = ray_bounds_from_points(
                origin.cpu().detach().numpy(), scene_points
            )
        else:
            depths = torch.linalg.norm(
                positions - origin[None, :].to(device), dim=1
            )
            if depths.numel() == 0:
                bounds = (0.3, 7.6)
            else:
                lower = float(
                    torch.quantile(depths.detach(), 0.05).item()
                )
                upper = float(
                    torch.quantile(depths.detach(), 0.95).item()
                )
                bounds = (
                    max(lower * 0.75, 1e-4),
                    max(upper * 1.35, lower * 1.5 + 1e-4),
                )
        if near is None:
            near = bounds[0]
        if far is None:
            far = bounds[1]
    camera_space, u, v, z, visible = _project_points(
        positions,
        rotation,
        origin,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        near,
    )
    if far is not None:
        visible = visible & (z < far)

    if pixels is None:
        uu, vv = torch.meshgrid(
            torch.arange(width, dtype=torch.float32, device=device),
            torch.arange(height, dtype=torch.float32, device=device),
            indexing="xy",
        )
        pixels = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
        pixel_count = pixels.shape[0]
        output_shape = (height, width, 3)
    else:
        pixels = pixels.to(device)
        pixel_count = pixels.shape[0]
        output_shape = (pixel_count, 3)

    sigma_u = scale[:, None] * fx / torch.clamp(z, min=1e-4)
    sigma_v = scale[:, None] * fy / torch.clamp(z, min=1e-4)
    radius_u = 2.8 * sigma_u[:, 0]
    radius_v = 2.8 * sigma_v[:, 0]
    margin = 4.0
    active = (
        visible[:, 0]
        & (u[:, 0] > -radius_u - margin)
        & (u[:, 0] < width + radius_u + margin)
        & (v[:, 0] > -radius_v - margin)
        & (v[:, 0] < height + radius_v + margin)
    )
    active_indices = torch.nonzero(active).reshape(-1)
    if active_indices.numel() == 0:
        rendered = background[None, :].expand(pixel_count, -1)
        return rendered.reshape(output_shape)

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    min_u = u[:, 0] - radius_u - margin
    max_u = u[:, 0] + radius_u + margin
    min_v = v[:, 0] - radius_v - margin
    max_v = v[:, 0] + radius_v + margin
    min_tile_u = torch.clamp(
        torch.floor(min_u / tile_size).long(), 0, tiles_x - 1
    )
    max_tile_u = torch.clamp(
        torch.floor(max_u / tile_size).long(), 0, tiles_x - 1
    )
    min_tile_v = torch.clamp(
        torch.floor(min_v / tile_size).long(), 0, tiles_y - 1
    )
    max_tile_v = torch.clamp(
        torch.floor(max_v / tile_size).long(), 0, tiles_y - 1
    )

    pixel_tile_u = torch.clamp(
        torch.floor(pixels[:, 0] / tile_size).long(), 0, tiles_x - 1
    )
    pixel_tile_v = torch.clamp(
        torch.floor(pixels[:, 1] / tile_size).long(), 0, tiles_y - 1
    )
    pixel_keys = pixel_tile_u + tiles_x * pixel_tile_v
    pixel_order = torch.argsort(pixel_keys)
    sorted_pixels = pixels[pixel_order]
    sorted_keys = pixel_keys[pixel_order]
    pixel_unique, pixel_counts = torch.unique_consecutive(
        sorted_keys, return_counts=True
    )
    pixel_starts = torch.cumsum(pixel_counts, 0) - pixel_counts

    a_min_u = min_tile_u[active_indices]
    a_max_u = max_tile_u[active_indices]
    a_min_v = min_tile_v[active_indices]
    a_max_v = max_tile_v[active_indices]
    span_u = a_max_u - a_min_u + 1
    span_v = a_max_v - a_min_v + 1
    products = span_u * span_v
    max_expand = max(
        1,
        min(
            4096,
            2_000_000 // max(int(active_indices.numel()), 1),
        ),
    )
    broad = products > max_expand
    broad_idx = active_indices[broad]
    small_mask = ~broad
    small_products = products[small_mask]
    small_idx = active_indices[small_mask]

    if small_idx.numel() > 0:
        small_span_u = span_u[small_mask]
        small_min_u = a_min_u[small_mask]
        small_min_v = a_min_v[small_mask]
        total_entries = int(small_products.sum().item())
        group_ids = torch.repeat_interleave(
            torch.arange(small_idx.numel(), device=device),
            small_products,
        )
        local = torch.arange(total_entries, device=device) - torch.repeat_interleave(
            torch.cumsum(small_products, 0) - small_products,
            small_products,
        )
        repeated_span = small_span_u[group_ids]
        entry_tile_u = small_min_u[group_ids] + local % repeated_span
        entry_tile_v = small_min_v[group_ids] + local // repeated_span
        entry_keys = entry_tile_u + tiles_x * entry_tile_v
        entry_splats = small_idx[group_ids]
        entry_order = torch.argsort(entry_keys)
        entry_keys = entry_keys[entry_order]
        entry_splats = entry_splats[entry_order]
    else:
        entry_keys = torch.empty(0, dtype=torch.long, device=device)
        entry_splats = torch.empty(0, dtype=torch.long, device=device)

    if broad_idx.numel() > 0:
        broad_min_u = min_tile_u[broad_idx]
        broad_max_u = max_tile_u[broad_idx]
        broad_min_v = min_tile_v[broad_idx]
        broad_max_v = max_tile_v[broad_idx]
    else:
        broad_min_u = broad_max_u = broad_min_v = broad_max_v = None

    entry_start = torch.searchsorted(entry_keys, pixel_unique)
    entry_end = torch.searchsorted(entry_keys, pixel_unique + 1)
    output = torch.zeros(
        (pixel_count, 3), dtype=torch.float32, device=device
    )

    broad_count = int(broad_idx.numel()) if broad_idx.numel() > 0 else 0
    splat_z = z[:, 0]
    tile_keys = pixel_unique
    tile_pixel_counts = pixel_counts
    tile_u = tile_keys % tiles_x
    tile_v = tile_keys // tiles_x
    small_counts = entry_end - entry_start
    broad_block = 4096

    # Count broad-splat coverage per tile in bounded GPU blocks so chunk
    # boundaries use the real pair load instead of a full B x T matrix.
    broad_counts = torch.zeros(
        tile_keys.numel(), dtype=torch.int32, device=device
    )
    if broad_count > 0:
        for start in range(0, broad_count, broad_block):
            stop = min(broad_count, start + broad_block)
            bu_min = broad_min_u[start:stop]
            bu_max = broad_max_u[start:stop]
            bv_min = broad_min_v[start:stop]
            bv_max = broad_max_v[start:stop]
            covered_u = (
                (bu_min[:, None] <= tile_u[None, :])
                & (bu_max[:, None] >= tile_u[None, :])
            )
            covered_v = (
                (bv_min[:, None] <= tile_v[None, :])
                & (bv_max[:, None] >= tile_v[None, :])
            )
            broad_counts += (covered_u & covered_v).sum(dim=0).to(
                torch.int32
            )

    # Tile pair estimates drive segment boundaries computed entirely on the
    # GPU: every segment fits the pair budget and the 4096-tile chunk limit,
    # and oversized tiles are isolated so expansion never exhausts VRAM.
    tile_entry_counts = small_counts + broad_counts
    tile_estimate = (
        tile_entry_counts.to(torch.float64)
        * tile_pixel_counts.to(torch.float64)
    )
    max_chunk_tiles = 4096
    segment_edges = _gpu_prefix_edges(
        tile_estimate,
        tile_pair_budget,
        max_count=max_chunk_tiles,
        hard_split=tile_entry_counts > tile_max_splats,
    ).tolist()

    for segment_index in range(len(segment_edges) - 1):
        t0 = segment_edges[segment_index]
        t1 = segment_edges[segment_index + 1]
        chunk_keys = tile_keys[t0:t1]
        chunk_tile_u = chunk_keys % tiles_x
        chunk_tile_v = chunk_keys // tiles_x
        p0 = int(pixel_starts[t0].item())
        p1 = int(
            pixel_starts[t0].item()
            + int(pixel_counts[t0 : t0 + (t1 - t0)].sum().item())
        )

        chunk_tile_entry_counts = entry_end[t0:t1] - entry_start[t0:t1]
        small_splat_count = int(chunk_tile_entry_counts.sum().item())
        if small_splat_count > 0:
            small_splats = entry_splats[
                int(entry_start[t0].item()) :
                int(entry_start[t0].item()) + small_splat_count
            ]
            small_splat_tiles = torch.repeat_interleave(
                torch.arange(t1 - t0, device=device),
                chunk_tile_entry_counts,
            )
        else:
            small_splats = torch.empty(0, dtype=torch.long, device=device)
            small_splat_tiles = torch.empty(0, dtype=torch.long, device=device)

        # Broad splats are expanded to (tile, splat) pairs in the same GPU
        # block scheme, replacing the old broad_count x chunk_tile matrix.
        broad_pair_splats: list[torch.Tensor] = []
        broad_pair_tiles: list[torch.Tensor] = []
        if broad_count > 0 and t1 > t0:
            for start in range(0, broad_count, broad_block):
                stop = min(broad_count, start + broad_block)
                bu_min = broad_min_u[start:stop]
                bu_max = broad_max_u[start:stop]
                bv_min = broad_min_v[start:stop]
                bv_max = broad_max_v[start:stop]
                covered_u = (
                    (bu_min[:, None] <= chunk_tile_u[None, :])
                    & (bu_max[:, None] >= chunk_tile_u[None, :])
                )
                covered_v = (
                    (bv_min[:, None] <= chunk_tile_v[None, :])
                    & (bv_max[:, None] >= chunk_tile_v[None, :])
                )
                covered_rows, covered_cols = torch.nonzero(
                    covered_u & covered_v, as_tuple=True
                )
                if covered_rows.numel() > 0:
                    broad_pair_splats.append(broad_idx[start + covered_rows])
                    broad_pair_tiles.append(covered_cols)
        if broad_pair_splats:
            broad_splats = torch.cat(broad_pair_splats, dim=0)
            broad_tiles = torch.cat(broad_pair_tiles, dim=0)
        else:
            broad_splats = torch.empty(0, dtype=torch.long, device=device)
            broad_tiles = torch.empty(0, dtype=torch.long, device=device)

        if small_splat_count > 0 and broad_splats.numel() > 0:
            chunk_splats = torch.cat([small_splats, broad_splats], dim=0)
            chunk_splat_tile = torch.cat(
                [small_splat_tiles, broad_tiles], dim=0
            )
        elif small_splat_count > 0:
            chunk_splats = small_splats
            chunk_splat_tile = small_splat_tiles
        elif broad_splats.numel() > 0:
            chunk_splats = broad_splats
            chunk_splat_tile = broad_tiles
        else:
            chunk_splats = torch.empty(0, dtype=torch.long, device=device)
            chunk_splat_tile = torch.empty(0, dtype=torch.long, device=device)

        if chunk_splats.numel() == 0 or p1 <= p0:
            output[pixel_order[p0:p1]] = background[None, :].expand(
                p1 - p0, -1
            )
            continue

        chunk_z = splat_z[chunk_splats]
        # One packed 64-bit key gives the (tile, depth) order: the low 32
        # bits preserve the positive float32 depth ordering, the high bits
        # sort by tile.
        depth_key = chunk_z.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        order_key = chunk_splat_tile.to(torch.int64) << 32 | depth_key
        order = torch.argsort(order_key, stable=True)
        ordered_splats = chunk_splats[order]
        ordered_tiles = chunk_splat_tile[order]
        chunk_u = u[ordered_splats, 0]
        chunk_v = v[ordered_splats, 0]
        chunk_sigma_u = sigma_u[ordered_splats, 0]
        chunk_sigma_v = sigma_v[ordered_splats, 0]
        chunk_opacity = opacity[ordered_splats]
        chunk_color = color[ordered_splats]

        chunk_entry_counts = (
            entry_end[t0:t1] - entry_start[t0:t1] + broad_counts[t0:t1]
        )
        entry_cum = torch.cumsum(chunk_entry_counts, 0)
        tile_pixel_cum = torch.cumsum(pixel_counts[t0:t1], 0)

        # The GPU segment boundaries already bound the real pair load, so
        # each segment is expanded as one group without a second Python pass.
        group_edges = [0, int(t1 - t0)]

        for group_index in range(len(group_edges) - 1):
            g0 = group_edges[group_index]
            g1 = group_edges[group_index + 1]
            group_splat_start = (
                int(entry_cum[g0 - 1].item()) if g0 > 0 else 0
            )
            group_splat_end = int(entry_cum[g1 - 1].item())
            group_splats = ordered_splats[group_splat_start:group_splat_end]
            group_tiles = ordered_tiles[group_splat_start:group_splat_end]
            if group_splats.numel() == 0:
                continue
            group_pixel_start = (
                int(tile_pixel_cum[g0 - 1].item()) if g0 > 0 else 0
            )
            group_pixel_end = int(tile_pixel_cum[g1 - 1].item())
            group_batch_count = group_pixel_end - group_pixel_start
            group_coords = sorted_pixels[
                p0 + group_pixel_start : p0 + group_pixel_end
            ]
            group_pixel_counts = pixel_counts[t0 + g0 : t0 + g1]
            group_tiles_local = group_tiles - g0
            num_group_tiles = g1 - g0
            tile_offsets = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int64, device=device),
                    torch.cumsum(
                        group_pixel_counts[:-1], 0
                    ),
                ],
                dim=0,
            )
            # Build pixel-major (tile, pixel, depth) pairs in one pass:
            # each pixel repeats all splats of its tile in depth order, so
            # same-pixel pairs stay contiguous without a global argsort.
            tile_splat_counts = torch.bincount(
                group_tiles_local, minlength=num_group_tiles
            )
            tile_splat_offsets = (
                torch.cumsum(tile_splat_counts, 0) - tile_splat_counts
            )
            tile_pair_counts = tile_splat_counts * group_pixel_counts
            pair_count = int(tile_pair_counts.sum().item())
            if pair_count == 0:
                continue
            pair_tile = torch.repeat_interleave(
                torch.arange(num_group_tiles, device=device),
                tile_pair_counts,
            )
            tile_pair_starts = (
                torch.cumsum(tile_pair_counts, 0) - tile_pair_counts
            )
            local_index = (
                torch.arange(pair_count, device=device)
                - torch.repeat_interleave(
                    tile_pair_starts, tile_pair_counts
                )
            )
            splat_per_tile = torch.repeat_interleave(
                tile_splat_counts, tile_pair_counts
            )
            pair_splat = (
                tile_splat_offsets[pair_tile]
                + local_index % splat_per_tile
            )
            pair_pixel = (
                tile_offsets[pair_tile]
                + local_index // splat_per_tile
            )
            group_u = chunk_u[group_splat_start:group_splat_end]
            group_v = chunk_v[group_splat_start:group_splat_end]
            group_sigma_u = chunk_sigma_u[
                group_splat_start:group_splat_end
            ]
            group_sigma_v = chunk_sigma_v[
                group_splat_start:group_splat_end
            ]
            group_opacity = chunk_opacity[
                group_splat_start:group_splat_end
            ]
            group_color = chunk_color[
                group_splat_start:group_splat_end
            ]
            pair_u = group_u[pair_splat]
            pair_v = group_v[pair_splat]
            pair_sigma_u = group_sigma_u[pair_splat]
            pair_sigma_v = group_sigma_v[pair_splat]
            pair_color = group_color[pair_splat]
            du = group_coords[:, 0][pair_pixel] - pair_u
            dv = group_coords[:, 1][pair_pixel] - pair_v
            squared = (
                du * du / (2.0 * pair_sigma_u * pair_sigma_u)
                + dv * dv / (2.0 * pair_sigma_v * pair_sigma_v)
            )
            if falloff_power == 1.0:
                weight = torch.exp(-squared)
            else:
                weight = torch.exp(-torch.pow(squared, falloff_power))
            weight = weight * (
                squared < 0.5 * disk_cutoff * disk_cutoff
            )
            alpha = torch.clamp(
                group_opacity[pair_splat] * weight, 0.0, 1.0
            )

            seg_pixel = pair_pixel
            seg_alpha = alpha
            seg_color = pair_color
            log_survive = torch.log1p(-seg_alpha).double()
            segment_start = torch.cat(
                [
                    torch.ones(1, dtype=torch.bool, device=device),
                    seg_pixel[1:] != seg_pixel[:-1],
                ],
                dim=0,
            )
            before, segment_end, segment_total = _segment_log_cumsum(
                log_survive, segment_start
            )
            contribution = seg_alpha[:, None] * before[:, None] * seg_color
            acc = torch.zeros(
                group_batch_count, 3, dtype=torch.float32, device=device
            )
            acc.index_add_(0, seg_pixel, contribution)
            transmittance = torch.ones(group_batch_count, device=device)
            transmittance[seg_pixel[segment_end]] = torch.exp(
                segment_total
            ).float()
            rendered = acc + transmittance[:, None] * background[None, :]
            output[
                pixel_order[
                    p0 + group_pixel_start : p0 + group_pixel_end
                ]
            ] = rendered

    return output.reshape(output_shape)


def _render_pixels_cuda(
    model: SplatFacetField,
    camera: dict,
    pixels: torch.Tensor | None,
    width: int,
    height: int,
    background: torch.Tensor,
    near: float | None = None,
    far: float | None = None,
    scene_points: np.ndarray | None = None,
    disk_cutoff: float = 2.2,
    falloff_power: float = 1.0,
    tile_size: int = 16,
    tile_pair_budget: int = 4_000_000,
    tile_max_splats: int = 8192,
    tile_work_budget: int = 2_000_000,
):
    """Render through the CUDA tile rasterizer with autograd support."""
    device = next(model.parameters()).device
    tile_size = max(1, int(tile_size))
    rotation, origin, fx, fy, cx, cy, width, height = _camera_geometry(
        camera, device
    )
    if near is None or far is None:
        if scene_points is not None and len(scene_points) > 0:
            from .main import ray_bounds_from_points

            bounds = ray_bounds_from_points(
                origin.cpu().detach().numpy(), scene_points
            )
        else:
            depths = torch.linalg.norm(
                model.positions - origin[None, :], dim=1
            )
            if depths.numel() == 0:
                bounds = (0.3, 7.6)
            else:
                lower = float(
                    torch.quantile(depths.detach(), 0.05).item()
                )
                upper = float(
                    torch.quantile(depths.detach(), 0.95).item()
                )
                bounds = (
                    max(lower * 0.75, 1e-4),
                    max(upper * 1.35, lower * 1.5 + 1e-4),
                )
        if near is None:
            near = bounds[0]
        if far is None:
            far = bounds[1]
    if pixels is None:
        uu, vv = torch.meshgrid(
            torch.arange(width, dtype=torch.float32, device=device),
            torch.arange(height, dtype=torch.float32, device=device),
            indexing="xy",
        )
        pixels = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
        output_shape = (height, width, 3)
    else:
        pixels = pixels.to(device)
        output_shape = (pixels.shape[0], 3)
    rendered = tile_render(
        model.positions,
        model.log_scales,
        model.opacity_logits,
        model.color_logits,
        rotation,
        origin,
        pixels,
        background,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        tile_size=tile_size,
        near=near,
        far=far,
        disk_cutoff=disk_cutoff,
        falloff_power=falloff_power,
        tile_pair_budget=tile_pair_budget,
        tile_max_splats=tile_max_splats,
        tile_work_budget=tile_work_budget,
    )
    return rendered.reshape(output_shape)


def _render_pixels(
    model: SplatFacetField,
    camera: dict,
    pixels: torch.Tensor | None,
    width: int,
    height: int,
    background: torch.Tensor,
    near: float | None = None,
    far: float | None = None,
    scene_points: np.ndarray | None = None,
    batch_size: int = 96,
    disk_cutoff: float = 2.2,
    falloff_power: float = 1.0,
    tile_size: int = 16,
    dense_renderer: bool = False,
    auto_dense_threshold: int = 20000,
    auto_dense_pixels: int = 1_000_000,
    tile_pair_budget: int = 4_000_000,
    tile_max_splats: int = 8192,
    tile_work_budget: int = 2_000_000,
    use_cuda_tile: bool = False,
):
    if use_cuda_tile:
        try:
            if _tile_cuda_loader() is not None:
                return _render_pixels_cuda(
                    model,
                    camera,
                    pixels,
                    width,
                    height,
                    background,
                    near=near,
                    far=far,
                    scene_points=scene_points,
                    disk_cutoff=disk_cutoff,
                    falloff_power=falloff_power,
                    tile_size=tile_size,
                    tile_pair_budget=tile_pair_budget,
                    tile_max_splats=tile_max_splats,
                    tile_work_budget=tile_work_budget,
                )
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:
            pass
    splat_count = int(model.positions.numel())
    pixel_count = (
        int(pixels.shape[0]) if pixels is not None else int(width) * int(height)
    )
    use_dense = dense_renderer or (
        splat_count <= auto_dense_threshold
        and pixel_count <= auto_dense_pixels
    )
    if use_dense:
        return _render_pixels_dense(
            model,
            camera,
            pixels,
            width,
            height,
            background,
            near=near,
            far=far,
            scene_points=scene_points,
            batch_size=batch_size,
            disk_cutoff=disk_cutoff,
            falloff_power=falloff_power,
        )
    try:
        return _render_pixels_tile(
            model,
            camera,
            pixels,
            width,
            height,
            background,
            near=near,
            far=far,
            scene_points=scene_points,
            batch_size=batch_size,
            disk_cutoff=disk_cutoff,
            falloff_power=falloff_power,
            tile_size=tile_size,
            tile_pair_budget=tile_pair_budget,
            tile_max_splats=tile_max_splats,
        )
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception:
        return _render_pixels_dense(
            model,
            camera,
            pixels,
            width,
            height,
            background,
            near=near,
            far=far,
            scene_points=scene_points,
            batch_size=batch_size,
            disk_cutoff=disk_cutoff,
            falloff_power=falloff_power,
        )


def render_splats_view(
    model: SplatFacetField,
    camera: dict,
    background: np.ndarray,
    scene_points: np.ndarray | None = None,
    batch_size: int = 96,
    disk_cutoff: float = 2.2,
    falloff_power: float = 1.0,
    tile_size: int = 16,
    dense_renderer: bool = False,
    auto_dense_threshold: int = 20000,
    auto_dense_pixels: int = 1_000_000,
    tile_pair_budget: int = 4_000_000,
    tile_max_splats: int = 8192,
    tile_work_budget: int = 2_000_000,
    use_cuda_tile: bool = False,
) -> np.ndarray:
    device = next(model.parameters()).device
    bg = torch.as_tensor(background, dtype=torch.float32, device=device)
    with torch.no_grad():
        image = _render_pixels(
            model,
            camera,
            None,
            camera["width"],
            camera["height"],
            bg,
            scene_points=scene_points,
            batch_size=batch_size,
            disk_cutoff=disk_cutoff,
            falloff_power=falloff_power,
            tile_size=tile_size,
            dense_renderer=dense_renderer,
            auto_dense_threshold=auto_dense_threshold,
            auto_dense_pixels=auto_dense_pixels,
            tile_pair_budget=tile_pair_budget,
            tile_max_splats=tile_max_splats,
            tile_work_budget=tile_work_budget,
            use_cuda_tile=use_cuda_tile,
        )
    return image.cpu().detach().numpy()


def _density_prior_loss(
    model: SplatFacetField,
    surface_points: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
    count: int = 1024,
):
    """Keep facets near the observed surface instead of drifting into haze."""
    if len(surface_points) == 0 or model.positions.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    position_count = model.positions.shape[0]
    position_indices = rng.integers(0, position_count, size=count)
    positions = model.positions[position_indices]
    surface_count = min(len(surface_points), 2048)
    surface = surface_points[
        rng.integers(0, len(surface_points), size=surface_count)
    ]
    target = torch.as_tensor(surface, dtype=torch.float32, device=device)
    diff = positions[:, None, :] - target[None, :, :]
    distance = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-8)
    nearest = torch.min(distance, dim=1).values
    return torch.mean(nearest * nearest)


def color_points_from_images(
    points: np.ndarray,
    cameras: list[dict],
    images: np.ndarray,
) -> np.ndarray:
    """Paint initialization facets with image samples at their projections.

    COLMAP sparse models do not always carry valid RGB, and a black initial
    cloud renders as a dark fog that is slow to recover from. Averaging the
    training images at each point's projection gives every facet a useful
    starting color without changing its geometry.
    """
    colors = np.zeros((len(points), 3), dtype=np.float32)
    counts = np.zeros(len(points), dtype=np.float32)
    for index, camera in enumerate(cameras):
        if index >= len(images):
            break
        c2w = np.asarray(camera["c2w"], dtype=np.float32)
        origin = c2w[:3, 3]
        rotation = c2w[:3, :3]
        k = camera["K"]
        width, height = camera["width"], camera["height"]
        fx, fy, cx, cy = (
            float(k[0][0]),
            float(k[1][1]),
            float(k[0][2]),
            float(k[1][2]),
        )
        relative = points - origin[None, :]
        camera_space = relative @ rotation
        z = camera_space[:, 2]
        visible = z > 0.0
        safe_z = np.maximum(z, 1e-4)
        u = fx * camera_space[:, 0] / safe_z + cx
        v = fy * camera_space[:, 1] / safe_z + cy
        visible &= (u >= 0.0) & (u < width - 1) & (v >= 0.0) & (v < height - 1)
        if not np.any(visible):
            continue
        ui = np.clip(np.round(u[visible]).astype(np.int32), 0, width - 1)
        vi = np.clip(np.round(v[visible]).astype(np.int32), 0, height - 1)
        colors[visible] += images[index][vi, ui]
        counts[visible] += 1.0
    covered = counts > 0
    if np.any(covered):
        colors[covered] /= counts[covered, None]
    return colors


def estimate_init_scale(
    points: np.ndarray,
    cameras: list[dict],
    target_px: float = 2.5,
) -> float:
    """Pick splat radii so a median point projects to roughly target_px."""
    if len(points) == 0 or len(cameras) == 0:
        return 0.05
    origins = np.asarray(
        [np.asarray(camera["c2w"], dtype=np.float32)[:3, 3] for camera in cameras]
    )
    if origins.ndim != 2 or origins.shape[0] == 0:
        return 0.05
    fxs = np.asarray(
        [float(camera["K"][0][0]) for camera in cameras], dtype=np.float32
    )
    if fxs.size == 0 or float(np.median(fxs)) <= 0:
        return 0.05
    depths = np.min(np.linalg.norm(points[:, None, :] - origins[None, :, :], axis=2), axis=1)
    median_depth = float(np.median(depths))
    median_fx = float(np.median(fxs))
    if not np.isfinite(median_depth) or median_depth <= 1e-4:
        return 0.05
    return max(median_depth * target_px / median_fx, 1e-3)


def _visible_mask(
    model: SplatFacetField,
    camera: dict,
    scene_points: np.ndarray,
    device: torch.device,
    near: float,
    far: float,
) -> torch.Tensor:
    rotation, origin, fx, fy, cx, cy, width, height = _camera_geometry(
        camera, device
    )
    positions = model.positions
    camera_space, u, v, z, visible = _project_points(
        positions,
        rotation,
        origin,
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        near,
    )
    visible = visible & (z < far)
    sigma_u = torch.exp(model.log_scales)[:, None] * fx / torch.clamp(z, min=1e-4)
    sigma_v = torch.exp(model.log_scales)[:, None] * fy / torch.clamp(z, min=1e-4)
    radius_u = 2.8 * sigma_u[:, 0]
    radius_v = 2.8 * sigma_v[:, 0]
    visible = (
        visible[:, 0]
        & (u[:, 0] > -radius_u - 4.0)
        & (u[:, 0] < width + radius_u + 4.0)
        & (v[:, 0] > -radius_v - 4.0)
        & (v[:, 0] < height + radius_v + 4.0)
    )
    return visible


def _opacity_reset(
    model: SplatFacetField,
    camera: dict,
    surface_points: np.ndarray,
    device: torch.device,
    opacity: float,
    near: float,
    far: float,
) -> None:
    """Re-raise opacity for visible facets so haze cannot persist."""
    visible = _visible_mask(
        model, camera, surface_points, device, near, far
    )
    if not torch.any(visible):
        return
    with torch.no_grad():
        model.opacity_logits[visible] = torch.as_tensor(
            [math.log(opacity / (1.0 - opacity))],
            dtype=torch.float32,
            device=device,
        )


def _densify_and_prune(
    model: SplatFacetField,
    camera: dict,
    surface_points: np.ndarray,
    device: torch.device,
    grad_accum: torch.Tensor,
    max_splats: int,
    prune_threshold: float,
    jitter_scale: float,
    near: float,
    far: float,
    rng: np.random.Generator,
) -> bool:
    """Clone high-gradient visible facets and prune dead ones."""
    with torch.no_grad():
        opacity = torch.sigmoid(model.opacity_logits)
        keep = opacity >= prune_threshold
        visible = _visible_mask(
            model, camera, surface_points, device, near, far
        )
        strength = grad_accum * visible
        threshold = float(torch.quantile(strength, 0.72).item()) if torch.any(strength > 0) else 0.0
        candidates = torch.nonzero(
            (strength >= threshold) & (strength > 0) & keep
        ).reshape(-1)
        remaining = int(torch.sum(keep).item())
        budget = max_splats - remaining
        count = min(int(candidates.numel()), max(budget, 0))
        if count <= 0:
            return False
        chosen = candidates[rng.permutation(int(candidates.numel()))[:count]]
        jitter = torch.as_tensor(
            rng.normal(scale=jitter_scale, size=(count, 3)),
            dtype=torch.float32,
            device=device,
        )
        parent_scale = torch.exp(model.log_scales[chosen])
        new_positions = torch.cat(
            [model.positions.detach(), (model.positions[chosen] + jitter).detach()],
            dim=0,
        )
        new_log_scales = torch.cat(
            [
                model.log_scales.detach(),
                (torch.log(parent_scale * 0.65)).detach(),
            ],
            dim=0,
        )
        new_opacity = torch.cat(
            [
                model.opacity_logits.detach(),
                math.log(0.9 / (1.0 - 0.9))
                + 0.2 * torch.randn(count, device=device),
            ],
            dim=0,
        )
        new_colors = torch.cat(
            [
                model.color_logits.detach(),
                model.color_logits[chosen].detach(),
            ],
            dim=0,
        )
        keep_full = torch.cat(
            [
                keep,
                torch.ones(count, dtype=torch.bool, device=device),
            ]
        )
        keep_indices = torch.nonzero(keep_full).reshape(-1)
        keep_positions = new_positions[keep_indices]
        keep_scales = new_log_scales[keep_indices]
        keep_opacity = new_opacity[keep_indices]
        keep_colors = new_colors[keep_indices]
        model.positions = torch.nn.Parameter(keep_positions)
        model.log_scales = torch.nn.Parameter(keep_scales)
        model.opacity_logits = torch.nn.Parameter(keep_opacity)
        model.color_logits = torch.nn.Parameter(keep_colors)
        return True


def train_tfrf_splat(
    images: np.ndarray,
    cameras: list[dict],
    config: dict,
    project_dir,
    task_id: str,
    sparse_points: list[list[float]] | None = None,
) -> tuple[dict, list[dict]]:
    from .main import (
        SCENE_BACKGROUND,
        compute_metrics,
        emit_status,
        emit_training_status,
        log,
        ray_bounds_from_points,
        scene_scale_from_cameras,
        status_after,
    )

    train_count = int(config.get("train_views", 16))
    steps = int(config.get("steps", 1200))
    patch = int(config.get("patch_size", 1024))
    eval_every = int(config.get("eval_every", 100))
    progress_every = int(config.get("progress_every", 10))
    learning_rate = float(config.get("learning_rate", 0.02))
    render_batch = int(config.get("render_batch_size", 256))
    render_tile = max(1, int(config.get("render_tile_size", 16)))
    dense_renderer = bool(config.get("dense_renderer", False))
    auto_dense_threshold = int(config.get("auto_dense_threshold", 20000))
    auto_dense_pixels = int(config.get("auto_dense_pixels", 1_000_000))
    tile_pair_budget = int(config.get("tile_pair_budget", 4_000_000))
    tile_work_budget = int(config.get("tile_work_budget", 2_000_000))
    tile_max_splats = int(config.get("tile_max_splats", 8192))
    disk_cutoff = float(config.get("disk_cutoff", 1.8))
    falloff_power = float(config.get("falloff_power", 1.0))
    train_original = bool(config.get("train_original_resolution", False))
    train_max_pixels = int(config.get("train_max_image_pixels", 2_000_000))
    use_cuda_tile = bool(config.get("use_cuda_tile", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_scale = scene_scale_from_cameras(cameras)

    if sparse_points:
        points = np.asarray(sparse_points, dtype=np.float32)[:, :3]
        colors = np.asarray(sparse_points, dtype=np.float32)[:, 3:6] / 255.0
        use_image_colors = bool(config.get("use_image_colors", True))
        if use_image_colors and colors.size > 0 and np.mean(colors) < 0.05:
            image_colors = color_points_from_images(points, cameras, images)
            covered = int(np.sum(np.any(image_colors > 0, axis=1)))
            log(
                f"[init] 稀疏点颜色缺失，从 {min(len(cameras), len(images))} 张"
                f"图片投影采样颜色，{covered}/{len(points)} 点获得颜色"
            )
            colors = image_colors
    else:
        points = np.asarray(
            [
                [math.cos(angle), math.sin(angle), 0.0]
                for angle in np.linspace(0.0, math.tau, 2400)
            ],
            dtype=np.float32,
        )
        colors = None
    init_scale = float(config.get("init_scale", 0.0))
    if init_scale <= 0:
        init_scale = estimate_init_scale(
            points,
            cameras,
            target_px=float(config.get("init_target_px", 1.3)),
        )
        log(f"[init] 自适应棱面半径 {init_scale:.4f}")
    point_scales = _estimate_point_scales(
        points,
        init_scale,
        multiplier=float(config.get("scale_neighbor_factor", 1.0)),
    )
    model = SplatFacetField(
        points,
        colors,
        scale=init_scale,
        scales=point_scales,
        opacity=float(config.get("init_opacity", 0.85)),
    ).to(device)

    def build_optimizer() -> torch.optim.Adam:
        return torch.optim.Adam(
            [
                {"params": [model.positions], "lr": learning_rate * 0.6},
                {"params": [model.color_logits], "lr": learning_rate},
                {"params": [model.log_scales], "lr": learning_rate * 0.6},
                {"params": [model.opacity_logits], "lr": learning_rate * 0.6},
            ],
        )

    optimizer = build_optimizer()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=learning_rate * 0.1
    )

    train_cameras = cameras[:train_count]
    eval_cameras = cameras[train_count:]
    if not eval_cameras:
        eval_cameras = cameras[-1:]
    camera_choices = np.random.default_rng(3).integers(
        0, len(train_cameras), size=steps
    )
    surface_points = (
        np.asarray(sparse_points, dtype=np.float32)[:, :3] if sparse_points else None
    )
    background = SCENE_BACKGROUND
    bg_tensor = torch.as_tensor(background, dtype=torch.float32, device=device)
    global_res = int(config.get("global_residual_resolution", 16))
    global_res_weight = float(config.get("global_residual_weight", 0.6))
    images_low = (
        _downsample_images(images, global_res) if global_res > 0 else None
    )
    if images_low is not None:
        images_low_t = torch.as_tensor(
            images_low, dtype=torch.float32, device=device
        )
    else:
        images_low_t = None
    cached_pixel_grid: torch.Tensor | None = None
    cached_grid_shape: tuple[int, int] | None = None

    def full_pixel_grid(grid_width: int, grid_height: int) -> torch.Tensor:
        nonlocal cached_pixel_grid, cached_grid_shape
        if (
            cached_pixel_grid is None
            or cached_grid_shape != (grid_width, grid_height)
        ):
            uu, vv = torch.meshgrid(
                torch.arange(grid_width, dtype=torch.float32, device=device),
                torch.arange(grid_height, dtype=torch.float32, device=device),
                indexing="xy",
            )
            cached_pixel_grid = torch.stack(
                [uu.reshape(-1), vv.reshape(-1)], dim=1
            )
            cached_grid_shape = (grid_width, grid_height)
        return cached_pixel_grid

    log(
        f"[train] TFRF-facets 开始，{len(train_cameras)} 训练视角，"
        f"{len(points)} 棱面，{steps} 步，设备 {device}，"
        f"场景尺度 {scene_scale:.4f}，原分辨率 {train_original}，"
        f"最大像素 {train_max_pixels}"
    )
    history: list[dict] = []
    start_wall = time.time()
    densify_every = int(config.get("densify_every", 50))
    densify_after = int(config.get("densify_after", 80))
    max_splats = int(config.get("max_splats", 12000))
    prune_opacity = float(config.get("prune_opacity", 0.04))
    reset_after = int(config.get("opacity_reset_after", 140))
    reset_every = int(config.get("opacity_reset_every", 100))
    reset_opacity = float(config.get("opacity_reset_value", 0.92))
    scale_upper = max(
        init_scale * float(config.get("scale_log_clamp", 2.5)), 0.2
    )
    densify_enabled = bool(config.get("densify", True))
    grad_accum = torch.zeros(len(points), dtype=torch.float32, device=device)

    def accumulate_gradients(current: torch.Tensor | None) -> None:
        if current is None:
            return
        if current.dim() > 1:
            current = current.norm(dim=1)
        if current.shape[0] != grad_accum.shape[0]:
            grad_accum.resize_(current.shape[0])
        grad_accum.mul_(0.95).add_(current)

    for step in range(steps):
        camera = train_cameras[int(camera_choices[step])]
        c2w = np.asarray(camera["c2w"], dtype=np.float32)
        width, height = camera["width"], camera["height"]
        k = camera["K"]
        fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
        near, far = ray_bounds_from_points(
            c2w[:3, 3], surface_points
        ) if surface_points is not None and len(surface_points) else (0.3, 7.6)
        rng = np.random.default_rng(step)
        if train_original:
            pixels = full_pixel_grid(width, height)
        else:
            ys = rng.integers(0, height, size=patch)
            xs = rng.integers(0, width, size=patch)
            pixels = torch.as_tensor(
                np.stack([xs, ys], axis=-1), dtype=torch.float32, device=device
            )
        rendered = _render_pixels(
            model,
            camera,
            pixels,
            width,
            height,
            bg_tensor,
            near=near,
            far=far,
            batch_size=render_batch,
            disk_cutoff=disk_cutoff,
            falloff_power=falloff_power,
            tile_size=render_tile,
            dense_renderer=dense_renderer,
            auto_dense_threshold=auto_dense_threshold,
            auto_dense_pixels=auto_dense_pixels,
            tile_pair_budget=tile_pair_budget,
            tile_max_splats=tile_max_splats,
            tile_work_budget=tile_work_budget,
            use_cuda_tile=use_cuda_tile,
        )
        if train_original:
            target = torch.as_tensor(
                images[int(camera_choices[step])],
                dtype=torch.float32,
                device=device,
            ).reshape(-1, 3)
        else:
            target = torch.as_tensor(
                images[int(camera_choices[step])][ys, xs],
                dtype=torch.float32,
                device=device,
            )
        loss = torch.mean((rendered - target) ** 2)
        if surface_points is not None:
            prior_rng = np.random.default_rng(step + 50_000)
            prior_weight = float(config.get("surface_prior_weight", 0.05))
            loss = loss + prior_weight * _density_prior_loss(
                model, surface_points, device, prior_rng
            )
        scale_target_count = min(
            model.positions.shape[0], point_scales.shape[0]
        )
        scale_loss = float(config.get("scale_prior_weight", 0.03)) * torch.mean(
            (
                model.log_scales[:scale_target_count]
                - torch.log(
                    torch.as_tensor(
                        point_scales[:scale_target_count],
                        dtype=torch.float32,
                        device=device,
                    )
                )
            )
            ** 2
        )
        loss = loss + scale_loss
        if float(config.get("opacity_concentration_weight", 0.01)) > 0:
            opac = torch.sigmoid(model.opacity_logits)
            loss = loss + float(config.get("opacity_concentration_weight", 0.01)) * torch.mean(
                opac * (1.0 - opac)
            )
        if images_low_t is not None and not train_original:
            low_camera = _lowres_camera(camera, global_res)
            low_pixels = torch.meshgrid(
                torch.arange(global_res, dtype=torch.float32, device=device),
                torch.arange(global_res, dtype=torch.float32, device=device),
                indexing="xy",
            )
            low_pixels = torch.stack(
                [low_pixels[0].reshape(-1), low_pixels[1].reshape(-1)],
                dim=1,
            )
            low_rendered = _render_pixels(
                model,
                low_camera,
                low_pixels,
                global_res,
                global_res,
                bg_tensor,
                near=near,
                far=far,
                batch_size=render_batch,
                disk_cutoff=disk_cutoff,
                falloff_power=falloff_power,
                tile_size=render_tile,
                dense_renderer=dense_renderer,
                auto_dense_threshold=auto_dense_threshold,
                auto_dense_pixels=auto_dense_pixels,
                tile_pair_budget=tile_pair_budget,
                tile_max_splats=tile_max_splats,
                tile_work_budget=tile_work_budget,
                use_cuda_tile=use_cuda_tile,
            )
            low_target = images_low_t[
                int(camera_choices[step])
            ].reshape(-1, 3)
            loss = loss + global_res_weight * torch.mean(
                (low_rendered - low_target) ** 2
            )
        if float(config.get("background_residual_weight", 0.0)) > 0:
            bg_count = min(int(config.get("background_residual_pixels", 256)), patch)
            bg_rng = np.random.default_rng(step + 101)
            bg_ys = bg_rng.integers(0, height, size=bg_count)
            bg_xs = bg_rng.integers(0, width, size=bg_count)
            bg_pixels = torch.as_tensor(
                np.stack([bg_xs, bg_ys], axis=-1),
                dtype=torch.float32,
                device=device,
            )
            bg_rendered = _render_pixels(
                model,
                camera,
                bg_pixels,
                width,
                height,
                bg_tensor,
                near=near,
                far=far,
                batch_size=render_batch,
                disk_cutoff=disk_cutoff,
                falloff_power=falloff_power,
                tile_size=render_tile,
                dense_renderer=dense_renderer,
                auto_dense_threshold=auto_dense_threshold,
                auto_dense_pixels=auto_dense_pixels,
                tile_pair_budget=tile_pair_budget,
                tile_max_splats=tile_max_splats,
                tile_work_budget=tile_work_budget,
                use_cuda_tile=use_cuda_tile,
            )
            bg_target = torch.as_tensor(
                images[int(camera_choices[step])][bg_ys, bg_xs],
                dtype=torch.float32,
                device=device,
            )
            loss = loss + float(config.get("background_residual_weight", 0.15)) * torch.mean(
                (bg_rendered - bg_target) ** 2
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        step_grad = (
            model.positions.grad.abs() if model.positions.grad is not None else None
        )
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            model.log_scales.clamp_(
                math.log(1e-4), math.log(scale_upper)
            )
            if step >= reset_after and step % reset_every == 0:
                _opacity_reset(
                    model,
                    camera,
                    surface_points,
                    device,
                    reset_opacity,
                    near,
                    far,
                )
                log(
                    f"[train] step {step} 重置可见棱面透明度到 {reset_opacity:.2f}"
                )
            if (
                densify_enabled
                and step >= densify_after
                and step % densify_every == 0
            ):
                accumulate_gradients(step_grad)
                densified = _densify_and_prune(
                    model,
                    camera,
                    surface_points,
                    device,
                    grad_accum,
                    max_splats,
                    prune_opacity,
                    jitter_scale=max(
                        float(np.median(point_scales)) * 0.25,
                        1e-4,
                    ),
                    near=near,
                    far=far,
                    rng=np.random.default_rng(step),
                )
                if densified:
                    optimizer = build_optimizer()
                    grad_accum = torch.zeros(
                        len(model.positions), dtype=torch.float32, device=device
                    )
                    log(
                        f"[train] 密化后 {len(model.positions)} 棱面，"
                        f"透明度 {float(torch.sigmoid(model.opacity_logits).mean()):.3f}"
                    )

        if step % eval_every == 0 or step == steps - 1:
            eval_index = (step // eval_every) % len(eval_cameras)
            eval_camera = eval_cameras[eval_index]
            with torch.no_grad():
                rendered_view = render_splats_view(
                    model,
                    eval_camera,
                    background,
                    scene_points=surface_points,
                    batch_size=render_batch,
                    disk_cutoff=disk_cutoff,
                    falloff_power=falloff_power,
                    tile_size=render_tile,
                    dense_renderer=dense_renderer,
                    auto_dense_threshold=auto_dense_threshold,
                    auto_dense_pixels=auto_dense_pixels,
                    tile_pair_budget=tile_pair_budget,
                    tile_max_splats=tile_max_splats,
                    tile_work_budget=tile_work_budget,
                    use_cuda_tile=use_cuda_tile,
                )
            target_view = images[len(cameras) - len(eval_cameras) + eval_index]
            metrics = compute_metrics(rendered_view, target_view)
            vram_mb = 0
            if torch.cuda.is_available():
                try:
                    vram_mb = int(
                        torch.cuda.memory_reserved(device) / (1024 * 1024)
                    )
                except Exception:
                    vram_mb = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            metrics.update(
                {
                    "step": step,
                    "totalSteps": steps,
                    "vramMB": vram_mb,
                    "elapsed": round(time.time() - start_wall, 2),
                }
            )
            history.append(metrics)
            log(
                f"[train] step {step} loss {metrics['loss']:.5f} "
                f"psnr {metrics['psnr']:.2f}"
            )
            emit_status(
                project_dir,
                {
                    "taskId": task_id,
                    "status": "running",
                    "stage": "training",
                    "progress": round((step + 1) / steps, 4),
                    "step": step,
                    "loss": metrics["loss"],
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "vramMB": vram_mb,
                    "metrics": history,
                },
            )
        elif step % progress_every == 0:
            emit_training_status(
                project_dir,
                task_id,
                step,
                steps,
                float(loss.detach().cpu()),
            )
        status_after(project_dir, task_id)

    return {"model": model, "history": history, "engine": "facets"}, history
