"""Autograd wrapper for the Titan CUDA tile rasterizer."""

import math

import torch

from . import tile_cuda


class _TileRasterFunction(torch.autograd.Function):
    """Differentiable tile rasterization over full pixel sets.

    Forward takes the raw model parameters (positions, log scales, logit
    opacity, logit color) and camera tensors, and returns the composited
    image for the requested pixels. The CUDA extension keeps the tile-splat
    pair expansion entirely on the GPU and returns the sorted pairs needed by
    the backward kernel.
    """

    @staticmethod
    def forward(
        ctx,
        positions,
        log_scales,
        opacity_logits,
        color_logits,
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
        tile_size,
        near,
        far,
        disk_cutoff,
        falloff_power,
        tile_pair_budget,
        tile_max_splats,
        tile_work_budget,
    ):
        if pixels is None:
            uu, vv = torch.meshgrid(
                torch.arange(int(width), dtype=torch.float32, device=positions.device),
                torch.arange(int(height), dtype=torch.float32, device=positions.device),
                indexing="xy",
            )
            pixels = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
        module = tile_cuda()
        if module is None:
            raise RuntimeError("tile_cuda 扩展不可用")
        scale = torch.exp(log_scales)
        opacity = torch.sigmoid(opacity_logits)
        colors = torch.sigmoid(color_logits)
        result = module.render_forward(
            positions.contiguous(),
            scale.contiguous(),
            opacity.contiguous(),
            colors.contiguous(),
            rotation.contiguous(),
            origin.contiguous(),
            pixels.contiguous(),
            background.contiguous(),
            float(fx),
            float(fy),
            float(cx),
            float(cy),
            int(width),
            int(height),
            int(tile_size),
            float(near),
            float(far),
            float(disk_cutoff),
            float(falloff_power),
            int(tile_pair_budget),
            int(tile_max_splats),
            int(tile_work_budget),
        )
        if len(result) != 17:
            raise RuntimeError(
                f"tile_cuda forward 返回 {len(result)} 个张量，预期 17"
            )
        pixel_order = result[1]
        sorted_output = result[0]
        output = torch.empty_like(sorted_output)
        output[pixel_order] = sorted_output
        ctx.save_for_backward(
            result[1],
            result[2],
            result[3],
            result[4],
            result[5],
            result[6],
            result[7],
            result[8],
            result[9],
            result[10],
            result[11],
            result[12],
            result[13],
            result[14],
            result[15],
            result[16],
            scale,
            opacity,
            colors,
            background,
            rotation,
        )
        ctx.fx = float(fx)
        ctx.fy = float(fy)
        ctx.cx = float(cx)
        ctx.cy = float(cy)
        ctx.tile_size = int(tile_size)
        ctx.tiles_x = math.ceil(int(width) / int(tile_size))
        ctx.tiles_y = math.ceil(int(height) / int(tile_size))
        ctx.disk_cutoff = float(disk_cutoff)
        ctx.falloff_power = float(falloff_power)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            pixel_order,
            sorted_pixels,
            tile_pixel_start,
            tile_pixel_count,
            tile_pair_start,
            tile_pair_count,
            sorted_splats,
            active_ids,
            compact_u,
            compact_v,
            compact_sigma_u,
            compact_sigma_v,
            compact_z,
            pixel_work_offset,
            segments,
            seg_trans,
            scale,
            opacity,
            colors,
            background,
            rotation,
        ) = ctx.saved_tensors
        module = tile_cuda()
        if module is None:
            raise RuntimeError("tile_cuda 扩展不可用")
        grads = module.render_backward(
            grad_output.contiguous(),
            pixel_order,
            sorted_pixels,
            tile_pair_start,
            tile_pair_count,
            sorted_splats,
            active_ids,
            compact_u,
            compact_v,
            compact_sigma_u,
            compact_sigma_v,
            compact_z,
            pixel_work_offset,
            opacity,
            colors,
            background,
            rotation,
            ctx.fx,
            ctx.fy,
            ctx.cx,
            ctx.cy,
            ctx.tile_size,
            ctx.tiles_x,
            ctx.tiles_y,
            ctx.disk_cutoff,
            ctx.falloff_power,
            segments,
            seg_trans,
        )
        if len(grads) != 4:
            raise RuntimeError(
                f"tile_cuda backward 返回 {len(grads)} 个张量，预期 4"
            )
        grad_positions, grad_scale, grad_opacity, grad_colors = grads
        grad_log_scales = grad_scale * scale
        grad_opacity_logits = grad_opacity * opacity * (1.0 - opacity)
        grad_color_logits = grad_colors * colors * (1.0 - colors)
        return (
            grad_positions,
            grad_log_scales,
            grad_opacity_logits,
            grad_color_logits,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def tile_render(
    positions,
    log_scales,
    opacity_logits,
    color_logits,
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
    tile_size=16,
    near=0.1,
    far=10.0,
    disk_cutoff=2.2,
    falloff_power=1.0,
    tile_pair_budget=4_000_000,
    tile_max_splats=8192,
    tile_work_budget=2_000_000,
):
    """Render the requested pixels with the CUDA tile rasterizer."""
    return _TileRasterFunction.apply(
        positions,
        log_scales,
        opacity_logits,
        color_logits,
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
        tile_size,
        near,
        far,
        disk_cutoff,
        falloff_power,
        tile_pair_budget,
        tile_max_splats,
        tile_work_budget,
    )
