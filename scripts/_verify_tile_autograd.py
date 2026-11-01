"""Small-scene autograd smoke test for the CUDA tile rasterizer."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
import torch

from titan_engine.tile_cuda import tile_cuda
from titan_engine.tile_cuda.autograd import tile_render


def main():
    module = tile_cuda()
    if module is None:
        raise RuntimeError("tile_cuda 扩展不可用")
    print("tile_cuda module ok")

    device = torch.device("cuda")
    tiny_pos = torch.zeros(1, 3, dtype=torch.float32, device=device)
    tiny_pos[0, 2] = 3.0
    tiny_scale = torch.full((1,), 0.05, dtype=torch.float32, device=device)
    tiny_opacity = torch.full((1,), 1.0, dtype=torch.float32, device=device)
    tiny_colors = torch.full((1, 3), 0.8, dtype=torch.float32, device=device)
    tiny_pixels = torch.tensor(
        [[63.0, 63.0]], dtype=torch.float32, device=device
    )
    tiny_result = module.render_forward(
        tiny_pos.contiguous(),
        tiny_scale.contiguous(),
        tiny_opacity.contiguous(),
        tiny_colors.contiguous(),
        torch.eye(3, dtype=torch.float32, device=device).contiguous(),
        torch.zeros(3, dtype=torch.float32, device=device).contiguous(),
        tiny_pixels.contiguous(),
        torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32, device=device).contiguous(),
        1.2 * 64,
        1.2 * 64,
        32.0,
        32.0,
        64,
        64,
        16,
        0.1,
        10.0,
        2.2,
        1.0,
        4_000_000,
        8192,
        2_000_000,
    )
    torch.cuda.synchronize()
    print("tiny active:", int(tiny_result[8].shape[0]))
    print("tiny segments:", int(tiny_result[15].shape[0]))
    print("tiny output:", tiny_result[0].tolist())

    torch.manual_seed(5)
    width = height = 128
    n = 3000
    rng = np.random.default_rng(9)
    points = rng.uniform(-0.4, 0.4, size=(n, 3)).astype(np.float32)
    points[:, 2] += 3.0
    colors = rng.uniform(0.2, 0.8, size=(n, 3)).astype(np.float32)

    positions = torch.tensor(points, dtype=torch.float32, device=device, requires_grad=True)
    log_scales = torch.full(
        (n,), float(np.log(0.02)), dtype=torch.float32, device=device, requires_grad=True
    )
    opacity_logits = torch.full(
        (n,), 1.5, dtype=torch.float32, device=device, requires_grad=True
    )
    color_logits = torch.logit(
        torch.tensor(colors, dtype=torch.float32, device=device),
        eps=1e-4,
    ).requires_grad_(True)
    rotation = torch.eye(3, dtype=torch.float32, device=device)
    origin = torch.zeros(3, dtype=torch.float32, device=device)
    background = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32, device=device)
    fx = fy = 1.2 * width
    cx = width / 2.0
    cy = height / 2.0

    pixels = None
    if pixels is None:
        uu, vv = torch.meshgrid(
            torch.arange(width, dtype=torch.float32, device=device),
            torch.arange(height, dtype=torch.float32, device=device),
            indexing="xy",
        )
        pixels = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
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
        fx,
        fy,
        cx,
        cy,
        width,
        height,
        16,
        0.1,
        10.0,
        2.2,
        1.0,
        4_000_000,
        8192,
        2_000_000,
    )
    torch.cuda.synchronize()
    print("result count:", len(result))
    for index, tensor in enumerate(result):
        value = (
            f"mean={float(tensor.mean()):.4f}"
            if tensor.dtype in (torch.float32, torch.float64)
            else f"sum={float(tensor.sum()):.1f}"
        )
        print(f"  [{index}] {str(tuple(tensor.shape)):>14} dtype={tensor.dtype} {value}")
    (
        output,
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
    ) = result
    print("active ids:", int(active_ids.shape[0]))
    print("segments:", int(segments.shape[0]))
    print("total tile pairs:", int(tile_pair_count.sum().item()))
    print("pixel order count:", int(pixel_order.shape[0]))
    print("output mean:", float(output.mean()))
    reordered = torch.empty_like(output)
    reordered[pixel_order] = output
    print("reordered mean:", float(reordered.mean()))
    if float(reordered.mean()) <= 0.01:
        raise AssertionError("reordered output is blank")

    output_flat = tile_render(
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
    )
    torch.cuda.synchronize()
    if output_flat.shape != (width * height, 3):
        raise AssertionError("tile_render should return flat pixel output")
    loss = torch.mean((output_flat - 0.5) ** 2)
    loss.backward()
    torch.cuda.synchronize()
    for name, grad in [
        ("positions", positions.grad),
        ("log_scales", log_scales.grad),
        ("opacity_logits", opacity_logits.grad),
        ("color_logits", color_logits.grad),
    ]:
        if grad is None:
            raise AssertionError(f"{name} grad is None")
        print(
            f"{name} grad: finite={torch.isfinite(grad).all().item()} "
            f"abs_mean={float(grad.abs().mean()):.6f} nonzero={int((grad.abs() > 0).sum())}"
        )
        if not torch.isfinite(grad).all() or float(grad.abs().mean()) <= 1e-12:
            raise AssertionError(f"{name} grad invalid")
    print("autograd smoke ok")


if __name__ == "__main__":
    main()
