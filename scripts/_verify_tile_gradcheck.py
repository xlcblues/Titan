"""Finite-difference gradient check for the CUDA tile rasterizer."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
import torch

from titan_engine.tile_cuda.autograd import tile_render
from titan_engine.tile_cuda import tile_cuda


def _cpu_render(positions, scale, opacity, colors, rotation, origin, pixels, fx, fy, cx, cy, disk_cutoff):
    rel = positions - origin[None, :]
    cam = rel @ rotation
    z = cam[:, 2]
    u = fx * cam[:, 0] / z + cx
    v = fy * cam[:, 1] / z + cy
    su = scale * fx / z
    sv = scale * fy / z
    out = np.zeros((pixels.shape[0], 3), dtype=np.float64)
    for p in range(pixels.shape[0]):
        du = pixels[p, 0] - u
        dv = pixels[p, 1] - v
        sq = du * du / (2.0 * su * su) + dv * dv / (2.0 * sv * sv)
        cutoff = 0.5 * disk_cutoff * disk_cutoff
        weight = np.where(sq < cutoff, np.exp(-sq), 0.0)
        alpha = np.clip(opacity * weight, 0.0, 1.0)
        order = np.argsort(z)
        trans = 1.0
        acc = np.zeros(3)
        for i in order:
            acc += alpha[i] * trans * colors[i]
            trans *= 1.0 - alpha[i]
        out[p] = acc + trans * 0.1
    return out


def main():
    torch.manual_seed(7)
    device = torch.device("cuda")
    n = 4
    positions = torch.tensor(
        [[-0.2, 0.1, 2.8], [0.15, -0.1, 3.0], [0.05, 0.2, 3.2], [-0.1, -0.2, 3.1]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    log_scales = torch.full((n,), np.log(0.06), dtype=torch.float32, device=device, requires_grad=True)
    opacity_logits = torch.tensor([1.0, -0.5, 0.5, 0.2], dtype=torch.float32, device=device, requires_grad=True)
    color_logits = torch.logit(
        torch.tensor([[0.9, 0.1, 0.2], [0.1, 0.8, 0.3], [0.2, 0.3, 0.9], [0.5, 0.5, 0.5]], dtype=torch.float32, device=device),
        eps=1e-4,
    ).requires_grad_(True)
    rotation = torch.eye(3, dtype=torch.float32, device=device)
    origin = torch.zeros(3, dtype=torch.float32, device=device)
    background = torch.full((3,), 0.1, dtype=torch.float32, device=device)
    width = height = 64
    fx = fy = 1.2 * width
    cx = cy = width / 2.0
    pixels = torch.tensor(
        [[31.0, 31.0], [32.0, 32.0], [33.0, 32.0], [30.0, 34.0], [35.0, 33.0]],
        dtype=torch.float32,
        device=device,
    )

    rendered = tile_render(
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
    cpu_ref = _cpu_render(
        positions.detach().cpu().numpy(),
        torch.exp(log_scales).detach().cpu().numpy(),
        torch.sigmoid(opacity_logits).detach().cpu().numpy(),
        torch.sigmoid(color_logits).detach().cpu().numpy(),
        rotation.cpu().numpy(),
        origin.cpu().numpy(),
        pixels.cpu().numpy(),
        fx,
        fy,
        cx,
        cy,
        2.2,
    )
    err = float(torch.abs(rendered.cpu() - torch.as_tensor(cpu_ref, dtype=torch.float32)).max())
    print("forward max err vs cpu:", err)
    diff = rendered.detach().cpu().numpy() - cpu_ref
    worst = np.argsort(-np.abs(diff).sum(axis=1))[:3]
    for p in worst:
        print(
            f"pixel {p}: cuda={rendered[p].detach().cpu().tolist()} "
            f"cpu={cpu_ref[p].tolist()}"
        )
    module = tile_cuda()
    raw = module.render_forward(
        positions.detach().contiguous(),
        torch.exp(log_scales.detach()).contiguous(),
        torch.sigmoid(opacity_logits.detach()).contiguous(),
        torch.sigmoid(color_logits.detach()).contiguous(),
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
    sorted_out = raw[0]
    order = raw[1]
    sorted_px = raw[2]
    manual = torch.empty_like(sorted_out)
    manual[order] = sorted_out
    print("sorted order:", order.detach().cpu().tolist())
    print("sorted pixels:", sorted_px.detach().cpu().tolist())
    print("sorted out:", sorted_out.detach().cpu().tolist())
    print("manual reorder:", manual.detach().cpu().tolist())
    print("tile pair start:", raw[5].detach().cpu().tolist())
    print("tile pair count:", raw[6].detach().cpu().tolist())
    print("sorted splats:", raw[7].detach().cpu().tolist())
    print("active ids:", raw[8].detach().cpu().tolist())
    print("compact u:", raw[9].detach().cpu().tolist())
    print("compact v:", raw[10].detach().cpu().tolist())
    print("segments:", raw[15].detach().cpu().tolist())
    print("pixel work offset:", raw[14].detach().cpu().tolist())
    if err > 2e-3:
        raise AssertionError("forward mismatch")

    loss = torch.sum((rendered - 0.5) ** 2)
    loss.backward()
    torch.cuda.synchronize()
    params = [
        ("positions", positions, 1e-3),
        ("log_scales", log_scales, 1e-3),
        ("opacity_logits", opacity_logits, 1e-3),
        ("color_logits", color_logits, 1e-3),
    ]
    for name, param, eps in params:
        analytic = param.grad.detach().cpu().numpy()
        numeric = np.zeros_like(analytic)
        base = param.detach().clone()
        for index in np.ndindex(base.shape):
            plus = base.clone()
            minus = base.clone()
            plus[index] += eps
            minus[index] -= eps

            def eval_at(value):
                candidate = value.clone().requires_grad_(False)
                current = {
                    "positions": positions.detach().clone(),
                    "log_scales": log_scales.detach().clone(),
                    "opacity_logits": opacity_logits.detach().clone(),
                    "color_logits": color_logits.detach().clone(),
                }
                current[name] = candidate
                with torch.no_grad():
                    out = tile_render(
                        current["positions"],
                        current["log_scales"],
                        current["opacity_logits"],
                        current["color_logits"],
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
                return float(torch.sum((out - 0.5) ** 2))

            numeric[index] = (eval_at(plus) - eval_at(minus)) / (2.0 * eps)
        denom = max(np.abs(numeric).max(), 1e-9)
        rel = np.abs(analytic - numeric).max() / denom
        print(f"{name}: rel err {rel:.6f}, analytic max {np.abs(analytic).max():.5f}, numeric max {np.abs(numeric).max():.5f}")
        if name == "positions":
            diff = analytic - numeric
            idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
            print(
                f"  worst idx {idx}: analytic {analytic[idx]:.8f} "
                f"numeric {numeric[idx]:.8f} diff {diff[idx]:.8f}"
            )
            print("  analytic:", analytic.tolist())
            print("  numeric:", numeric.tolist())
        if rel > 0.08:
            raise AssertionError(f"{name} gradient mismatch")
    print("gradcheck ok")


if __name__ == "__main__":
    main()
