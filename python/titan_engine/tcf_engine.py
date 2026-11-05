"""Titan Coherence Field (TCF) engine.

TCF is a self-designed implicit scene representation. It stores a
multi-resolution coherence lattice and answers appearance queries with a
small shared readout. Rendering is a coherence voting process: every
sample along a ray casts an appearance vote, and the votes are weighted by
how coherent the local field is and whether the ray has already been
carried to that point.
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn


def log(message: str) -> None:
    print(message, flush=True)


def _read_control(project_dir: Path) -> str | None:
    control = project_dir / "control.json"
    try:
        return json.loads(control.read_text(encoding="utf-8")).get("command")
    except (OSError, ValueError, KeyError):
        return None


def _device_peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.max_memory_reserved(0) / (1024 * 1024)
    except Exception:
        return 0.0


def _device_reserved_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.memory_reserved(0) / (1024 * 1024)
    except Exception:
        return 0.0


class CoherentLattice(nn.Module):
    """Multi-resolution hash lattice used as the scene's memory."""

    def __init__(
        self,
        levels: int = 8,
        base_resolution: int = 32,
        scale: float = 1.6,
        max_resolution: int = 1024,
        entries_per_level: int = 196608,
        feature_dim: int = 16,
        seed: int = 11,
    ) -> None:
        super().__init__()
        self.levels = max(2, int(levels))
        self.base_resolution = int(base_resolution)
        self.scale = float(scale)
        self.max_resolution = int(max_resolution)
        self.entries_per_level = int(entries_per_level)
        self.feature_dim = int(feature_dim)

        growth = self.scale ** np.arange(self.levels)
        resolutions = np.minimum(
            self.max_resolution,
            np.floor(self.base_resolution * growth).astype(np.int64),
        )
        resolutions = np.maximum(resolutions, 2)
        self.register_buffer("resolutions", torch.from_numpy(resolutions).long())
        self.tables = nn.ModuleList(
            [
                nn.Embedding(self.entries_per_level, self.feature_dim)
                for _ in range(self.levels)
            ]
        )
        generator = torch.Generator().manual_seed(seed)
        for table in self.tables:
            nn.init.normal_(table.weight, mean=0.0, std=0.08, generator=generator)

    def params(self) -> list[nn.Parameter]:
        return [table.weight for table in self.tables]


class CoherenceReadout(nn.Module):
    """Shared tiny readout that turns lattice context into a vote."""

    def __init__(
        self,
        feature_dim: int = 8,
        levels: int = 8,
        position_levels: int = 6,
        direction_levels: int = 3,
        hidden: int = 144,
        seed: int = 12,
        context_scale: float = 8.0,
        coherence_bias: float = 0.0,
        capture_bias: float = 0.5,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.levels = int(levels)
        self.position_levels = int(position_levels)
        self.direction_levels = int(direction_levels)
        self.context_scale = float(context_scale)
        self.coherence_bias = float(coherence_bias)
        self.capture_bias = float(capture_bias)
        position_in = 3 * (1 + 2 * self.position_levels)
        direction_in = 3 * (1 + 2 * self.direction_levels)
        in_dim = self.feature_dim * self.levels + position_in + direction_in
        self.linear_in = nn.Linear(in_dim, hidden, bias=True)
        self.linear_hidden = nn.Linear(hidden, hidden, bias=True)
        self.linear_out = nn.Linear(hidden, 5, bias=True)
        generator = torch.Generator().manual_seed(seed)
        nn.init.xavier_uniform_(self.linear_in.weight, generator=generator)
        nn.init.xavier_uniform_(self.linear_hidden.weight, generator=generator)
        nn.init.xavier_uniform_(self.linear_out.weight, generator=generator)
        nn.init.zeros_(self.linear_in.bias)
        nn.init.zeros_(self.linear_hidden.bias)
        nn.init.zeros_(self.linear_out.bias)
        with torch.no_grad():
            self.linear_out.bias[3] = self.coherence_bias
            self.linear_out.bias[4] = self.capture_bias

    def forward(
        self,
        context: torch.Tensor,
        positions: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            context = context.float() * self.context_scale
            pos = self._encode(positions.float(), self.position_levels)
            direction = self._encode(directions.float(), self.direction_levels)
            hidden = torch.relu(
                self.linear_in(torch.cat([context, pos, direction], dim=-1))
            )
            hidden = torch.relu(self.linear_hidden(hidden))
            return self.linear_out(hidden)

    @staticmethod
    def _encode(values: torch.Tensor, levels: int) -> torch.Tensor:
        encoded = [values]
        for level in range(levels):
            freq = (2.0**level) * math.pi
            encoded.append(torch.sin(values * freq))
            encoded.append(torch.cos(values * freq))
        return torch.cat(encoded, dim=-1)


class TitanCoherenceField(nn.Module):
    """The complete TCF scene model."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        levels = int(config.get("lattice_levels", 8))
        self.lattice = CoherentLattice(
            levels=levels,
            base_resolution=int(config.get("lattice_base_resolution", 32)),
            scale=float(config.get("lattice_scale", 1.6)),
            max_resolution=int(config.get("lattice_max_resolution", 1024)),
            entries_per_level=int(config.get("lattice_entries", 196608)),
            feature_dim=int(config.get("lattice_feature_dim", 16)),
        )
        self.readout = CoherenceReadout(
            feature_dim=int(config.get("lattice_feature_dim", 16)),
            levels=levels,
            position_levels=int(config.get("position_levels", 6)),
            direction_levels=int(config.get("direction_levels", 3)),
            hidden=int(config.get("readout_hidden", 144)),
            context_scale=float(config.get("lattice_context_scale", 0.5)),
            coherence_bias=float(
                config.get("coherence_bias", config.get("density_bias", 0.0))
            ),
            capture_bias=float(config.get("capture_bias", 0.0)),
        )
        self.voting_temperature_start = float(
            config.get("voting_temperature_start", 0.5)
        )
        self.voting_temperature_end = float(
            config.get("voting_temperature_end", 3.0)
        )
        self.anchor_blend_start = float(
            config.get("anchor_blend_start", 1.0)
        )
        self.anchor_blend_end = float(config.get("anchor_blend_end", 0.5))
        self.anchor_blend_fraction = float(
            config.get("anchor_blend_fraction", 1.0)
        )
        anchor_peak = float(config.get("anchor_peak", 6.0))
        self.logit_floor = float(config.get("logit_floor", -3.0))
        self.logit_floor_slope = float(
            config.get("logit_floor_slope", 0.1)
        )
        self.consensus_temperature_start = float(
            config.get("consensus_temperature_start", 1.5)
        )
        self.consensus_temperature_end = float(
            config.get("consensus_temperature_end", 0.6)
        )
        self.consensus_near_bias = float(
            config.get("consensus_near_bias", 0.5)
        )
        self.consensus_top_k = int(config.get("consensus_top_k", 2))
        self.render_top_k = max(1, int(config.get("render_top_k", 1)))
        self.coarse_samples = int(config.get("coarse_samples", 32))
        self.fine_samples = int(config.get("fine_samples", 16))
        self.consensus_sharp_start = float(
            config.get("consensus_sharp_start", 0.0)
        )
        self.consensus_sharp_end = float(
            config.get("consensus_sharp_end", 1.0)
        )
        self.register_parameter(
            "background",
            nn.Parameter(
                torch.tensor([0.08, 0.09, 0.10], dtype=torch.float32)
            ),
        )
        if "background_color" in config:
            with torch.no_grad():
                self.background.copy_(
                    torch.tensor(
                        [float(v) for v in config["background_color"]],
                        dtype=torch.float32,
                    )
                )
        self.register_buffer(
            "coherence_temperature",
            torch.tensor(self.voting_temperature_end, dtype=torch.float32),
        )
        self.register_buffer(
            "anchor_blend",
            torch.tensor(self.anchor_blend_end, dtype=torch.float32),
        )
        self.register_buffer(
            "anchor_peak",
            torch.tensor(anchor_peak, dtype=torch.float32),
        )
        self.register_buffer(
            "alpha_scale",
            torch.tensor(float(config.get("alpha_scale", 6.0)), dtype=torch.float32),
        )
        self.capture_floor_start = float(config.get("capture_floor_start", 1.2))
        self.capture_floor_end = float(config.get("capture_floor_end", 0.06))
        self.register_buffer(
            "capture_floor",
            torch.tensor(self.capture_floor_end, dtype=torch.float32),
        )
        self.register_buffer(
            "consensus_temperature",
            torch.tensor(
                self.consensus_temperature_end, dtype=torch.float32
            ),
        )
        self.register_buffer(
            "consensus_sharp",
            torch.tensor(self.consensus_sharp_end, dtype=torch.float32),
        )
        self.register_buffer(
            "logit_clamp",
            torch.tensor(float(config.get("logit_clamp", 10.0)), dtype=torch.float32),
        )
        self.register_buffer(
            "background_target",
            torch.tensor(
                [float(v) for v in config.get("background_target", [0.08, 0.09, 0.10])],
                dtype=torch.float32,
            ),
        )
        self.peak_render_start = float(config.get("peak_render_start", 1.0))
        self.peak_render_end = float(config.get("peak_render_end", 3.0))
        self.register_buffer(
            "peak_power",
            torch.tensor(self.peak_render_end, dtype=torch.float32),
        )
        self.fine_anchor_fraction = float(
            config.get("fine_anchor_fraction", 0.4)
        )
        self.coarse_sampling_mode = str(
            config.get("coarse_sampling_mode", "physical")
        )
        self.fine_sampling_mode = str(
            config.get("fine_sampling_mode", "importance")
        )
        self.anchor_spread_fraction = float(
            config.get("anchor_spread_fraction", 0.0)
        )
        self.consensus_include_coarse = bool(
            config.get("consensus_include_coarse", False)
        )

    def lattice_mb(self) -> float:
        total = sum(p.numel() for p in self.lattice.params())
        return total * 4 / (1024 * 1024)

    def hash_feature(self, points: torch.Tensor) -> torch.Tensor:
        """Trilinear lookup over every lattice level."""
        if points.numel() == 0:
            return torch.empty(
                (0, self.lattice.levels * self.lattice.feature_dim),
                dtype=points.dtype,
                device=points.device,
            )
        levels = self.lattice.levels
        feature_dim = self.lattice.feature_dim
        points = points.reshape(-1, 3)
        resolution = self.lattice.resolutions
        scaled = points[None, :, :] * resolution[:, None, None].to(points.dtype)
        base = torch.floor(scaled - 0.5).long()
        fraction = scaled - 0.5 - base.to(points.dtype)
        lo = torch.remainder(base, resolution[:, None, None].long())
        hi = torch.remainder(lo + 1, resolution[:, None, None].long())

        level_index = torch.arange(
            levels, device=points.device, dtype=torch.long
        ).reshape(levels, 1)
        x, y, z = lo[..., 0], lo[..., 1], lo[..., 2]
        xh, yh, zh = hi[..., 0], hi[..., 1], hi[..., 2]
        table_size = torch.tensor(
            self.lattice.entries_per_level,
            device=points.device,
            dtype=torch.long,
        )
        index000 = self._hash_index(level_index, x, y, z, table_size)
        index001 = self._hash_index(level_index, x, y, zh, table_size)
        index010 = self._hash_index(level_index, x, yh, z, table_size)
        index011 = self._hash_index(level_index, x, yh, zh, table_size)
        index100 = self._hash_index(level_index, xh, y, z, table_size)
        index101 = self._hash_index(level_index, xh, y, zh, table_size)
        index110 = self._hash_index(level_index, xh, yh, z, table_size)
        index111 = self._hash_index(level_index, xh, yh, zh, table_size)

        table_weights = torch.stack([table.weight for table in self.lattice.tables])
        f000 = table_weights[level_index, index000]
        f001 = table_weights[level_index, index001]
        f010 = table_weights[level_index, index010]
        f011 = table_weights[level_index, index011]
        f100 = table_weights[level_index, index100]
        f101 = table_weights[level_index, index101]
        f110 = table_weights[level_index, index110]
        f111 = table_weights[level_index, index111]

        xd = fraction[..., 0:1]
        yd = fraction[..., 1:2]
        zd = fraction[..., 2:3]
        c00 = f000 * (1 - xd) + f100 * xd
        c01 = f001 * (1 - xd) + f101 * xd
        c10 = f010 * (1 - xd) + f110 * xd
        c11 = f011 * (1 - xd) + f111 * xd
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        result = c0 * (1 - zd) + c1 * zd
        result = result.permute(1, 0, 2).reshape(-1, levels * feature_dim)
        return result

    @staticmethod
    def _hash_index(
        level: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        table_size: torch.Tensor,
    ) -> torch.Tensor:
        hashed = torch.bitwise_xor(
            torch.bitwise_xor(
                torch.remainder(x * 73856093 + level * 2654435761, 2**31 - 1),
                y * 19349663,
            ),
            z * 83492791,
        )
        return torch.remainder(hashed, table_size)

    def readout_logits(
        self,
        points: torch.Tensor,
        directions: torch.Tensor,
        chunk: int = 262144,
    ) -> torch.Tensor:
        context = self.hash_feature(points)
        if context.shape[0] <= chunk:
            return self.readout(context, points, directions)
        outputs = []
        for start in range(0, context.shape[0], chunk):
            stop = min(start + chunk, context.shape[0])
            outputs.append(
                self.readout(
                    context[start:stop],
                    points[start:stop],
                    directions[start:stop],
                )
            )
        return torch.cat(outputs, dim=0)


def _capture_logit(
    coherence: torch.Tensor,
    gates: torch.Tensor,
    temperature: float | torch.Tensor = 1.0,
    inverse_depth: torch.Tensor | None = None,
    anchor_inverse: torch.Tensor | None = None,
    anchor_spread: float | torch.Tensor | None = None,
    anchor_blend: float | torch.Tensor = 0.0,
    anchor_peak: float = 0.0,
    anchor_mask: torch.Tensor | None = None,
    logit_clamp: float = 10.0,
    logit_floor: float = -3.0,
    logit_floor_slope: float = 0.1,
) -> torch.Tensor:
    """Compute the capture score before it is turned into opacity.

    The gate is a positive vote strength and the coherence field sharpens it
    along the ray. A sparse point-cloud anchor raises the score around the
    expected surface during early training. A leaky lower bound keeps gates
    from drifting so far negative that opacity and its gradients collapse.
    """
    coherence = coherence.float()
    gates = gates.float()
    temperature = temperature.float()
    capture_logit = gates + 0.5 * temperature * torch.tanh(coherence)
    if inverse_depth is not None and anchor_inverse is not None:
        anchor_inverse = anchor_inverse.to(capture_logit.dtype)
        if anchor_inverse.dim() == 2 and anchor_inverse.shape[-1] != 1:
            anchor_inverse = anchor_inverse[:, :1]
        while anchor_inverse.dim() > 1 and anchor_inverse.shape[-1] == 1:
            anchor_inverse = anchor_inverse.squeeze(-1)
        if anchor_inverse.dim() == 1:
            anchor_inverse = anchor_inverse.reshape(-1, 1, 1)
        if anchor_spread is None:
            anchor_spread = 0.15
        if isinstance(anchor_spread, (int, float)):
            spread = torch.as_tensor(
                anchor_spread,
                device=capture_logit.device,
                dtype=capture_logit.dtype,
            )
        else:
            spread = anchor_spread.to(capture_logit.dtype)
            if spread.dim() == 2 and spread.shape[-1] != 1:
                spread = spread[:, :1]
            while spread.dim() > 1 and spread.shape[-1] == 1:
                spread = spread.squeeze(-1)
            if spread.dim() == 1:
                spread = spread.reshape(-1, 1, 1)
        anchor_score = -(
            (inverse_depth.unsqueeze(-1) - anchor_inverse) ** 2
            / (2.0 * spread**2 + 1e-6)
        )
        if isinstance(anchor_blend, (int, float)):
            blend = torch.full_like(
                anchor_inverse,
                float(anchor_blend),
                dtype=capture_logit.dtype,
            )
        else:
            blend = anchor_blend.to(capture_logit.dtype).reshape(-1, 1, 1)
        if anchor_mask is not None:
            mask = (anchor_mask > 0).to(capture_logit.dtype).reshape(-1, 1, 1)
            blend = blend * mask
        capture_logit = capture_logit + blend * (
            float(anchor_peak) + anchor_score
        )
    capture_logit = capture_logit.clamp(
        min=-float(logit_clamp), max=float(logit_clamp)
    )
    floor = float(logit_floor)
    if floor > -float(logit_clamp):
        below = torch.relu(
            torch.full_like(capture_logit, floor) - capture_logit
        )
        capture_logit = torch.clamp(capture_logit, min=floor) - (
            float(logit_floor_slope) * below
        )
    return capture_logit


def _soft_top_k(
    weights: torch.Tensor,
    top_k: int,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Differentiable approximation of keeping the top-k positive weights.

    A hard ``torch.topk`` + scatter blocks gradients from reaching samples
    that are not selected, so the color loss cannot teach the vote which
    depth to choose. This version gates by the (k+1)-th largest value with a
    smooth sigmoid, preserving near top-k sparsity while keeping selection
    gradients alive.
    """
    if top_k <= 0 or weights.shape[1] <= top_k:
        return weights
    flat = weights[..., 0]
    sorted_values, _ = torch.sort(flat, dim=1, descending=True)
    threshold = sorted_values[:, top_k : top_k + 1]
    scaled = (flat - threshold) / max(float(temperature), 1e-4)
    mask = torch.sigmoid(scaled).clamp(min=0.0, max=1.0)
    masked = weights * mask.unsqueeze(-1)
    return masked / masked.sum(dim=1, keepdim=True).clamp(min=1e-6)


def _consensus_weights(
    capture_logit: torch.Tensor,
    inverse_depth: torch.Tensor | None,
    temperature: float = 1.0,
    near_bias: float = 0.0,
    top_k: int = 2,
) -> torch.Tensor:
    """Turn capture scores into a small set of coherent votes per ray.

    Every sample along a ray competes for one consensus vote. The scene is
    one continuous volume with many possible depth layers; there is no
    separate background slot. Keeping only the top-k sample votes and
    renormalizing them prevents many small opacity layers from averaging
    colors into fog.
    """
    temperature = max(float(temperature), 0.25)
    scores = capture_logit[..., 0] / temperature
    if float(near_bias) > 0.0 and inverse_depth is not None:
        inv = inverse_depth.float()
        inv_min = inv.min(dim=1, keepdim=True).values
        inv_max = inv.max(dim=1, keepdim=True).values
        near_norm = (inv - inv_min) / (inv_max - inv_min + 1e-6)
        scores = scores + float(near_bias) * near_norm
    probs = torch.softmax(scores, dim=1)
    sample_weights = probs.unsqueeze(-1)
    sample_weights = _soft_top_k(sample_weights, int(top_k))
    return sample_weights


def _voting_weights(
    coherence: torch.Tensor,
    gates: torch.Tensor,
    temperature: float | torch.Tensor = 1.0,
    inverse_depth: torch.Tensor | None = None,
    anchor_inverse: torch.Tensor | None = None,
    anchor_spread: float | torch.Tensor | None = None,
    anchor_blend: float | torch.Tensor = 0.0,
    anchor_peak: float = 0.0,
    anchor_mask: torch.Tensor | None = None,
    segment_lengths: torch.Tensor | None = None,
    alpha_scale: float = 1.0,
    capture_floor: float = 0.0,
    logit_clamp: float = 10.0,
    logit_floor: float = -3.0,
    logit_floor_slope: float = 0.1,
    return_alpha: bool = False,
) -> torch.Tensor:
    """Compute alpha-composited sample weights.

    Opacity is the physical absorption of a ray segment:
    ``alpha = 1 - exp(-strength * length * alpha_scale)``. Weights are the
    volume-rendering product of opacity and accumulated transmittance. The
    final color chain uses :func:`_consensus_weights` instead, so these
    weights primarily drive geometry, visibility and point extraction.
    """
    capture_logit = _capture_logit(
        coherence,
        gates,
        temperature,
        inverse_depth=inverse_depth,
        anchor_inverse=anchor_inverse,
        anchor_spread=anchor_spread,
        anchor_blend=anchor_blend,
        anchor_peak=anchor_peak,
        anchor_mask=anchor_mask,
        logit_clamp=logit_clamp,
        logit_floor=logit_floor,
        logit_floor_slope=logit_floor_slope,
    )
    strength = torch.nn.functional.softplus(capture_logit)
    if float(capture_floor) > 0.0:
        strength = strength + float(capture_floor)

    if segment_lengths is None:
        if inverse_depth is not None:
            inv = inverse_depth.float()
            segment_lengths = torch.abs(
                inv[:, 1:] - inv[:, :-1]
            ) / inv[:, :-1].clamp(min=1e-6).square()
            segment_lengths = torch.cat(
                [segment_lengths, segment_lengths[:, -1:]], dim=1
            )
        else:
            segment_lengths = torch.ones_like(strength)
    segment_lengths = segment_lengths.to(strength.dtype)
    if segment_lengths.dim() == strength.dim() - 1:
        segment_lengths = segment_lengths.unsqueeze(-1)
    exponent = (strength * segment_lengths * float(alpha_scale)).clamp(
        max=30.0
    )
    alpha = 1.0 - torch.exp(-exponent)
    transmittance = torch.cumprod(1.0 - alpha, dim=1)
    transmittance = torch.cat(
        [torch.ones_like(transmittance[:, :1]), transmittance[:, :-1]],
        dim=1,
    )
    weights = alpha * transmittance
    if return_alpha:
        return weights, alpha
    return weights


def _visible_topk_render_weights(
    consensus_weights: torch.Tensor,
    alpha: torch.Tensor,
    physical_weights: torch.Tensor,
    peak_power: float,
    top_k: int,
) -> torch.Tensor:
    """Scale consensus votes by physical visibility, then keep the top-k.

    A ray is one continuous scene path: physical alpha tells which depth
    layers are actually visible, and the consensus vote decides which of
    those visible layers best explain the color. Samples hidden behind an
    opaque layer no longer compete for the color vote, so a multi-object
    scene composites front-to-back instead of averaging unrelated layers.

    The forward pass keeps exactly one winning sample (top-1) so a pixel is
    explained by a single surface color instead of a foggy average. The
    backward pass substitutes a soft vote so the color loss can move which
    sample wins; a hard argmax normalization has no gradient at all and
    leaves the vote frozen at whatever depth the geometry terms happened to
    prefer.
    """
    visible = consensus_weights * physical_weights
    visible = visible / visible.sum(dim=1, keepdim=True).clamp(min=1e-6)
    visible = _soft_top_k(visible, int(top_k))
    if float(peak_power) <= 1.0:
        return visible
    peak_scale = visible.max(dim=1, keepdim=True).values.clamp(min=1e-6)
    surface = (visible / peak_scale).clamp(min=0.0) ** float(peak_power)
    hard = torch.zeros_like(surface)
    max_index = surface.argmax(dim=1, keepdim=True)
    hard.scatter_(1, max_index, 1.0)
    soft = surface / surface.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return hard + soft - soft.detach()


def _sample_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
    near: float,
    far: float,
    samples: int,
    fine_inverse: torch.Tensor | None = None,
    fine_spread: float | None = None,
    jitter: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    inverse_near = 1.0 / max(float(near), 1e-4)
    inverse_far = 1.0 / max(float(far), 1e-3)
    if fine_inverse is None:
        spacing = (inverse_far - inverse_near) / float(samples)
        inverse_depth = torch.linspace(
            inverse_near,
            inverse_far,
            samples,
            device=origins.device,
            dtype=origins.dtype,
        )
        inverse_depth = inverse_depth.unsqueeze(0).expand(
            origins.shape[0], samples
        )
        if jitter:
            inverse_depth = inverse_depth + (
                torch.rand(
                    origins.shape[0],
                    samples,
                    device=origins.device,
                    dtype=origins.dtype,
                )
                - 0.5
            ) * spacing
    else:
        spread = (
            fine_spread
            if fine_spread is not None
            else max((inverse_far - inverse_near) * 0.5, 1e-3)
        )
        offsets = torch.linspace(
            1.0,
            -1.0,
            samples,
            device=origins.device,
            dtype=origins.dtype,
        )
        inverse_depth = fine_inverse + offsets[None, :] * spread
        inverse_depth = torch.clamp(
            inverse_depth,
            min=inverse_far * 0.98,
            max=inverse_near * 1.02,
        )
    positions = (
        origins.unsqueeze(1) + directions.unsqueeze(1) / inverse_depth.unsqueeze(-1)
    )
    return positions, inverse_depth


def _segment_lengths(
    positions: torch.Tensor,
    inverse_depths: torch.Tensor,
) -> torch.Tensor:
    """Physical path length between samples along each ray.

    Samples are parameterized by inverse depth, so the difference in
    inverse-depth values is not a distance. The actual segment length is the
    Euclidean distance between neighboring sample positions.
    """
    segment = torch.linalg.norm(
        positions[:, 1:] - positions[:, :-1],
        dim=-1,
    )
    last = segment[:, -1:]
    return torch.cat([segment, last], dim=1)


def _importance_sample_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
    samples: int,
    coarse_weights: torch.Tensor,
    coarse_inverse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Importance-sample fine points from the coarse coherence votes."""
    weights = coarse_weights[..., 0].clamp(min=0.0)
    cdf = torch.cumsum(weights, dim=1)
    total = cdf[:, -1:].clamp(min=1e-6)
    cdf = cdf / total
    bins = (
        torch.arange(samples, device=origins.device, dtype=origins.dtype)
        .unsqueeze(0)
        .expand(origins.shape[0], samples)
    )
    u = bins + torch.rand(
        origins.shape[0], samples, device=origins.device, dtype=origins.dtype
    )
    u = (u / samples).clamp(0.0, 1.0 - 1e-6)
    indices = torch.searchsorted(cdf, u)
    hi = indices.clamp(1, coarse_inverse.shape[1] - 1)
    lo = hi - 1
    cdf_lo = cdf.gather(1, lo)
    cdf_hi = cdf.gather(1, hi)
    fraction = ((u - cdf_lo) / (cdf_hi - cdf_lo).clamp(min=1e-6)).clamp(0.0, 1.0)
    inverse_depth = (
        coarse_inverse.gather(1, lo) * (1.0 - fraction)
        + coarse_inverse.gather(1, hi) * fraction
    )
    degenerate = (total < 1e-4).squeeze(-1)
    if bool(degenerate.any()):
        uniform = torch.linspace(
            float(coarse_inverse[0, 0]),
            float(coarse_inverse[0, -1]),
            samples,
            device=origins.device,
            dtype=origins.dtype,
        ).unsqueeze(0).expand(origins.shape[0], samples)
        inverse_depth = torch.where(
            degenerate.unsqueeze(1).expand_as(inverse_depth),
            uniform,
            inverse_depth,
        )
    positions = (
        origins.unsqueeze(1)
        + directions.unsqueeze(1) / inverse_depth.unsqueeze(-1)
    )
    return positions, inverse_depth


def _coherence_peak_sample_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
    samples: int,
    coarse_weights: torch.Tensor,
    coarse_inverse: torch.Tensor,
    near: float,
    far: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a tight band around the strongest coarse coherence vote.

    The coarse pass can still be broad, so this keeps fine samples focused
    on the current peak instead of splitting them across every secondary
    shell. The band width adapts to the local inverse-depth spacing so an
    early diffuse vote still spans enough space to be corrected.
    """
    weights = coarse_weights[..., 0].clamp(min=0.0)
    peak_index = weights.argmax(dim=1, keepdim=True)
    peak_inverse = coarse_inverse.gather(1, peak_index)
    degenerate = weights.max(dim=1).values < 1e-3
    if coarse_inverse.shape[1] > 1:
        spacing = (coarse_inverse[:, 1:] - coarse_inverse[:, :-1]).abs()
        median_spacing = torch.median(spacing, dim=1, keepdim=True).values
        local_index = peak_index.clamp(max=coarse_inverse.shape[1] - 2)
        local_spacing = spacing.gather(1, local_index).clamp(
            min=median_spacing * 0.5
        )
        spread = torch.maximum(local_spacing, median_spacing * 0.75)
    else:
        spread = torch.ones_like(peak_inverse) * 1e-3
    inverse_near = 1.0 / max(float(near), 1e-4)
    inverse_far = 1.0 / max(float(far), 1e-3)
    offsets = torch.linspace(
        1.0,
        -1.0,
        samples,
        device=origins.device,
        dtype=origins.dtype,
    )
    inverse_depth = peak_inverse + offsets[None, :] * (spread * 1.25)
    inverse_depth = torch.clamp(
        inverse_depth,
        min=inverse_far * 0.98,
        max=inverse_near * 1.02,
    )
    if bool(degenerate.any()):
        uniform = torch.linspace(
            inverse_near,
            inverse_far,
            samples,
            device=origins.device,
            dtype=origins.dtype,
        ).unsqueeze(0).expand(origins.shape[0], samples)
        inverse_depth = torch.where(
            degenerate.unsqueeze(1).expand_as(inverse_depth),
            uniform,
            inverse_depth,
        )
    positions = (
        origins.unsqueeze(1)
        + directions.unsqueeze(1) / inverse_depth.unsqueeze(-1)
    )
    return positions, inverse_depth


def _anchor_sample_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
    near: float,
    far: float,
    samples: int,
    anchor_inverse: torch.Tensor,
    spread: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a tight band around a per-ray inverse-depth anchor.

    Sparse point evidence gives an approximate surface depth for many rays.
    Placing a subset of fine samples directly around that anchor guarantees
    the readout can build a sharp vote at the target instead of having to
    rediscover the surface from a broad coarse distribution.
    """
    inverse_near = 1.0 / max(float(near), 1e-4)
    inverse_far = 1.0 / max(float(far), 1e-3)
    offsets = torch.linspace(
        1.0,
        -1.0,
        samples,
        device=origins.device,
        dtype=origins.dtype,
    )
    if isinstance(spread, torch.Tensor):
        inverse_depth = (
            anchor_inverse.unsqueeze(1)
            + offsets[None, :] * spread.unsqueeze(1)
        )
    else:
        inverse_depth = anchor_inverse.unsqueeze(1) + offsets[None, :] * spread
    inverse_depth = torch.clamp(
        inverse_depth,
        min=inverse_far * 0.98,
        max=inverse_near * 1.02,
    )
    positions = (
        origins.unsqueeze(1)
        + directions.unsqueeze(1) / inverse_depth.unsqueeze(-1)
    )
    return positions, inverse_depth


def _fine_sample_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
    near: float,
    far: float,
    samples: int,
    coarse_weights: torch.Tensor,
    coarse_inverse: torch.Tensor,
    anchor_inverse: torch.Tensor | None = None,
    anchor_spread: torch.Tensor | float | None = None,
    anchor_fraction: float = 0.4,
    sampling_mode: str = "importance",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Importance-sample fine points, optionally around point-cloud anchors.

    The fine chain follows the coarse vote distribution instead of being
    locked to a single coarse peak, so a wrong early peak can still be
    corrected later. A minority of samples stay around sparse point depth
    when anchors are available, which keeps the surface from drifting off
    the evidence.
    """
    samples = max(1, int(samples))
    anchor_samples = 0
    if anchor_inverse is not None and samples > 1:
        anchor_samples = max(
            1,
            min(
                samples - 1,
                int(round(samples * float(anchor_fraction))),
            ),
        )
    importance_samples = samples - anchor_samples
    if sampling_mode == "peak":
        fine_importance, fine_importance_inv = _coherence_peak_sample_rays(
            origins,
            directions,
            importance_samples,
            coarse_weights,
            coarse_inverse,
            near,
            far,
        )
    else:
        fine_importance, fine_importance_inv = _importance_sample_rays(
            origins,
            directions,
            importance_samples,
            coarse_weights,
            coarse_inverse,
        )
    if anchor_samples > 0:
        fine_anchor, fine_anchor_inv = _anchor_sample_rays(
            origins,
            directions,
            near,
            far,
            anchor_samples,
            anchor_inverse,
            anchor_spread,
        )
        fine_positions = torch.cat(
            [fine_importance, fine_anchor], dim=1
        )
        fine_inverse = torch.cat(
            [fine_importance_inv, fine_anchor_inv], dim=1
        )
    else:
        fine_positions = fine_importance
        fine_inverse = fine_importance_inv
    return fine_positions, fine_inverse


def render_rays_chunk(
    model: TitanCoherenceField,
    origins: torch.Tensor,
    directions: torch.Tensor,
    near: float,
    far: float,
    coarse_samples: int = 32,
    fine_samples: int = 16,
    readout_chunk: int = 131072,
    use_amp: bool = True,
    jitter: bool = False,
    anchor_depth: torch.Tensor | None = None,
    anchor_spread: float = 0.08,
    anchor_spread_fraction: float | None = None,
    anchor_mask: torch.Tensor | None = None,
    return_detail: bool = False,
) -> torch.Tensor | tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Render a batch of rays with coherence voting.

    A coarse pass locates the coherent region and guides importance
    sampling. The final color vote is computed only from the fine chain:
    half of the fine samples follow the coarse importance distribution and
    the other half are anchored around per-ray point-cloud depth when
    available. Keeping the final chain thin prevents broad density shells
    from averaging colors into fog.
    """
    if anchor_spread_fraction is None:
        anchor_spread_fraction = float(
            getattr(model, "anchor_spread_fraction", 0.0)
        )
    device = origins.device
    with torch.autocast("cuda", enabled=use_amp and device.type == "cuda"):
        coarse_positions, coarse_inverse = _sample_rays(
            origins, directions, near, far, coarse_samples, jitter=jitter
        )
        inverse_span = float(coarse_inverse[0, 0] - coarse_inverse[0, -1])
        coarse_logits = model.readout_logits(
            coarse_positions.reshape(-1, 3),
            directions.unsqueeze(1).expand_as(coarse_positions).reshape(-1, 3),
            chunk=readout_chunk,
        )
        coarse_logits = coarse_logits.reshape(
            origins.shape[0], coarse_samples, 5
        )
        temperature = model.coherence_temperature.to(coarse_logits.dtype)
        alpha_scale = float(model.alpha_scale)
        coarse_segments = _segment_lengths(coarse_positions, coarse_inverse)
        anchor_inverse = None
        anchor_spread_t = None
        anchor_mask_float = None
        if anchor_depth is not None:
            inverse_near = 1.0 / max(float(near), 1e-4)
            inverse_far = 1.0 / max(float(far), 1e-3)
            hit_spread = float(anchor_spread)
            fallback_spread = max((inverse_far - inverse_near) * 0.25, 1e-3)
            if float(anchor_spread_fraction) > 0.0:
                max_spread = inverse_span * float(anchor_spread_fraction)
                hit_spread = min(float(anchor_spread), max_spread)
                fallback_spread = min(fallback_spread, max_spread)
            hit_inverse = 1.0 / anchor_depth.clamp(min=1e-4)
            fallback_inverse = coarse_inverse[:, coarse_samples // 2]
            if anchor_mask is not None:
                anchor_mask_float = (anchor_mask > 0).to(torch.float32)
                mask = (
                    (anchor_mask > 0).squeeze(-1)
                    if anchor_mask.dim() > 1
                    else (anchor_mask > 0)
                )
                hit_inverse_flat = (
                    hit_inverse.squeeze(-1)
                    if hit_inverse.dim() > 1
                    else hit_inverse
                )
                anchor_inverse = torch.where(mask, hit_inverse_flat, fallback_inverse)
                anchor_spread_t = torch.where(
                    mask,
                    torch.full_like(hit_inverse_flat, hit_spread),
                    torch.full_like(fallback_inverse, fallback_spread),
                )
            else:
                anchor_inverse = (
                    hit_inverse.squeeze(-1)
                    if hit_inverse.dim() > 1
                    else hit_inverse
                )
                anchor_spread_t = torch.full_like(anchor_inverse, hit_spread)
        coarse_weights = _voting_weights(
            coarse_logits[..., 3:4],
            coarse_logits[..., 4:5],
            temperature,
            inverse_depth=coarse_inverse,
            anchor_inverse=anchor_inverse,
            anchor_spread=anchor_spread_t,
            anchor_blend=model.anchor_blend,
            anchor_peak=model.anchor_peak,
            anchor_mask=anchor_mask_float,
            segment_lengths=coarse_segments,
            alpha_scale=alpha_scale,
            capture_floor=float(model.capture_floor),
            logit_clamp=float(model.logit_clamp),
        )

        sampling_weights = coarse_weights
        if model.coarse_sampling_mode == "consensus":
            coarse_capture_logit = _capture_logit(
                coarse_logits[..., 3:4],
                coarse_logits[..., 4:5],
                temperature,
                inverse_depth=coarse_inverse,
                anchor_inverse=anchor_inverse,
                anchor_spread=anchor_spread_t,
                anchor_blend=model.anchor_blend,
                anchor_peak=model.anchor_peak,
                anchor_mask=anchor_mask_float,
                logit_clamp=float(model.logit_clamp),
                logit_floor=float(model.logit_floor),
                logit_floor_slope=float(model.logit_floor_slope),
            )
            sampling_weights = _consensus_weights(
                coarse_capture_logit,
                coarse_inverse,
                float(model.consensus_temperature),
                near_bias=0.0,
                top_k=4,
            )

        if anchor_depth is not None:
            if anchor_inverse is None:
                anchor_inverse = 1.0 / anchor_depth.clamp(min=1e-4)
            if anchor_spread_t is None:
                anchor_spread_t = torch.full_like(
                    anchor_inverse, anchor_spread
                )
        fine_positions, fine_inverse = _fine_sample_rays(
            origins,
            directions,
            near,
            far,
            fine_samples,
            sampling_weights,
            coarse_inverse,
            anchor_inverse=anchor_inverse,
            anchor_spread=anchor_spread_t,
            anchor_fraction=float(model.fine_anchor_fraction),
            sampling_mode=str(model.fine_sampling_mode),
        )
        fine_logits = model.readout_logits(
            fine_positions.reshape(-1, 3),
            directions.unsqueeze(1).expand_as(fine_positions).reshape(-1, 3),
            chunk=readout_chunk,
        )
        fine_logits = fine_logits.reshape(origins.shape[0], fine_samples, 5)

        order = torch.argsort(fine_inverse, dim=1, descending=True)
        fine_inverse = torch.gather(fine_inverse, 1, order)
        fine_positions = torch.gather(
            fine_positions, 1, order.unsqueeze(-1).expand_as(fine_positions)
        )
        fine_logits = torch.gather(
            fine_logits, 1, order.unsqueeze(-1).expand_as(fine_logits)
        )

        sample_colors = torch.sigmoid(fine_logits[..., 0:3])
        fine_segments = _segment_lengths(fine_positions, fine_inverse)
        weights, sample_alpha = _voting_weights(
            fine_logits[..., 3:4],
            fine_logits[..., 4:5],
            temperature,
            inverse_depth=fine_inverse,
            anchor_inverse=anchor_inverse,
            anchor_spread=anchor_spread_t,
            anchor_blend=model.anchor_blend,
            anchor_peak=model.anchor_peak,
            anchor_mask=anchor_mask_float,
            segment_lengths=fine_segments,
            alpha_scale=alpha_scale,
            capture_floor=float(model.capture_floor),
            logit_clamp=float(model.logit_clamp),
            return_alpha=True,
        )
        sample_gates = fine_logits[..., 4:5].float()
        capture_logit = _capture_logit(
            fine_logits[..., 3:4],
            fine_logits[..., 4:5],
            temperature,
            inverse_depth=fine_inverse,
            anchor_inverse=anchor_inverse,
            anchor_spread=anchor_spread_t,
            anchor_blend=model.anchor_blend,
            anchor_peak=model.anchor_peak,
            anchor_mask=anchor_mask_float,
            logit_clamp=float(model.logit_clamp),
            logit_floor=float(model.logit_floor),
            logit_floor_slope=float(model.logit_floor_slope),
        )
        consensus_weights = _consensus_weights(
            capture_logit,
            fine_inverse,
            float(model.consensus_temperature),
            near_bias=model.consensus_near_bias,
            top_k=model.consensus_top_k,
        )
        render_weights = _visible_topk_render_weights(
            consensus_weights,
            sample_alpha,
            weights,
            float(model.peak_power),
            int(model.render_top_k),
        )
        color = torch.sum(render_weights * sample_colors, dim=1)
        # If every visible vote is suppressed (for example an entirely
        # transparent ray), fall back to the physical composite color so a
        # partially visible ray never turns black before exit blending.
        vote_total = render_weights.sum(dim=1)
        color = torch.where(
            vote_total > 1e-6,
            color,
            torch.sum(weights * sample_colors, dim=1),
        )
        # Physical transparency decides how much of the ray is empty. The
        # scene is continuous: rays with opacity keep the surface vote, and
        # only truly empty rays fall back to the scene exit color.
        alpha_visibility = weights.sum(dim=1).clamp(max=1.0)
        physical_color = torch.sum(weights * sample_colors, dim=1)
        visibility = alpha_visibility.clamp(0.0, 1.0)
        background = model.background.to(color.dtype)
        rendered = color * visibility + background * (1.0 - visibility)
        physical_rendered = (
            physical_color + background * (1.0 - alpha_visibility)
        )

        coarse_colors = torch.sigmoid(coarse_logits[..., 0:3])
        coarse_color = torch.sum(coarse_weights * coarse_colors, dim=1)
        coarse_visibility = coarse_weights.sum(dim=1)
        coarse_residual = torch.clamp(1.0 - coarse_visibility, min=0.0)
        coarse_rendered = (
            coarse_color + coarse_residual * background
        )
    if return_detail:
        return (
            rendered,
            weights,
            sample_alpha,
            visibility,
            fine_positions,
            coarse_rendered,
            sample_gates,
            consensus_weights,
            physical_rendered,
            color,
        )
    return rendered


def render_rays_chunk_geometry(
    model: TitanCoherenceField,
    origins: torch.Tensor,
    directions: torch.Tensor,
    near: float,
    far: float,
    coarse_samples: int = 32,
    fine_samples: int = 16,
    readout_chunk: int = 131072,
    use_amp: bool = True,
    jitter: bool = False,
    anchor_depth: torch.Tensor | None = None,
    anchor_spread: float = 0.08,
    anchor_spread_fraction: float | None = None,
    anchor_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render one ray batch and return the 3D vote position.

    Like ``render_rays_chunk``, the final color and expected position come
    from the fine chain only, so extraction stays consistent with the
    rendered image.
    """
    if anchor_spread_fraction is None:
        anchor_spread_fraction = float(
            getattr(model, "anchor_spread_fraction", 0.0)
        )
    device = origins.device
    with torch.autocast("cuda", enabled=use_amp and device.type == "cuda"):
        coarse_positions, coarse_inverse = _sample_rays(
            origins, directions, near, far, coarse_samples, jitter=jitter
        )
        inverse_span = float(coarse_inverse[0, 0] - coarse_inverse[0, -1])
        coarse_logits = model.readout_logits(
            coarse_positions.reshape(-1, 3),
            directions.unsqueeze(1).expand_as(coarse_positions).reshape(-1, 3),
            chunk=readout_chunk,
        )
        coarse_logits = coarse_logits.reshape(
            origins.shape[0], coarse_samples, 5
        )
        temperature = model.coherence_temperature.to(coarse_logits.dtype)
        alpha_scale = float(model.alpha_scale)
        coarse_segments = _segment_lengths(coarse_positions, coarse_inverse)
        anchor_inverse = None
        anchor_spread_t = None
        anchor_mask_float = None
        if anchor_depth is not None:
            inverse_near = 1.0 / max(float(near), 1e-4)
            inverse_far = 1.0 / max(float(far), 1e-3)
            hit_spread = float(anchor_spread)
            fallback_spread = max((inverse_far - inverse_near) * 0.25, 1e-3)
            if float(anchor_spread_fraction) > 0.0:
                max_spread = inverse_span * float(anchor_spread_fraction)
                hit_spread = min(float(anchor_spread), max_spread)
                fallback_spread = min(fallback_spread, max_spread)
            hit_inverse = 1.0 / anchor_depth.clamp(min=1e-4)
            fallback_inverse = coarse_inverse[:, coarse_samples // 2]
            if anchor_mask is not None:
                anchor_mask_float = (anchor_mask > 0).to(torch.float32)
                mask = (
                    (anchor_mask > 0).squeeze(-1)
                    if anchor_mask.dim() > 1
                    else (anchor_mask > 0)
                )
                hit_inverse_flat = (
                    hit_inverse.squeeze(-1)
                    if hit_inverse.dim() > 1
                    else hit_inverse
                )
                anchor_inverse = torch.where(mask, hit_inverse_flat, fallback_inverse)
                anchor_spread_t = torch.where(
                    mask,
                    torch.full_like(hit_inverse_flat, hit_spread),
                    torch.full_like(fallback_inverse, fallback_spread),
                )
            else:
                anchor_inverse = (
                    hit_inverse.squeeze(-1)
                    if hit_inverse.dim() > 1
                    else hit_inverse
                )
                anchor_spread_t = torch.full_like(anchor_inverse, hit_spread)
        coarse_weights = _voting_weights(
            coarse_logits[..., 3:4],
            coarse_logits[..., 4:5],
            temperature,
            inverse_depth=coarse_inverse,
            anchor_inverse=anchor_inverse,
            anchor_spread=anchor_spread_t,
            anchor_blend=model.anchor_blend,
            anchor_peak=model.anchor_peak,
            anchor_mask=anchor_mask_float,
            segment_lengths=coarse_segments,
            alpha_scale=alpha_scale,
            capture_floor=float(model.capture_floor),
            logit_clamp=float(model.logit_clamp),
        )

        sampling_weights = coarse_weights
        if model.coarse_sampling_mode == "consensus":
            coarse_capture_logit = _capture_logit(
                coarse_logits[..., 3:4],
                coarse_logits[..., 4:5],
                temperature,
                inverse_depth=coarse_inverse,
                anchor_inverse=anchor_inverse,
                anchor_spread=anchor_spread_t,
                anchor_blend=model.anchor_blend,
                anchor_peak=model.anchor_peak,
                anchor_mask=anchor_mask_float,
                logit_clamp=float(model.logit_clamp),
                logit_floor=float(model.logit_floor),
                logit_floor_slope=float(model.logit_floor_slope),
            )
            sampling_weights = _consensus_weights(
                coarse_capture_logit,
                coarse_inverse,
                float(model.consensus_temperature),
                near_bias=0.0,
                top_k=4,
            )

        if anchor_depth is not None:
            if anchor_inverse is None:
                anchor_inverse = 1.0 / anchor_depth.clamp(min=1e-4)
            if anchor_spread_t is None:
                anchor_spread_t = torch.full_like(
                    anchor_inverse, anchor_spread
                )
        fine_positions, fine_inverse = _fine_sample_rays(
            origins,
            directions,
            near,
            far,
            fine_samples,
            sampling_weights,
            coarse_inverse,
            anchor_inverse=anchor_inverse,
            anchor_spread=anchor_spread_t,
            anchor_fraction=float(model.fine_anchor_fraction),
            sampling_mode=str(model.fine_sampling_mode),
        )
        fine_logits = model.readout_logits(
            fine_positions.reshape(-1, 3),
            directions.unsqueeze(1).expand_as(fine_positions).reshape(-1, 3),
            chunk=readout_chunk,
        )
        fine_logits = fine_logits.reshape(origins.shape[0], fine_samples, 5)

        order = torch.argsort(fine_inverse, dim=1, descending=True)
        fine_inverse = torch.gather(fine_inverse, 1, order)
        fine_positions = torch.gather(
            fine_positions, 1, order.unsqueeze(-1).expand_as(fine_positions)
        )
        fine_logits = torch.gather(
            fine_logits, 1, order.unsqueeze(-1).expand_as(fine_logits)
        )
        sample_colors = torch.sigmoid(fine_logits[..., 0:3])
        fine_segments = _segment_lengths(fine_positions, fine_inverse)
        weights, sample_alpha = _voting_weights(
            fine_logits[..., 3:4],
            fine_logits[..., 4:5],
            temperature,
            inverse_depth=fine_inverse,
            anchor_inverse=anchor_inverse,
            anchor_spread=anchor_spread_t,
            anchor_blend=model.anchor_blend,
            anchor_peak=model.anchor_peak,
            anchor_mask=anchor_mask_float,
            segment_lengths=fine_segments,
            alpha_scale=alpha_scale,
            capture_floor=float(model.capture_floor),
            logit_clamp=float(model.logit_clamp),
            return_alpha=True,
        )
        capture_logit = _capture_logit(
            fine_logits[..., 3:4],
            fine_logits[..., 4:5],
            temperature,
            inverse_depth=fine_inverse,
            anchor_inverse=anchor_inverse,
            anchor_spread=anchor_spread_t,
            anchor_blend=model.anchor_blend,
            anchor_peak=model.anchor_peak,
            anchor_mask=anchor_mask_float,
            logit_clamp=float(model.logit_clamp),
            logit_floor=float(model.logit_floor),
            logit_floor_slope=float(model.logit_floor_slope),
        )
        consensus_weights = _consensus_weights(
            capture_logit,
            fine_inverse,
            float(model.consensus_temperature),
            near_bias=model.consensus_near_bias,
            top_k=model.consensus_top_k,
        )
        render_weights = _visible_topk_render_weights(
            consensus_weights,
            sample_alpha,
            weights,
            float(model.peak_power),
            int(model.render_top_k),
        )
        color = torch.sum(render_weights * sample_colors, dim=1)
        weights_sum = render_weights.sum(dim=1).clamp(min=1e-6)
        expected_position = torch.sum(
            render_weights * fine_positions, dim=1
        ) / weights_sum
        alpha_visibility = weights.sum(dim=1).clamp(max=1.0)
        visibility = alpha_visibility.clamp(0.0, 1.0)
        background = model.background.to(color.dtype)
        rendered = color * visibility + background * (1.0 - visibility)
    return rendered, expected_position, visibility


def render_rays(
    model: TitanCoherenceField,
    origins: torch.Tensor,
    directions: torch.Tensor,
    near: float,
    far: float,
    ray_batch: int = 4096,
    coarse_samples: int = 32,
    fine_samples: int = 16,
    use_amp: bool = True,
    jitter: bool = False,
    scene_points: torch.Tensor | None = None,
    depth_tolerance: float = 0.25,
    anchor_spread: float = 0.08,
) -> torch.Tensor:
    """Render arbitrary ray counts in memory-bounded batches.

    When sparse point evidence is supplied, each ray batch is anchored to its
    nearest projected point so training and inference share the same focused
    fine sampling strategy.
    """
    outputs = []
    for start in range(0, origins.shape[0], ray_batch):
        stop = min(start + ray_batch, origins.shape[0])
        chunk_origins = origins[start:stop]
        chunk_directions = directions[start:stop]
        anchor_depth = None
        anchor_mask = None
        if scene_points is not None and scene_points.shape[0] > 0:
            target_depth, hit_mask, _ = _ray_point_targets(
                chunk_origins,
                chunk_directions,
                scene_points,
                tolerance=depth_tolerance,
            )
            anchor_depth = target_depth
            anchor_mask = hit_mask
        outputs.append(
            render_rays_chunk(
                model,
                chunk_origins,
                chunk_directions,
                near,
                far,
                coarse_samples=coarse_samples,
                fine_samples=fine_samples,
                use_amp=use_amp,
                jitter=jitter,
                anchor_depth=anchor_depth,
                anchor_mask=anchor_mask,
                anchor_spread=anchor_spread,
            )
        )
    return torch.cat(outputs, dim=0)


def _camera_rays_from_pose(
    camera: dict,
    device: torch.device,
    ys: torch.Tensor | None = None,
    xs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    c2w = np.asarray(camera["c2w"], dtype=np.float32)
    k = camera["K"]
    width, height = camera["width"], camera["height"]
    fx, fy, cx, cy = float(k[0][0]), float(k[1][1]), float(k[0][2]), float(k[1][2])
    if ys is None:
        ys_flat = np.arange(height)
        xs_flat = np.arange(width)
    else:
        ys_flat = ys
        xs_flat = xs
    if ys is None:
        grid_y, grid_x = np.meshgrid(ys_flat, xs_flat, indexing="ij")
        local = np.stack(
            [
                (grid_x - cx) / fx,
                (grid_y - cy) / fy,
                np.ones_like(grid_x),
            ],
            axis=-1,
        )
        directions = local @ np.asarray(c2w[:3, :3], dtype=np.float32).T
        flat = directions.reshape(-1, 3)
    else:
        local = np.stack(
            [(xs_flat - cx) / fx, (ys_flat - cy) / fy, np.ones_like(xs_flat)],
            axis=-1,
        )
        flat = local @ np.asarray(c2w[:3, :3], dtype=np.float32).T
    flat = flat / np.linalg.norm(flat, axis=-1, keepdims=True)
    origins = torch.tensor(
        np.asarray(c2w[:3, 3], dtype=np.float32),
        device=device,
        dtype=torch.float32,
    ).expand(flat.shape[0], 3)
    directions = torch.tensor(flat, device=device, dtype=torch.float32)
    return origins, directions


def _scene_transform(
    cameras: list[dict],
    scene_points: np.ndarray | None = None,
) -> tuple[torch.Tensor, float]:
    positions = np.asarray(
        [np.asarray(camera["c2w"], dtype=np.float32)[:3, 3] for camera in cameras]
    )
    anchors = positions
    if scene_points is not None and len(scene_points):
        points = np.asarray(scene_points, dtype=np.float32).reshape(-1, 3)
        if len(points):
            anchors = np.concatenate([positions, points], axis=0)
    if len(anchors) == 0:
        return (
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            1.0,
        )
    shift = np.median(anchors, axis=0)
    radius = float(np.median(np.linalg.norm(anchors - shift, axis=1)))
    if not np.isfinite(radius) or radius <= 1e-6:
        radius = 1.0
    return torch.tensor(shift, dtype=torch.float32), radius


def _scene_exit_color(images: list[np.ndarray]) -> list[float]:
    """Physical far-field color for rays that leave the reconstructed volume.

    Bounded scene datasets contain scene content everywhere, so this is only
    a weak prior for genuinely empty rays; the optimizer can move it and the
    volume material fills the actual image regions. The estimate blends the
    image border and the whole-view statistics so neither a borderless scene
    nor an open scene is forced into a foreground/background dichotomy.
    """
    border_samples: list[np.ndarray] = []
    view_samples: list[np.ndarray] = []
    for image in images:
        height, width = image.shape[:2]
        border = np.concatenate(
            [
                image[0, :, :],
                image[-1, :, :],
                image[:, 0, :],
                image[:, -1, :],
            ],
            axis=0,
        )
        border_samples.append(border)
        view_samples.append(image.reshape(-1, 3))
    if not border_samples:
        return [0.5, 0.5, 0.5]
    border_pixels = np.concatenate(border_samples, axis=0)
    view_pixels = np.concatenate(view_samples, axis=0)
    border_color = np.median(border_pixels.reshape(-1, 3), axis=0)
    view_color = np.median(view_pixels.reshape(-1, 3), axis=0)
    exit_color = 0.6 * border_color + 0.4 * view_color
    return [float(exit_color[0]), float(exit_color[1]), float(exit_color[2])]


def _ray_bounds(
    origins: torch.Tensor,
    scene_points: np.ndarray,
    radius: float,
    scale: float,
    shift: torch.Tensor,
) -> tuple[float, float]:
    if len(scene_points) == 0:
        return 0.3, 7.6
    points = np.asarray(scene_points, dtype=np.float32)[:, :3]
    points = (points - shift.cpu().numpy()) * scale
    distances = np.linalg.norm(points - origins.cpu().numpy()[0], axis=1)
    distances = distances[distances > 1e-4]
    if distances.size == 0:
        return 0.3, 7.6
    lower = float(np.percentile(distances, 5.0))
    upper = float(np.percentile(distances, 95.0))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= 1e-6:
        return 0.3, 7.6
    return max(lower * 0.6, 0.05), upper * 1.5


def _ray_point_targets(
    origins: torch.Tensor,
    directions: torch.Tensor,
    points: torch.Tensor,
    tolerance: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Anchor ray geometry to the sparse point cloud.

    For every ray, find the nearest point projected onto the ray. Rays with a
    point within ``tolerance`` are treated as surface hits; their expected
    depth is anchored to that point and confidence scales with proximity.
    """
    if points.shape[0] == 0:
        batch = origins.shape[0]
        return (
            torch.zeros(batch, device=origins.device, dtype=origins.dtype),
            torch.zeros(batch, device=origins.device, dtype=origins.dtype),
            torch.zeros(batch, device=origins.device, dtype=origins.dtype),
        )
    oc = points.unsqueeze(0) - origins.unsqueeze(1)
    projection = torch.sum(oc * directions.unsqueeze(1), dim=-1)
    projection = projection.clamp(min=0.0)
    closest = origins.unsqueeze(1) + projection.unsqueeze(-1) * directions.unsqueeze(1)
    distance = torch.linalg.norm(closest - points.unsqueeze(0), dim=-1)
    distance_min, _ = distance.min(dim=1)
    hit_mask = (distance_min <= tolerance).float()
    confidence = torch.clamp(1.0 - distance_min / max(tolerance, 1e-6), min=0.0)
    # Prefer the nearest surface along the ray. A sphere seen edge-on has
    # near and far points at similar ray distance, and picking the nearest
    # projection avoids anchoring geometry to the far side of the object.
    invalid = distance > tolerance
    masked_projection = torch.where(
        invalid,
        torch.full_like(projection, float("inf")),
        projection,
    )
    target_depth, _ = masked_projection.min(dim=1)
    target_depth = torch.where(
        hit_mask.bool(),
        target_depth.clamp(min=0.0),
        torch.zeros_like(target_depth),
    )
    return target_depth, hit_mask, confidence


def _build_scaffold(
    points: torch.Tensor,
    resolution: int = 64,
    dilation: int = 2,
    bounds: float = 1.4,
) -> torch.Tensor:
    """Build a dilated occupancy scaffold from the sparse point cloud.

    The scaffold is a coarse 3D grid marking where surface evidence exists.
    It is used as a soft prior: density is discouraged in cells that contain
    no point evidence, which prevents the model from hiding behind a shell.
    """
    resolution = max(16, int(resolution))
    device = points.device
    grid = torch.zeros(
        resolution * resolution * resolution,
        device=device,
        dtype=torch.float32,
    )
    if points.shape[0] == 0:
        return grid
    coords = (points + bounds) / (2.0 * bounds) * float(resolution - 1)
    cells = coords.long().clamp(0, resolution - 1)
    index = (
        cells[:, 0] * resolution * resolution
        + cells[:, 1] * resolution
        + cells[:, 2]
    )
    index = index.unique()
    grid.index_fill_(0, index, 1.0)
    if dilation > 0:
        volume = grid.view(1, 1, resolution, resolution, resolution)
        volume = torch.nn.functional.max_pool3d(
            volume,
            kernel_size=2 * dilation + 1,
            stride=1,
            padding=dilation,
        )
        grid = volume.reshape(-1)
    return grid.detach()


def _sample_scaffold(
    grid: torch.Tensor,
    positions: torch.Tensor,
    resolution: int = 64,
    bounds: float = 1.4,
) -> torch.Tensor:
    """Trilinear sample the scaffold at arbitrary normalized positions."""
    resolution = max(16, int(resolution))
    flat = positions.reshape(-1, 3)
    coords = (flat + bounds) / (2.0 * bounds) * float(resolution - 1)
    lo = torch.floor(coords).long().clamp(0, resolution - 2)
    hi = (lo + 1).clamp(0, resolution - 1)
    fraction = (coords - lo.float()).clamp(0.0, 1.0)
    fx = fraction[:, 0:1]
    fy = fraction[:, 1:2]
    fz = fraction[:, 2:3]
    stride_y = resolution
    stride_x = resolution * resolution
    idx_lo = lo[..., 0] * stride_x + lo[..., 1] * stride_y + lo[..., 2]
    idx_hi = hi[..., 0] * stride_x + hi[..., 1] * stride_y + hi[..., 2]
    grid_flat = grid
    v000 = grid_flat[idx_lo].unsqueeze(-1)
    v001 = grid_flat[
        lo[..., 0] * stride_x + lo[..., 1] * stride_y + hi[..., 2]
    ].unsqueeze(-1)
    v010 = grid_flat[
        lo[..., 0] * stride_x + hi[..., 1] * stride_y + lo[..., 2]
    ].unsqueeze(-1)
    v011 = grid_flat[
        lo[..., 0] * stride_x + hi[..., 1] * stride_y + hi[..., 2]
    ].unsqueeze(-1)
    v100 = grid_flat[
        hi[..., 0] * stride_x + lo[..., 1] * stride_y + lo[..., 2]
    ].unsqueeze(-1)
    v101 = grid_flat[
        hi[..., 0] * stride_x + lo[..., 1] * stride_y + hi[..., 2]
    ].unsqueeze(-1)
    v110 = grid_flat[
        hi[..., 0] * stride_x + hi[..., 1] * stride_y + lo[..., 2]
    ].unsqueeze(-1)
    v111 = grid_flat[
        hi[..., 0] * stride_x + hi[..., 1] * stride_y + hi[..., 2]
    ].unsqueeze(-1)
    c00 = v000 * (1.0 - fx) + v100 * fx
    c01 = v001 * (1.0 - fx) + v101 * fx
    c10 = v010 * (1.0 - fx) + v110 * fx
    c11 = v011 * (1.0 - fx) + v111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    values = c0 * (1.0 - fz) + c1 * fz
    return values.reshape(positions.shape[:-1])


def _voxel_sample(points: np.ndarray, max_points: int) -> np.ndarray:
    """Downsample a dense point cloud with a deterministic voxel grid."""
    if len(points) <= max_points:
        return points
    lower = points[:, :3].min(axis=0)
    upper = points[:, :3].max(axis=0)
    span = np.maximum(upper - lower, 1e-6)
    resolution = max(8, int(np.cbrt(max_points * 0.75)))
    index = np.floor((points[:, :3] - lower) / span * resolution).astype(np.int64)
    index = np.clip(index, 0, resolution - 1)
    keys = (
        index[:, 0] * resolution * resolution
        + index[:, 1] * resolution
        + index[:, 2]
    )
    _, first = np.unique(keys, return_index=True)
    sampled = points[first]
    if len(sampled) > max_points:
        rng = np.random.default_rng(7)
        sampled = sampled[rng.choice(len(sampled), max_points, replace=False)]
    return sampled


def extract_dense_points(
    model: TitanCoherenceField,
    cameras: list[dict],
    shift: torch.Tensor,
    scale: float,
    scene_points: np.ndarray,
    radius: float,
    device: torch.device,
    stride: int = 4,
    ray_batch: int = 4096,
    coarse_samples: int = 32,
    fine_samples: int = 16,
    visibility_threshold: float = 0.2,
    max_points: int = 200000,
    use_amp: bool = True,
    depth_tolerance: float = 0.25,
) -> list[float]:
    """Extract a dense colored 3D point cloud from the trained field.

    Each camera renders a subsampled ray grid and the expected vote position
    of every ray is unprojected back into world space. Points are returned as
    a flat [x, y, z, r, g, b, ...] array to keep scene.json compact.
    """
    model.to(device)
    shift = shift.to(device)
    anchor_points = None
    if len(scene_points):
        anchor_points = torch.from_numpy(
            _voxel_sample(scene_points.astype(np.float32), 20000)
        ).to(device)
        anchor_points = (anchor_points - shift) * scale
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for camera in cameras:
            height, width = int(camera["height"]), int(camera["width"])
            step = max(1, int(stride))
            ys = np.arange(0, height, step)
            xs = np.arange(0, width, step)
            grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
            ys_flat = grid_y.reshape(-1)
            xs_flat = grid_x.reshape(-1)
            origins, directions = _camera_rays_from_pose(
                camera, device, ys_flat, xs_flat
            )
            origins_scaled = (origins - shift) * scale
            near, far = _ray_bounds(
                origins_scaled, scene_points, radius, scale, shift
            )
            for start in range(0, origins_scaled.shape[0], ray_batch):
                stop = min(start + ray_batch, origins_scaled.shape[0])
                chunk_origins = origins_scaled[start:stop]
                chunk_directions = directions[start:stop]
                anchor_depth = None
                anchor_mask = None
                if anchor_points is not None:
                    target_depth, hit_mask, _ = _ray_point_targets(
                        chunk_origins,
                        chunk_directions,
                        anchor_points,
                        tolerance=depth_tolerance,
                    )
                    anchor_depth = target_depth
                    anchor_mask = hit_mask
                rendered, expected_position, visibility = (
                    render_rays_chunk_geometry(
                        model,
                        chunk_origins,
                        chunk_directions,
                        near,
                        far,
                        coarse_samples=coarse_samples,
                        fine_samples=fine_samples,
                        use_amp=use_amp,
                        anchor_depth=anchor_depth,
                        anchor_mask=anchor_mask,
                    )
                )
                rendered = rendered.float().cpu().numpy()
                expected_position = expected_position.float().cpu().numpy()
                visibility = visibility.float().cpu().numpy()[:, 0]
                mask = visibility >= float(visibility_threshold)
                if not mask.any():
                    continue
                positions_world = (
                    expected_position[mask] / scale + shift.cpu().numpy()
                )
                colors = rendered[mask]
                chunks.append(
                    np.concatenate([positions_world, colors], axis=1).astype(
                        np.float32
                    )
                )
    if not chunks:
        return []
    points = np.concatenate(chunks, axis=0)
    points = _voxel_sample(points, int(max_points))
    return [float(value) for value in points.reshape(-1)]


def train_tcf(
    images: np.ndarray,
    cameras: list[dict],
    config: dict,
    project_dir: Path,
    task_id: str,
    device_name: str = "cuda",
    sparse_points: list[list[float]] | None = None,
    status_cb: Callable[[dict], None] | None = None,
) -> tuple[dict, list[dict]]:
    """Train a TCF scene model with per-step progress reporting."""
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log(f"[tcf] 使用 CUDA：{torch.cuda.get_device_name(0)}")
    else:
        log("[tcf] 使用 CPU 训练（未检测到 CUDA）")

    seed = int(config.get("seed", 2026))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = TitanCoherenceField(config).to(device)
    view_images = (
        [images[index] for index in range(images.shape[0])]
        if isinstance(images, np.ndarray)
        else list(images)
    )
    with torch.no_grad():
        exit_color = torch.tensor(
            _scene_exit_color(view_images),
            dtype=torch.float32,
            device=device,
        )
        model.background.copy_(exit_color)
        model.background_target.copy_(exit_color)
    log(f"[tcf] 场景远场颜色 {exit_color.cpu().tolist()}")
    scene_points = (
        np.asarray(sparse_points, dtype=np.float32)[:, :3]
        if sparse_points is not None and len(sparse_points)
        else np.empty((0, 3), dtype=np.float32)
    )
    shift, radius = _scene_transform(cameras, scene_points)
    shift = shift.to(device)
    scale = 1.0 / max(float(radius), 1e-6)
    log(
        f"[tcf] 场景归一化 shift {shift.cpu().numpy().tolist()} "
        f"radius {radius:.4f} scale {scale:.6f}"
    )

    train_count = max(1, int(config.get("train_views", 16)))
    train_cameras = cameras[:train_count]
    if not train_cameras:
        train_cameras = cameras[:1]
    eval_cameras = cameras[train_count:]
    if not eval_cameras:
        eval_cameras = cameras[-1:]

    steps = max(1, int(config.get("steps", 1200)))
    half_step = max(0, steps // 2)
    patch = max(64, int(config.get("patch_size", 1024)))
    coarse_samples = int(config.get("coarse_samples", 32))
    fine_samples = int(config.get("fine_samples", 16))
    ray_batch = int(config.get("ray_batch_size", 4096))
    readout_chunk = int(config.get("readout_chunk", 131072))
    progress_every = max(1, int(config.get("training_progress_every", 10)))
    learning_rate = float(config.get("learning_rate", 0.01))
    lr_end = float(config.get("learning_rate_end", 0.002))
    use_amp = bool(config.get("mixed_precision", True))
    background_weight = float(config.get("background_weight", 0.1))
    occupancy_weight = float(config.get("occupancy_regularization", 0.002))
    visibility_weight = float(config.get("visibility_regularization", 0.02))
    visibility_target_min = float(
        config.get("visibility_target_min", 0.75)
    )
    visibility_target_max = float(
        config.get("visibility_target_max", 0.95)
    )
    visibility_upper_cap = float(config.get("visibility_upper_cap", 1.0))
    visibility_upper_weight = float(
        config.get("visibility_upper_regularization", 0.0)
    )
    hole_weight = float(config.get("hole_regularization", 0.25))
    winner_color_weight = float(config.get("winner_color_regularization", 1.0))
    consensus_entropy_weight = float(
        config.get("consensus_entropy_regularization", 0.02)
    )
    consensus_margin_weight = float(
        config.get("consensus_margin_regularization", 0.05)
    )
    sharpness_weight = float(config.get("sharpness_regularization", 0.05))
    geometry_weight = float(config.get("geometry_regularization", 1.0))
    scaffold_weight = float(config.get("scaffold_regularization", 0.5))
    scaffold_resolution = int(config.get("scaffold_resolution", 64))
    depth_dist_weight = float(config.get("depth_distribution_regularization", 2.0))
    depth_sigma = float(config.get("depth_sigma", 0.1))
    surface_weight = float(config.get("surface_regularization", 1.0))
    peak_weight = float(config.get("peak_regularization", 0.6))
    surface_alpha_weight = float(
        config.get("surface_alpha_regularization", 0.6)
    )
    surface_alpha_target = float(config.get("surface_alpha_target", 0.85))
    surface_density = float(config.get("surface_density", 5.0))
    surface_density_weight = float(
        config.get("surface_density_regularization", 0.0)
    )
    surface_density_outside = float(
        config.get("surface_density_outside", 0.2)
    )
    geometry_warmup = max(0, int(config.get("geometry_warmup", 120)))
    density_floor = float(config.get("density_floor", 0.3))
    density_floor_weight = float(config.get("density_floor_regularization", 0.0))
    readout_lr_scale = float(config.get("readout_lr_scale", 0.3))
    depth_tolerance = float(config.get("depth_tolerance", 0.25))
    spatial_patch = bool(config.get("spatial_patch", True))
    anchor_spread = float(config.get("anchor_spread", 0.08))
    voting_temp_start = float(config.get("voting_temperature_start", 0.5))
    voting_temp_end = float(config.get("voting_temperature_end", 6.0))
    consensus_temp_start = float(
        config.get("consensus_temperature_start", 1.5)
    )
    consensus_temp_end = float(
        config.get("consensus_temperature_end", 0.6)
    )
    anchor_blend_start = float(config.get("anchor_blend_start", 1.0))
    anchor_blend_end = float(config.get("anchor_blend_end", 0.0))
    anchor_blend_fraction = float(config.get("anchor_blend_fraction", 0.6))
    coarse_color_weight = float(config.get("coarse_color_weight", 0.5))
    physical_color_weight = float(config.get("physical_color_weight", 0.0))

    train_parameters = [p for p in model.parameters() if p.requires_grad]
    lattice_ids = {id(p) for p in model.lattice.params()}
    readout_parameters = [
        p
        for p in train_parameters
        if id(p) not in lattice_ids and p is not model.background
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": list(model.lattice.params()), "lr": learning_rate},
            {"params": readout_parameters, "lr": learning_rate * readout_lr_scale},
        ],
        betas=(0.9, 0.99),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr_end
    )
    training_seed = int(config.get("training_seed", 2026))
    torch.manual_seed(training_seed)
    torch.cuda.manual_seed_all(training_seed)
    np.random.seed(training_seed)
    rng = np.random.default_rng(training_seed)
    history: list[dict] = []
    start_wall = time.time()
    loss_ema: float | None = None

    scene_points_t = (
        torch.from_numpy(
            _voxel_sample(scene_points.astype(np.float32), 30000)
        ).to(device)
        if len(scene_points)
        else torch.empty((0, 3), device=device, dtype=torch.float32)
    )
    scaffold_grid = _build_scaffold(
        (scene_points_t - shift) * scale,
        resolution=scaffold_resolution,
    )

    log(
        f"[tcf] 开始：{len(train_cameras)} 训练视角，{steps} 步，"
        f"lattice {model.lattice_mb():.1f} MB，patch {patch} px"
    )
    if status_cb:
        status_cb(
            {
                "taskId": task_id,
                "status": "running",
                "stage": "training",
                "progress": 0.12,
                "step": 0,
                "totalSteps": steps,
                "loss": None,
                "vramMB": _device_reserved_mb(),
                "latticeMB": model.lattice_mb(),
            }
        )

    for step in range(steps):
        command = _read_control(project_dir)
        if command == "cancel":
            log("[tcf] 收到取消指令，安全停止训练")
            if status_cb:
                status_cb(
                    {
                        "taskId": task_id,
                        "status": "cancelled",
                        "stage": "cancel",
                        "progress": 0.0,
                        "step": step,
                    }
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model.to("cpu")
            return {
                "cancelled": True,
                "model": model,
                "shift": shift.cpu(),
                "scale": scale,
                "radius": radius,
                "engine": "tcf",
                "history": history,
            }, history
        if command == "pause":
            log("[tcf] 训练已暂停")
            if status_cb:
                status_cb(
                    {
                        "taskId": task_id,
                        "status": "paused",
                        "stage": "training",
                        "progress": round(0.12 + 0.78 * ((step + 1) / steps), 4),
                        "step": step,
                        "totalSteps": steps,
                        "loss": None,
                    }
                )
            while _read_control(project_dir) == "pause":
                time.sleep(0.4)
            log("[tcf] 训练已恢复")
            if status_cb:
                status_cb(
                    {
                        "taskId": task_id,
                        "status": "running",
                        "stage": "training",
                        "progress": round(0.12 + 0.78 * ((step + 1) / steps), 4),
                        "step": step,
                        "totalSteps": steps,
                        "loss": None,
                        "vramMB": _device_reserved_mb(),
                        "latticeMB": model.lattice_mb(),
                    }
                )
        temp_fraction = (step + 1) / max(steps, 1)
        voting_temp = voting_temp_start + 0.5 * (
            voting_temp_end - voting_temp_start
        ) * (1.0 - math.cos(math.pi * temp_fraction))
        model.coherence_temperature.fill_(voting_temp)
        anchor_frac = min(
            1.0, temp_fraction / max(anchor_blend_fraction, 1e-3)
        )
        anchor_blend = anchor_blend_start + 0.5 * (
            anchor_blend_end - anchor_blend_start
        ) * (1.0 - math.cos(math.pi * anchor_frac))
        model.anchor_blend.fill_(anchor_blend)
        peak_power = model.peak_render_start + 0.5 * (
            model.peak_render_end - model.peak_render_start
        ) * (1.0 - math.cos(math.pi * temp_fraction))
        model.peak_power.fill_(peak_power)
        capture_floor = model.capture_floor_start + 0.5 * (
            model.capture_floor_end - model.capture_floor_start
        ) * (1.0 - math.cos(math.pi * temp_fraction))
        model.capture_floor.fill_(capture_floor)
        consensus_temp = consensus_temp_start + 0.5 * (
            consensus_temp_end - consensus_temp_start
        ) * (1.0 - math.cos(math.pi * temp_fraction))
        model.consensus_temperature.fill_(consensus_temp)

        camera_index = int(rng.integers(0, len(train_cameras)))
        camera = train_cameras[camera_index]
        image = images[camera_index] if camera_index < len(images) else images[0]
        height, width = int(camera["height"]), int(camera["width"])
        total_pixels = height * width
        sample_count = min(patch, total_pixels)
        if spatial_patch:
            side = max(8, int(round(math.sqrt(sample_count))))
            side = min(side, height, width)
            y0 = int(rng.integers(0, max(1, height - side + 1)))
            x0 = int(rng.integers(0, max(1, width - side + 1)))
            ys_grid, xs_grid = np.meshgrid(
                np.arange(y0, y0 + side),
                np.arange(x0, x0 + side),
                indexing="ij",
            )
            ys = ys_grid.reshape(-1).astype(np.int64)
            xs = xs_grid.reshape(-1).astype(np.int64)
        else:
            flat_indices = rng.choice(
                total_pixels, size=sample_count, replace=False
            )
            ys = (flat_indices // width).astype(np.int64)
            xs = (flat_indices % width).astype(np.int64)
        origins, directions = _camera_rays_from_pose(camera, device, ys, xs)
        origins_scaled = (origins - shift) * scale
        near, far = _ray_bounds(origins_scaled, scene_points, radius, scale, shift)
        target_depth, hit_mask, depth_confidence = _ray_point_targets(
            origins_scaled,
            directions,
            (scene_points_t - shift) * scale,
            tolerance=depth_tolerance,
        )
        visibility_target = visibility_target_min + (
            visibility_target_max - visibility_target_min
        ) * depth_confidence
        target = torch.tensor(
            image[ys, xs].astype(np.float32), device=device, dtype=torch.float32
        )
        target_luminance = (
            0.2126 * target[..., 0:1]
            + 0.7152 * target[..., 1:2]
            + 0.0722 * target[..., 2:3]
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            "cuda", enabled=use_amp and device.type == "cuda"
        ):
            (
                rendered,
                weights,
                sample_alpha,
                visibility,
                sample_positions,
                coarse_rendered,
                sample_gates,
                consensus_weights,
                physical_rendered,
                color,
            ) = render_rays_chunk(
                model,
                origins_scaled,
                directions,
                near,
                far,
                coarse_samples=coarse_samples,
                fine_samples=fine_samples,
                readout_chunk=readout_chunk,
                use_amp=use_amp,
                jitter=True,
                anchor_depth=target_depth,
                anchor_spread=anchor_spread,
                anchor_mask=hit_mask,
                return_detail=True,
            )
            sample_depths = torch.sum(
                (sample_positions - origins_scaled.unsqueeze(1))
                * directions.unsqueeze(1),
                dim=-1,
            )
            render_weights = _visible_topk_render_weights(
                consensus_weights,
                sample_alpha,
                weights,
                float(model.peak_power),
                int(model.render_top_k),
            )
            consensus_vote = render_weights[..., 0]
            weights_sum = consensus_vote.sum(dim=1).clamp(min=1e-6)
            expected_depth = (
                torch.sum(consensus_vote * sample_depths, dim=1) / weights_sum
            )
            depth_sq = (
                torch.sum(
                    consensus_vote * sample_depths**2,
                    dim=1,
                )
                / weights_sum
            )
            depth_variance = (depth_sq - expected_depth**2).clamp(min=0.0)
            scaffold_values = _sample_scaffold(
                scaffold_grid,
                sample_positions,
                resolution=scaffold_resolution,
            )
            scaffold_penalty = torch.mean(
                sample_alpha[..., 0] * (1.0 - scaffold_values)
            )
            color_loss = torch.mean((rendered - target) ** 2)
            winner_color_loss = torch.mean((color - target) ** 2)
            coarse_color_loss = torch.mean((coarse_rendered - target) ** 2)
            physical_color_loss = torch.mean(
                (physical_rendered - target) ** 2
            )
            occupancy_penalty = torch.mean(
                sample_alpha * (1.0 - sample_alpha)
            )
            alpha_visibility = weights.sum(dim=1).clamp(max=1.0)
            bright_mask = (target_luminance > 0.25).to(rendered.dtype)
            visibility_deficit = torch.relu(
                visibility_target - alpha_visibility
            )
            hole_penalty = torch.mean(
                hit_mask * bright_mask * visibility_deficit**2
            )
            alpha_peak = sample_alpha.max(dim=1).values[..., 0]
            surface_alpha_penalty = torch.mean(
                hit_mask
                * depth_confidence
                * torch.relu(surface_alpha_target - alpha_peak) ** 2
            )
            visibility_penalty = torch.mean(
                hit_mask * torch.relu(visibility_target - visibility) ** 2
            )
            visibility_upper_penalty = torch.mean(
                hit_mask
                * torch.relu(visibility - visibility_upper_cap) ** 2
            )
            weights_norm = consensus_vote / weights_sum.unsqueeze(1)
            raw_consensus = consensus_weights[..., 0]
            sharpness_penalty = torch.mean(
                hit_mask * (1.0 - raw_consensus.max(dim=1).values)
            )
            consensus_entropy = -torch.sum(
                raw_consensus * torch.log(raw_consensus.clamp(min=1e-6)),
                dim=1,
            )
            entropy_penalty = torch.mean(hit_mask * consensus_entropy)
            consensus_top2 = torch.topk(raw_consensus, 2, dim=1).values
            consensus_margin_penalty = torch.mean(
                hit_mask
                * torch.relu(
                    0.12 - (consensus_top2[:, 0] - consensus_top2[:, 1])
                )
            )
            target_gauss = torch.exp(
                -0.5
                * (
                    (sample_depths - target_depth.unsqueeze(1))
                    / max(float(depth_sigma), 1e-3)
                )
                ** 2
            )
            target_weights = target_gauss / target_gauss.sum(
                dim=1, keepdim=True
            ).clamp(min=1e-6)
            surface_energy = (
                torch.sum(consensus_vote * target_gauss, dim=1) / weights_sum
            )
            peak_penalty = torch.mean(
                hit_mask
                * depth_confidence
                * (1.0 - surface_energy.clamp(min=0.0, max=1.0))
            )
            depth_scale = max(float(far - near), 1e-3)
            geometry_penalty = torch.mean(
                depth_confidence
                * hit_mask
                * (
                    ((expected_depth - target_depth) / depth_scale) ** 2
                    + depth_variance / depth_scale**2
                    + surface_weight * (1.0 - surface_energy) ** 2
                )
            )
            depth_dist_penalty = torch.mean(
                hit_mask
                * depth_confidence
                * torch.sum(
                    (weights_norm - target_weights) ** 2,
                    dim=1,
                )
            )
            depth_segments = torch.abs(
                sample_depths[:, 1:] - sample_depths[:, :-1]
            )
            depth_segments = torch.cat(
                [
                    depth_segments,
                    depth_segments[:, -1:],
                ],
                dim=1,
            ).unsqueeze(-1)
            density = torch.nn.functional.softplus(sample_gates) / depth_segments.clamp(
                min=1e-6
            )
            density_floor_penalty = torch.mean(
                hit_mask
                * torch.relu(
                    density_floor - density.max(dim=1).values.unsqueeze(-1)
                )
            )
            surface_mask = hit_mask * depth_confidence
            target_gauss_3d = target_gauss.unsqueeze(-1)
            surface_density_penalty = torch.mean(
                surface_mask.unsqueeze(-1).unsqueeze(-1)
                * (
                    torch.relu(surface_density - density) * target_gauss_3d
                    + surface_density_outside
                    * density
                    * (1.0 - target_gauss_3d)
                )
            )
            bg_target = model.background_target.to(model.background.dtype)
            bg_penalty = (
                torch.mean((model.background - bg_target) ** 2)
                + 0.02 * torch.mean(torch.relu(-model.background))
            ) * background_weight
            geometry_scale = min(1.0, (step + 1) / max(geometry_warmup, 1))
            reg_scale = 1.0 - 0.6 * temp_fraction
            physical_color_scale = physical_color_weight * (
                1.0 - 0.8 * temp_fraction
            )
            loss = (
                color_loss
                + winner_color_weight * winner_color_loss
                + coarse_color_weight * coarse_color_loss
                + physical_color_scale * physical_color_loss
                + occupancy_weight * occupancy_penalty
                + hole_weight * hole_penalty
                + visibility_weight * visibility_penalty
                + visibility_upper_weight * visibility_upper_penalty
                + reg_scale * sharpness_weight * sharpness_penalty
                + reg_scale * consensus_entropy_weight * entropy_penalty
                + reg_scale * consensus_margin_weight * consensus_margin_penalty
                + reg_scale * geometry_weight * geometry_scale * geometry_penalty
                + reg_scale * peak_weight * geometry_scale * peak_penalty
                + reg_scale * surface_alpha_weight * geometry_scale * surface_alpha_penalty
                + reg_scale * scaffold_weight * geometry_scale * scaffold_penalty
                + reg_scale * depth_dist_weight * geometry_scale * depth_dist_penalty
                + reg_scale * density_floor_weight * density_floor_penalty
                + reg_scale * surface_density_weight * geometry_scale * surface_density_penalty
                + bg_penalty
            )
        if not torch.isfinite(loss):
            components = {
                "rendered": rendered,
                "weights": weights,
                "sample_alpha": sample_alpha,
                "visibility": visibility,
                "visibility_upper": visibility_upper_penalty,
                "sample_positions": sample_positions,
                "coarse_rendered": coarse_rendered,
                "sample_gates": sample_gates,
                "consensus_weights": consensus_weights,
                "color": color,
            }
            bad = [
                name
                for name, tensor in components.items()
                if not torch.isfinite(tensor.float()).all()
            ]
            log(
                f"[tcf] non-finite loss at step {step} "
                f"loss={float(loss)} bad={bad} "
                f"color={float(color_loss):.6f} "
                f"winner={float(winner_color_loss):.6f} "
                f"coarse={float(coarse_color_loss):.6f} "
                f"occ={float(occupancy_penalty):.6f} "
                f"vis={float(visibility_penalty):.6f} "
                f"sharp={float(sharpness_penalty):.6f} "
                f"entropy={float(entropy_penalty):.6f} "
                f"margin={float(consensus_margin_penalty):.6f} "
                f"geo={float(geometry_penalty):.6f} "
                f"peak={float(peak_penalty):.6f} "
                f"scaf={float(scaffold_penalty):.6f} "
                f"salpha={float(surface_alpha_penalty):.6f} "
                f"dist={float(depth_dist_penalty):.6f} "
                f"dfloor={float(density_floor_penalty):.6f} "
                f"sdfd={float(surface_density_penalty):.6f}"
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_parameters, 1.0)
        if os.environ.get("TITAN_DBG_GRAD") and step == 0:
            with torch.no_grad():
                grad_items = []
                bias_grad = model.readout.linear_out.bias.grad
                if bias_grad is not None:
                    grad_items.append(
                        "out_bias_grad="
                        + str(bias_grad.detach().cpu().numpy().tolist())
                    )
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_items.append(
                            f"{name}:norm={float(param.grad.norm()):.4f}"
                            f",mean={float(param.grad.mean()):.6f}"
                        )
                log("[tcf:dbg:grad] " + " | ".join(grad_items))
        optimizer.step()
        scheduler.step()

        raw_loss_value = float(loss.detach().cpu())
        if loss_ema is None:
            loss_ema = raw_loss_value
        else:
            loss_ema = 0.97 * loss_ema + 0.03 * raw_loss_value
        loss_value = loss_ema
        if os.environ.get("TITAN_DEBUG") and step % 10 == 0:
            with torch.no_grad():
                log(
                    "[tcf:dbg] step {} hit {:.3f} vis {:.3f} alpha {:.4f} "
                    "ed {:.3f} td {:.3f} loss c {:.5f} win {:.5f} hole {:.6f} "
                    "occ {:.6f} "
                    "visp {:.6f} sharp {:.6f} "
                    "entropy {:.6f} margin {:.6f} geo {:.6f} "
                    "peak {:.6f} "
                    "scaf {:.6f} salpha {:.6f} dist {:.6f} dfloor {:.6f} "
                    "sdfd {:.6f} conf {:.3f} gate {:.3f} "
                    "floor {:.3f} cwmax {:.3f} "
                    "ctemp {:.3f}".format(
                        step,
                        float(hit_mask.mean()),
                        float(visibility.mean()),
                        float(sample_alpha.mean()),
                        float(expected_depth.mean()),
                        float(target_depth.mean()),
                        float(color_loss),
                        float(winner_color_loss),
                        float(hole_penalty),
                        float(occupancy_penalty),
                        float(visibility_penalty),
                        float(visibility_upper_penalty),
                        float(sharpness_penalty),
                        float(entropy_penalty),
                        float(consensus_margin_penalty),
                        float(geometry_penalty),
                        float(peak_penalty),
                        float(scaffold_penalty),
                        float(surface_alpha_penalty),
                        float(depth_dist_penalty),
                        float(density_floor_penalty),
                        float(surface_density_penalty),
                        float(depth_confidence.mean()),
                        float(sample_gates.mean()),
                        float(model.background.mean()),
                        float(model.capture_floor),
                        float(
                            consensus_weights.max(dim=1).values.mean()
                        ),
                        float(model.consensus_temperature),
                    )
                )
        if step % progress_every == 0 or step == steps - 1:
            payload = {
                "taskId": task_id,
                "status": "running",
                "stage": "training",
                "progress": round(0.12 + 0.78 * ((step + 1) / steps), 4),
                "step": step,
                "totalSteps": steps,
                "loss": loss_value,
                "lossRaw": raw_loss_value,
                "vramMB": _device_reserved_mb(),
                "latticeMB": model.lattice_mb(),
            }
            if status_cb:
                status_cb(payload)

        if step == half_step or step == steps - 1:
            eval_index = 0 if step == half_step else len(eval_cameras) - 1
            eval_camera = eval_cameras[eval_index]
            target_index = len(cameras) - len(eval_cameras) + eval_index
            if target_index < len(images):
                target_view = images[target_index]
            else:
                target_view = image
            rendered_view = render_tcf_view(
                model,
                eval_camera,
                shift,
                scale,
                scene_points,
                radius,
                ray_batch=ray_batch,
                coarse_samples=coarse_samples,
                fine_samples=fine_samples,
                use_amp=use_amp,
            )
            mse = float(np.mean((rendered_view - target_view) ** 2))
            from .metrics import psnr as compute_psnr, ssim as compute_ssim

            metrics = {
                "step": step,
                "totalSteps": steps,
                "loss": mse,
                "psnr": compute_psnr(rendered_view, target_view),
                "ssim": compute_ssim(rendered_view, target_view),
                "vramMB": int(_device_reserved_mb()),
                "latticeMB": model.lattice_mb(),
                "elapsed": round(time.time() - start_wall, 2),
            }
            history.append(metrics)
            log(
                f"[tcf] step {step} loss {mse:.5f} "
                f"psnr {metrics['psnr']:.2f} ssim {metrics['ssim']:.3f}"
            )
            if status_cb:
                status_cb(
                    {
                        **payload,
                        "psnr": metrics["psnr"],
                        "ssim": metrics["ssim"],
                        "vramMB": int(_device_reserved_mb()),
                        "metrics": history,
                    }
                )

    model.coherence_temperature.fill_(voting_temp_end)
    model.anchor_blend.fill_(anchor_blend_end)
    model.peak_power.fill_(model.peak_render_end)
    model.capture_floor.fill_(model.capture_floor_end)
    model.consensus_temperature.fill_(consensus_temp_end)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.to("cpu")
    return {
        "model": model,
        "shift": shift.cpu(),
        "scale": scale,
        "radius": radius,
        "engine": "tcf",
        "history": history,
        "latticeMB": model.lattice_mb(),
        "peakVramMB": _device_peak_mb(),
    }, history


def render_tcf_view(
    model: TitanCoherenceField,
    camera: dict,
    shift: torch.Tensor,
    scale: float,
    scene_points: np.ndarray,
    radius: float,
    ray_batch: int = 4096,
    coarse_samples: int | None = None,
    fine_samples: int | None = None,
    use_amp: bool = True,
    depth_tolerance: float = 0.25,
    anchor_spread: float = 0.08,
) -> np.ndarray:
    """Render a full view with a memory-bounded ray loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if coarse_samples is None:
        coarse_samples = int(getattr(model, "coarse_samples", 32))
    if fine_samples is None:
        fine_samples = int(getattr(model, "fine_samples", 16))
    model.coherence_temperature.fill_(
        getattr(model, "voting_temperature_end", 6.0)
    )
    model.anchor_blend.fill_(getattr(model, "anchor_blend_end", 0.0))
    model.peak_power.fill_(
        getattr(model, "peak_render_end", 3.0)
    )
    model.capture_floor.fill_(
        getattr(model, "capture_floor_end", 0.06)
    )
    model.consensus_temperature.fill_(
        getattr(model, "consensus_temperature_end", 0.6)
    )
    shift = shift.to(device)
    anchor_points = None
    if len(scene_points):
        anchor_points = torch.from_numpy(
            _voxel_sample(scene_points.astype(np.float32), 20000)
        ).to(device)
        anchor_points = (anchor_points - shift) * scale
    origins, directions = _camera_rays_from_pose(camera, device)
    origins_scaled = (origins - shift) * scale
    near, far = _ray_bounds(origins_scaled, scene_points, radius, scale, shift)
    with torch.no_grad():
        if os.environ.get("TITAN_DBG_MEM"):
            log(
                f"[tcf:mem] view rays {origins_scaled.shape[0]} "
                f"alloc {torch.cuda.memory_allocated() / 2**20:.0f} MiB "
                f"reserved {torch.cuda.memory_reserved() / 2**20:.0f} MiB"
            )
        rendered = render_rays(
            model,
            origins_scaled,
            directions,
            near,
            far,
            ray_batch=ray_batch,
            coarse_samples=coarse_samples,
            fine_samples=fine_samples,
            use_amp=use_amp,
            scene_points=anchor_points,
            depth_tolerance=depth_tolerance,
            anchor_spread=anchor_spread,
        )
        if os.environ.get("TITAN_DBG_MEM"):
            log(
                f"[tcf:mem] view done alloc {torch.cuda.memory_allocated() / 2**20:.0f} MiB "
                f"reserved {torch.cuda.memory_reserved() / 2**20:.0f} MiB"
            )
    image = rendered.detach().cpu().numpy()
    height, width = int(camera["height"]), int(camera["width"])
    return image.reshape(height, width, 3)
