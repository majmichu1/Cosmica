"""Auto-Stretch — statistical stretch with midtone transfer function.

GPU-accelerated via the device manager.
Inspired by PixInsight's STF and Screen Transfer Function approach.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from cosmica.core.device_manager import get_device_manager

log = logging.getLogger(__name__)


@dataclass
class StretchParams:
    """Parameters for the midtone stretch."""

    shadow_clip: float = -2.8  # MAD units below median for black point
    highlight_clip: float = 1.0  # maximum white point (fraction of range)
    midtone: float = 0.25  # target midtone balance (0-1)
    linked: bool = True  # link RGB channels (same stretch for all)


def compute_channel_stats(data: np.ndarray) -> dict:
    """Compute statistics for a single channel (CPU — small reduction overhead)."""
    valid = data[data > 0] if np.any(data > 0) else data.ravel()
    if len(valid) == 0:
        return {"median": 0, "mad": 0, "mean": 0, "std": 0, "min": 0, "max": 0}

    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    return {
        "median": median,
        "mad": mad,
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
    }


def midtone_transfer_function(x: np.ndarray, midtone: float) -> np.ndarray:
    """Apply midtone transfer function (MTF) on GPU.

    MTF(x, m) = (m - 1) * x / ((2m - 1) * x - m)
    where m is the midtone balance parameter.
    """
    dm = get_device_manager()
    t = dm.from_numpy(x)  # to GPU or CPU tensor

    x_c = torch.clamp(t, 0.0, 1.0)
    mask = x_c > 0
    denom = (2.0 * midtone - 1.0) * x_c - midtone

    result = torch.zeros_like(x_c)
    safe_mask = torch.abs(denom) > 1e-10
    combined = mask & safe_mask
    if combined.any():
        result[combined] = (midtone - 1.0) * x_c[combined] / denom[combined]

    return torch.clamp(result, 0.0, 1.0).cpu().numpy().astype(np.float32)


def auto_stretch(
    data: np.ndarray,
    params: StretchParams | None = None,
) -> np.ndarray:
    """Apply auto-stretch to image data (GPU-accelerated).

    Input: float32 in [0, 1], shape (H, W) for mono or (C, H, W) for color.
    Output: same shape, stretched to [0, 1].
    """
    if params is None:
        params = StretchParams()

    if data.ndim == 2:
        return _stretch_channel_gpu(data, params)

    n_ch = data.shape[0]
    result = np.empty_like(data)

    if params.linked and n_ch >= 3:
        # Compute combined statistics from all channels (CPU — small reduction)
        all_stats = [compute_channel_stats(data[ch]) for ch in range(n_ch)]
        ref_ch = max(range(n_ch), key=lambda i: all_stats[i]["median"])
        ref_stats = all_stats[ref_ch]

        shadow = max(0.0, ref_stats["median"] + params.shadow_clip * max(ref_stats["mad"], 1e-8))
        scale = 1.0 / max(1e-10, params.highlight_clip - shadow)

        # GPU-accelerated per-channel stretch
        for ch in range(n_ch):
            stretched = (data[ch] - shadow) * scale
            result[ch] = midtone_transfer_function(stretched, params.midtone)
    else:
        for ch in range(n_ch):
            result[ch] = _stretch_channel_gpu(data[ch], params)

    return result


def _stretch_channel_gpu(channel: np.ndarray, params: StretchParams) -> np.ndarray:
    """Stretch a single channel using GPU acceleration."""
    stats = compute_channel_stats(channel)
    shadow = max(0.0, stats["median"] + params.shadow_clip * max(stats["mad"], 1e-8))
    scale = 1.0 / max(1e-10, params.highlight_clip - shadow)

    # GPU: apply linear stretch
    dm = get_device_manager()
    t = dm.from_numpy(channel)
    stretched_t = (t - shadow) * scale
    stretched_np = stretched_t.cpu().numpy()

    return midtone_transfer_function(stretched_np, params.midtone)


@dataclass
class GHSParams:
    """Parameters for Generalized Hyperbolic Stretch (GHS).

    GHS is a more flexible stretch function that provides separate control
    over stretch intensity, symmetry, and shadow/highlight protection.
    """

    D: float = 5.0  # stretch factor (intensity), 0 = identity
    b: float = 0.0  # asymmetry (-5 to 5), 0 = symmetric
    SP: float = 0.0  # symmetry point (0-1)
    shadow_protection: float = 0.0  # 0-1
    highlight_protection: float = 0.0  # 0-1
    linked: bool = True


def generalized_hyperbolic_stretch(
    data: np.ndarray,
    params: GHSParams | None = None,
) -> np.ndarray:
    """Apply Generalized Hyperbolic Stretch (GHS) — GPU-accelerated."""
    if params is None:
        params = GHSParams()

    if params.D == 0:
        return data.copy()

    if data.ndim == 2:
        return _ghs_channel_gpu(data, params)

    result = np.empty_like(data)
    for ch in range(data.shape[0]):
        result[ch] = _ghs_channel_gpu(data[ch], params)

    return result


def _ghs_channel_gpu(channel: np.ndarray, params: GHSParams) -> np.ndarray:
    """Apply GHS to a single channel using GPU acceleration."""
    dm = get_device_manager()
    t = dm.from_numpy(channel)

    D = float(params.D)
    b = float(params.b)
    SP = float(params.SP)

    if abs(b) < 0.01:
        centered = t - SP
        stretched = SP + torch.sign(centered) * torch.pow(
            torch.abs(centered) + 1e-10, 1.0 / max(D, 0.1)
        )
    else:
        centered = (t - SP) * D
        b_tensor = torch.tensor(b, device=t.device)
        stretched = SP + torch.sinh(centered * b_tensor) / (
            2.0 * b_tensor * torch.cosh(centered * b_tensor / 2.0) ** 2 + 1e-10
        )

    # Normalize to [0, 1]
    s_min = stretched.amin()
    s_max = stretched.amax()
    if s_max > s_min:
        stretched = (stretched - s_min) / (s_max - s_min)

    # Shadow protection
    sp = params.shadow_protection
    if sp > 0:
        shadow_blend = torch.clamp(1.0 - t / max(float(sp), 1e-10), 0.0, 1.0)
        stretched = stretched * (1.0 - shadow_blend) + t * shadow_blend

    # Highlight protection
    hp = params.highlight_protection
    if hp > 0:
        highlight_blend = torch.clamp(
            (t - (1.0 - float(hp))) / max(float(hp), 1e-10), 0.0, 1.0
        )
        stretched = stretched * (1.0 - highlight_blend) + t * highlight_blend

    return torch.clamp(stretched, 0.0, 1.0).cpu().numpy().astype(np.float32)


def compute_histogram(
    data: np.ndarray, bins: int = 256, range_min: float = 0.0, range_max: float = 1.0
) -> dict:
    """Compute histogram data for display.

    Stats are computed on a thumbnail (≤1024px longest side) for speed.
    Returns dict with 'edges' (bin edges) and per-channel counts.
    """
    # Downsample for speed — histogram shape is identical, counts scale proportionally
    h, w = data.shape[-2], data.shape[-1]
    max_side = 1024
    if max(h, w) > max_side:
        sr = max(1, h // max_side)
        sc = max(1, w // max_side)
        if data.ndim == 2:
            data = data[::sr, ::sc]
        else:
            data = data[:, ::sr, ::sc]

    edges = np.linspace(range_min, range_max, bins + 1)
    result = {"edges": edges}

    if data.ndim == 2:
        counts, _ = np.histogram(data.ravel(), bins=edges)
        result["gray"] = counts
    elif data.ndim == 3:
        colors = ["red", "green", "blue"]
        for ch in range(min(data.shape[0], 3)):
            counts, _ = np.histogram(data[ch].ravel(), bins=edges)
            result[colors[ch]] = counts
        if data.shape[0] >= 3:
            lum = 0.2126 * data[0] + 0.7152 * data[1] + 0.0722 * data[2]
            counts, _ = np.histogram(lum.ravel(), bins=edges)
            result["luminance"] = counts

    return result
