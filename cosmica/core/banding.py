"""Banding Reduction — remove horizontal and vertical banding artifacts.

Banding is common in CMOS sensors, appearing as faint stripes in the
background. This module detects and subtracts per-row/column offsets
using sigma-clipped median analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from cosmica.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class BandingParams:
    """Parameters for banding reduction."""

    horizontal: bool = True  # remove horizontal banding (row offsets)
    vertical: bool = False  # remove vertical banding (column offsets)
    amount: float = 1.0  # correction strength (0-1)
    protection_sigma: float = 3.0  # sigma clip to reject bright objects


def _compute_line_offsets(
    data_2d: np.ndarray,
    axis: int,
    protection_sigma: float,
) -> np.ndarray:
    """Compute per-line (row or column) offset from neighbors.

    Uses sigma clipping along each line to reject bright objects (stars, etc.),
    then computes the median offset of each line relative to a smoothed version.

    Parameters
    ----------
    data_2d : ndarray
        2D channel data, shape (H, W).
    axis : int
        0 = compute row offsets (horizontal banding),
        1 = compute column offsets (vertical banding).
    protection_sigma : float
        Sigma clipping threshold for object rejection.

    Returns
    -------
    ndarray
        Per-line offsets to subtract.
    """
    if axis == 0:
        n_lines = data_2d.shape[0]
    else:
        n_lines = data_2d.shape[1]

    line_medians = np.zeros(n_lines, dtype=np.float32)

    for i in range(n_lines):
        if axis == 0:
            line = data_2d[i, :]
        else:
            line = data_2d[:, i]

        # Iterative sigma clip
        valid = line.copy()
        for _ in range(3):
            med = np.median(valid)
            mad = np.median(np.abs(valid - med))
            sigma = max(mad * 1.4826, 1e-10)
            keep = np.abs(valid - med) < protection_sigma * sigma
            if keep.sum() < 3:
                break
            valid = valid[keep]

        line_medians[i] = np.median(valid)

    # Compute offsets: deviation from smoothed version
    # Use a windowed running median for the smooth reference
    window = max(5, n_lines // 20)
    if window % 2 == 0:
        window += 1

    smooth = _running_median(line_medians, window)
    offsets = line_medians - smooth

    return offsets


def _running_median(data: np.ndarray, window: int) -> np.ndarray:
    """Compute a running median filter.

    Parameters
    ----------
    data : ndarray
        1D input array.
    window : int
        Window size (must be odd).

    Returns
    -------
    ndarray
        Smoothed array.
    """
    n = len(data)
    half = window // 2
    result = np.zeros_like(data)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result[i] = np.median(data[lo:hi])

    return result


def banding_reduction(
    image: np.ndarray,
    params: BandingParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Remove horizontal and/or vertical banding from an image.

    Parameters
    ----------
    image : ndarray
        Image data, shape (H, W) or (C, H, W), values in [0, 1].
    params : BandingParams, optional
        Banding reduction parameters.
    mask : Mask, optional
        Selective processing mask.
    progress : callable
        Progress callback.

    Returns
    -------
    ndarray
        Corrected image.
    """
    if params is None:
        params = BandingParams()

    if not params.horizontal and not params.vertical:
        return image

    original = image.copy()
    result = image.copy()

    if result.ndim == 2:
        channels = [(result, lambda d: None)]
    else:
        channels = [(result[ch], ch) for ch in range(result.shape[0])]

    n_ch = len(channels)
    step = 0
    total_steps = n_ch * (int(params.horizontal) + int(params.vertical))

    for ch_idx, (ch_data, _) in enumerate(channels):
        if params.horizontal:
            progress(step / total_steps, f"Horizontal banding ch{ch_idx + 1}...")
            offsets = _compute_line_offsets(ch_data, axis=0, protection_sigma=params.protection_sigma)
            correction = offsets[:, np.newaxis] * params.amount
            ch_data -= correction
            step += 1

        if params.vertical:
            progress(step / total_steps, f"Vertical banding ch{ch_idx + 1}...")
            offsets = _compute_line_offsets(ch_data, axis=1, protection_sigma=params.protection_sigma)
            correction = offsets[np.newaxis, :] * params.amount
            ch_data -= correction
            step += 1

        # Write back
        if result.ndim == 2:
            result[:] = ch_data
        else:
            result[ch_idx] = ch_data

    result = np.clip(result, 0, 1)
    progress(1.0, "Banding reduction complete")
    return apply_mask(original, result, mask)
