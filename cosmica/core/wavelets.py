"""Wavelets — GPU-accelerated a trous wavelet decomposition and reconstruction.

Uses PyTorch conv2d for GPU acceleration. Falls back to CPU when no GPU.
The B3 spline kernel is used for the a trous algorithm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from cosmica.core.device_manager import get_device_manager
from cosmica.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

# B3 spline 1D kernel for a trous wavelet transform
_B3_KERNEL_1D = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0


def _atrous_kernel_2d(scale: int) -> np.ndarray:
    """Create 2D a trous B3 spline kernel at given scale.

    The kernel is an upsampled version of the B3 spline where
    zeros are inserted between coefficients (a trous = with holes).
    """
    k1d = _B3_KERNEL_1D
    step = 2**scale
    size = (len(k1d) - 1) * step + 1
    padded = np.zeros(size, dtype=np.float32)
    for i, v in enumerate(k1d):
        padded[i * step] = v
    return np.outer(padded, padded)


def _smooth_gpu(data_t: torch.Tensor, kernel_t: torch.Tensor) -> torch.Tensor:
    """Apply 2D smoothing convolution on GPU using torch conv2d."""
    # data_t: (1, 1, H, W), kernel_t: (1, 1, kH, kW)
    pad_h = kernel_t.shape[2] // 2
    pad_w = kernel_t.shape[3] // 2
    return F.conv2d(data_t, kernel_t, padding=(pad_h, pad_w))


@dataclass
class WaveletParams:
    """Parameters for wavelet processing."""

    n_scales: int = 4  # number of wavelet scales
    scale_weights: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0]
    )
    residual_weight: float = 1.0


def wavelet_decompose(
    data: np.ndarray,
    n_scales: int = 4,
) -> list[np.ndarray]:
    """Decompose a 2D array into wavelet scales + residual using a trous.

    Parameters
    ----------
    data : ndarray
        Single-channel 2D array, float32.
    n_scales : int
        Number of wavelet detail scales.

    Returns
    -------
    list[ndarray]
        List of (n_scales + 1) arrays: detail scales [0..n_scales-1] and
        the residual (smooth) at index n_scales.
    """
    dm = get_device_manager()
    device = dm.device

    current = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    scales = []

    for s in range(n_scales):
        kernel_np = _atrous_kernel_2d(s)
        kernel_t = torch.from_numpy(kernel_np).unsqueeze(0).unsqueeze(0).to(device)
        smoothed = _smooth_gpu(current, kernel_t)
        detail = current - smoothed
        scales.append(detail.squeeze().cpu().numpy())
        current = smoothed

    # Residual (low-frequency content)
    scales.append(current.squeeze().cpu().numpy())
    return scales


def wavelet_reconstruct(scales: list[np.ndarray]) -> np.ndarray:
    """Reconstruct an image from wavelet scales.

    Parameters
    ----------
    scales : list[ndarray]
        List from wavelet_decompose: detail scales + residual.

    Returns
    -------
    ndarray
        Reconstructed 2D array.
    """
    result = np.zeros_like(scales[0])
    for s in scales:
        result = result + s
    return result


def wavelet_sharpen(
    data: np.ndarray,
    params: WaveletParams | None = None,
    mask: Mask | None = None,
) -> np.ndarray:
    """Sharpen or smooth image using per-scale wavelet weights.

    Weights > 1.0 sharpen that scale, < 1.0 smooth it.

    Parameters
    ----------
    data : ndarray
        Image data, shape (H, W) or (C, H, W), float32 in [0, 1].
    params : WaveletParams, optional
        Processing parameters.
    mask : Mask, optional
        Processing mask.

    Returns
    -------
    ndarray
        Processed image.
    """
    if params is None:
        params = WaveletParams()

    original = data.copy()

    # Ensure scale_weights length matches n_scales
    weights = list(params.scale_weights)
    while len(weights) < params.n_scales:
        weights.append(1.0)

    def _process_channel(ch: np.ndarray) -> np.ndarray:
        scales = wavelet_decompose(ch, n_scales=params.n_scales)
        for i in range(params.n_scales):
            scales[i] = scales[i] * weights[i]
        scales[-1] = scales[-1] * params.residual_weight
        return np.clip(wavelet_reconstruct(scales), 0, 1).astype(np.float32)

    if data.ndim == 2:
        result = _process_channel(data)
    else:
        result = np.empty_like(data)
        for ch in range(data.shape[0]):
            result[ch] = _process_channel(data[ch])

    return apply_mask(original, result, mask)
