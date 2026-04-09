"""Noise Reduction — NLM and wavelet denoising.

Uses OpenCV's fastNlMeansDenoising (BSD) for quick NLM
and PyWavelets (MIT) for wavelet-based denoising.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import cv2
import numpy as np
import pywt

from cosmica.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


class DenoiseMethod(Enum):
    NLM = auto()  # Non-local means (OpenCV)
    WAVELET = auto()  # Wavelet thresholding (PyWavelets)


@dataclass
class DenoiseParams:
    """Parameters for noise reduction."""

    method: DenoiseMethod = DenoiseMethod.WAVELET
    strength: float = 0.5  # 0-1, overall denoising strength
    detail_preservation: float = 0.5  # 0-1, how much fine detail to keep
    chrominance_only: bool = False  # only denoise color, not luminance
    # NLM-specific
    nlm_h: float = 10.0  # filter strength for NLM
    nlm_template_size: int = 7  # template patch size
    nlm_search_size: int = 21  # search window size
    # Wavelet-specific
    wavelet: str = "db4"  # wavelet family
    wavelet_levels: int = 4  # decomposition levels


def _denoise_nlm_channel(channel: np.ndarray, params: DenoiseParams) -> np.ndarray:
    """Apply OpenCV Non-Local Means denoising to a single channel."""
    # Convert to uint8 for OpenCV NLM (it works on 8-bit images)
    img_u8 = np.clip(channel * 255, 0, 255).astype(np.uint8)
    h = params.nlm_h * params.strength
    denoised = cv2.fastNlMeansDenoising(
        img_u8,
        None,
        h=h,
        templateWindowSize=params.nlm_template_size,
        searchWindowSize=params.nlm_search_size,
    )
    return denoised.astype(np.float32) / 255.0


def _denoise_wavelet_channel(channel: np.ndarray, params: DenoiseParams) -> np.ndarray:
    """Apply wavelet denoising to a single channel using PyWavelets.

    Uses the à trous (stationary wavelet transform) for translation-invariant denoising,
    with BayesShrink thresholding.
    """
    # Estimate noise sigma using MAD of finest wavelet coefficients
    coeffs = pywt.wavedec2(channel, params.wavelet, level=params.wavelet_levels)

    # MAD-based noise estimation from the finest detail coefficients
    detail_finest = coeffs[-1]
    # detail_finest is a tuple of (LH, HL, HH) arrays
    hh = detail_finest[2]  # diagonal detail
    sigma_noise = np.median(np.abs(hh)) / 0.6745

    # Threshold scale: strength controls how aggressively we threshold
    threshold_scale = params.strength * 3.0

    # Apply soft thresholding to detail coefficients
    # Skip the approximation (coeffs[0]) — keep it intact
    denoised_coeffs = [coeffs[0]]
    for level_idx, detail in enumerate(coeffs[1:], 1):
        # Progressive thresholding: less aggressive at coarser scales
        level_factor = 1.0 - (level_idx - 1) / max(params.wavelet_levels, 1)
        # detail_preservation reduces the threshold
        effective_factor = level_factor * (1.0 - params.detail_preservation * 0.8)

        thresholded = []
        for d in detail:
            # BayesShrink: threshold = sigma_noise^2 / sigma_signal
            sigma_d = max(np.std(d), 1e-10)
            sigma_signal = max(np.sqrt(max(sigma_d**2 - sigma_noise**2, 0)), 1e-10)
            thresh = (sigma_noise**2 / sigma_signal) * threshold_scale * effective_factor
            thresholded.append(pywt.threshold(d, thresh, mode='soft'))
        denoised_coeffs.append(tuple(thresholded))

    result = pywt.waverec2(denoised_coeffs, params.wavelet)
    # Trim to original size (wavelet reconstruction may pad)
    result = result[:channel.shape[0], :channel.shape[1]]
    return np.clip(result, 0, 1).astype(np.float32)


def denoise(
    image: np.ndarray,
    params: DenoiseParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Apply noise reduction to an image.

    Parameters
    ----------
    image : ndarray
        Image data, shape (H, W) or (C, H, W), values in [0, 1].
    params : DenoiseParams, optional
        Denoising parameters.
    mask : Mask, optional
        Selective processing mask.
    progress : callable
        Progress callback.

    Returns
    -------
    ndarray
        Denoised image.
    """
    if params is None:
        params = DenoiseParams()

    original = image.copy()
    denoise_fn = _denoise_nlm_channel if params.method == DenoiseMethod.NLM else _denoise_wavelet_channel

    if image.ndim == 2:
        progress(0.1, "Denoising mono image...")
        result = denoise_fn(image, params)
    elif params.chrominance_only and image.shape[0] >= 3:
        progress(0.1, "Denoising chrominance only...")
        result = _denoise_chrominance_only(image, params, denoise_fn, progress)
    else:
        result = np.empty_like(image)
        n_ch = image.shape[0]
        for ch in range(n_ch):
            progress(ch / n_ch, f"Denoising channel {ch + 1}/{n_ch}...")
            result[ch] = denoise_fn(image[ch], params)

    progress(1.0, "Noise reduction complete")
    return apply_mask(original, result, mask)


def _denoise_chrominance_only(
    image: np.ndarray,
    params: DenoiseParams,
    denoise_fn,
    progress: ProgressCallback,
) -> np.ndarray:
    """Denoise only the chrominance, preserving luminance detail."""
    # Convert to YCbCr-like space
    r, g, b = image[0], image[1], image[2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = b - y
    cr = r - y

    progress(0.3, "Denoising Cb channel...")
    cb_dn = denoise_fn(cb + 0.5, params) - 0.5  # shift to [0,1] for denoising
    progress(0.6, "Denoising Cr channel...")
    cr_dn = denoise_fn(cr + 0.5, params) - 0.5

    # Reconstruct
    result = np.empty_like(image)
    result[0] = np.clip(y + cr_dn, 0, 1)
    result[1] = np.clip(y - 0.2126 / 0.7152 * cr_dn - 0.0722 / 0.7152 * cb_dn, 0, 1)
    result[2] = np.clip(y + cb_dn, 0, 1)

    # Copy extra channels unchanged
    if image.shape[0] > 3:
        result[3:] = image[3:]

    return result
