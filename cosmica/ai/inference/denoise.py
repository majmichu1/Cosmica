"""AI Denoise — deep learning noise reduction using U-Net.

Uses tiled inference to handle large astro images.
Model weights are downloaded on first use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from cosmica.ai.inference.tiled import tiled_inference
from cosmica.ai.model_manager import ModelType, get_model_manager
from cosmica.ai.models.unet import UNet
from cosmica.core.device_manager import get_device_manager
from cosmica.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class AIDenoiseParams:
    """Parameters for AI denoising."""

    strength: float = 1.0  # 0-1, blend between original and denoised
    tile_size: int = 512
    overlap: int = 64


def _load_trained_model() -> UNet | None:
    """Try to load a trained N2S model from disk."""
    from pathlib import Path

    # Check bundled models directory
    local_path = Path(__file__).resolve().parent.parent / "models" / "cosmica_denoise_n2s_v1.pt"
    if local_path.exists():
        try:
            model = UNet(in_channels=1, out_channels=1, base_features=32, depth=4)
            model.load_state_dict(torch.load(local_path, map_location="cpu", weights_only=True))
            model.eval()
            log.info("Loaded trained denoise model from %s", local_path)
            return model
        except Exception as e:
            log.debug("Could not load local model: %s", e)

    # Also check model manager cache
    try:
        mm = get_model_manager()
        model_path = mm.get_cache_path(ModelType.DENOISE)
        if model_path and model_path.exists():
            model = UNet(
                in_channels=1, out_channels=1,
                base_features=32, depth=4,
            )
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            model.eval()
            log.info("Loaded trained denoise model from %s", model_path)
            return model
    except Exception as e:
        log.debug("Could not load trained model from cache: %s", e)

    return None


def ai_denoise(
    data: np.ndarray,
    model: torch.nn.Module | None = None,
    params: AIDenoiseParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback | None = None,
) -> np.ndarray:
    """Denoise an image using a trained U-Net model.

    Automatically loads trained model if available, falls back to wavelet denoise.
    """
    if params is None:
        params = AIDenoiseParams()
    if progress is None:
        progress = _noop_progress

    if model is None:
        model = _load_trained_model()

    if model is None:
        # No trained model — fall back to traditional wavelet denoise
        log.info("No AI model available, falling back to wavelet denoise")
        return _traditional_denoise(data, params, mask, progress)

    dm = get_device_manager()
    model = model.to(dm.device)
    original = data.copy()

    if data.ndim == 2:
        result = tiled_inference(
            data, model,
            tile_size=params.tile_size,
            overlap=params.overlap,
            progress=progress,
        )
    else:
        result = np.empty_like(data)
        n_ch = data.shape[0]
        for ch in range(n_ch):
            def ch_progress(frac, msg, c=ch):
                total_frac = (c + frac) / n_ch
                progress(total_frac, f"Channel {c + 1}/{n_ch}: {msg}")

            result[ch] = tiled_inference(
                data[ch], model,
                tile_size=params.tile_size,
                overlap=params.overlap,
                progress=ch_progress,
            )

    # Blend with original based on strength
    if params.strength < 1.0:
        result = data * (1 - params.strength) + result * params.strength

    result = np.clip(result, 0, 1).astype(np.float32)
    return apply_mask(original, result, mask)


def _traditional_denoise(
    data: np.ndarray,
    params: AIDenoiseParams,
    mask: Mask | None,
    progress: ProgressCallback,
) -> np.ndarray:
    """Traditional wavelet denoise as fallback when no AI model is available."""
    from cosmica.core.denoise import DenoiseParams, denoise

    # Map AI strength (0-1) to wavelet denoise strength
    strength = max(0.1, params.strength * 0.8)
    wp = DenoiseParams(
        method="wavelet",
        strength=strength,
        detail_preservation=0.7,
        wavelet="db4",
        wavelet_levels=4,
    )
    return denoise(data, params=wp, mask=mask, progress=progress)


def _create_default_model() -> UNet:
    """Create a default U-Net model (untrained, for testing)."""
    return UNet(in_channels=1, out_channels=1, base_features=16, depth=3)
