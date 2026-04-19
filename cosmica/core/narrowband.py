"""Narrowband Processing — palette mapping and continuum subtraction.

Maps narrowband filter data (Ha, OIII, SII) into RGB composites
using standard or custom palette matrices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import torch

from cosmica.core.device_manager import get_device_manager

log = logging.getLogger(__name__)


class NarrowbandPalette(Enum):
    SHO = auto()  # Hubble palette: S=R, H=G, O=B
    HOO = auto()  # Ha=R, OIII=G+B
    HOS = auto()  # Ha=R, OIII=G, SII=B (natural-ish)
    CUSTOM = auto()


# Predefined palette matrices: each row is [Ha_weight, OIII_weight, SII_weight]
# for [R, G, B] output channels
PALETTE_MATRICES = {
    NarrowbandPalette.SHO: np.array([
        [0.0, 0.0, 1.0],  # R = SII
        [1.0, 0.0, 0.0],  # G = Ha
        [0.0, 1.0, 0.0],  # B = OIII
    ], dtype=np.float32),
    NarrowbandPalette.HOO: np.array([
        [1.0, 0.0, 0.0],  # R = Ha
        [0.0, 1.0, 0.0],  # G = OIII
        [0.0, 1.0, 0.0],  # B = OIII
    ], dtype=np.float32),
    NarrowbandPalette.HOS: np.array([
        [1.0, 0.0, 0.0],  # R = Ha
        [0.0, 1.0, 0.0],  # G = OIII
        [0.0, 0.0, 1.0],  # B = SII
    ], dtype=np.float32),
}


@dataclass
class NarrowbandParams:
    """Parameters for narrowband combination."""

    palette: NarrowbandPalette = NarrowbandPalette.SHO
    custom_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float32)
    )
    normalize: bool = True  # normalize output to [0, 1]


def combine_narrowband(
    channels: dict[str, np.ndarray],
    params: NarrowbandParams | None = None,
) -> np.ndarray:
    """Combine narrowband channels into an RGB image.

    Parameters
    ----------
    channels : dict
        Mapping of filter names to 2D arrays. Expected keys: "ha", "oiii", "sii".
        At least "ha" and one other must be present.
    params : NarrowbandParams, optional
        Combination parameters.

    Returns
    -------
    ndarray
        RGB image of shape (3, H, W), values in [0, 1].
    """
    if params is None:
        params = NarrowbandParams()

    # Get channels, defaulting missing ones to zeros
    ha = channels.get("ha")
    oiii = channels.get("oiii")
    sii = channels.get("sii")

    if ha is None:
        raise ValueError("Ha channel is required for narrowband combination")

    h, w = ha.shape
    if oiii is None:
        oiii = np.zeros((h, w), dtype=np.float32)
    if sii is None:
        sii = np.zeros((h, w), dtype=np.float32)

    # Stack inputs: (3, H, W) — [Ha, OIII, SII]
    stack = np.stack([ha, oiii, sii], axis=0)  # (3, H, W)

    # Get palette matrix
    if params.palette == NarrowbandPalette.CUSTOM:
        matrix = params.custom_matrix
    else:
        matrix = PALETTE_MATRICES[params.palette]

    # Apply: result[c,H,W] = einsum("ci,ihw->chw", matrix, stack) on GPU
    dm = get_device_manager()
    with torch.no_grad():
        matrix_t = torch.from_numpy(matrix).to(dm.device)  # (3, 3)
        stack_t = torch.from_numpy(stack).to(dm.device)    # (3, H, W)
        result_t = torch.einsum("ci,ihw->chw", matrix_t, stack_t)
        if params.normalize:
            max_val = result_t.max()
            if max_val > 1e-10:
                result_t = result_t / max_val
        result = result_t.clamp(0, 1).cpu().numpy().astype(np.float32)

    return result


def continuum_subtraction(
    narrowband: np.ndarray,
    broadband: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Subtract scaled broadband from narrowband to isolate emission.

    Parameters
    ----------
    narrowband : ndarray
        Narrowband channel, shape (H, W).
    broadband : ndarray
        Broadband channel (e.g., R for Ha), same shape.
    scale : float
        Scale factor for broadband before subtraction.

    Returns
    -------
    ndarray
        Emission-only image, clipped to [0, 1].
    """
    result = narrowband - scale * broadband
    return np.clip(result, 0, 1).astype(np.float32)
