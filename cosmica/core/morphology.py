"""Morphological Operations — erode, dilate, open, close.

Uses OpenCV morphology (Apache 2.0). Applied to images or masks.
Supports circular, square, and diamond structuring elements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np

from cosmica.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)


class StructuringElement(Enum):
    CIRCLE = auto()
    SQUARE = auto()
    DIAMOND = auto()  # cross/diamond shape


class MorphOp(Enum):
    ERODE = auto()
    DILATE = auto()
    OPEN = auto()  # erode then dilate
    CLOSE = auto()  # dilate then erode


@dataclass
class MorphologyParams:
    """Parameters for morphological operations."""

    operation: MorphOp = MorphOp.DILATE
    element: StructuringElement = StructuringElement.CIRCLE
    kernel_size: int = 3  # must be odd
    iterations: int = 1


def _get_kernel(element: StructuringElement, size: int) -> np.ndarray:
    """Create a structuring element kernel."""
    size = max(3, size | 1)  # ensure odd and >= 3
    shape_map = {
        StructuringElement.CIRCLE: cv2.MORPH_ELLIPSE,
        StructuringElement.SQUARE: cv2.MORPH_RECT,
        StructuringElement.DIAMOND: cv2.MORPH_CROSS,
    }
    return cv2.getStructuringElement(shape_map[element], (size, size))


def morphology_transform(
    data: np.ndarray,
    params: MorphologyParams | None = None,
    mask: Mask | None = None,
) -> np.ndarray:
    """Apply morphological operation to an image.

    Parameters
    ----------
    data : ndarray
        Image data, shape (H, W) or (C, H, W), float32 in [0, 1].
    params : MorphologyParams, optional
        Operation parameters.
    mask : Mask, optional
        Processing mask.

    Returns
    -------
    ndarray
        Transformed image.
    """
    if params is None:
        params = MorphologyParams()

    original = data.copy()
    kernel = _get_kernel(params.element, params.kernel_size)

    op_map = {
        MorphOp.ERODE: cv2.MORPH_ERODE,
        MorphOp.DILATE: cv2.MORPH_DILATE,
        MorphOp.OPEN: cv2.MORPH_OPEN,
        MorphOp.CLOSE: cv2.MORPH_CLOSE,
    }
    cv_op = op_map[params.operation]

    def _process_channel(ch: np.ndarray) -> np.ndarray:
        return cv2.morphologyEx(ch, cv_op, kernel, iterations=params.iterations)

    if data.ndim == 2:
        result = _process_channel(data)
    else:
        result = np.empty_like(data)
        for ch in range(data.shape[0]):
            result[ch] = _process_channel(data[ch])

    result = np.clip(result, 0, 1).astype(np.float32)
    return apply_mask(original, result, mask)


def morphology_mask(
    mask: Mask,
    params: MorphologyParams | None = None,
) -> Mask:
    """Apply morphological operation to a mask.

    Parameters
    ----------
    mask : Mask
        Input mask.
    params : MorphologyParams, optional
        Operation parameters.

    Returns
    -------
    Mask
        Transformed mask.
    """
    if params is None:
        params = MorphologyParams()

    kernel = _get_kernel(params.element, params.kernel_size)

    op_map = {
        MorphOp.ERODE: cv2.MORPH_ERODE,
        MorphOp.DILATE: cv2.MORPH_DILATE,
        MorphOp.OPEN: cv2.MORPH_OPEN,
        MorphOp.CLOSE: cv2.MORPH_CLOSE,
    }
    cv_op = op_map[params.operation]
    result = cv2.morphologyEx(mask.data, cv_op, kernel, iterations=params.iterations)

    return Mask(
        data=np.clip(result, 0, 1).astype(np.float32),
        name=f"{mask.name} ({params.operation.name})",
        mask_type=mask.mask_type,
    )
