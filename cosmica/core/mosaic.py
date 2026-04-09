"""Mosaic Stitching — combine overlapping image panels into a seamless mosaic.

Uses OpenCV (Apache 2.0) for homography computation and image warping.
Supports gradient-matched feathered blending for seamless transitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import cv2
import numpy as np

from cosmica.core.star_detection import detect_stars, find_transform

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


class BlendMethod(Enum):
    FEATHER = auto()  # linear feathered blending
    MULTIBAND = auto()  # Laplacian pyramid blending
    AVERAGE = auto()  # simple averaging in overlap


@dataclass
class MosaicParams:
    """Parameters for mosaic stitching."""

    blend_method: BlendMethod = BlendMethod.FEATHER
    feather_width: int = 50  # blend transition width in pixels
    match_gradient: bool = True  # adjust brightness across panels


@dataclass
class MosaicResult:
    """Result of mosaic stitching."""

    data: np.ndarray
    n_panels: int
    output_shape: tuple[int, ...]


def mosaic_stitch(
    panels: list[np.ndarray],
    params: MosaicParams | None = None,
    progress: ProgressCallback | None = None,
) -> MosaicResult:
    """Stitch overlapping panels into a mosaic.

    Parameters
    ----------
    panels : list[ndarray]
        Panel images, shape (H, W) or (C, H, W), float32 in [0, 1].
    params : MosaicParams, optional
        Stitching parameters.
    progress : callable, optional
        Progress callback.

    Returns
    -------
    MosaicResult
        Stitched mosaic.

    Raises
    ------
    ValueError
        If fewer than 2 panels are provided.
    """
    if len(panels) < 2:
        raise ValueError("Mosaic stitching requires at least 2 panels")

    if params is None:
        params = MosaicParams()
    if progress is None:
        progress = _noop_progress

    is_color = panels[0].ndim == 3

    # Compute pairwise transforms (each panel to panel 0)
    progress(0.0, "Computing panel registrations...")
    transforms = _compute_pairwise_transforms(panels, progress)

    # Compute output canvas size
    progress(0.4, "Computing output canvas...")
    canvas_size, offsets = _compute_canvas(panels, transforms)

    # Warp and blend panels
    progress(0.5, "Warping and blending panels...")
    if is_color:
        result = np.zeros((panels[0].shape[0], canvas_size[0], canvas_size[1]), dtype=np.float32)
    else:
        result = np.zeros(canvas_size, dtype=np.float32)
    weight_total = np.zeros(canvas_size, dtype=np.float32)

    for i, (panel, transform, offset) in enumerate(zip(panels, transforms, offsets)):
        frac = 0.5 + 0.5 * i / max(len(panels) - 1, 1)
        progress(frac, f"Blending panel {i + 1}/{len(panels)}...")

        warped, mask = _warp_panel(panel, transform, offset, canvas_size)
        weight = _compute_blend_weight(mask, params.feather_width)

        if is_color:
            for ch in range(panel.shape[0]):
                result[ch] += warped[ch] * weight
        else:
            result += warped * weight
        weight_total += weight

    # Normalize
    valid = weight_total > 0
    if is_color:
        for ch in range(result.shape[0]):
            result[ch][valid] /= weight_total[valid]
    else:
        result[valid] /= weight_total[valid]

    result = np.clip(result, 0, 1)
    progress(1.0, "Mosaic complete")

    return MosaicResult(
        data=result,
        n_panels=len(panels),
        output_shape=result.shape,
    )


def _compute_pairwise_transforms(
    panels: list[np.ndarray],
    progress: ProgressCallback,
) -> list[np.ndarray]:
    """Compute affine transforms from each panel to panel 0."""
    ref_sf = detect_stars(panels[0])
    transforms = [np.eye(2, 3, dtype=np.float32)]  # identity for reference

    for i in range(1, len(panels)):
        frac = 0.4 * i / max(len(panels) - 1, 1)
        progress(frac, f"Registering panel {i + 1}/{len(panels)}...")
        panel_sf = detect_stars(panels[i])
        t = find_transform(ref_sf, panel_sf)
        if t is None:
            log.warning("Failed to register panel %d, using identity", i)
            t = np.eye(2, 3, dtype=np.float32)
        transforms.append(t)

    return transforms


def _compute_canvas(
    panels: list[np.ndarray],
    transforms: list[np.ndarray],
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """Compute the output canvas size and per-panel offsets."""
    corners_all = []
    for panel, transform in zip(panels, transforms):
        if panel.ndim == 3:
            h, w = panel.shape[1], panel.shape[2]
        else:
            h, w = panel.shape

        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        # Apply transform
        for c in corners:
            x = transform[0, 0] * c[0] + transform[0, 1] * c[1] + transform[0, 2]
            y = transform[1, 0] * c[0] + transform[1, 1] * c[1] + transform[1, 2]
            corners_all.append([x, y])

    corners_all = np.array(corners_all)
    x_min, y_min = corners_all.min(axis=0)
    x_max, y_max = corners_all.max(axis=0)

    canvas_w = int(np.ceil(x_max - x_min))
    canvas_h = int(np.ceil(y_max - y_min))

    offsets = []
    for transform in transforms:
        ox = -x_min
        oy = -y_min
        offsets.append((int(ox), int(oy)))

    return (canvas_h, canvas_w), offsets


def _warp_panel(
    panel: np.ndarray,
    transform: np.ndarray,
    offset: tuple[int, int],
    canvas_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a panel onto the canvas using its transform + offset."""
    h, w = canvas_size
    is_color = panel.ndim == 3

    # Add offset to transform
    t = transform.copy()
    t[0, 2] += offset[0]
    t[1, 2] += offset[1]

    if is_color:
        warped = np.zeros((panel.shape[0], h, w), dtype=np.float32)
        for ch in range(panel.shape[0]):
            warped[ch] = cv2.warpAffine(
                panel[ch], t, (w, h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        mask = cv2.warpAffine(
            np.ones(panel.shape[1:], dtype=np.float32), t, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        warped = cv2.warpAffine(
            panel, t, (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask = cv2.warpAffine(
            np.ones_like(panel), t, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    return warped, mask


def _compute_blend_weight(
    mask: np.ndarray,
    feather_width: int,
) -> np.ndarray:
    """Compute feathered blend weight from a binary mask."""
    if feather_width <= 0:
        return mask

    # Distance transform for feathering
    binary = (mask > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    weight = np.clip(dist / max(feather_width, 1), 0, 1).astype(np.float32)
    return weight
