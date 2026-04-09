"""Star Detection — shared star detection and registration utilities.

Extracted from stacking.py so it can be reused by star masks, PCC,
plate solving, and other features that need star positions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Star:
    """A detected star with position and measured properties."""

    x: float
    y: float
    flux: float  # peak pixel value
    fwhm: float = 0.0  # full width at half maximum (pixels)
    roundness: float = 0.0  # 0 = perfect circle, 1 = elongated


@dataclass
class StarField:
    """Collection of detected stars from a single image."""

    stars: list[Star]
    image_width: int
    image_height: int

    @property
    def positions(self) -> np.ndarray:
        """Return Nx2 array of (x, y) positions."""
        if not self.stars:
            return np.empty((0, 2), dtype=np.float32)
        return np.array([(s.x, s.y) for s in self.stars], dtype=np.float32)

    @property
    def fluxes(self) -> np.ndarray:
        """Return N-element array of star fluxes."""
        if not self.stars:
            return np.empty(0, dtype=np.float32)
        return np.array([s.flux for s in self.stars], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.stars)


def detect_stars(
    image: np.ndarray,
    max_stars: int = 200,
    sigma_threshold: float = 5.0,
    min_area: int = 3,
    max_area: int = 500,
) -> StarField:
    """Detect stars using thresholding and contour analysis.

    Parameters
    ----------
    image : ndarray
        Image data, shape (H, W) for mono or (C, H, W) for color.
        Values in [0, 1] float32.
    max_stars : int
        Maximum number of stars to return (brightest first).
    sigma_threshold : float
        Detection threshold in MAD units above median.
    min_area : int
        Minimum contour area in pixels.
    max_area : int
        Maximum contour area in pixels.

    Returns
    -------
    StarField
        Detected stars sorted by brightness (brightest first).
    """
    # Work on a single-channel version
    if image.ndim == 3:
        gray = np.mean(image, axis=0)
    else:
        gray = image

    h, w = gray.shape

    # Threshold based on median + sigma * MAD-estimated noise
    med = np.median(gray)
    mad = np.median(np.abs(gray - med))
    noise_est = max(mad * 1.4826, 0.01)  # floor prevents near-zero threshold on clean images
    threshold = med + sigma_threshold * noise_est
    threshold = min(threshold, 0.95)  # never so high we miss everything
    binary = (gray > threshold).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stars: list[Star] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Peak intensity
        ix, iy = int(round(cx)), int(round(cy))
        if not (0 <= iy < h and 0 <= ix < w):
            continue
        flux = float(gray[iy, ix])

        # Estimate FWHM from contour area (approximation: area ~ pi * (FWHM/2)^2)
        fwhm = 2.0 * np.sqrt(area / np.pi) if area > 0 else 0.0

        # Roundness from bounding rect
        _, _, bw, bh = cv2.boundingRect(cnt)
        if bw > 0 and bh > 0:
            roundness = 1.0 - min(bw, bh) / max(bw, bh)
        else:
            roundness = 0.0

        stars.append(Star(x=cx, y=cy, flux=flux, fwhm=fwhm, roundness=roundness))

    # Sort by brightness (brightest first) and take top N
    stars.sort(key=lambda s: -s.flux)
    stars = stars[:max_stars]

    log.debug("Detected %d stars in %dx%d image", len(stars), w, h)
    return StarField(stars=stars, image_width=w, image_height=h)


def find_transform(
    ref_stars: StarField | np.ndarray,
    target_stars: StarField | np.ndarray,
    max_match: int = 50,
    max_dist_fraction: float = 0.1,
) -> np.ndarray | None:
    """Find affine transform from target to reference coordinate system.

    Parameters
    ----------
    ref_stars : StarField or ndarray
        Reference star positions. If ndarray, shape (N, 2) with (x, y).
    target_stars : StarField or ndarray
        Target star positions. Same format as ref_stars.
    max_match : int
        Maximum number of stars to use for matching.
    max_dist_fraction : float
        Maximum matching distance as fraction of image diagonal.

    Returns
    -------
    ndarray or None
        2x3 affine transform matrix, or None if matching failed.
    """
    from scipy.spatial.distance import cdist

    ref_pts = ref_stars.positions if isinstance(ref_stars, StarField) else ref_stars
    tgt_pts = target_stars.positions if isinstance(target_stars, StarField) else target_stars

    if len(ref_pts) < 3 or len(tgt_pts) < 3:
        return None

    n_ref = min(len(ref_pts), max_match)
    n_tgt = min(len(tgt_pts), max_match)
    ref = ref_pts[:n_ref]
    tgt = tgt_pts[:n_tgt]

    # Distance matrix
    dists = cdist(ref, tgt)

    # Maximum match distance
    max_coord = max(ref.max(), tgt.max(), 1.0)
    max_dist = max(max_coord * max_dist_fraction, 50)

    # Greedy nearest-neighbor matching
    matched_ref = []
    matched_tgt = []
    used_tgt: set[int] = set()
    for i in range(len(ref)):
        best_j = -1
        best_d = float("inf")
        for j in range(len(tgt)):
            if j not in used_tgt and dists[i, j] < best_d:
                best_d = dists[i, j]
                best_j = j
        if best_j >= 0 and best_d < max_dist:
            matched_ref.append(ref[i])
            matched_tgt.append(tgt[best_j])
            used_tgt.add(best_j)

    if len(matched_ref) < 3:
        return None

    ref_matched = np.array(matched_ref, dtype=np.float32)
    tgt_matched = np.array(matched_tgt, dtype=np.float32)

    transform, inliers = cv2.estimateAffinePartial2D(
        tgt_matched, ref_matched, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    return transform


def align_image(
    image: np.ndarray,
    transform: np.ndarray,
    ref_shape: tuple[int, ...],
) -> np.ndarray:
    """Apply affine transform to align an image to the reference.

    Parameters
    ----------
    image : ndarray
        Image to warp. Shape (H, W) or (C, H, W).
    transform : ndarray
        2x3 affine transform matrix.
    ref_shape : tuple
        Shape of the reference image.

    Returns
    -------
    ndarray
        Aligned image with same dtype.
    """
    if image.ndim == 3:
        h, w = ref_shape[-2], ref_shape[-1]
        result = np.zeros((image.shape[0], h, w), dtype=np.float32)
        for ch in range(image.shape[0]):
            result[ch] = cv2.warpAffine(
                image[ch],
                transform,
                (w, h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return result
    else:
        h, w = ref_shape[-2], ref_shape[-1]
        return cv2.warpAffine(
            image,
            transform,
            (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
