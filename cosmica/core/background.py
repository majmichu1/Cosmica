"""Background Extraction — automated gradient removal.

Independently implements polynomial surface fitting for background modeling.
Samples the background at grid points, fits a polynomial surface, and subtracts it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

log = logging.getLogger(__name__)


@dataclass
class BackgroundParams:
    grid_size: int = 8  # NxN sample grid
    box_size: int = 32  # sample box size in pixels
    polynomial_order: int = 3  # polynomial surface degree
    sigma_clip: float = 2.5  # reject bright samples (stars)
    smoothing: float = 0.5  # Gaussian smoothing of the model (fraction of image size)
    manual_points: list[tuple[int, int]] = field(default_factory=list)  # user-placed sample points
    object_aware: bool = False  # if True, use exclusion_mask to protect target signal
    exclusion_mask: np.ndarray | None = field(default=None, repr=False)  # (H,W) float32 mask, 1=exclude from sampling


def extract_background(
    data: np.ndarray,
    params: BackgroundParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and subtract background gradient.

    Input: float32 (H, W) for mono or (C, H, W) for color.
    Returns: (corrected_image, background_model)
    """
    if params is None:
        params = BackgroundParams()

    if data.ndim == 3:
        n_ch = data.shape[0]
        corrected = np.empty_like(data)
        bg_model = np.empty_like(data)
        for ch in range(n_ch):
            corrected[ch], bg_model[ch] = _extract_single_channel(data[ch], params)
        return corrected, bg_model
    else:
        return _extract_single_channel(data, params)


def _extract_single_channel(
    channel: np.ndarray,
    params: BackgroundParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract background from a single channel."""
    h, w = channel.shape

    # Generate sample points
    sample_points = _generate_sample_points(h, w, params)

    # Measure background level at each sample point
    samples_x = []
    samples_y = []
    samples_val = []

    for py, px in sample_points:
        # Object-aware: skip points inside the exclusion mask
        if params.object_aware and params.exclusion_mask is not None:
            if 0 <= py < h and 0 <= px < w:
                if params.exclusion_mask[py, px] > 0.5:
                    continue

        val = _measure_sample(channel, py, px, params.box_size)
        if val is not None:
            samples_x.append(px)
            samples_y.append(py)
            samples_val.append(val)

    if len(samples_val) < 6:
        log.warning("Too few background samples (%d), returning original", len(samples_val))
        return channel.copy(), np.zeros_like(channel)

    samples_x = np.array(samples_x, dtype=np.float64)
    samples_y = np.array(samples_y, dtype=np.float64)
    samples_val = np.array(samples_val, dtype=np.float64)

    # Sigma-clip to reject star-contaminated samples
    median_val = np.median(samples_val)
    mad = np.median(np.abs(samples_val - median_val))
    sigma = max(mad * 1.4826, 1e-10)  # MAD to sigma conversion

    keep = np.abs(samples_val - median_val) < params.sigma_clip * sigma
    samples_x = samples_x[keep]
    samples_y = samples_y[keep]
    samples_val = samples_val[keep]

    log.info("Background samples: %d kept, %d rejected", keep.sum(), (~keep).sum())

    if len(samples_val) < 6:
        log.warning("Too few samples after clipping, returning original")
        return channel.copy(), np.zeros_like(channel)

    # Normalize coordinates to [-1, 1]
    x_norm = (samples_x / w) * 2 - 1
    y_norm = (samples_y / h) * 2 - 1

    # Fit polynomial surface
    coeffs = _fit_polynomial_surface(x_norm, y_norm, samples_val, params.polynomial_order)

    # Evaluate model over full image
    yy, xx = np.mgrid[0:h, 0:w]
    xx_norm = (xx.astype(np.float64) / w) * 2 - 1
    yy_norm = (yy.astype(np.float64) / h) * 2 - 1
    bg_model = _evaluate_polynomial(xx_norm, yy_norm, coeffs, params.polynomial_order)

    # Optional smoothing
    if params.smoothing > 0:
        smooth_sigma = params.smoothing * min(h, w) / 10
        bg_model = ndimage.gaussian_filter(bg_model, sigma=smooth_sigma)

    bg_model = bg_model.astype(np.float32)

    # Subtract and re-normalize
    corrected = channel - bg_model
    # Shift so minimum is at or near 0
    c_min = np.percentile(corrected, 0.1)
    corrected -= c_min
    corrected = np.clip(corrected, 0, 1).astype(np.float32)

    return corrected, bg_model


def create_object_exclusion_mask(
    image_shape: tuple[int, int],
    objects: list[dict],
    plate_scale: float,
    margin_factor: float = 1.3,
) -> np.ndarray:
    """Create an exclusion mask from known object positions.

    Parameters
    ----------
    image_shape : (H, W)
        Image dimensions.
    objects : list of dict
        Each dict has 'center_x', 'center_y' (pixel coords) and
        'radius_arcmin' (angular radius of the object).
    plate_scale : float
        Arcseconds per pixel.
    margin_factor : float
        Expand object radius by this factor for safety margin.

    Returns
    -------
    ndarray
        Float32 mask, 1.0 inside objects, 0.0 outside.
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.ogrid[0:h, 0:w]

    for obj in objects:
        cx = obj["center_x"]
        cy = obj["center_y"]
        # Convert angular radius to pixels
        radius_px = (obj["radius_arcmin"] * 60.0 / plate_scale) * margin_factor
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        mask[dist_sq <= radius_px**2] = 1.0

    return mask


def _generate_sample_points(
    h: int, w: int, params: BackgroundParams
) -> list[tuple[int, int]]:
    """Generate grid sample points plus any manual points."""
    points = []
    margin = params.box_size
    grid = params.grid_size

    ys = np.linspace(margin, h - margin, grid).astype(int)
    xs = np.linspace(margin, w - margin, grid).astype(int)

    for y in ys:
        for x in xs:
            points.append((int(y), int(x)))

    points.extend(params.manual_points)
    return points


def _measure_sample(
    channel: np.ndarray, cy: int, cx: int, box_size: int
) -> float | None:
    """Measure the background level in a box centered at (cy, cx).

    Uses the median of the darkest 50% of pixels in the box to avoid stars.
    """
    h, w = channel.shape
    half = box_size // 2
    y0 = max(0, cy - half)
    y1 = min(h, cy + half)
    x0 = max(0, cx - half)
    x1 = min(w, cx + half)

    box = channel[y0:y1, x0:x1]
    if box.size == 0:
        return None

    # Use the darkest 50% to avoid stars
    sorted_vals = np.sort(box.ravel())
    n = len(sorted_vals)
    dark_half = sorted_vals[: n // 2]
    if len(dark_half) == 0:
        return None

    return float(np.median(dark_half))


def _fit_polynomial_surface(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, order: int
) -> np.ndarray:
    """Fit a 2D polynomial surface to scattered data points."""
    # Build Vandermonde-like matrix
    terms = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            terms.append(x**i * y**j)

    A = np.column_stack(terms)

    # Least squares fit
    result, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    return result


def _evaluate_polynomial(
    x: np.ndarray, y: np.ndarray, coeffs: np.ndarray, order: int
) -> np.ndarray:
    """Evaluate 2D polynomial surface on a grid."""
    result = np.zeros_like(x, dtype=np.float64)
    idx = 0
    for i in range(order + 1):
        for j in range(order + 1 - i):
            result += coeffs[idx] * x**i * y**j
            idx += 1
    return result
