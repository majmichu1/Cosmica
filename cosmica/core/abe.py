"""Automatic Background Extraction — RBF-based gradient removal.

Uses Radial Basis Function interpolation to model and remove large-scale
background gradients (light pollution, vignetting, sky glow) from
astrophotography images.  Unlike polynomial fitting, RBF interpolation
adapts locally to complex, non-polynomial gradient shapes.

Requires scipy >= 1.7 for ``scipy.interpolate.RBFInterpolator``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

from cosmica.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)


@dataclass
class ABEParams:
    """Parameters for Automatic Background Extraction.

    Attributes
    ----------
    grid_size : int
        Number of sample points along each axis (NxN grid).
    box_size : int
        Side length in pixels of the measurement box at each sample point.
    sigma_clip : float
        Sigma-clipping threshold for rejecting star-contaminated samples.
        Samples whose measured value exceeds ``median + sigma_clip * MAD``
        are discarded.
    rbf_kernel : str
        Kernel function passed to ``RBFInterpolator``.  Valid choices
        include ``"thin_plate_spline"``, ``"multiquadric"``,
        ``"inverse_multiquadric"``, ``"gaussian"``, ``"linear"``,
        ``"cubic"``, and ``"quintic"``.
    rbf_smoothing : float
        Smoothing parameter for the RBF interpolator.  Higher values
        produce a smoother background model at the cost of less fidelity
        to the sample points.
    correction_mode : str
        How the background model is removed from the image.
        ``"subtraction"`` subtracts the model; ``"division"`` divides.
    iterations : int
        Number of refinement iterations.  Each iteration re-samples on
        the corrected image, builds a new model, and applies it to the
        *original* accumulated model so that the total correction is the
        sum (or product) of all per-iteration models.
    """

    grid_size: int = 10
    box_size: int = 48
    sigma_clip: float = 2.0
    rbf_kernel: str = "thin_plate_spline"
    rbf_smoothing: float = 0.5
    correction_mode: str = "subtraction"
    iterations: int = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def abe_extract(
    data: np.ndarray,
    params: ABEParams | None = None,
    mask: Mask | None = None,
    progress=None,  # accepts but ignores — keeps ProcessingWorker happy
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and remove background using RBF interpolation.

    Parameters
    ----------
    data : ndarray
        Input image as float32 in [0, 1].  Shape ``(H, W)`` for mono or
        ``(C, H, W)`` for color.
    params : ABEParams, optional
        Extraction parameters.  Uses defaults when *None*.
    mask : Mask, optional
        Selective-processing mask.  Where the mask is 0 the original
        pixel values are preserved; where 1 the corrected values are
        used.

    Returns
    -------
    corrected : ndarray
        Background-corrected image, same shape and dtype as *data*,
        clipped to [0, 1].
    background_model : ndarray
        The estimated background model, same shape as *data*.
    """
    if params is None:
        params = ABEParams()

    log.info(
        "ABE: grid=%d, box=%d, kernel=%s, smoothing=%.2f, mode=%s, iters=%d",
        params.grid_size,
        params.box_size,
        params.rbf_kernel,
        params.rbf_smoothing,
        params.correction_mode,
        params.iterations,
    )

    if data.ndim == 3:
        n_channels = data.shape[0]
        corrected = np.empty_like(data)
        bg_model = np.empty_like(data)
        for ch in range(n_channels):
            log.info("ABE: processing channel %d/%d", ch + 1, n_channels)
            corrected[ch], bg_model[ch] = _extract_channel(data[ch], params)
    else:
        corrected, bg_model = _extract_channel(data, params)

    corrected = apply_mask(data, corrected, mask)
    return corrected, bg_model


# ---------------------------------------------------------------------------
# Single-channel pipeline
# ---------------------------------------------------------------------------


def _extract_channel(
    channel: np.ndarray,
    params: ABEParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the full ABE pipeline on a single (H, W) channel.

    When ``params.iterations > 1`` the algorithm iteratively refines its
    estimate: each pass re-samples on the *latest corrected* image, but
    the individual per-iteration models are accumulated so that the final
    model represents the total background removed.
    """
    h, w = channel.shape

    # Accumulate the total background model across iterations.
    if params.correction_mode == "subtraction":
        total_bg = np.zeros((h, w), dtype=np.float32)
    else:
        total_bg = np.ones((h, w), dtype=np.float32)

    working = channel.copy()

    for it in range(params.iterations):
        log.info("ABE iteration %d/%d", it + 1, params.iterations)

        # 1. Sample the background on the current working image.
        points, values = _sample_background(working, params)

        if len(values) < 6:
            log.warning(
                "ABE: only %d samples survived clipping at iteration %d — "
                "skipping further iterations",
                len(values),
                it + 1,
            )
            break

        # 2. Build the RBF model and evaluate it over the full image.
        iter_bg = _build_rbf_model(points, values, h, w, params)

        # 3. Apply correction to the working image.
        if params.correction_mode == "division":
            safe_bg = np.maximum(iter_bg, 1e-7)
            working = np.clip(working / safe_bg, 0.0, 1.0).astype(np.float32)
            total_bg *= iter_bg
        else:
            working = np.clip(working - iter_bg, 0.0, 1.0).astype(np.float32)
            total_bg += iter_bg

    corrected = working
    return corrected, total_bg


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _generate_grid_points(
    h: int,
    w: int,
    grid_size: int,
    box_size: int,
) -> np.ndarray:
    """Return an (N, 2) array of (row, col) sample-point coordinates.

    Points are placed on a regular ``grid_size x grid_size`` lattice
    with a margin of ``box_size // 2`` from the image edges so that
    measurement boxes are fully inside the image.
    """
    margin = box_size // 2
    ys = np.linspace(margin, h - margin - 1, grid_size).astype(int)
    xs = np.linspace(margin, w - margin - 1, grid_size).astype(int)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.column_stack([yy.ravel(), xx.ravel()])


def _measure_background_at(
    channel: np.ndarray,
    row: int,
    col: int,
    box_size: int,
) -> float | None:
    """Measure background level in a box centred at (*row*, *col*).

    Takes the ``box_size x box_size`` neighbourhood, sorts the pixel
    values, and returns the median of the darkest 50 %.  This rejects
    stars that may fall inside the box.

    Returns *None* when the box is empty (should not happen with proper
    margin handling, but guarded defensively).
    """
    h, w = channel.shape
    half = box_size // 2
    r0 = max(0, row - half)
    r1 = min(h, row + half)
    c0 = max(0, col - half)
    c1 = min(w, col + half)

    box = channel[r0:r1, c0:c1]
    if box.size == 0:
        return None

    sorted_vals = np.sort(box.ravel())
    dark_half = sorted_vals[: len(sorted_vals) // 2]
    if dark_half.size == 0:
        return None

    return float(np.median(dark_half))


def _sample_background(
    channel: np.ndarray,
    params: ABEParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate grid samples and sigma-clip to reject star contamination.

    Returns
    -------
    points : ndarray, shape (M, 2)
        Row/col coordinates of the surviving sample points.
    values : ndarray, shape (M,)
        Measured background value at each surviving point.
    """
    h, w = channel.shape
    grid_pts = _generate_grid_points(h, w, params.grid_size, params.box_size)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for row, col in grid_pts:
        val = _measure_background_at(channel, int(row), int(col), params.box_size)
        if val is not None:
            rows.append(int(row))
            cols.append(int(col))
            vals.append(val)

    if len(vals) == 0:
        log.warning("ABE: no valid background samples obtained")
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64)

    points = np.column_stack([rows, cols]).astype(np.float64)
    values = np.asarray(vals, dtype=np.float64)

    # Sigma-clip: reject points where value > median + sigma_clip * MAD.
    median_val = np.median(values)
    mad = np.median(np.abs(values - median_val))
    # Convert MAD to a standard-deviation-like scale (* 1.4826) is the
    # classic estimator, but the spec says "median + sigma_clip * MAD"
    # directly, so we honour that literally.
    threshold = median_val + params.sigma_clip * mad
    keep = values <= threshold

    n_rejected = int((~keep).sum())
    log.info(
        "ABE sampling: %d measured, %d kept, %d rejected (threshold=%.6f)",
        len(values),
        int(keep.sum()),
        n_rejected,
        threshold,
    )

    return points[keep], values[keep]


# ---------------------------------------------------------------------------
# RBF model construction
# ---------------------------------------------------------------------------

_DOWNSAMPLE_STEP = 8  # evaluate RBF every Nth pixel, then upscale


def _add_boundary_anchors(
    points: np.ndarray,
    values: np.ndarray,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend the sample set with anchor points at the image boundary.

    RBF kernels like thin-plate spline extrapolate wildly outside the
    convex hull of sample points.  The measurement grid has a ``box_size/2``
    inward margin, so the image edges and corners sit outside that hull —
    leading to runaway extrapolation and a bright-edge halo artefact.

    Anchors are placed at the 4 corners and along each edge.  Each anchor
    takes the value of the nearest real sample so the RBF is guided toward
    measured data at the boundary rather than inventing values.
    """
    if len(points) == 0:
        return points, values

    n_edge = 5  # evenly-spaced points per side (excluding corners)

    anchors: list[tuple[float, float]] = []

    # 4 corners
    for r in (0.0, float(h - 1)):
        for c in (0.0, float(w - 1)):
            anchors.append((r, c))

    # Points along each of the 4 edges (corners already added)
    for frac in np.linspace(0, 1, n_edge + 2)[1:-1]:
        anchors.append((0.0,           frac * (w - 1)))   # top edge
        anchors.append((float(h - 1),  frac * (w - 1)))   # bottom edge
        anchors.append((frac * (h - 1), 0.0))             # left edge
        anchors.append((frac * (h - 1), float(w - 1)))    # right edge

    anchors_arr = np.array(anchors, dtype=np.float64)

    # Each anchor gets the value of the nearest real sample.
    anchor_vals: list[float] = []
    for ry, cx in anchors_arr:
        dists = np.sqrt((points[:, 0] - ry) ** 2 + (points[:, 1] - cx) ** 2)
        anchor_vals.append(float(values[int(np.argmin(dists))]))

    all_points = np.vstack([points, anchors_arr])
    all_values = np.concatenate([values, np.array(anchor_vals, dtype=np.float64)])
    return all_points, all_values


def _build_rbf_model(
    points: np.ndarray,
    values: np.ndarray,
    h: int,
    w: int,
    params: ABEParams,
) -> np.ndarray:
    """Fit an RBF to the sample points and evaluate over the full image.

    For efficiency the RBF is evaluated on a coarse grid (every
    ``_DOWNSAMPLE_STEP`` pixels) and then upscaled to the full resolution
    with bicubic interpolation.

    Boundary anchor points are added before fitting so that the RBF hull
    covers the full image extent, eliminating edge-extrapolation artefacts.

    Parameters
    ----------
    points : ndarray, shape (M, 2)
        Sample coordinates (row, col).
    values : ndarray, shape (M,)
        Background values at the sample points.
    h, w : int
        Full image height and width.
    params : ABEParams
        Kernel and smoothing configuration.

    Returns
    -------
    ndarray, shape (h, w), float32
        Background model at full resolution.
    """
    step = _DOWNSAMPLE_STEP

    # Add boundary anchors to prevent wild RBF extrapolation at image edges.
    aug_points, aug_values = _add_boundary_anchors(points, values, h, w)

    # Fit the RBF interpolator on the augmented point set.
    rbf = RBFInterpolator(
        aug_points,
        aug_values,
        kernel=params.rbf_kernel,
        smoothing=params.rbf_smoothing,
    )

    # Coarse evaluation grid.
    coarse_rows = np.arange(0, h, step)
    coarse_cols = np.arange(0, w, step)
    cr, cc = np.meshgrid(coarse_rows, coarse_cols, indexing="ij")
    eval_pts = np.column_stack([cr.ravel(), cc.ravel()])

    coarse_vals = rbf(eval_pts).reshape(cr.shape)

    # Upscale to full resolution using bicubic interpolation.
    coarse_f32 = coarse_vals.astype(np.float32)
    bg_full = cv2.resize(coarse_f32, (w, h), interpolation=cv2.INTER_CUBIC)

    # Final safety clamp: model must stay within the measured value range.
    # Lower bound is clamped to 0 so we never subtract a negative background.
    val_min = float(values.min())
    val_max = float(values.max())
    margin = max(0.01, (val_max - val_min) * 0.1)
    bg_full = np.clip(bg_full, max(0.0, val_min - margin), val_max + margin)

    return bg_full.astype(np.float32)
