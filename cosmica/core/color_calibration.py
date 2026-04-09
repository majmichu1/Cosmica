"""Photometric Color Calibration — calibrate color balance using star photometry.

Uses detected star fluxes and optional catalog data to compute a
color correction matrix. Falls back to statistical white balance
when no catalog is available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from cosmica.core.masks import Mask, apply_mask
from cosmica.core.star_detection import detect_stars

log = logging.getLogger(__name__)


@dataclass
class ColorCalibrationParams:
    """Parameters for photometric color calibration."""

    white_reference: str = "average"  # "average", "G2V" (solar type), "custom"
    custom_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    neutralize_background: bool = True
    background_percentile: float = 10.0  # use darkest N% for background


@dataclass
class ColorCalibrationResult:
    """Result of color calibration."""

    data: np.ndarray
    correction_factors: tuple[float, float, float]  # R, G, B multipliers
    background_offset: tuple[float, float, float] | None = None


def color_calibrate(
    image: np.ndarray,
    params: ColorCalibrationParams | None = None,
    mask: Mask | None = None,
) -> ColorCalibrationResult:
    """Calibrate color balance using star photometry.

    For images without WCS/catalog data, uses statistical methods:
    1. Background neutralization (make background neutral gray)
    2. White balance using average star color or specified reference

    Parameters
    ----------
    image : ndarray
        Color image, shape (C, H, W) with C >= 3, values in [0, 1].
    params : ColorCalibrationParams, optional
        Calibration parameters.
    mask : Mask, optional
        Selective processing mask.

    Returns
    -------
    ColorCalibrationResult
        Calibrated image and correction factors.
    """
    if params is None:
        params = ColorCalibrationParams()

    if image.ndim != 3 or image.shape[0] < 3:
        log.warning("Color calibration requires a color image with >= 3 channels")
        return ColorCalibrationResult(data=image, correction_factors=(1.0, 1.0, 1.0))

    original = image.copy()
    result = image.copy()

    bg_offset = None

    # Step 1: Background neutralization
    if params.neutralize_background:
        bg_offset = _neutralize_background(result, params.background_percentile)

    # Step 2: White balance using star photometry
    correction = _compute_white_balance(result, params)
    for ch in range(3):
        result[ch] = np.clip(result[ch] * correction[ch], 0, 1)

    result = apply_mask(original, result, mask)

    return ColorCalibrationResult(
        data=result,
        correction_factors=correction,
        background_offset=bg_offset,
    )


def _neutralize_background(
    image: np.ndarray,
    percentile: float,
) -> tuple[float, float, float]:
    """Neutralize the background by subtracting per-channel offsets.

    Uses the darkest pixels as the background reference.
    Modifies image in-place.

    Returns the offsets that were subtracted.
    """
    offsets = []
    for ch in range(3):
        # Use percentile of non-zero pixels as background
        channel = image[ch]
        valid = channel[channel > 0]
        if len(valid) == 0:
            offsets.append(0.0)
            continue
        bg_val = float(np.percentile(valid, percentile))
        offsets.append(bg_val)

    # Find the minimum offset (darkest channel background)
    min_offset = min(offsets)

    # Subtract so all channels have the same background level
    for ch in range(3):
        excess = offsets[ch] - min_offset
        image[ch] = np.clip(image[ch] - excess, 0, 1)

    return (offsets[0], offsets[1], offsets[2])


def _compute_white_balance(
    image: np.ndarray,
    params: ColorCalibrationParams,
) -> tuple[float, float, float]:
    """Compute per-channel correction factors for white balance.

    Uses star photometry: detects stars, measures their average color,
    and computes correction to make them match the reference.
    """
    if params.white_reference == "custom":
        return params.custom_rgb

    # Detect stars and measure their average RGB
    sf = detect_stars(image, max_stars=100, sigma_threshold=5.0)

    if len(sf) < 5:
        log.info("Too few stars for photometric calibration, using global statistics")
        return _global_white_balance(image)

    # Measure star fluxes in each channel
    star_colors = []
    for star in sf.stars:
        ix, iy = int(round(star.x)), int(round(star.y))
        # Aperture photometry: sum within small radius
        r = max(int(star.fwhm), 2)
        y_lo = max(0, iy - r)
        y_hi = min(image.shape[1], iy + r + 1)
        x_lo = max(0, ix - r)
        x_hi = min(image.shape[2], ix + r + 1)

        if y_hi - y_lo < 2 or x_hi - x_lo < 2:
            continue

        rgb = [float(image[ch, y_lo:y_hi, x_lo:x_hi].sum()) for ch in range(3)]
        if all(v > 0 for v in rgb):
            star_colors.append(rgb)

    if len(star_colors) < 3:
        return _global_white_balance(image)

    colors = np.array(star_colors)
    avg_rgb = np.median(colors, axis=0)

    if params.white_reference == "G2V":
        # G2V stars (solar type) have roughly equal RGB
        target = np.array([1.0, 1.0, 1.0])
    else:  # "average"
        # Make the average star color neutral
        target = np.array([1.0, 1.0, 1.0])

    # Correction factors to make avg_rgb match target
    correction = target / np.maximum(avg_rgb, 1e-10)
    # Normalize so the brightest channel has factor 1.0
    correction = correction / correction.max()

    log.info("PCC correction factors: R=%.3f G=%.3f B=%.3f", *correction)
    return (float(correction[0]), float(correction[1]), float(correction[2]))


def _global_white_balance(image: np.ndarray) -> tuple[float, float, float]:
    """Fallback: use global median for white balance."""
    medians = [float(np.median(image[ch])) for ch in range(3)]
    max_med = max(medians)
    if max_med < 1e-10:
        return (1.0, 1.0, 1.0)
    correction = [max_med / max(m, 1e-10) for m in medians]
    # Normalize
    max_corr = max(correction)
    correction = [c / max_corr for c in correction]
    return (correction[0], correction[1], correction[2])


def photometric_color_calibrate(
    image: np.ndarray,
    catalog_stars: list | None = None,
    wcs: dict | None = None,
    neutralize_bg: bool = True,
    background_percentile: float = 10.0,
) -> ColorCalibrationResult:
    """Photometric color calibration using star catalog data.

    If catalog stars are provided with WCS, performs true PCC using
    Gaia DR3 color information. Otherwise falls back to statistical
    white balance.

    Parameters
    ----------
    image : ndarray
        Color image, shape (C, H, W) with C >= 3, values in [0, 1].
    catalog_stars : list, optional
        List of StarCatalogEntry from Gaia query.
    wcs : dict, optional
        WCS info with 'ra', 'dec' keys for pixel-to-sky conversion.
    neutralize_bg : bool
        Whether to neutralize background.
    background_percentile : float
        Percentile for background estimation.

    Returns
    -------
    ColorCalibrationResult
        Calibrated image and correction factors.
    """
    if catalog_stars is None or wcs is None:
        params = ColorCalibrationParams(
            neutralize_background=neutralize_bg,
            background_percentile=background_percentile,
        )
        return color_calibrate(image, params)

    if image.ndim != 3 or image.shape[0] < 3:
        log.warning("PCC requires a color image with >= 3 channels")
        return ColorCalibrationResult(data=image, correction_factors=(1.0, 1.0, 1.0))

    result = image.copy()
    bg_offset = None

    if neutralize_bg:
        bg_offset = _neutralize_background(result, background_percentile)

    det_stars = detect_stars(result, max_stars=100, sigma_threshold=5.0)
    if len(det_stars.stars) < 5:
        log.warning("Too few detected stars for PCC, using statistical calibration")
        return color_calibrate(result, ColorCalibrationParams(neutralize_background=False))

    matched_stars = _match_stars_to_catalog(
        det_stars, catalog_stars, wcs, result.shape[1], result.shape[2]
    )

    if len(matched_stars) < 5:
        log.warning("Not enough matched stars for PCC, using statistical calibration")
        return color_calibrate(result, ColorCalibrationParams(neutralize_background=False))

    correction = _compute_ccm_from_matches(matched_stars)

    for ch in range(3):
        result[ch] = np.clip(result[ch] * correction[ch], 0, 1)

    log.info("PCC with catalog: correction factors R=%.3f G=%.3f B=%.3f", *correction)
    return ColorCalibrationResult(
        data=result,
        correction_factors=correction,
        background_offset=bg_offset,
    )


def _match_stars_to_catalog(detected, catalog_stars, wcs, height, width):
    """Match detected stars to catalog stars using WCS."""
    if "ra" not in wcs or "dec" not in wcs:
        return []

    ra0, dec0 = wcs["ra"], wcs["dec"]
    scale = wcs.get("scale", 1.0)
    if scale is None or scale <= 0:
        scale = 1.0

    matches = []

    for star in detected.stars:
        px, py = star.x, star.y

        center_ra = ra0 + (px - width / 2) * scale / 3600
        center_dec = dec0 + (py - height / 2) * scale / 3600

        best_match = None
        best_sep = float("inf")
        for cat_star in catalog_stars:
            sep = ((cat_star.ra_deg - center_ra) ** 2 + (cat_star.dec_deg - center_dec) ** 2) ** 0.5
            if sep < best_sep and sep < 0.01:
                best_sep = sep
                best_match = cat_star

        if best_match is not None:
            r = max(int(star.fwhm), 2)
            y_lo = max(0, int(py) - r)
            y_hi = min(height, int(py) + r + 1)
            x_lo = max(0, int(px) - r)
            x_hi = min(width, int(px) + r + 1)

            if y_hi - y_lo < 2 or x_hi - x_lo < 2:
                continue

            fluxes = [float(image[ch, y_lo:y_hi, x_lo:x_hi].sum()) for ch in range(3)]

            if all(f > 0 for f in fluxes):
                matches.append(
                    {
                        "instrumental": fluxes,
                        "catalog": best_match,
                    }
                )

    return matches


def _compute_ccm_from_matches(matches):
    """Compute 3x3 color correction matrix from matched stars."""
    if len(matches) < 3:
        return (1.0, 1.0, 1.0)

    A = []
    b = []

    for m in matches:
        inst = np.array(m["instrumental"])
        cat = m["catalog"]

        if cat.bp_mag is None or cat.rp_mag is None or cat.g_mag is None:
            continue

        bp_rp = cat.bp_mag - cat.rp_mag

        g_inst = inst[0] + inst[1] + inst[2]
        if g_inst <= 0:
            continue

        r_inst = inst[0] / g_inst
        g_inst_norm = inst[1] / g_inst
        b_inst = inst[2] / g_inst

        A.append([r_inst - 1, g_inst_norm - 1, b_inst - 1])
        b.append(-bp_rp)

    if len(A) < 3:
        return (1.0, 1.0, 1.0)

    A = np.array(A)
    b = np.array(b)

    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return (1.0, 1.0, 1.0)

    ccm = np.array([1.0 + coeffs[0], 1.0 + coeffs[1], 1.0 + coeffs[2]])
    ccm = ccm / np.max(ccm)

    return (float(ccm[0]), float(ccm[1]), float(ccm[2]))
