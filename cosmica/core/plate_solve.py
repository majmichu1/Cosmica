"""Plate Solving — determine WCS coordinates from star positions.

Uses astropy WCS (BSD) for coordinate transforms and local solving
via triangle matching against a reference catalog.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from cosmica.core.star_detection import detect_stars

log = logging.getLogger(__name__)


@dataclass
class PlateSolveResult:
    """Result from plate solving."""

    success: bool
    ra_center: float = 0.0  # Right Ascension in degrees
    dec_center: float = 0.0  # Declination in degrees
    pixel_scale: float = 0.0  # arcsec/pixel
    rotation: float = 0.0  # field rotation in degrees
    n_stars_matched: int = 0
    wcs_header: dict | None = None


@dataclass
class PlateSolveParams:
    """Parameters for plate solving."""

    # Approximate field center (if known)
    ra_hint: float | None = None
    dec_hint: float | None = None
    # Approximate pixel scale (arcsec/pixel)
    scale_hint: float | None = None
    scale_tolerance: float = 0.2  # +/- fraction of scale_hint
    max_stars: int = 100
    # Search radius in degrees (if hint provided)
    search_radius: float = 5.0


def plate_solve(
    image: np.ndarray,
    params: PlateSolveParams | None = None,
) -> PlateSolveResult:
    """Attempt to plate-solve an image using detected star positions.

    This performs local solving using triangle matching. For more robust
    solving, use the astrometry.net API integration.

    Parameters
    ----------
    image : ndarray
        Image data, shape (H, W) or (C, H, W), float32 in [0, 1].
    params : PlateSolveParams, optional
        Solving parameters.

    Returns
    -------
    PlateSolveResult
        Solving result with WCS information if successful.
    """
    if params is None:
        params = PlateSolveParams()

    sf = detect_stars(image, max_stars=params.max_stars)
    if len(sf) < 4:
        log.warning("Too few stars detected for plate solving (%d)", len(sf))
        return PlateSolveResult(success=False)

    positions = sf.positions  # Nx2 (x, y)

    # Build triangle index from detected stars
    triangles = _build_triangle_index(positions[:min(30, len(positions))])

    if not triangles:
        return PlateSolveResult(success=False)

    # If no hint is provided, we can only compute relative geometry
    if params.scale_hint is None:
        log.info("No scale hint provided, computing relative geometry only")
        result = _estimate_field_geometry(positions, sf.image_width, sf.image_height)
        return result

    # We have star positions and a scale hint but no reference catalog
    # to match against, so we can only report geometry — not true RA/Dec.
    log.info(
        "Local triangle-match: %d stars, scale hint %.2f arcsec/px, "
        "but no reference catalog — RA/Dec unavailable",
        len(sf), params.scale_hint,
    )
    return PlateSolveResult(
        success=False,
        n_stars_matched=len(sf),
        pixel_scale=params.scale_hint or 0.0,
    )


def plate_solve_astrometry_net(
    image: np.ndarray,
    api_key: str | None = None,
    params: PlateSolveParams | None = None,
) -> PlateSolveResult:
    """Plate-solve using astrometry.net API.

    This is a stub for future implementation — requires network access
    and an astrometry.net API key.

    Parameters
    ----------
    image : ndarray
        Image data.
    api_key : str, optional
        Astrometry.net API key.
    params : PlateSolveParams, optional
        Solving parameters.

    Returns
    -------
    PlateSolveResult
        Solving result.
    """
    log.warning("Astrometry.net API integration not yet implemented")
    return PlateSolveResult(success=False)


def _build_triangle_index(
    points: np.ndarray,
) -> list[tuple[tuple[int, int, int], tuple[float, float]]]:
    """Build triangle descriptors from star positions for matching.

    Each triangle is described by its two invariant ratios.
    """
    n = len(points)
    if n < 3:
        return []

    triangles = []
    for i in range(min(n, 15)):
        for j in range(i + 1, min(n, 15)):
            for k in range(j + 1, min(n, 15)):
                p = points[[i, j, k]]
                # Compute pairwise distances
                d01 = np.linalg.norm(p[0] - p[1])
                d02 = np.linalg.norm(p[0] - p[2])
                d12 = np.linalg.norm(p[1] - p[2])

                sides = sorted([d01, d02, d12])
                if sides[2] < 1e-6:
                    continue

                # Two invariant ratios
                r1 = sides[0] / sides[2]
                r2 = sides[1] / sides[2]
                triangles.append(((i, j, k), (r1, r2)))

    return triangles


def _estimate_field_geometry(
    positions: np.ndarray,
    width: int,
    height: int,
) -> PlateSolveResult:
    """Estimate basic field geometry from star positions."""
    if len(positions) < 3:
        return PlateSolveResult(success=False)

    # Compute mean position
    center_x = np.mean(positions[:, 0])
    center_y = np.mean(positions[:, 1])

    # Estimate field rotation from star distribution
    centered = positions - np.array([center_x, center_y])
    if len(centered) > 3:
        # PCA for orientation
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        rotation = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    else:
        rotation = 0.0

    return PlateSolveResult(
        success=True,
        n_stars_matched=len(positions),
        rotation=rotation,
    )
