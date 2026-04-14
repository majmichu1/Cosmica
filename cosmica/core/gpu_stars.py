"""GPU-Accelerated Star Detection and Alignment using PyTorch.

Star class is imported from star_detection to avoid duplication.
GPU functions operate on tensors and call back to OpenCV RANSAC
for robust transform estimation.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

from cosmica.core.device_manager import get_device_manager
from cosmica.core.star_detection import Star  # single definition, re-exported here

__all__ = [
    "Star",
    "detect_stars_gpu",
    "match_stars_gpu",
    "estimate_transform_gpu",
    "warp_image_gpu",
    "compose_affine_transforms",
]

log = logging.getLogger(__name__)


def detect_stars_gpu(
    data: torch.Tensor,
    threshold_sigma: float = 5.0,
    max_stars: int = 500,
    min_distance: int = 5,
) -> list[Star]:
    """Detect stars using GPU max-pooling for local maxima.

    Parameters
    ----------
    data : torch.Tensor
        Input image (H, W) or (C, H, W), float32 in [0, 1].
    threshold_sigma : float
        Detection threshold in sigmas above background.
    max_stars : int
        Maximum number of stars to return (brightest first).
    min_distance : int
        Minimum separation between stars in pixels.
    """
    image = data.mean(dim=0) if data.dim() == 3 else data

    # Noise estimation via median + MAD (same as CPU detect_stars)
    med = torch.median(image)
    mad = torch.median(torch.abs(image - med))
    noise_est = max(mad.item() * 1.4826, 0.01)  # floor prevents near-zero threshold
    thresh = med.item() + threshold_sigma * noise_est

    masked_image = torch.where(image > thresh, image, torch.zeros_like(image))

    # Local maxima via max pooling
    kernel = min_distance
    local_max = functional.max_pool2d(
        masked_image.unsqueeze(0).unsqueeze(0),
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
    ).squeeze()

    is_max = (masked_image == local_max) & (masked_image > thresh)
    coords = torch.nonzero(is_max, as_tuple=False)

    if len(coords) == 0:
        return []

    rows = coords[:, 0]
    cols = coords[:, 1]
    fluxes = image[rows, cols]

    # Sort by flux descending and truncate BEFORE the patch loop —
    # avoids computing FWHM for stars we'd discard anyway.
    order = torch.argsort(fluxes, descending=True)[:max_stars]
    rows = rows[order]
    cols = cols[order]
    fluxes = fluxes[order]

    H, W = image.shape
    patch_r = 5
    img_cpu = image.cpu()
    stars: list[Star] = []
    for i in range(rows.shape[0]):
        r_i = int(rows[i].item())
        c_i = int(cols[i].item())
        flux = float(fluxes[i].item())
        r0, r1 = max(0, r_i - patch_r), min(H, r_i + patch_r + 1)
        c0, c1 = max(0, c_i - patch_r), min(W, c_i + patch_r + 1)
        patch = img_cpu[r0:r1, c0:c1]
        area = float((patch > flux * 0.5).sum().item())
        fwhm = 2.0 * (area / 3.14159) ** 0.5 if area > 0 else 3.0
        stars.append(Star(x=float(c_i), y=float(r_i), flux=flux, fwhm=fwhm))

    return stars


def match_stars_gpu(
    ref_stars: list[Star],
    target_stars: list[Star],
    max_dist: float,
) -> list[tuple[Star, Star]]:
    """Greedy nearest-neighbour star matching using GPU distance matrix.

    Returns list of (ref_star, target_star) pairs.
    """
    if not ref_stars or not target_stars:
        return []

    dm = get_device_manager()
    ref_pos = torch.tensor([[s.x, s.y] for s in ref_stars], device=dm.device)
    tgt_pos = torch.tensor([[s.x, s.y] for s in target_stars], device=dm.device)

    # Pairwise squared distances (N_ref, N_tgt)
    diff = ref_pos.unsqueeze(1) - tgt_pos.unsqueeze(0)
    dist_sq = torch.sum(diff**2, dim=2)

    dist_matrix = dist_sq.cpu().numpy()
    max_dist_sq = max_dist**2

    matches: list[tuple[Star, Star]] = []
    used_targets: set[int] = set()

    # Match brightest ref stars first
    ref_indices = np.argsort([-s.flux for s in ref_stars])
    for i in ref_indices:
        row = dist_matrix[i]
        best_j = int(np.argmin(row))
        if row[best_j] < max_dist_sq and best_j not in used_targets:
            matches.append((ref_stars[i], target_stars[best_j]))
            used_targets.add(best_j)

    return matches


def estimate_transform_gpu(
    matches: list[tuple[Star, Star]],
    ransac_threshold: float = 3.0,
) -> np.ndarray | None:
    """Estimate 2D similarity transform using OpenCV RANSAC for robustness.

    Uses GPU-computed matches (from match_stars_gpu) but runs RANSAC on CPU
    via OpenCV's estimateAffinePartial2D (translation + rotation + uniform scale).

    Parameters
    ----------
    matches : list of (ref_star, target_star) pairs
    ransac_threshold : float
        Max reprojection error in pixels for RANSAC inliers.

    Returns
    -------
    2×3 affine transform matrix (float32) or None if estimation fails.
    """
    if len(matches) < 3:
        return None

    src_pts = np.array([[m[1].x, m[1].y] for m in matches], dtype=np.float32)
    dst_pts = np.array([[m[0].x, m[0].y] for m in matches], dtype=np.float32)

    # OpenCV RANSAC is robust against mismatches (unlike pure least-squares)
    transform, inliers = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
    )

    if transform is None:
        log.warning("RANSAC transform estimation failed")
        return None

    n_inliers = int(inliers.sum()) if inliers is not None else 0
    log.debug("Transform estimated: %d/%d inliers", n_inliers, len(matches))
    return transform.astype(np.float32)


def warp_image_gpu(
    image: torch.Tensor,
    matrix: np.ndarray,
    mode: str = "bicubic",
) -> torch.Tensor:
    """Apply 2×3 affine transform to image on GPU.

    Parameters
    ----------
    image : torch.Tensor
        (H, W) or (C, H, W), float32.
    matrix : ndarray
        2×3 affine transform in pixel coordinates.
    mode : str
        Interpolation mode: "bilinear" (fast) or "bicubic" (quality).
    """
    original_2d = False
    if image.dim() == 2:
        image = image.unsqueeze(0)
        original_2d = True
    elif image.dim() != 3:
        raise ValueError(f"Unexpected image dims: {image.dim()}")

    c, h, w = image.shape

    # Convert pixel-space translations to normalised [-1, 1] coordinates
    # required by affine_grid with align_corners=True
    norm_matrix = matrix.copy()
    if h > 1:
        norm_matrix[1, 2] = matrix[1, 2] * 2.0 / (h - 1)
    if w > 1:
        norm_matrix[0, 2] = matrix[0, 2] * 2.0 / (w - 1)

    theta = torch.from_numpy(norm_matrix).to(image.device).unsqueeze(0)  # [1, 2, 3]
    grid = functional.affine_grid(theta, (1, c, h, w), align_corners=True)
    warped = functional.grid_sample(
        image.unsqueeze(0), grid, mode=mode, align_corners=True, padding_mode="zeros"
    )
    warped = warped.squeeze(0)

    if original_2d:
        warped = warped.squeeze(0)

    return warped


def compose_affine_transforms(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Compose two 2×3 affine transforms: apply m1 first, then m2.

    Works for general 2×3 affine matrices (rotation, scale, shear, translation).
    Returns m_combined such that m_combined(p) = m2(m1(p)).
    """
    # Promote to 3×3 homogeneous
    mat1 = np.vstack([m1, [0, 0, 1]])
    mat2 = np.vstack([m2, [0, 0, 1]])
    mat = mat2 @ mat1
    return mat[:2, :].astype(np.float32)
