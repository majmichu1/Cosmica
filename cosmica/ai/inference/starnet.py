"""StarNet Integration — GPL-isolated star removal via subprocess.

StarNet is GPL licensed, so it MUST be run as a subprocess only.
No imports of StarNet code are allowed. Communication is via temp files.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cosmica.core.image_io import ImageData, load_image, save_fits

log = logging.getLogger(__name__)


@dataclass
class StarNetResult:
    """Result from StarNet processing."""

    starless: np.ndarray  # image with stars removed
    stars_only: np.ndarray | None = None  # extracted stars (original - starless)
    success: bool = True
    message: str = ""


def find_starnet_binary() -> Path | None:
    """Find the StarNet binary on the system.

    Searches common installation locations and PATH.

    Returns
    -------
    Path or None
        Path to StarNet binary, or None if not found.
    """
    # Check common names
    for name in ("starnet++", "starnet2", "StarNetv2CLI", "starnet"):
        path = shutil.which(name)
        if path:
            return Path(path)

    # Check common install locations
    common_paths = [
        Path.home() / "StarNet",
        Path.home() / ".local" / "bin",
        Path("/usr/local/bin"),
        Path("/opt/starnet"),
    ]
    for base in common_paths:
        for name in ("starnet++", "starnet2", "StarNetv2CLI"):
            candidate = base / name
            if candidate.exists() and candidate.is_file():
                return candidate

    return None


def run_starnet(
    image: np.ndarray,
    starnet_path: Path | str | None = None,
    extract_stars: bool = True,
) -> StarNetResult:
    """Run StarNet as a subprocess for GPL isolation.

    Parameters
    ----------
    image : ndarray
        Image data, shape (H, W) or (C, H, W), float32 in [0, 1].
    starnet_path : Path or str, optional
        Path to StarNet binary. If None, auto-detected.
    extract_stars : bool
        If True, also compute stars-only image (original - starless).

    Returns
    -------
    StarNetResult
        Result containing starless image and optionally stars-only.
    """
    if starnet_path is None:
        starnet_path = find_starnet_binary()
    else:
        starnet_path = Path(starnet_path)

    if starnet_path is None or not starnet_path.exists():
        return StarNetResult(
            starless=image.copy(),
            success=False,
            message="StarNet binary not found. Install StarNet++ and ensure it's in PATH.",
        )

    try:
        with tempfile.TemporaryDirectory(prefix="cosmica_starnet_") as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.fits"
            output_path = tmpdir / "starless.fits"

            # Save input as FITS
            img_data = ImageData(data=image, header={})
            save_fits(img_data, input_path)

            # Run StarNet subprocess
            cmd = [str(starnet_path), str(input_path), str(output_path)]
            log.info("Running StarNet: %s", " ".join(cmd))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if result.returncode != 0:
                return StarNetResult(
                    starless=image.copy(),
                    success=False,
                    message=f"StarNet failed: {result.stderr}",
                )

            if not output_path.exists():
                return StarNetResult(
                    starless=image.copy(),
                    success=False,
                    message="StarNet produced no output file",
                )

            # Load result
            starless_img = load_image(str(output_path))
            starless = starless_img.data

            # Compute stars-only if requested
            stars_only = None
            if extract_stars:
                stars_only = np.clip(image - starless, 0, 1).astype(np.float32)

            return StarNetResult(
                starless=starless,
                stars_only=stars_only,
                success=True,
                message="StarNet processing complete",
            )

    except subprocess.TimeoutExpired:
        return StarNetResult(
            starless=image.copy(),
            success=False,
            message="StarNet timed out after 10 minutes",
        )
    except Exception as e:
        return StarNetResult(
            starless=image.copy(),
            success=False,
            message=f"StarNet error: {e}",
        )
