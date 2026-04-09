"""Star Catalog — query Vizier for reference stars (Gaia DR3).

Provides access to star catalogs for photometry and plate solving.
Uses TAP queries to Vizier or direct HTTP fallback.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

CATALOG_CACHE_DIR = Path.home() / ".cosmica" / "catalogs"
CATALOG_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class StarCatalogEntry:
    """Single star from a catalog query."""

    ra_deg: float
    dec_deg: float
    g_mag: float | None
    bp_mag: float | None
    rp_mag: float | None
    source_id: str | None


def query_gaia_dr3(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float = 0.5,
    max_stars: int = 500,
) -> list[StarCatalogEntry]:
    """Query Gaia DR3 stars around a given position.

    Uses the Vizier TAP service for Gaia DR3 (I/355/gaiadr3).

    Parameters
    ----------
    ra_deg : float
        Right Ascension of field center in degrees (J2000).
    dec_deg : float
        Declination of field center in degrees (J2000).
    radius_deg : float
        Search radius in degrees.
    max_stars : int
        Maximum number of stars to return.

    Returns
    -------
    list[StarCatalogEntry]
        Stars in the field with available photometry.
    """
    log.info(
        "Querying Gaia DR3: RA=%.4f, Dec=%.4f, radius=%.3f deg",
        ra_deg,
        dec_deg,
        radius_deg,
    )

    if shutil.which("astap_cli") is not None:
        return _query_gaia_vizier_tap(ra_deg, dec_deg, radius_deg, max_stars)
    else:
        return _query_gaia_vizier_tap(ra_deg, dec_deg, radius_deg, max_stars)


def _query_gaia_vizier_tap(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    max_stars: int,
) -> list[StarCatalogEntry]:
    """Query Gaia DR3 via Vizier TAP service."""
    import xml.etree.ElementTree as ET

    tap_url = "https://vizier.u-strasbg.fr/vizier/sl/VizierCmd.py"

    radius_deg = min(radius_deg, 2.0)

    params = {
        "-source": "I/355/gaiadr3",
        "-info": "all",
        "-out": "RAJ2000,DEJ2000,Gmag,BPmag,RPmag,Source",
        "-c": f"{ra_deg} {dec_deg}",
        "-c.rs": str(radius_deg),
        "-c.eq": "J2000",
        "-out.max": str(max_stars),
    }

    url = f"{tap_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"

    log.debug("Vizier query URL: %s", url)

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Cosmica/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        log.error("Vizier TAP query failed: %s", e)
        return []

    stars: list[StarCatalogEntry] = []
    try:
        root = ET.fromstring(data)
        for resource in root.findall(".//RESOURCE"):
            for table in resource.findall("TABLE"):
                for data_elem in table.findall("DATA/TABLEDATA/TR"):
                    cells = data_elem.findall("TD")
                    if len(cells) >= 6:
                        try:
                            ra = float(cells[0].text or 0)
                            dec = float(cells[1].text or 0)
                            g_mag = float(cells[2].text) if cells[2].text else None
                            bp_mag = float(cells[3].text) if cells[3].text else None
                            rp_mag = float(cells[4].text) if cells[4].text else None
                            source_id = cells[5].text
                            stars.append(
                                StarCatalogEntry(
                                    ra_deg=ra,
                                    dec_deg=dec,
                                    g_mag=g_mag,
                                    bp_mag=bp_mag,
                                    rp_mag=rp_mag,
                                    source_id=source_id,
                                )
                            )
                        except (ValueError, TypeError):
                            continue
    except ET.ParseError:
        log.error("Failed to parse Vizier response")
        return []

    log.info("Retrieved %d stars from Gaia DR3", len(stars))
    return stars


def plate_solve_astap(
    image_path: Path,
    ra_hint: float | None = None,
    dec_hint: float | None = None,
    scale_hint: float | None = None,
) -> dict | None:
    """Plate-solve using ASTAP CLI.

    Requires astap_cli to be installed and in PATH.

    Parameters
    ----------
    image_path : Path
        Path to the FITS image to solve.
    ra_hint : float, optional
        Approximate RA in degrees.
    dec_hint : float, optional
        Approximate Dec in degrees.
    scale_hint : float, optional
        Approximate pixel scale in arcsec/px.

    Returns
    -------
    dict or None
        Dict with 'ra', 'dec', 'scale', 'rotation', 'wcs' keys, or None if failed.
    """
    astap_path = shutil.which("astap_cli")
    if astap_path is None:
        log.error("ASTAP CLI not found in PATH")
        return None

    cmd = [astap_path, "-f", str(image_path)]

    if ra_hint is not None and dec_hint is not None:
        cmd.extend(["-ra", f"{ra_hint:.6f}", "-dec", f"{dec_hint:.6f}"])

    if scale_hint is not None:
        cmd.extend(["-scale", f"{scale_hint:.2f}"])

    log.info("Running ASTAP: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            log.error("ASTAP failed: %s", result.stderr)
            return None

        wcs = _parse_astap_output(result.stdout, result.stderr)
        return wcs
    except subprocess.TimeoutExpired:
        log.error("ASTAP timed out")
        return None
    except Exception as e:
        log.error("ASTAP error: %s", e)
        return None


def _parse_astap_output(stdout: str, stderr: str) -> dict | None:
    """Parse ASTAP output to extract WCS information."""
    output = stdout + "\n" + stderr

    wcs = {"ra": None, "dec": None, "scale": None, "rotation": None, "wcs": {}}

    for line in output.split("\n"):
        line = line.strip()
        if "RA:" in line:
            try:
                ra_str = line.split("RA:")[-1].strip().split()[0]
                wcs["ra"] = float(ra_str)
            except (IndexError, ValueError):
                pass
        if "Dec:" in line:
            try:
                dec_str = line.split("Dec:")[-1].strip().split()[0]
                wcs["dec"] = float(dec_str)
            except (IndexError, ValueError):
                pass
        if "scale:" in line.lower():
            try:
                scale_str = line.split(":")[-1].strip().split()[0]
                wcs["scale"] = float(scale_str)
            except (IndexError, ValueError):
                pass

    if wcs["ra"] is not None and wcs["dec"] is not None:
        log.info("ASTAP solved: RA=%.4f, Dec=%.4f", wcs["ra"], wcs["dec"])
        return wcs

    return None
