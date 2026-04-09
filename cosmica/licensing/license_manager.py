"""License Manager — Lemon Squeezy / Gumroad key validation.

Validates license keys, caches activation status, and gates Pro features.
No invasive DRM — simple key-based activation with periodic online checks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import requests
from platformdirs import user_data_dir

log = logging.getLogger(__name__)

LEMON_SQUEEZY_API = "https://api.lemonsqueezy.com/v1"
GUMROAD_API = "https://api.gumroad.com/v2/licenses/verify"
CACHE_FILE = "license_cache.json"
REVALIDATION_INTERVAL = 7 * 24 * 3600  # 7 days


class LicenseTier(Enum):
    FREE = auto()
    PRO = auto()


class LicenseProvider(Enum):
    LEMON_SQUEEZY = auto()
    GUMROAD = auto()


@dataclass
class LicenseStatus:
    tier: LicenseTier
    key: str = ""
    email: str = ""
    valid: bool = False
    provider: LicenseProvider | None = None
    last_validated: float = 0.0
    error: str = ""


PRO_FEATURES = {
    "ai_denoise",
    "ai_sharpen",
    "advanced_masking",
    "batch_processing",
    "cloud_sync",
    "star_removal",
}


class LicenseManager:
    """Manages license activation and Pro feature gating."""

    def __init__(self, product_id: str = ""):
        self._product_id = product_id
        self._status = LicenseStatus(tier=LicenseTier.FREE)
        self._cache_path = Path(user_data_dir("Cosmica", "Cosmica")) / CACHE_FILE
        self._load_cache()

    @property
    def status(self) -> LicenseStatus:
        return self._status

    @property
    def is_pro(self) -> bool:
        return self._status.tier == LicenseTier.PRO and self._status.valid

    def is_feature_available(self, feature: str) -> bool:
        """Check if a specific feature is available with the current license."""
        if feature not in PRO_FEATURES:
            return True  # free feature
        return self.is_pro

    def activate(self, key: str) -> LicenseStatus:
        """Attempt to activate a license key."""
        key = key.strip()
        if not key:
            return LicenseStatus(tier=LicenseTier.FREE, error="No key provided")

        # Try Lemon Squeezy format first, then Gumroad
        status = self._validate_lemon_squeezy(key)
        if not status.valid:
            status = self._validate_gumroad(key)

        self._status = status
        if status.valid:
            self._save_cache()
            log.info("License activated: %s (%s)", status.email, status.provider)
        else:
            log.warning("License activation failed: %s", status.error)

        return status

    def deactivate(self):
        """Remove the current license."""
        self._status = LicenseStatus(tier=LicenseTier.FREE)
        if self._cache_path.exists():
            self._cache_path.unlink()
        log.info("License deactivated")

    def check_revalidation(self) -> bool:
        """Check if the license needs online revalidation."""
        if not self._status.valid or not self._status.key:
            return False
        elapsed = time.time() - self._status.last_validated
        if elapsed > REVALIDATION_INTERVAL:
            log.info("License revalidation needed (%.0f hours since last check)", elapsed / 3600)
            return True
        return False

    def revalidate(self) -> LicenseStatus:
        """Revalidate the current license key online."""
        if self._status.key:
            return self.activate(self._status.key)
        return self._status

    def _validate_lemon_squeezy(self, key: str) -> LicenseStatus:
        """Validate against Lemon Squeezy API."""
        try:
            resp = requests.post(
                f"{LEMON_SQUEEZY_API}/licenses/validate",
                json={"license_key": key},
                headers={"Accept": "application/json"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("valid"):
                    return LicenseStatus(
                        tier=LicenseTier.PRO,
                        key=key,
                        email=data.get("meta", {}).get("customer_email", ""),
                        valid=True,
                        provider=LicenseProvider.LEMON_SQUEEZY,
                        last_validated=time.time(),
                    )
                return LicenseStatus(
                    tier=LicenseTier.FREE,
                    key=key,
                    error=data.get("error", "Invalid key"),
                )
        except requests.RequestException as e:
            log.debug("Lemon Squeezy validation failed: %s", e)

        return LicenseStatus(tier=LicenseTier.FREE, key=key, error="Could not validate")

    def _validate_gumroad(self, key: str) -> LicenseStatus:
        """Validate against Gumroad API."""
        try:
            resp = requests.post(
                GUMROAD_API,
                data={
                    "product_id": self._product_id,
                    "license_key": key,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    purchase = data.get("purchase", {})
                    return LicenseStatus(
                        tier=LicenseTier.PRO,
                        key=key,
                        email=purchase.get("email", ""),
                        valid=True,
                        provider=LicenseProvider.GUMROAD,
                        last_validated=time.time(),
                    )
        except requests.RequestException as e:
            log.debug("Gumroad validation failed: %s", e)

        return LicenseStatus(tier=LicenseTier.FREE, key=key, error="Could not validate")

    def _save_cache(self):
        """Cache the license status locally."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "key_hash": hashlib.sha256(self._status.key.encode()).hexdigest(),
            "key": self._status.key,  # stored locally only
            "tier": self._status.tier.name,
            "email": self._status.email,
            "provider": self._status.provider.name if self._status.provider else "",
            "last_validated": self._status.last_validated,
        }
        with open(self._cache_path, "w") as f:
            json.dump(data, f)

    def _load_cache(self):
        """Load cached license status."""
        if not self._cache_path.exists():
            return
        try:
            with open(self._cache_path) as f:
                data = json.load(f)
            key = data.get("key", "")
            if not key:
                return
            provider = None
            if data.get("provider"):
                provider = LicenseProvider[data["provider"]]
            self._status = LicenseStatus(
                tier=LicenseTier[data.get("tier", "FREE")],
                key=key,
                email=data.get("email", ""),
                valid=True,
                provider=provider,
                last_validated=data.get("last_validated", 0),
            )
            log.info("License loaded from cache: %s tier", self._status.tier.name)
        except Exception as e:
            log.debug("Could not load license cache: %s", e)
