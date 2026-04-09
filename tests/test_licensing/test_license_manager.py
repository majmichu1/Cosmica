"""Tests for the license manager with mocked HTTP requests."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from cosmica.licensing.license_manager import (
    LicenseManager,
    LicenseProvider,
    LicenseTier,
)


def _make_response(status_code, json_data):
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    return resp


@pytest.fixture
def mock_cache_dir(tmp_path, monkeypatch):
    """Redirect platformdirs to a temp directory."""
    monkeypatch.setattr(
        "platformdirs.user_data_dir",
        lambda *a, **k: str(tmp_path),
    )


class TestLicenseManagerActivation:
    def test_empty_key_returns_free(self, mock_cache_dir):
        mgr = LicenseManager()
        result = mgr.activate("")
        assert not result.valid
        assert result.tier == LicenseTier.FREE

    def test_lemon_squeezy_valid_key(self, mock_cache_dir):
        mock_resp = _make_response(200, {
            "valid": True,
            "meta": {"customer_email": "user@example.com"},
        })

        mgr = LicenseManager()
        with patch("requests.post", return_value=mock_resp):
            result = mgr.activate("COSMICA-TEST-VALID-1234")

        assert result.valid
        assert result.tier == LicenseTier.PRO
        assert result.email == "user@example.com"
        assert result.provider == LicenseProvider.LEMON_SQUEEZY
        assert mgr.is_pro

    def test_lemon_squeezy_invalid_key(self, mock_cache_dir):
        mock_resp = _make_response(200, {
            "valid": False,
            "error": "Invalid key",
        })

        mgr = LicenseManager()
        with patch("requests.post", return_value=mock_resp):
            result = mgr.activate("COSMICA-FAKE-1234")

        assert not result.valid
        assert result.tier == LicenseTier.FREE
        assert not mgr.is_pro

    def test_gumroad_valid_key(self, mock_cache_dir):
        # Lemon Squeezy fails first, then Gumroad is tried
        ls_resp = _make_response(200, {"valid": False})
        gr_resp = _make_response(200, {
            "success": True,
            "purchase": {"email": "gumroad@example.com"},
        })

        call_count = {"count": 0}

        def mock_post(*args, **kwargs):
            call_count["count"] += 1
            if call_count["count"] == 1:
                return ls_resp
            return gr_resp

        mgr = LicenseManager(product_id="gumroad_product_id")
        with patch("requests.post", side_effect=mock_post):
            result = mgr.activate("GUMROAD-TEST-KEY")

        assert result.valid
        assert result.email == "gumroad@example.com"
        assert result.provider == LicenseProvider.GUMROAD

    def test_network_error_falls_back_gracefully(self, mock_cache_dir):
        import requests as req
        mgr = LicenseManager()
        with patch("requests.post", side_effect=req.RequestException("Network error")):
            result = mgr.activate("ANY-KEY")

        assert not result.valid
        assert result.tier == LicenseTier.FREE
        assert not mgr.is_pro


class TestLicenseManagerCache:
    def test_cache_roundtrip(self, mock_cache_dir):
        mock_resp = _make_response(200, {
            "valid": True,
            "meta": {"customer_email": "cache@test.com"},
        })

        mgr = LicenseManager()
        with patch("requests.post", return_value=mock_resp):
            mgr.activate("CACHE-TEST-KEY")

        # Create new instance — should load from cache
        mgr2 = LicenseManager()
        assert mgr2.is_pro

    def test_deactivate_clears_cache(self, mock_cache_dir):
        mock_resp = _make_response(200, {
            "valid": True,
            "meta": {"customer_email": "a@b.com"},
        })

        mgr = LicenseManager()
        with patch("requests.post", return_value=mock_resp):
            mgr.activate("DEACT-TEST")

        assert mgr.is_pro
        mgr.deactivate()
        assert not mgr.is_pro


class TestLicenseFeatureGating:
    def test_free_user_no_access_to_pro_features(self, mock_cache_dir):
        mgr = LicenseManager()
        assert not mgr.is_feature_available("ai_denoise")
        assert not mgr.is_feature_available("ai_sharpen")
        assert not mgr.is_feature_available("batch_processing")

    def test_free_user_has_free_features(self, mock_cache_dir):
        mgr = LicenseManager()
        # Unknown/non-gated features are always available
        assert mgr.is_feature_available("some_random_feature")

    def test_pro_user_has_all_features(self, mock_cache_dir):
        mock_resp = _make_response(200, {
            "valid": True,
            "meta": {"customer_email": "pro@test.com"},
        })

        mgr = LicenseManager()
        with patch("requests.post", return_value=mock_resp):
            mgr.activate("PRO-TEST-KEY")

        assert mgr.is_feature_available("ai_denoise")
        assert mgr.is_feature_available("ai_sharpen")
        assert mgr.is_feature_available("advanced_masking")
        assert mgr.is_feature_available("batch_processing")
        assert mgr.is_feature_available("cloud_sync")
        assert mgr.is_feature_available("star_removal")


class TestLicenseRevalidation:
    def test_revalidation_not_needed_yet(self, mock_cache_dir):
        mock_resp = _make_response(200, {
            "valid": True,
            "meta": {"customer_email": "r@test.com"},
        })

        mgr = LicenseManager()
        with patch("requests.post", return_value=mock_resp):
            mgr.activate("REVAL-TEST")

        # Just activated — no revalidation needed
        assert not mgr.check_revalidation()

    def test_revalidation_expired(self, mock_cache_dir):
        mgr = LicenseManager()
        # Manually set a very old timestamp
        mgr._status.valid = True
        mgr._status.key = "OLD-KEY"
        mgr._status.last_validated = time.time() - (8 * 24 * 3600)  # 8 days ago

        assert mgr.check_revalidation()

    def test_revalidation_free_user(self, mock_cache_dir):
        mgr = LicenseManager()
        assert not mgr.check_revalidation()
