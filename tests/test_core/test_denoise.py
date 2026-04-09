"""Tests for noise reduction."""

import numpy as np

from cosmica.core.denoise import DenoiseMethod, DenoiseParams, denoise
from cosmica.core.masks import Mask


def _noisy_image(h=100, w=100, noise_level=0.1):
    """Create an image with additive Gaussian noise."""
    clean = np.ones((h, w), dtype=np.float32) * 0.3
    rng = np.random.default_rng(42)
    noise = rng.normal(0, noise_level, (h, w)).astype(np.float32)
    return np.clip(clean + noise, 0, 1)


class TestDenoise:
    def test_nlm_reduces_noise(self):
        noisy = _noisy_image(noise_level=0.1)
        params = DenoiseParams(method=DenoiseMethod.NLM, strength=0.5)
        result = denoise(noisy, params)
        assert result.shape == noisy.shape
        # Standard deviation should decrease
        assert result.std() < noisy.std()

    def test_wavelet_reduces_noise(self):
        noisy = _noisy_image(noise_level=0.1)
        params = DenoiseParams(method=DenoiseMethod.WAVELET, strength=0.5)
        result = denoise(noisy, params)
        assert result.shape == noisy.shape
        assert result.std() < noisy.std()

    def test_output_in_range(self):
        noisy = _noisy_image()
        result = denoise(noisy)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_color_image(self):
        rng = np.random.default_rng(42)
        noisy = np.clip(0.3 + rng.normal(0, 0.05, (3, 64, 64)), 0, 1).astype(np.float32)
        result = denoise(noisy, DenoiseParams(method=DenoiseMethod.WAVELET))
        assert result.shape == (3, 64, 64)

    def test_chrominance_only(self):
        rng = np.random.default_rng(42)
        noisy = np.clip(0.3 + rng.normal(0, 0.05, (3, 64, 64)), 0, 1).astype(np.float32)
        params = DenoiseParams(chrominance_only=True)
        result = denoise(noisy, params)
        assert result.shape == (3, 64, 64)

    def test_mask_support(self):
        noisy = _noisy_image()
        mask_data = np.zeros((100, 100), dtype=np.float32)
        mask_data[50:, :] = 1.0
        mask = Mask(data=mask_data)
        result = denoise(noisy, mask=mask)
        # Top half should be unchanged
        np.testing.assert_array_almost_equal(result[:50, :], noisy[:50, :])
