"""Tests for narrowband processing."""

import numpy as np

from cosmica.core.narrowband import (
    NarrowbandPalette,
    NarrowbandParams,
    combine_narrowband,
    continuum_subtraction,
)


class TestCombineNarrowband:
    def test_sho_palette(self):
        ha = np.ones((100, 100), dtype=np.float32) * 0.5
        oiii = np.ones((100, 100), dtype=np.float32) * 0.3
        sii = np.ones((100, 100), dtype=np.float32) * 0.2

        result = combine_narrowband(
            {"ha": ha, "oiii": oiii, "sii": sii},
            NarrowbandParams(palette=NarrowbandPalette.SHO),
        )
        assert result.shape == (3, 100, 100)
        # SHO: R=SII, G=Ha, B=OIII
        # After normalization, G should be brightest (Ha=0.5)
        assert result[1].mean() > result[0].mean()

    def test_hoo_palette(self):
        ha = np.ones((100, 100), dtype=np.float32) * 0.8
        oiii = np.ones((100, 100), dtype=np.float32) * 0.4

        result = combine_narrowband(
            {"ha": ha, "oiii": oiii},
            NarrowbandParams(palette=NarrowbandPalette.HOO),
        )
        # HOO: R=Ha, G=B=OIII
        assert result.shape == (3, 100, 100)

    def test_output_normalized(self):
        ha = np.ones((100, 100), dtype=np.float32) * 0.9
        oiii = np.ones((100, 100), dtype=np.float32) * 0.5
        sii = np.ones((100, 100), dtype=np.float32) * 0.3

        result = combine_narrowband({"ha": ha, "oiii": oiii, "sii": sii})
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_missing_sii(self):
        ha = np.ones((100, 100), dtype=np.float32) * 0.5
        oiii = np.ones((100, 100), dtype=np.float32) * 0.3

        result = combine_narrowband({"ha": ha, "oiii": oiii})
        assert result.shape == (3, 100, 100)

    def test_ha_required(self):
        import pytest
        oiii = np.ones((100, 100), dtype=np.float32) * 0.3
        with pytest.raises(ValueError):
            combine_narrowband({"oiii": oiii})


class TestContinuumSubtraction:
    def test_subtracts_continuum(self):
        narrowband = np.ones((100, 100), dtype=np.float32) * 0.8
        broadband = np.ones((100, 100), dtype=np.float32) * 0.3

        result = continuum_subtraction(narrowband, broadband, scale=1.0)
        np.testing.assert_allclose(result.mean(), 0.5, atol=0.01)

    def test_output_clipped(self):
        narrowband = np.ones((100, 100), dtype=np.float32) * 0.2
        broadband = np.ones((100, 100), dtype=np.float32) * 0.5

        result = continuum_subtraction(narrowband, broadband)
        assert result.min() >= 0.0

    def test_scale_factor(self):
        narrowband = np.ones((100, 100), dtype=np.float32) * 0.8
        broadband = np.ones((100, 100), dtype=np.float32) * 0.4

        r1 = continuum_subtraction(narrowband, broadband, scale=0.5)
        r2 = continuum_subtraction(narrowband, broadband, scale=1.0)
        assert r1.mean() > r2.mean()  # less subtracted with lower scale
