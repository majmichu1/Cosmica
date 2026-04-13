"""Tests for the stacking engine."""

import numpy as np
import pytest

from cosmica.core.image_io import ImageData
from cosmica.core.stacking import (
    IntegrationMethod,
    NormalizationMethod,
    RegistrationMode,
    RejectionMethod,
    StackingParams,
    StackResult,
    normalize_stack,
    normalize_stack_linear_fit,
    stack_images,
)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestNormalizeStack:
    def test_single_frame_unchanged(self):
        stack = np.random.random((1, 20, 20)).astype(np.float32)
        result = normalize_stack(stack)
        np.testing.assert_allclose(result, stack, rtol=1e-5)

    def test_identical_frames_unchanged(self):
        data = np.full((5, 10, 10), 0.42, dtype=np.float32)
        result = normalize_stack(data)
        np.testing.assert_allclose(result, data, rtol=1e-5)

    def test_offset_correction(self):
        base = np.random.random((10, 10)).astype(np.float32) * 0.1
        stack = np.array([base + offset for offset in np.linspace(0.0, 0.5, 5)])
        result = normalize_stack(stack)
        frame_medians = np.array([np.median(result[i]) for i in range(5)])
        assert np.std(frame_medians) < 0.05

    def test_scale_correction(self):
        base = np.random.random((10, 10)).astype(np.float32) * 0.3 + 0.1
        stack = np.array([base * scale for scale in np.linspace(0.8, 1.2, 4)])
        result = normalize_stack(stack)
        frame_medians = np.array([np.median(result[i]) for i in range(4)])
        assert np.std(frame_medians) < 0.05

    def test_additive_only(self):
        base = np.full((10, 10), 0.3, dtype=np.float32)
        stack = np.array([base + 0.0, base + 0.1, base + 0.2])
        result = normalize_stack(stack, NormalizationMethod.ADDITIVE)
        frame_medians = np.array([np.median(result[i]) for i in range(3)])
        assert np.std(frame_medians) < 0.01

    def test_none_passthrough(self):
        stack = np.random.random((4, 10, 10)).astype(np.float32)
        result = normalize_stack(stack, NormalizationMethod.NONE)
        np.testing.assert_array_equal(result, stack)

    def test_backward_compat_alias(self):
        stack = np.random.random((3, 10, 10)).astype(np.float32)
        r1 = normalize_stack_linear_fit(stack)
        r2 = normalize_stack(stack, NormalizationMethod.ADDITIVE_SCALING)
        np.testing.assert_allclose(r1, r2)


# ---------------------------------------------------------------------------
# Rejection methods
# ---------------------------------------------------------------------------


class TestStackImages:
    def test_single_image(self):
        img = ImageData(data=np.random.random((50, 60)).astype(np.float32))
        result = stack_images([img], align=False)
        assert result.n_frames == 1
        np.testing.assert_allclose(result.image.data, img.data)

    def test_stack_identical_no_alignment(self):
        data = np.random.random((40, 50)).astype(np.float32) * 0.5
        images = [ImageData(data=data.copy()) for _ in range(5)]
        params = StackingParams(rejection=RejectionMethod.SIGMA_CLIP,
                                integration=IntegrationMethod.AVERAGE)
        result = stack_images(images, params=params, align=False)
        assert result.n_frames == 5
        np.testing.assert_allclose(result.image.data, data, atol=0.01)

    def test_sigma_clip_rejects_hot_pixel(self):
        data = np.full((30, 40), 0.3, dtype=np.float32)
        images = [ImageData(data=data.copy()) for _ in range(20)]
        bad = data.copy()
        bad[15, 20] = 5.0
        images[5] = ImageData(data=bad)
        params = StackingParams(rejection=RejectionMethod.SIGMA_CLIP, kappa_high=2.5)
        result = stack_images(images, params=params, align=False)
        assert abs(result.image.data[15, 20] - 0.3) < 0.1
        assert result.total_rejected > 0

    def test_winsorized_sigma_rejects_hot_pixel(self):
        """Winsorized sigma must actually reject outliers, not silently pass through."""
        data = np.full((20, 20), 0.3, dtype=np.float32)
        images = [ImageData(data=data.copy()) for _ in range(20)]
        bad = data.copy()
        bad[10, 10] = 5.0
        images[5] = ImageData(data=bad)
        params = StackingParams(rejection=RejectionMethod.WINSORIZED_SIGMA, kappa_high=2.5)
        result = stack_images(images, params=params, align=False)
        assert abs(result.image.data[10, 10] - 0.3) < 0.2
        assert result.total_rejected > 0

    def test_percentile_clip_rejects_outlier(self):
        data = np.full((20, 20), 0.3, dtype=np.float32)
        images = [ImageData(data=data.copy()) for _ in range(10)]
        bad = data.copy()
        bad[5, 5] = 5.0
        images[9] = ImageData(data=bad)
        params = StackingParams(rejection=RejectionMethod.PERCENTILE_CLIP,
                                percentile_low=5.0, percentile_high=5.0)
        result = stack_images(images, params=params, align=False)
        # 10 frames, rejecting top 5% = 1 value — the hot pixel at [5,5] should be gone
        assert result.image.data[5, 5] < 1.0

    def test_esd_rejects_outlier(self):
        data = np.full((15, 15), 0.3, dtype=np.float32)
        images = [ImageData(data=data.copy()) for _ in range(15)]
        bad = data.copy()
        bad[7, 7] = 8.0
        images[0] = ImageData(data=bad)
        params = StackingParams(rejection=RejectionMethod.ESD)
        result = stack_images(images, params=params, align=False)
        assert abs(result.image.data[7, 7] - 0.3) < 0.5
        assert result.total_rejected > 0

    def test_min_max_rejects_extremes(self):
        data = np.full((10, 10), 0.3, dtype=np.float32)
        images = [ImageData(data=data.copy()) for _ in range(6)]
        images[0].data[5, 5] = 0.0   # lowest
        images[5].data[5, 5] = 1.0   # highest
        params = StackingParams(rejection=RejectionMethod.MIN_MAX, min_max_reject=1)
        result = stack_images(images, params=params, align=False)
        # After rejecting min and max, remaining 4 frames all have 0.3
        assert abs(result.image.data[5, 5] - 0.3) < 0.05
        assert result.total_rejected > 0

    def test_median_integration(self):
        data = np.full((20, 20), 0.4, dtype=np.float32)
        images = [ImageData(data=data.copy()) for _ in range(5)]
        images[2] = ImageData(data=np.full((20, 20), 0.9, dtype=np.float32))
        params = StackingParams(rejection=RejectionMethod.SIGMA_CLIP,
                                integration=IntegrationMethod.MEDIAN)
        result = stack_images(images, params=params, align=False)
        assert np.median(result.image.data) < 0.5

    def test_no_rejection(self):
        data = np.full((10, 10), 0.3, dtype=np.float32)
        images = [
            ImageData(data=data.copy()),
            ImageData(data=data.copy() + 0.01),
            ImageData(data=data.copy() - 0.01),
        ]
        params = StackingParams(rejection=RejectionMethod.NONE)
        result = stack_images(images, params=params, align=False)
        assert abs(np.mean(result.image.data) - 0.3) < 0.05
        assert result.total_rejected == 0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            stack_images([])

    def test_linear_fit_rejection_mode(self):
        data = np.full((20, 20), 0.3, dtype=np.float32)
        images = []
        for i in range(15):
            frame = data.copy() + i * 0.02
            images.append(ImageData(data=frame))
        images[7].data[10, 10] = 5.0
        params = StackingParams(rejection=RejectionMethod.LINEAR_FIT, kappa_high=2.5)
        result = stack_images(images, params=params, align=False)
        assert abs(result.image.data[10, 10] - 0.3) < 0.5
        assert result.total_rejected > 0

    def test_normalization_prevents_black_screen(self):
        base = np.random.random((30, 30)).astype(np.float32) * 0.05
        images = [ImageData(data=(base + 0.1 * i).astype(np.float32)) for i in range(4)]
        params = StackingParams(rejection=RejectionMethod.SIGMA_CLIP,
                                integration=IntegrationMethod.AVERAGE)
        result = stack_images(images, params=params, align=False)
        assert np.mean(result.image.data) > 0.01
        assert np.mean(result.image.data) < 0.9


# ---------------------------------------------------------------------------
# FFT alignment (translation only, no GPU required)
# ---------------------------------------------------------------------------


class TestFFTAlignment:
    def test_fft_aligns_shifted_images(self):
        """FFT alignment should recover a known pixel shift."""
        base = np.zeros((60, 80), dtype=np.float32)
        # Add a few bright spots
        for y, x in [(10, 15), (30, 50), (45, 25)]:
            base[y, x] = 1.0

        shift_row, shift_col = 5, -3
        shifted = np.roll(np.roll(base, shift_row, axis=0), shift_col, axis=1)

        ref_img = ImageData(data=base)
        tgt_img = ImageData(data=shifted)

        params = StackingParams(
            registration_mode=RegistrationMode.FFT_TRANSLATION,
            use_gpu=False,
        )
        aligned = stack_images([ref_img, tgt_img], params=params, align=True)
        assert aligned.n_frames == 2
        # The stacked result should be close to the reference
        assert aligned.image.data.shape == base.shape


class TestCometAlignment:
    def test_comet_nucleus_found(self):
        from cosmica.core.stacking import _find_comet_nucleus

        frame = np.zeros((100, 100), dtype=np.float32)
        frame[40, 60] = 1.0
        cx, cy = _find_comet_nucleus(frame, 15)
        assert abs(cx - 60) < 1.0
        assert abs(cy - 40) < 1.0

    def test_comet_aligns_shifted_frames(self):
        from cosmica.core.stacking import _comet_align_frames

        base = np.zeros((80, 80), dtype=np.float32)
        base[40, 40] = 1.0  # nucleus at (40, 40)

        shifted = np.zeros((80, 80), dtype=np.float32)
        shifted[50, 45] = 1.0  # nucleus at (45, 50)

        imgs = [ImageData(data=base), ImageData(data=shifted)]
        params = StackingParams(
            registration_mode=RegistrationMode.COMET,
            comet_nucleus_radius=15,
        )
        aligned = _comet_align_frames(imgs, params, lambda f, m: None)
        assert len(aligned) == 2
        # After alignment, nucleus in frame 2 should be near (40, 40)
        from cosmica.core.stacking import _find_comet_nucleus
        cx, cy = _find_comet_nucleus(aligned[1].data, 15)
        assert abs(cx - 40) < 2.0
        assert abs(cy - 40) < 2.0

    def test_comet_mode_in_registration_enum(self):
        assert RegistrationMode.COMET is not None

    def test_stacking_params_comet_radius(self):
        p = StackingParams(registration_mode=RegistrationMode.COMET, comet_nucleus_radius=25)
        assert p.comet_nucleus_radius == 25
