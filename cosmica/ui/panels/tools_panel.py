"""Tools Panel — right sidebar with tabbed processing controls.

Organized into tabs:
1. Pre-Process — Calibration, Cosmetic Correction, Subframe Selector
2. Stacking — Registration + integration controls, Batch Processing
3. Stretch — Auto-Stretch, GHS, Histogram Transform, Curves
4. Background — Background Extraction, ABE, Vignette Correction, Banding Reduction
5. Transform — Crop, Rotate, Flip, Resize, Bin, Invert
6. Color — SCNR, Color Adjustment, Color Calibration, PCC
7. Detail — Deconvolution, Noise Reduction, Star Reduction, Wavelets, Local Contrast, Morphology
8. AI Tools — AI Denoise, AI Sharpen, StarNet Star Removal
9. Utility — Narrowband, Pixel Math, Channels, HDR, Macros
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from cosmica.ai.inference.denoise import AIDenoiseParams
from cosmica.ai.inference.sharpen import AISharpenParams
from cosmica.core.abe import ABEParams
from cosmica.core.background import BackgroundParams
from cosmica.core.banding import BandingParams
from cosmica.core.chromatic_aberration import CAParams
from cosmica.core.color_calibration import ColorCalibrationParams
from cosmica.core.color_tools import ColorAdjustParams, SCNRMethod, SCNRParams
from cosmica.core.cosmetic import CosmeticParams
from cosmica.core.curves import CurvesParams
from cosmica.core.deconvolution import DeconvolutionParams, SpatialDeconvParams
from cosmica.core.denoise import DenoiseMethod, DenoiseParams
from cosmica.core.filters import MedianFilterParams, UnsharpMaskParams
from cosmica.core.histogram_transform import HistogramTransformParams
from cosmica.core.local_contrast import LocalContrastParams
from cosmica.core.morphology import MorphologyParams, MorphOp, StructuringElement
from cosmica.core.stacking import (
    IntegrationMethod,
    RegistrationMode,
    RejectionMethod,
    StackingParams,
)
from cosmica.core.star_reduction import StarReductionParams
from cosmica.core.stretch import GHSParams, StretchParams
from cosmica.core.transforms import (
    BinMode,
    BinParams,
    CropParams,
    FlipAxis,
    FlipParams,
    InterpolationMethod,
    ResizeParams,
    RotateAngle,
    RotateParams,
)
from cosmica.core.vignette import VignetteParams
from cosmica.core.wavelets import WaveletParams
from cosmica.ui.widgets.curves_widget import CurveEditor


class ResettableSlider(QSlider):
    """Slider that resets to its default value on double-click."""

    def __init__(self, orientation, default_value: int = 0, parent=None):
        super().__init__(orientation, parent)
        self._default_value = default_value

    def mouseDoubleClickEvent(self, event):
        self.setValue(self._default_value)
        super().mouseDoubleClickEvent(event)


def _info_label(text: str) -> QLabel:
    """Create a grey description label used across all tool groups."""
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color: #969696; font-size: 11px;")
    return lbl


def _h_row(label_text: str, widget: QWidget) -> QHBoxLayout:
    """Create a horizontal label + widget row."""
    row = QHBoxLayout()
    row.addWidget(QLabel(label_text))
    row.addWidget(widget)
    return row


def _scrollable_tab(layout: QVBoxLayout) -> QScrollArea:
    """Wrap a layout in a scrollable area for use as a tab."""
    container = QWidget()
    container.setLayout(layout)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(container)
    return scroll


class ToolsPanel(QWidget):
    """Right panel: tabbed processing tool controls."""

    # Signals — existing (keep backward compat)
    run_calibration = pyqtSignal()
    run_stacking = pyqtSignal()
    run_alignment = pyqtSignal()
    run_stretch = pyqtSignal()
    run_background = pyqtSignal()
    stretch_params_changed = pyqtSignal()

    # Phase A signals
    run_cosmetic = pyqtSignal()
    run_banding = pyqtSignal()
    run_histogram_transform = pyqtSignal()
    run_curves = pyqtSignal()
    run_scnr = pyqtSignal()
    run_color_adjust = pyqtSignal()
    run_deconvolution = pyqtSignal()

    # Phase B signals
    run_ghs = pyqtSignal()
    run_color_calibration = pyqtSignal()
    run_pcc = pyqtSignal()
    run_denoise = pyqtSignal()
    run_star_reduction = pyqtSignal()
    open_narrowband_dialog = pyqtSignal()
    open_pixelmath_dialog = pyqtSignal()
    run_split_channels = pyqtSignal()
    run_extract_luminance = pyqtSignal()

    # Phase C signals
    run_wavelet_sharpen = pyqtSignal()
    run_local_contrast = pyqtSignal()
    run_morphology = pyqtSignal()
    open_hdr_dialog = pyqtSignal()

    # Phase D signals
    run_ai_denoise = pyqtSignal()
    run_ai_sharpen = pyqtSignal()
    run_starnet = pyqtSignal()
    open_batch_dialog = pyqtSignal()
    start_macro_recording = pyqtSignal()
    stop_macro_recording = pyqtSignal()
    play_macro = pyqtSignal()
    save_macro = pyqtSignal()
    load_macro = pyqtSignal()

    # New tool signals
    run_unsharp_mask = pyqtSignal()
    run_median_filter = pyqtSignal()
    run_abe = pyqtSignal()
    run_vignette_correction = pyqtSignal()
    run_chromatic_aberration = pyqtSignal()
    show_image_statistics = pyqtSignal()
    curves_histogram_changed = pyqtSignal()  # checkbox toggled or channel changed
    measure_psf = pyqtSignal()
    run_continuum_subtraction = pyqtSignal()
    toggle_sample_mode = pyqtSignal(bool)
    clear_bg_samples = pyqtSignal()
    toggle_wcs_overlay = pyqtSignal(bool)
    open_python_console = pyqtSignal()
    run_mlt = pyqtSignal()
    open_star_mask_dialog = pyqtSignal()
    open_subframe_selector = pyqtSignal()

    # Transform signals
    run_crop = pyqtSignal()
    run_rotate = pyqtSignal()
    run_flip = pyqtSignal()
    run_resize = pyqtSignal()
    run_bin = pyqtSignal()
    run_invert = pyqtSignal()

    # Preview signal: emitted with tool name when preview is requested
    preview_requested = pyqtSignal(str)  # tool_name
    preview_cancelled = pyqtSignal()

    # Smart Processor signals
    open_smart_processor = pyqtSignal()
    open_equipment_dialog = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(420)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)
        self._tabs.setUsesScrollButtons(True)
        self._tabs.tabBar().setExpanding(False)
        outer.addWidget(self._tabs)

        self._preview_checks: dict[str, QCheckBox] = {}

        self._build_preprocess_tab()
        self._build_stacking_tab()
        self._build_stretch_tab()
        self._build_background_tab()
        self._build_transform_tab()
        self._build_color_tab()
        self._build_detail_tab()
        self._build_ai_pro_tab()
        self._build_utility_tab()

    def _add_preview_checkbox(self, layout: QVBoxLayout, tool_name: str) -> QCheckBox:
        """Add a preview toggle checkbox to a tool group."""
        cb = QCheckBox("Live preview (split view)")
        cb.setToolTip("Show before/after comparison on canvas at reduced resolution")
        cb.toggled.connect(lambda checked, t=tool_name: self._on_preview_toggled(t, checked))
        layout.addWidget(cb)
        self._preview_checks[tool_name] = cb
        return cb

    def _on_preview_toggled(self, tool_name: str, checked: bool):
        if checked:
            self.preview_requested.emit(tool_name)
        else:
            self.preview_cancelled.emit()

    def _emit_if_preview_enabled(self, tool_name: str):
        """Emit preview_requested if the preview checkbox for tool is checked."""
        cb = self._preview_checks.get(tool_name)
        if cb is not None and cb.isChecked():
            self.preview_requested.emit(tool_name)

    # ================================================================
    # TAB 1: Pre-Process
    # ================================================================
    def _build_preprocess_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- Calibration ---
        cal_group = QGroupBox("Calibration")
        cal_layout = QVBoxLayout(cal_group)
        cal_layout.addWidget(
            _info_label("Create master frames from imported darks, flats, and bias.")
        )
        self._btn_calibrate = QPushButton("Run Calibration")
        self._btn_calibrate.setToolTip(
            "Creates master dark, flat, and bias from your imported calibration frames, "
            "then applies them to your light frames."
        )
        self._btn_calibrate.clicked.connect(self.run_calibration.emit)
        cal_layout.addWidget(self._btn_calibrate)
        layout.addWidget(cal_group)

        # --- Cosmetic Correction ---
        cos_group = QGroupBox("Cosmetic Correction")
        cos_layout = QVBoxLayout(cos_group)
        cos_layout.addWidget(
            _info_label("Detect and remove hot, cold, and dead pixels from sensor defects.")
        )

        self._hot_sigma_spin = QDoubleSpinBox()
        self._hot_sigma_spin.setRange(1.0, 20.0)
        self._hot_sigma_spin.setValue(5.0)
        self._hot_sigma_spin.setSingleStep(0.5)
        self._hot_sigma_spin.setToolTip("Sigma threshold for hot pixel detection")
        cos_layout.addLayout(_h_row("Hot sigma:", self._hot_sigma_spin))

        self._cold_sigma_spin = QDoubleSpinBox()
        self._cold_sigma_spin.setRange(1.0, 20.0)
        self._cold_sigma_spin.setValue(5.0)
        self._cold_sigma_spin.setSingleStep(0.5)
        self._cold_sigma_spin.setToolTip("Sigma threshold for cold pixel detection")
        cos_layout.addLayout(_h_row("Cold sigma:", self._cold_sigma_spin))

        self._dead_pixel_check = QCheckBox("Detect dead pixels (value = 0)")
        self._dead_pixel_check.setChecked(True)
        cos_layout.addWidget(self._dead_pixel_check)

        self._add_preview_checkbox(cos_layout, "cosmetic")
        self._btn_cosmetic = QPushButton("Apply Cosmetic Correction")
        self._btn_cosmetic.clicked.connect(self.run_cosmetic.emit)
        cos_layout.addWidget(self._btn_cosmetic)
        layout.addWidget(cos_group)

        # --- Subframe Selector ---
        sub_group = QGroupBox("Subframe Selector")
        sub_layout = QVBoxLayout(sub_group)
        sub_layout.addWidget(
            _info_label("Score and reject light frames by FWHM, eccentricity, SNR, and star count.")
        )
        btn_subframe = QPushButton("Open Subframe Selector...")
        btn_subframe.clicked.connect(self.open_subframe_selector.emit)
        sub_layout.addWidget(btn_subframe)
        layout.addWidget(sub_group)

        layout.addStretch()
        self._tabs.addTab(_scrollable_tab(layout), "Pre-Process")

    # ================================================================
    # TAB 2: Transform
    # ================================================================
    def _build_transform_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- Crop ---
        crop_group = QGroupBox("Crop")
        crop_layout = QVBoxLayout(crop_group)
        crop_layout.addWidget(_info_label("Crop image to a rectangular region."))

        self._crop_x_spin = QSpinBox()
        self._crop_x_spin.setRange(0, 99999)
        crop_layout.addLayout(_h_row("X offset:", self._crop_x_spin))

        self._crop_y_spin = QSpinBox()
        self._crop_y_spin.setRange(0, 99999)
        crop_layout.addLayout(_h_row("Y offset:", self._crop_y_spin))

        self._crop_w_spin = QSpinBox()
        self._crop_w_spin.setRange(0, 99999)
        self._crop_w_spin.setSpecialValueText("Full")
        crop_layout.addLayout(_h_row("Width:", self._crop_w_spin))

        self._crop_h_spin = QSpinBox()
        self._crop_h_spin.setRange(0, 99999)
        self._crop_h_spin.setSpecialValueText("Full")
        crop_layout.addLayout(_h_row("Height:", self._crop_h_spin))

        btn_crop = QPushButton("Apply Crop")
        btn_crop.clicked.connect(self.run_crop.emit)
        crop_layout.addWidget(btn_crop)
        layout.addWidget(crop_group)

        # --- Rotate ---
        rot_group = QGroupBox("Rotate")
        rot_layout = QVBoxLayout(rot_group)

        self._rotate_combo = QComboBox()
        self._rotate_combo.addItems(["90\u00b0 CW", "180\u00b0", "270\u00b0 CW", "Custom angle"])
        rot_layout.addLayout(_h_row("Angle:", self._rotate_combo))

        self._rotate_angle_spin = QDoubleSpinBox()
        self._rotate_angle_spin.setRange(-360, 360)
        self._rotate_angle_spin.setValue(0)
        self._rotate_angle_spin.setDecimals(1)
        self._rotate_angle_spin.setSuffix("\u00b0")
        rot_layout.addLayout(_h_row("Custom:", self._rotate_angle_spin))

        self._rotate_expand_check = QCheckBox("Expand canvas")
        self._rotate_expand_check.setChecked(True)
        rot_layout.addWidget(self._rotate_expand_check)

        btn_rotate = QPushButton("Apply Rotation")
        btn_rotate.clicked.connect(self.run_rotate.emit)
        rot_layout.addWidget(btn_rotate)
        layout.addWidget(rot_group)

        # --- Flip ---
        flip_group = QGroupBox("Flip")
        flip_layout = QVBoxLayout(flip_group)

        self._flip_combo = QComboBox()
        self._flip_combo.addItems(["Horizontal", "Vertical", "Both"])
        flip_layout.addLayout(_h_row("Axis:", self._flip_combo))

        btn_flip = QPushButton("Apply Flip")
        btn_flip.clicked.connect(self.run_flip.emit)
        flip_layout.addWidget(btn_flip)
        layout.addWidget(flip_group)

        # --- Resize ---
        resize_group = QGroupBox("Resize / Resample")
        resize_layout = QVBoxLayout(resize_group)

        self._resize_scale_spin = QDoubleSpinBox()
        self._resize_scale_spin.setRange(0.1, 10.0)
        self._resize_scale_spin.setValue(1.0)
        self._resize_scale_spin.setSingleStep(0.1)
        self._resize_scale_spin.setDecimals(2)
        resize_layout.addLayout(_h_row("Scale:", self._resize_scale_spin))

        self._resize_interp_combo = QComboBox()
        self._resize_interp_combo.addItems(["Lanczos", "Bicubic", "Bilinear", "Nearest"])
        resize_layout.addLayout(_h_row("Interpolation:", self._resize_interp_combo))

        btn_resize = QPushButton("Apply Resize")
        btn_resize.clicked.connect(self.run_resize.emit)
        resize_layout.addWidget(btn_resize)
        layout.addWidget(resize_group)

        # --- Bin ---
        bin_group = QGroupBox("Bin")
        bin_layout = QVBoxLayout(bin_group)
        bin_layout.addWidget(_info_label("Combine pixels to increase SNR at lower resolution."))

        self._bin_factor_combo = QComboBox()
        self._bin_factor_combo.addItems(["2x2", "3x3", "4x4"])
        bin_layout.addLayout(_h_row("Factor:", self._bin_factor_combo))

        self._bin_mode_combo = QComboBox()
        self._bin_mode_combo.addItems(["Average", "Sum"])
        bin_layout.addLayout(_h_row("Mode:", self._bin_mode_combo))

        btn_bin = QPushButton("Apply Bin")
        btn_bin.clicked.connect(self.run_bin.emit)
        bin_layout.addWidget(btn_bin)
        layout.addWidget(bin_group)

        # --- Invert ---
        btn_invert = QPushButton("Invert Image")
        btn_invert.setToolTip("Invert all pixel values (1 - image)")
        btn_invert.clicked.connect(self.run_invert.emit)
        layout.addWidget(btn_invert)

        layout.addStretch()
        self._tabs.addTab(_scrollable_tab(layout), "Transform")

    # ================================================================
    # TAB 3: Stacking
    # ================================================================
    def _build_stacking_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- Section 1: Registration / Alignment ---
        reg_group = QGroupBox("Registration (Alignment)")
        reg_layout = QVBoxLayout(reg_group)
        reg_layout.addWidget(
            _info_label("Detect stars and align light frames to the reference frame.")
        )

        self._reg_mode_combo = QComboBox()
        self._reg_mode_combo.addItems(["Star (1-Pass)", "Star (2-Pass)", "FFT Translation"])
        self._reg_mode_combo.setToolTip(
            "Star 1-Pass: GPU star detection + RANSAC, single pass.\n"
            "Star 2-Pass: as above with refinement pass (large rotations/drift).\n"
            "FFT Translation: phase cross-correlation, GPU-accelerated (pure shift only)."
        )
        reg_layout.addLayout(_h_row("Mode:", self._reg_mode_combo))

        self._star_sens_spin = QDoubleSpinBox()
        self._star_sens_spin.setRange(1.0, 20.0)
        self._star_sens_spin.setValue(5.0)
        self._star_sens_spin.setSingleStep(1.0)
        self._star_sens_spin.setToolTip(
            "Sigma threshold for star detection (lower = more stars detected)"
        )
        reg_layout.addLayout(_h_row("Star Sensitivity:", self._star_sens_spin))

        self._max_shift_spin = QDoubleSpinBox()
        self._max_shift_spin.setRange(10, 500)
        self._max_shift_spin.setValue(50)
        self._max_shift_spin.setSingleStep(10)
        self._max_shift_spin.setToolTip("Maximum allowed star shift in pixels for matching")
        reg_layout.addLayout(_h_row("Max Match Distance (px):", self._max_shift_spin))

        self._ransac_thresh_spin = QDoubleSpinBox()
        self._ransac_thresh_spin.setRange(1.0, 10.0)
        self._ransac_thresh_spin.setValue(3.0)
        self._ransac_thresh_spin.setSingleStep(0.5)
        self._ransac_thresh_spin.setToolTip("RANSAC inlier threshold for transform estimation")
        reg_layout.addLayout(_h_row("RANSAC Threshold:", self._ransac_thresh_spin))

        self._btn_align = QPushButton("Align Frames")
        self._btn_align.setToolTip("Detect stars and align all frames without stacking")
        self._btn_align.clicked.connect(self.run_alignment.emit)
        reg_layout.addWidget(self._btn_align)
        layout.addWidget(reg_group)

        # --- Section 1b: Frame Quality Filter ---
        quality_group = QGroupBox("Frame Quality Filter")
        quality_layout = QVBoxLayout(quality_group)
        quality_layout.addWidget(
            _info_label(
                "Reject poor-quality frames before alignment/stacking based on FWHM and SNR."
            )
        )

        self._quality_filter_check = QCheckBox("Filter by quality (subframe selector)")
        self._quality_filter_check.setChecked(False)
        quality_layout.addWidget(self._quality_filter_check)

        quality_metric_row = QHBoxLayout()
        quality_metric_row.addWidget(QLabel("Metric:"))
        self._quality_metric_combo = QComboBox()
        self._quality_metric_combo.addItems(["FWHM (sharpness)", "SNR (signal)", "Quality Score"])
        self._quality_metric_combo.setToolTip("Quality metric to use for filtering")
        quality_metric_row.addWidget(self._quality_metric_combo)
        quality_layout.addLayout(quality_metric_row)

        quality_mode_row = QHBoxLayout()
        quality_mode_row.addWidget(QLabel("Mode:"))
        self._quality_mode_combo = QComboBox()
        self._quality_mode_combo.addItems(["Keep best N frames", "Keep best %", "Sigma rejection"])
        self._quality_mode_combo.setToolTip("Filtering mode")
        self._quality_mode_combo.currentIndexChanged.connect(self._on_quality_mode_changed)
        quality_mode_row.addWidget(self._quality_mode_combo)
        quality_layout.addLayout(quality_mode_row)

        self._quality_n_spin = QSpinBox()
        self._quality_n_spin.setRange(1, 100)
        self._quality_n_spin.setValue(5)
        self._quality_n_spin.setToolTip("Number of best frames to keep")
        quality_layout.addLayout(_h_row("Keep N:", self._quality_n_spin))

        self._quality_percent_spin = QDoubleSpinBox()
        self._quality_percent_spin.setRange(10.0, 100.0)
        self._quality_percent_spin.setValue(80.0)
        self._quality_percent_spin.setSingleStep(5.0)
        self._quality_percent_spin.setToolTip("Percentage of best frames to keep")
        self._quality_percent_spin.setEnabled(False)
        quality_layout.addLayout(_h_row("Keep %:", self._quality_percent_spin))

        self._quality_sigma_spin = QDoubleSpinBox()
        self._quality_sigma_spin.setRange(0.5, 5.0)
        self._quality_sigma_spin.setValue(1.5)
        self._quality_sigma_spin.setSingleStep(0.5)
        self._quality_sigma_spin.setToolTip(
            "Frames scoring more than this many sigmas below the mean quality score are rejected"
        )
        self._quality_sigma_spin.setEnabled(False)
        quality_layout.addLayout(_h_row("Rejection Sigma:", self._quality_sigma_spin))

        self._quality_filter_check.toggled.connect(self._on_quality_filter_toggled)

        layout.addWidget(quality_group)

        # --- Section 2: Integration / Stacking ---
        stack_group = QGroupBox("Integration (Stacking)")
        stack_layout = QVBoxLayout(stack_group)
        stack_layout.addWidget(
            _info_label("Combine aligned frames using rejection to increase signal-to-noise ratio.")
        )

        self._rejection_combo = QComboBox()
        self._rejection_combo.addItems(
            [
                "Sigma Clipping",
                "Winsorized Sigma",
                "Linear Fit",
                "Percentile Clip",
                "ESD (Generalized)",
                "Min/Max",
                "None",
            ]
        )
        self._rejection_combo.setToolTip(
            "Method to reject outlier pixels (satellites, cosmic rays)"
        )
        stack_layout.addLayout(_h_row("Rejection:", self._rejection_combo))

        self._integration_combo = QComboBox()
        self._integration_combo.addItems(["Average", "Median"])
        self._integration_combo.setToolTip("How to combine remaining pixel values after rejection")
        stack_layout.addLayout(_h_row("Integration:", self._integration_combo))

        self._kappa_spin = QDoubleSpinBox()
        self._kappa_spin.setRange(0.5, 10.0)
        self._kappa_spin.setValue(3.0)
        self._kappa_spin.setSingleStep(0.1)
        self._kappa_spin.setToolTip("Sigma clipping threshold — lower = more aggressive rejection")
        stack_layout.addLayout(_h_row("Kappa:", self._kappa_spin))

        self._btn_stack = QPushButton("Stack Images")
        self._btn_stack.setToolTip("Align and stack all calibrated light frames")
        self._btn_stack.clicked.connect(self.run_stacking.emit)
        stack_layout.addWidget(self._btn_stack)
        layout.addWidget(stack_group)

        # --- Drizzle Integration ---
        drizzle_group = QGroupBox("Drizzle Integration")
        drizzle_layout = QVBoxLayout(drizzle_group)
        drizzle_layout.addWidget(
            _info_label(
                "Sub-pixel resolution enhancement. Stack frames at higher output resolution."
            )
        )

        self._drizzle_check = QCheckBox("Enable Drizzle")
        self._drizzle_check.setChecked(False)
        self._drizzle_check.setToolTip(
            "Use drizzle algorithm instead of standard stacking.\n"
            "Requires well-dithered frames for best results."
        )
        drizzle_layout.addWidget(self._drizzle_check)

        self._drizzle_scale_combo = QComboBox()
        self._drizzle_scale_combo.addItems(["2× (recommended)", "3×"])
        self._drizzle_scale_combo.setToolTip("Output scale factor (2× = double resolution)")
        drizzle_layout.addLayout(_h_row("Output Scale:", self._drizzle_scale_combo))

        self._drizzle_drop_spin = QDoubleSpinBox()
        self._drizzle_drop_spin.setRange(0.5, 1.0)
        self._drizzle_drop_spin.setValue(0.7)
        self._drizzle_drop_spin.setSingleStep(0.05)
        self._drizzle_drop_spin.setDecimals(2)
        self._drizzle_drop_spin.setToolTip(
            "Pixel fraction (drop shrink). 0.7 is standard; lower = sharper but noisier."
        )
        drizzle_layout.addLayout(_h_row("Drop Shrink:", self._drizzle_drop_spin))

        self._drizzle_check.toggled.connect(
            lambda en: (
                self._drizzle_scale_combo.setEnabled(en),
                self._drizzle_drop_spin.setEnabled(en),
            )
        )
        self._drizzle_scale_combo.setEnabled(False)
        self._drizzle_drop_spin.setEnabled(False)

        layout.addWidget(drizzle_group)

        # --- Batch Processing ---
        batch_group = QGroupBox("Batch Processing")
        batch_layout = QVBoxLayout(batch_group)
        batch_layout.addWidget(
            _info_label("Apply a processing pipeline to multiple images at once.")
        )
        self._btn_batch = QPushButton("Open Batch Dialog...")
        self._btn_batch.clicked.connect(self.open_batch_dialog.emit)
        batch_layout.addWidget(self._btn_batch)
        layout.addWidget(batch_group)

        layout.addStretch()
        self._tabs.addTab(_scrollable_tab(layout), "Stacking")

    # ================================================================
    # TAB 3: Stretch
    # ================================================================
    def _build_stretch_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- Auto-Stretch ---
        stretch_group = QGroupBox("Auto-Stretch")
        stretch_layout = QVBoxLayout(stretch_group)
        stretch_layout.addWidget(
            _info_label("Apply statistical midtone stretch to bring out faint details.")
        )

        self._midtone_slider = ResettableSlider(Qt.Orientation.Horizontal, default_value=25)
        self._midtone_slider.setRange(1, 99)
        self._midtone_slider.setValue(25)
        self._midtone_slider.setToolTip("Controls the brightness of midtones (lower = brighter). Double-click to reset.")
        self._midtone_value = QLabel("0.25")
        self._midtone_slider.valueChanged.connect(
            lambda v: self._midtone_value.setText(f"{v / 100:.2f}")
        )
        self._midtone_slider.valueChanged.connect(lambda: self.stretch_params_changed.emit())
        row = QHBoxLayout()
        row.addWidget(QLabel("Midtone:"))
        row.addWidget(self._midtone_slider)
        row.addWidget(self._midtone_value)
        stretch_layout.addLayout(row)

        self._shadow_spin = QDoubleSpinBox()
        self._shadow_spin.setRange(-10.0, 0.0)
        self._shadow_spin.setValue(-2.8)
        self._shadow_spin.setSingleStep(0.1)
        self._shadow_spin.setToolTip("Black point clipping in MAD units below median")
        self._shadow_spin.valueChanged.connect(lambda: self.stretch_params_changed.emit())
        stretch_layout.addLayout(_h_row("Shadow:", self._shadow_spin))

        self._linked_check = QCheckBox("Link RGB channels")
        self._linked_check.setChecked(True)
        self._linked_check.setToolTip(
            "Apply the same stretch to all channels (preserves color balance)"
        )
        self._linked_check.stateChanged.connect(lambda: self.stretch_params_changed.emit())
        stretch_layout.addWidget(self._linked_check)

        self._split_check = QCheckBox("Before/After split preview")
        self._split_check.setToolTip("Show original and stretched side by side")
        stretch_layout.addWidget(self._split_check)
        self._split_check.stateChanged.connect(lambda: self.stretch_params_changed.emit())

        stretch_btn_row = QHBoxLayout()
        self._btn_stretch = QPushButton("Apply Stretch")
        self._btn_stretch.setToolTip("Apply the stretch to the current image permanently")
        self._btn_stretch.clicked.connect(self.run_stretch.emit)
        stretch_btn_row.addWidget(self._btn_stretch)
        self._btn_stretch_reset = QPushButton("Reset")
        self._btn_stretch_reset.setToolTip("Reset midtone and shadow to defaults")
        self._btn_stretch_reset.clicked.connect(self.reset_stretch_params)
        stretch_btn_row.addWidget(self._btn_stretch_reset)
        stretch_layout.addLayout(stretch_btn_row)
        layout.addWidget(stretch_group)

        # --- GHS (Generalized Hyperbolic Stretch) ---
        ghs_group = QGroupBox("Generalized Hyperbolic Stretch")
        ghs_layout = QVBoxLayout(ghs_group)
        ghs_layout.addWidget(
            _info_label(
                "Advanced non-linear stretch with fine control over shadows, midtones, and highlights."
            )
        )

        self._ghs_d_spin = QDoubleSpinBox()
        self._ghs_d_spin.setRange(0.0, 20.0)
        self._ghs_d_spin.setValue(5.0)
        self._ghs_d_spin.setSingleStep(0.5)
        self._ghs_d_spin.setToolTip("Stretch factor (0 = no stretch)")
        ghs_layout.addLayout(_h_row("Stretch (D):", self._ghs_d_spin))

        self._ghs_b_spin = QDoubleSpinBox()
        self._ghs_b_spin.setRange(-5.0, 5.0)
        self._ghs_b_spin.setValue(0.0)
        self._ghs_b_spin.setSingleStep(0.1)
        self._ghs_b_spin.setToolTip("Asymmetry (-5 to 5, 0 = symmetric)")
        ghs_layout.addLayout(_h_row("Asymmetry (b):", self._ghs_b_spin))

        self._ghs_sp_spin = QDoubleSpinBox()
        self._ghs_sp_spin.setRange(0.0, 1.0)
        self._ghs_sp_spin.setValue(0.0)
        self._ghs_sp_spin.setSingleStep(0.05)
        self._ghs_sp_spin.setDecimals(3)
        self._ghs_sp_spin.setToolTip("Symmetry point (center of stretch)")
        ghs_layout.addLayout(_h_row("Sym. point:", self._ghs_sp_spin))

        self._ghs_shadow_slider = ResettableSlider(Qt.Orientation.Horizontal, default_value=0)
        self._ghs_shadow_slider.setRange(0, 100)
        self._ghs_shadow_slider.setValue(0)
        self._ghs_shadow_slider.setToolTip("Double-click to reset to 0")
        self._ghs_shadow_label = QLabel("0.00")
        self._ghs_shadow_slider.valueChanged.connect(
            lambda v: self._ghs_shadow_label.setText(f"{v / 100:.2f}")
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Shadow prot.:"))
        row.addWidget(self._ghs_shadow_slider)
        row.addWidget(self._ghs_shadow_label)
        ghs_layout.addLayout(row)

        self._ghs_highlight_slider = ResettableSlider(Qt.Orientation.Horizontal, default_value=0)
        self._ghs_highlight_slider.setRange(0, 100)
        self._ghs_highlight_slider.setValue(0)
        self._ghs_highlight_slider.setToolTip("Double-click to reset to 0")
        self._ghs_highlight_label = QLabel("0.00")
        self._ghs_highlight_slider.valueChanged.connect(
            lambda v: self._ghs_highlight_label.setText(f"{v / 100:.2f}")
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Highlight prot.:"))
        row.addWidget(self._ghs_highlight_slider)
        row.addWidget(self._ghs_highlight_label)
        ghs_layout.addLayout(row)

        ghs_btn_row = QHBoxLayout()
        self._btn_ghs = QPushButton("Apply GHS")
        self._btn_ghs.clicked.connect(self.run_ghs.emit)
        ghs_btn_row.addWidget(self._btn_ghs)
        self._btn_ghs_reset = QPushButton("Reset")
        self._btn_ghs_reset.setToolTip("Reset to default values")
        self._btn_ghs_reset.clicked.connect(self.reset_ghs_params)
        ghs_btn_row.addWidget(self._btn_ghs_reset)
        ghs_layout.addLayout(ghs_btn_row)

        self._add_preview_checkbox(ghs_layout, "ghs")
        self._ghs_d_spin.valueChanged.connect(lambda: self._emit_if_preview_enabled("ghs"))
        self._ghs_b_spin.valueChanged.connect(lambda: self._emit_if_preview_enabled("ghs"))
        self._ghs_sp_spin.valueChanged.connect(lambda: self._emit_if_preview_enabled("ghs"))
        self._ghs_shadow_slider.valueChanged.connect(lambda: self._emit_if_preview_enabled("ghs"))
        self._ghs_highlight_slider.valueChanged.connect(
            lambda: self._emit_if_preview_enabled("ghs")
        )

        layout.addWidget(ghs_group)

        # --- Histogram Transform ---
        ht_group = QGroupBox("Histogram Transform")
        ht_layout = QVBoxLayout(ht_group)
        ht_layout.addWidget(
            _info_label("Interactive black point, midtone, and white point adjustment.")
        )

        self._ht_black_spin = QDoubleSpinBox()
        self._ht_black_spin.setRange(0.0, 0.99)
        self._ht_black_spin.setValue(0.0)
        self._ht_black_spin.setSingleStep(0.01)
        self._ht_black_spin.setDecimals(3)
        self._ht_black_spin.setToolTip("Shadow clipping point")
        ht_layout.addLayout(_h_row("Black point:", self._ht_black_spin))

        self._ht_midtone_slider = ResettableSlider(Qt.Orientation.Horizontal, default_value=50)
        self._ht_midtone_slider.setRange(1, 99)
        self._ht_midtone_slider.setValue(50)
        self._ht_midtone_slider.setToolTip("Double-click to reset to default (0.50)")
        self._ht_midtone_label = QLabel("0.50")
        self._ht_midtone_slider.valueChanged.connect(
            lambda v: self._ht_midtone_label.setText(f"{v / 100:.2f}")
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Midtone:"))
        row.addWidget(self._ht_midtone_slider)
        row.addWidget(self._ht_midtone_label)
        ht_layout.addLayout(row)

        self._ht_white_spin = QDoubleSpinBox()
        self._ht_white_spin.setRange(0.01, 1.0)
        self._ht_white_spin.setValue(1.0)
        self._ht_white_spin.setSingleStep(0.01)
        self._ht_white_spin.setDecimals(3)
        self._ht_white_spin.setToolTip("White point clipping")
        ht_layout.addLayout(_h_row("White point:", self._ht_white_spin))

        ht_btn_row = QHBoxLayout()
        self._btn_ht = QPushButton("Apply Histogram Transform")
        self._btn_ht.clicked.connect(self.run_histogram_transform.emit)
        ht_btn_row.addWidget(self._btn_ht)
        self._btn_ht_reset = QPushButton("Reset")
        self._btn_ht_reset.setToolTip("Reset to default values")
        self._btn_ht_reset.clicked.connect(self.reset_histogram_transform_params)
        ht_btn_row.addWidget(self._btn_ht_reset)
        ht_layout.addLayout(ht_btn_row)

        self._add_preview_checkbox(ht_layout, "histogram_transform")
        self._ht_black_spin.valueChanged.connect(
            lambda: self._emit_if_preview_enabled("histogram_transform")
        )
        self._ht_midtone_slider.valueChanged.connect(
            lambda: self._emit_if_preview_enabled("histogram_transform")
        )
        self._ht_white_spin.valueChanged.connect(
            lambda: self._emit_if_preview_enabled("histogram_transform")
        )

        layout.addWidget(ht_group)

        # --- Curves ---
        curves_group = QGroupBox("Curves")
        curves_layout = QVBoxLayout(curves_group)
        curves_layout.addWidget(
            _info_label("Click to add points, drag to adjust. Right-click to remove.")
        )

        self._curve_channel_combo = QComboBox()
        self._curve_channel_combo.addItems(["Master (L)", "Red", "Green", "Blue"])
        self._curve_channel_combo.currentIndexChanged.connect(self._on_curve_channel_changed)
        self._curve_channel_combo.currentIndexChanged.connect(
            lambda: self.curves_histogram_changed.emit()
        )
        curves_layout.addLayout(_h_row("Channel:", self._curve_channel_combo))

        self._curve_editor = CurveEditor()
        self._curve_editor.setMinimumHeight(180)
        self._curve_editor.curve_changed.connect(self._on_curves_changed)
        curves_layout.addWidget(self._curve_editor)

        self._add_preview_checkbox(curves_layout, "curves")

        self._curves_histogram_check = QCheckBox("Show histogram")
        self._curves_histogram_check.setToolTip("Display image histogram behind curve")
        self._curves_histogram_check.stateChanged.connect(
            lambda: self.curves_histogram_changed.emit()
        )
        curves_layout.addWidget(self._curves_histogram_check)

        btn_row = QHBoxLayout()
        self._btn_curves_apply = QPushButton("Apply Curves")
        self._btn_curves_apply.clicked.connect(self.run_curves.emit)
        btn_row.addWidget(self._btn_curves_apply)
        self._btn_curves_reset = QPushButton("Reset")
        self._btn_curves_reset.clicked.connect(self._curve_editor.reset)
        btn_row.addWidget(self._btn_curves_reset)
        curves_layout.addLayout(btn_row)

        layout.addWidget(curves_group)

        layout.addStretch()
        self._tabs.addTab(_scrollable_tab(layout), "Stretch")

    # ================================================================
    # TAB 4: Background
    # ================================================================
    def _build_background_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- Background Extraction ---
        bg_group = QGroupBox("Background Extraction")
        bg_layout = QVBoxLayout(bg_group)
        bg_layout.addWidget(
            _info_label(
                "Remove light pollution gradients by fitting and subtracting a background model."
            )
        )

        self._bg_grid_spin = QSpinBox()
        self._bg_grid_spin.setRange(4, 32)
        self._bg_grid_spin.setValue(8)
        self._bg_grid_spin.setToolTip("Number of sample points per axis for background measurement")
        bg_layout.addLayout(_h_row("Grid size:", self._bg_grid_spin))

        self._bg_order_spin = QSpinBox()
        self._bg_order_spin.setRange(1, 6)
        self._bg_order_spin.setValue(3)
        self._bg_order_spin.setToolTip("Polynomial degree for the background surface model")
        bg_layout.addLayout(_h_row("Poly order:", self._bg_order_spin))

        # Interactive sample placement
        self._btn_place_samples = QPushButton("Place Samples")
        self._btn_place_samples.setCheckable(True)
        self._btn_place_samples.setToolTip(
            "Click on the image to place background sample points.\n"
            "Left-click adds, right-click removes nearest."
        )
        self._btn_place_samples.toggled.connect(self.toggle_sample_mode.emit)
        bg_layout.addWidget(self._btn_place_samples)

        self._bg_sample_label = QLabel("0 manual samples")
        self._bg_sample_label.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        bg_layout.addWidget(self._bg_sample_label)

        btn_clear_samples = QPushButton("Clear Samples")
        btn_clear_samples.setToolTip("Remove all manually placed sample points")
        btn_clear_samples.clicked.connect(self.clear_bg_samples.emit)
        bg_layout.addWidget(btn_clear_samples)

        self._btn_background = QPushButton("Extract Background")
        self._btn_background.setToolTip("Compute and subtract the background gradient")
        self._btn_background.clicked.connect(self.run_background.emit)
        bg_layout.addWidget(self._btn_background)
        layout.addWidget(bg_group)

        # --- ABE (RBF) ---
        abe_group = QGroupBox("ABE (Advanced)")
        abe_layout = QVBoxLayout(abe_group)
        abe_layout.addWidget(
            _info_label("RBF-based background extraction — handles complex LP gradients better.")
        )

        self._abe_grid_spin = QSpinBox()
        self._abe_grid_spin.setRange(5, 20)
        self._abe_grid_spin.setValue(10)
        abe_layout.addLayout(_h_row("Grid size:", self._abe_grid_spin))

        self._abe_kernel_combo = QComboBox()
        self._abe_kernel_combo.addItems(["Thin Plate Spline", "Multiquadric", "Gaussian"])
        abe_layout.addLayout(_h_row("Kernel:", self._abe_kernel_combo))

        self._abe_smoothing_spin = QDoubleSpinBox()
        self._abe_smoothing_spin.setRange(0.0, 5.0)
        self._abe_smoothing_spin.setValue(0.5)
        self._abe_smoothing_spin.setSingleStep(0.1)
        abe_layout.addLayout(_h_row("Smoothing:", self._abe_smoothing_spin))

        self._abe_mode_combo = QComboBox()
        self._abe_mode_combo.addItems(["Subtraction", "Division"])
        abe_layout.addLayout(_h_row("Mode:", self._abe_mode_combo))

        btn_abe = QPushButton("Run ABE")
        btn_abe.clicked.connect(self.run_abe.emit)
        abe_layout.addWidget(btn_abe)
        layout.addWidget(abe_group)

        # --- Vignette Correction ---
        vig_group = QGroupBox("Vignette Correction")
        vig_layout = QVBoxLayout(vig_group)
        vig_layout.addWidget(_info_label("Synthetic flat field for uncalibrated images."))

        self._vig_strength_spin = QDoubleSpinBox()
        self._vig_strength_spin.setRange(0.0, 2.0)
        self._vig_strength_spin.setValue(1.0)
        self._vig_strength_spin.setSingleStep(0.1)
        vig_layout.addLayout(_h_row("Strength:", self._vig_strength_spin))

        self._vig_falloff_spin = QDoubleSpinBox()
        self._vig_falloff_spin.setRange(0.5, 5.0)
        self._vig_falloff_spin.setValue(2.0)
        self._vig_falloff_spin.setSingleStep(0.1)
        vig_layout.addLayout(_h_row("Falloff:", self._vig_falloff_spin))

        btn_vig = QPushButton("Correct Vignette")
        btn_vig.clicked.connect(self.run_vignette_correction.emit)
        vig_layout.addWidget(btn_vig)
        layout.addWidget(vig_group)

        # --- Banding Reduction ---
        band_group = QGroupBox("Banding Reduction")
        band_layout = QVBoxLayout(band_group)
        band_layout.addWidget(
            _info_label("Remove horizontal/vertical banding artifacts common in CMOS sensors.")
        )

        self._band_h_check = QCheckBox("Horizontal banding")
        self._band_h_check.setChecked(True)
        band_layout.addWidget(self._band_h_check)

        self._band_v_check = QCheckBox("Vertical banding")
        band_layout.addWidget(self._band_v_check)

        self._band_amount_slider = QSlider(Qt.Orientation.Horizontal)
        self._band_amount_slider.setRange(0, 100)
        self._band_amount_slider.setValue(100)
        self._band_amount_label = QLabel("1.00")
        self._band_amount_slider.valueChanged.connect(
            lambda v: self._band_amount_label.setText(f"{v / 100:.2f}")
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Amount:"))
        row.addWidget(self._band_amount_slider)
        row.addWidget(self._band_amount_label)
        band_layout.addLayout(row)

        self._band_sigma_spin = QDoubleSpinBox()
        self._band_sigma_spin.setRange(1.0, 10.0)
        self._band_sigma_spin.setValue(3.0)
        self._band_sigma_spin.setSingleStep(0.5)
        self._band_sigma_spin.setToolTip("Sigma clipping to protect stars/objects from correction")
        band_layout.addLayout(_h_row("Protection:", self._band_sigma_spin))

        self._add_preview_checkbox(band_layout, "banding")
        self._btn_banding = QPushButton("Remove Banding")
        self._btn_banding.clicked.connect(self.run_banding.emit)
        band_layout.addWidget(self._btn_banding)
        layout.addWidget(band_group)

        layout.addStretch()
        self._tabs.addTab(_scrollable_tab(layout), "Background")

    # ================================================================
    # TAB 5: Transform
    # ================================================================
    # TAB 4: Color
    # ================================================================
    def _build_color_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- Color Calibration ---
        cc_group = QGroupBox("Color Calibration")
        cc_layout = QVBoxLayout(cc_group)
        cc_layout.addWidget(
            _info_label(
                "Calibrate white balance using star photometry and background neutralization."
            )
        )

        self._cc_reference_combo = QComboBox()
        self._cc_reference_combo.addItems(["Average Star", "G2V (Solar)", "Custom"])
        self._cc_reference_combo.setToolTip("White reference for calibration")
        cc_layout.addLayout(_h_row("Reference:", self._cc_reference_combo))

        self._cc_neutralize_bg = QCheckBox("Neutralize background")
        self._cc_neutralize_bg.setChecked(True)
        self._cc_neutralize_bg.setToolTip("Remove color cast from the background")
        cc_layout.addWidget(self._cc_neutralize_bg)

        self._cc_bg_percentile = QDoubleSpinBox()
        self._cc_bg_percentile.setRange(1.0, 50.0)
        self._cc_bg_percentile.setValue(10.0)
        self._cc_bg_percentile.setSingleStep(1.0)
        self._cc_bg_percentile.setToolTip("Use darkest N% of pixels for background reference")
        cc_layout.addLayout(_h_row("BG percentile:", self._cc_bg_percentile))

        self._btn_cc = QPushButton("Run Color Calibration")
        self._btn_cc.clicked.connect(self.run_color_calibration.emit)
        cc_layout.addWidget(self._btn_cc)
        layout.addWidget(cc_group)

        # --- Photometric Color Calibration (PCC) ---
        pcc_group = QGroupBox("Photometric Color Calibration (PCC)")
        pcc_layout = QVBoxLayout(pcc_group)
        pcc_layout.addWidget(
            _info_label(
                "Plate-solve and use Gaia DR3 star catalog for accurate color calibration. "
                "Requires ASTAP or astrometry.net."
            )
        )

        pcc_row1 = QHBoxLayout()
        pcc_row1.addWidget(QLabel("RA (deg):"))
        self._pcc_ra_spin = QDoubleSpinBox()
        self._pcc_ra_spin.setRange(0.0, 360.0)
        self._pcc_ra_spin.setValue(0.0)
        self._pcc_ra_spin.setDecimals(4)
        self._pcc_ra_spin.setToolTip("Right ascension hint in degrees (optional)")
        pcc_row1.addWidget(self._pcc_ra_spin)
        pcc_row1.addWidget(QLabel("Dec (deg):"))
        self._pcc_dec_spin = QDoubleSpinBox()
        self._pcc_dec_spin.setRange(-90.0, 90.0)
        self._pcc_dec_spin.setValue(0.0)
        self._pcc_dec_spin.setDecimals(4)
        self._pcc_dec_spin.setToolTip("Declination hint in degrees (optional)")
        pcc_row1.addWidget(self._pcc_dec_spin)
        pcc_layout.addLayout(pcc_row1)

        pcc_row2 = QHBoxLayout()
        pcc_row2.addWidget(QLabel("Solver:"))
        self._pcc_solver_combo = QComboBox()
        self._pcc_solver_combo.addItems(
            ["Auto (ASTAP → astrometry.net)", "ASTAP only", "astrometry.net only"]
        )
        self._pcc_solver_combo.setToolTip("Plate solver to use")
        pcc_row2.addWidget(self._pcc_solver_combo)
        pcc_layout.addLayout(pcc_row2)

        self._btn_pcc = QPushButton("Solve & Calibrate (PCC)")
        self._btn_pcc.setToolTip("Plate-solve and perform photometric color calibration")
        self._btn_pcc.clicked.connect(self.run_pcc.emit)
        pcc_layout.addWidget(self._btn_pcc)

        self._btn_wcs_overlay = QPushButton("Show WCS Overlay")
        self._btn_wcs_overlay.setCheckable(True)
        self._btn_wcs_overlay.setToolTip(
            "Display catalog star positions on the image after plate solving"
        )
        self._btn_wcs_overlay.toggled.connect(self.toggle_wcs_overlay.emit)
        pcc_layout.addWidget(self._btn_wcs_overlay)

        layout.addWidget(pcc_group)

        # --- SCNR ---
        scnr_group = QGroupBox("SCNR (Green Noise)")
        scnr_layout = QVBoxLayout(scnr_group)
        scnr_layout.addWidget(
            _info_label(
                "Remove excess green cast from narrowband composites or light-polluted images."
            )
        )

        self._scnr_method_combo = QComboBox()
        self._scnr_method_combo.addItems(["Average Neutral", "Maximum Neutral"])
        self._scnr_method_combo.setToolTip("Method to compute neutral green reference")
        scnr_layout.addLayout(_h_row("Method:", self._scnr_method_combo))

        self._scnr_amount_slider = QSlider(Qt.Orientation.Horizontal)
        self._scnr_amount_slider.setRange(0, 100)
        self._scnr_amount_slider.setValue(100)
        self._scnr_amount_label = QLabel("1.00")
        self._scnr_amount_slider.valueChanged.connect(
            lambda v: self._scnr_amount_label.setText(f"{v / 100:.2f}")
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Amount:"))
        row.addWidget(self._scnr_amount_slider)
        row.addWidget(self._scnr_amount_label)
        scnr_layout.addLayout(row)

        self._scnr_preserve_lum = QCheckBox("Preserve luminance")
        self._scnr_preserve_lum.setChecked(True)
        scnr_layout.addWidget(self._scnr_preserve_lum)

        self._add_preview_checkbox(scnr_layout, "scnr")
        self._btn_scnr = QPushButton("Apply SCNR")
        self._btn_scnr.clicked.connect(self.run_scnr.emit)
        scnr_layout.addWidget(self._btn_scnr)
        layout.addWidget(scnr_group)

        # --- Color Adjustment ---
        color_group = QGroupBox("Color Adjustment")
        color_layout = QVBoxLayout(color_group)
        color_layout.addWidget(_info_label("Adjust saturation, hue, and vibrance of the image."))

        self._saturation_slider = QSlider(Qt.Orientation.Horizontal)
        self._saturation_slider.setRange(0, 300)
        self._saturation_slider.setValue(100)
        self._saturation_label = QLabel("1.00")
        self._saturation_slider.valueChanged.connect(
            lambda v: self._saturation_label.setText(f"{v / 100:.2f}")
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Saturation:"))
        row.addWidget(self._saturation_slider)
        row.addWidget(self._saturation_label)
        color_layout.addLayout(row)

        self._hue_slider = QSlider(Qt.Orientation.Horizontal)
        self._hue_slider.setRange(-180, 180)
        self._hue_slider.setValue(0)
        self._hue_label = QLabel("0\u00b0")
        self._hue_slider.valueChanged.connect(lambda v: self._hue_label.setText(f"{v}\u00b0"))
        row = QHBoxLayout()
        row.addWidget(QLabel("Hue shift:"))
        row.addWidget(self._hue_slider)
        row.addWidget(self._hue_label)
        color_layout.addLayout(row)

        self._vibrance_slider = QSlider(Qt.Orientation.Horizontal)
        self._vibrance_slider.setRange(0, 100)
        self._vibrance_slider.setValue(0)
        self._vibrance_label = QLabel("0.00")
        self._vibrance_slider.valueChanged.connect(
            lambda v: self._vibrance_label.setText(f"{v / 100:.2f}")
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Vibrance:"))
        row.addWidget(self._vibrance_slider)
        row.addWidget(self._vibrance_label)
        color_layout.addLayout(row)

        self._add_preview_checkbox(color_layout, "color_adjust")
        self._btn_color = QPushButton("Apply Color Adjustment")
        self._btn_color.clicked.connect(self.run_color_adjust.emit)
        color_layout.addWidget(self._btn_color)
        layout.addWidget(color_group)

        layout.addStretch()
        self._tabs.addTab(_scrollable_tab(layout), "Color")

    # ================================================================
    # TAB 5: Detail
    # ================================================================
    def _build_detail_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- Deconvolution ---
        decon_group = QGroupBox("Deconvolution (Richardson-Lucy)")
        decon_layout = QVBoxLayout(decon_group)
        decon_layout.addWidget(
            _info_label(
                "Sharpen images by reversing atmospheric and optical blurring using iterative deconvolution."
            )
        )

        self._decon_fwhm_spin = QDoubleSpinBox()
        self._decon_fwhm_spin.setRange(0.5, 20.0)
        self._decon_fwhm_spin.setValue(3.0)
        self._decon_fwhm_spin.setSingleStep(0.1)
        self._decon_fwhm_spin.setToolTip("PSF full width at half maximum in pixels")

        fwhm_row = QHBoxLayout()
        fwhm_row.addWidget(QLabel("PSF FWHM:"))
        fwhm_row.addWidget(self._decon_fwhm_spin)
        self._btn_measure_psf = QPushButton("Measure")
        self._btn_measure_psf.setToolTip(
            "Automatically measure PSF FWHM from stars in the current image"
        )
        self._btn_measure_psf.clicked.connect(self.measure_psf.emit)
        fwhm_row.addWidget(self._btn_measure_psf)
        decon_layout.addLayout(fwhm_row)

        self._psf_result_label = QLabel("")
        self._psf_result_label.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        decon_layout.addWidget(self._psf_result_label)

        self._decon_iter_spin = QSpinBox()
        self._decon_iter_spin.setRange(1, 500)
        self._decon_iter_spin.setValue(50)
        self._decon_iter_spin.setToolTip("Number of RL iterations (more = sharper, but slower)")
        decon_layout.addLayout(_h_row("Iterations:", self._decon_iter_spin))

        self._decon_reg_slider = QSlider(Qt.Orientation.Horizontal)
        self._decon_reg_slider.setRange(0, 100)
        self._decon_reg_slider.setValue(10)
        self._decon_reg_label = QLabel("0.001")
        self._decon_reg_slider.valueChanged.connect(
            lambda v: self._decon_reg_label.setText(f"{v / 10000:.4f}")
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Regularization:"))
        row.addWidget(self._decon_reg_slider)
        row.addWidget(self._decon_reg_label)
        decon_layout.addLayout(row)

        self._decon_dering_check = QCheckBox("Deringing protection")
        self._decon_dering_check.setChecked(True)
        self._decon_dering_check.setToolTip("Reduce ringing artifacts around bright stars")
        decon_layout.addWidget(self._decon_dering_check)

        self._decon_dering_amount = QDoubleSpinBox()
        self._decon_dering_amount.setRange(0.0, 1.0)
        self._decon_dering_amount.setValue(0.5)
        self._decon_dering_amount.setSingleStep(0.1)
        decon_layout.addLayout(_h_row("Dering amount:", self._decon_dering_amount))

        self._decon_spatial_check = QCheckBox("Spatially-varying PSF (zone-based)")
        self._decon_spatial_check.setChecked(False)
        self._decon_spatial_check.setToolTip(
            "Measure PSF separately in 3x3 grid zones and deconvolve each zone "
            "with its local PSF. Handles field curvature and coma at edges. "
            "Requires sufficient stars across the image."
        )
        decon_layout.addWidget(self._decon_spatial_check)

        self._add_preview_checkbox(decon_layout, "deconvolution")
        self._btn_decon = QPushButton("Run Deconvolution")
        self._btn_decon.setToolTip("GPU-accelerated Richardson-Lucy deconvolution")
        self._btn_decon.clicked.connect(self.run_deconvolution.emit)
        decon_layout.addWidget(self._btn_decon)
        layout.addWidget(decon_group)

        # --- Noise Reduction ---
        nr_group = QGroupBox("Noise Reduction")
        nr_layout = QVBoxLayout(nr_group)
        nr_layout.addWidget(
            _info_label("Reduce noise using Non-Local Means or wavelet thresholding.")
        )

        self._nr_method_combo = QComboBox()
        self._nr_method_combo.addItems(["Wavelet", "Non-Local Means"])
        self._nr_method_combo.setToolTip("Wavelet preserves more structure; NLM is faster")
        nr_layout.addLayout(_h_row("Method:", self._nr_method_combo))

        self._nr_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self._nr_strength_slider.setRange(0, 100)
        self._nr_strength_slider.setValue(50)
        self._nr_strength_label = QLabel("0.50")
        self._nr_strength_slider.valueChanged.connect(
            lambda v: self._nr_strength_label.setText(f"{v / 100:.2f}")
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Strength:"))
        row.addWidget(self._nr_strength_slider)
        row.addWidget(self._nr_strength_label)
        nr_layout.addLayout(row)

        self._nr_detail_slider = QSlider(Qt.Orientation.Horizontal)
        self._nr_detail_slider.setRange(0, 100)
        self._nr_detail_slider.setValue(50)
        self._nr_detail_label = QLabel("0.50")
        self._nr_detail_slider.valueChanged.connect(
            lambda v: self._nr_detail_label.setText(f"{v / 100:.2f}")
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Detail:"))
        row.addWidget(self._nr_detail_slider)
        row.addWidget(self._nr_detail_label)
        nr_layout.addLayout(row)

        self._nr_chrom_check = QCheckBox("Chrominance only")
        self._nr_chrom_check.setToolTip("Only reduce color noise, preserving luminance detail")
        nr_layout.addWidget(self._nr_chrom_check)

        self._add_preview_checkbox(nr_layout, "denoise")
        self._btn_denoise = QPushButton("Reduce Noise")
        self._btn_denoise.clicked.connect(self.run_denoise.emit)
        nr_layout.addWidget(self._btn_denoise)
        layout.addWidget(nr_group)

        # --- Star Reduction ---
        sr_group = QGroupBox("Star Reduction")
        sr_layout = QVBoxLayout(sr_group)
        sr_layout.addWidget(
            _info_label(
                "Reduce star sizes using morphological erosion within an auto-generated star mask."
            )
        )

        self._sr_amount_slider = QSlider(Qt.Orientation.Horizontal)
        self._sr_amount_slider.setRange(0, 100)
        self._sr_amount_slider.setValue(50)
        self._sr_amount_label = QLabel("0.50")
        self._sr_amount_slider.valueChanged.connect(
            lambda v: self._sr_amount_label.setText(f"{v / 100:.2f}")
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Amount:"))
        row.addWidget(self._sr_amount_slider)
        row.addWidget(self._sr_amount_label)
        sr_layout.addLayout(row)

        self._sr_iterations_spin = QSpinBox()
        self._sr_iterations_spin.setRange(1, 10)
        self._sr_iterations_spin.setValue(2)
        self._sr_iterations_spin.setToolTip("Erosion iterations (more = smaller stars)")
        sr_layout.addLayout(_h_row("Iterations:", self._sr_iterations_spin))

        self._sr_kernel_spin = QSpinBox()
        self._sr_kernel_spin.setRange(3, 11)
        self._sr_kernel_spin.setValue(3)
        self._sr_kernel_spin.setSingleStep(2)
        self._sr_kernel_spin.setToolTip("Erosion kernel size")
        sr_layout.addLayout(_h_row("Kernel:", self._sr_kernel_spin))

        self._btn_star_reduce = QPushButton("Reduce Stars")
        self._btn_star_reduce.clicked.connect(self.run_star_reduction.emit)
        sr_layout.addWidget(self._btn_star_reduce)
        layout.addWidget(sr_group)

        # --- Wavelet Sharpening ---
        wav_group = QGroupBox("Wavelet Sharpening")
        wav_layout = QVBoxLayout(wav_group)
        wav_layout.addWidget(
            _info_label(
                "GPU-accelerated wavelet decomposition. Adjust per-scale weights to sharpen or smooth."
            )
        )

        self._wav_scales_spin = QSpinBox()
        self._wav_scales_spin.setRange(2, 8)
        self._wav_scales_spin.setValue(4)
        self._wav_scales_spin.setToolTip("Number of wavelet decomposition scales")
        wav_layout.addLayout(_h_row("Scales:", self._wav_scales_spin))

        self._wav_fine_slider = QSlider(Qt.Orientation.Horizontal)
        self._wav_fine_slider.setRange(0, 300)
        self._wav_fine_slider.setValue(150)
        self._wav_fine_label = QLabel("1.50")
        self._wav_fine_slider.valueChanged.connect(
            lambda v: self._wav_fine_label.setText(f"{v / 100:.2f}")
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Fine detail:"))
        row.addWidget(self._wav_fine_slider)
        row.addWidget(self._wav_fine_label)
        wav_layout.addLayout(row)

        self._wav_medium_slider = QSlider(Qt.Orientation.Horizontal)
        self._wav_medium_slider.setRange(0, 300)
        self._wav_medium_slider.setValue(120)
        self._wav_medium_label = QLabel("1.20")
        self._wav_medium_slider.valueChanged.connect(
            lambda v: self._wav_medium_label.setText(f"{v / 100:.2f}")
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Medium:"))
        row.addWidget(self._wav_medium_slider)
        row.addWidget(self._wav_medium_label)
        wav_layout.addLayout(row)

        self._wav_coarse_slider = QSlider(Qt.Orientation.Horizontal)
        self._wav_coarse_slider.setRange(0, 300)
        self._wav_coarse_slider.setValue(100)
        self._wav_coarse_label = QLabel("1.00")
        self._wav_coarse_slider.valueChanged.connect(
            lambda v: self._wav_coarse_label.setText(f"{v / 100:.2f}")
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Coarse:"))
        row.addWidget(self._wav_coarse_slider)
        row.addWidget(self._wav_coarse_label)
        wav_layout.addLayout(row)

        self._add_preview_checkbox(wav_layout, "wavelet")
        self._btn_wavelet = QPushButton("Apply Wavelet Sharpening")
        self._btn_wavelet.setToolTip("GPU-accelerated wavelet transform")
        self._btn_wavelet.clicked.connect(self.run_wavelet_sharpen.emit)
        wav_layout.addWidget(self._btn_wavelet)
        layout.addWidget(wav_group)

        # --- MLT (Multi-Scale Linear Transform) ---
        mlt_group = QGroupBox("MLT (Multi-Scale Linear Transform)")
        mlt_layout = QVBoxLayout(mlt_group)
        mlt_layout.addWidget(
            _info_label(
                "Full per-band control with noise thresholding. "
                "Scale 1 = finest detail (stars/noise). Scale 6 = large structures."
            )
        )

        self._mlt_sliders: list[tuple[QSlider, QLabel, QSlider, QLabel]] = []
        band_names = ["Scale 1 (finest)", "Scale 2", "Scale 3", "Scale 4", "Scale 5", "Scale 6 (coarsest)"]

        for i, name in enumerate(band_names):
            band_box = QGroupBox(name)
            band_box.setStyleSheet("QGroupBox { font-size: 10px; }")
            band_layout = QVBoxLayout(band_box)
            band_layout.setSpacing(2)
            band_layout.setContentsMargins(6, 12, 6, 4)

            # Boost slider
            boost_row = QHBoxLayout()
            boost_row.addWidget(QLabel("Boost:"))
            boost_sl = QSlider(Qt.Orientation.Horizontal)
            boost_sl.setRange(0, 400)
            boost_sl.setValue(100)
            boost_lbl = QLabel("1.00")
            boost_sl.valueChanged.connect(lambda v, lbl=boost_lbl: lbl.setText(f"{v/100:.2f}"))
            boost_row.addWidget(boost_sl)
            boost_row.addWidget(boost_lbl)
            band_layout.addLayout(boost_row)

            # Threshold slider
            thr_row = QHBoxLayout()
            thr_row.addWidget(QLabel("Denoise:"))
            thr_sl = QSlider(Qt.Orientation.Horizontal)
            thr_sl.setRange(0, 200)
            thr_sl.setValue(0)
            thr_lbl = QLabel("0.000")
            thr_sl.valueChanged.connect(lambda v, lbl=thr_lbl: lbl.setText(f"{v/10000:.4f}"))
            thr_row.addWidget(thr_sl)
            thr_row.addWidget(thr_lbl)
            band_layout.addLayout(thr_row)

            mlt_layout.addWidget(band_box)
            self._mlt_sliders.append((boost_sl, boost_lbl, thr_sl, thr_lbl))

        # Residual
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Residual weight:"))
        self._mlt_residual_spin = QDoubleSpinBox()
        self._mlt_residual_spin.setRange(0.0, 2.0)
        self._mlt_residual_spin.setValue(1.0)
        self._mlt_residual_spin.setSingleStep(0.05)
        res_row.addWidget(self._mlt_residual_spin)
        mlt_layout.addLayout(res_row)

        self._add_preview_checkbox(mlt_layout, "mlt")
        btn_mlt = QPushButton("Apply MLT")
        btn_mlt.setToolTip("Apply multi-scale linear transform")
        btn_mlt.clicked.connect(self.run_mlt.emit)
        mlt_layout.addWidget(btn_mlt)
        layout.addWidget(mlt_group)

        # --- Local Contrast ---
        lc_group = QGroupBox("Local Contrast (CLAHE)")
        lc_layout = QVBoxLayout(lc_group)
        lc_layout.addWidget(_info_label("Enhance local contrast using CLAHE on luminance channel."))

        self._lc_clip_spin = QDoubleSpinBox()
        self._lc_clip_spin.setRange(1.0, 10.0)
        self._lc_clip_spin.setValue(2.0)
        self._lc_clip_spin.setSingleStep(0.5)
        self._lc_clip_spin.setToolTip("CLAHE clip limit (higher = more contrast)")
        lc_layout.addLayout(_h_row("Clip limit:", self._lc_clip_spin))

        self._lc_tile_spin = QSpinBox()
        self._lc_tile_spin.setRange(4, 32)
        self._lc_tile_spin.setValue(8)
        self._lc_tile_spin.setToolTip("Tile grid size for local histogram")
        lc_layout.addLayout(_h_row("Tile size:", self._lc_tile_spin))

        self._lc_amount_slider = QSlider(Qt.Orientation.Horizontal)
        self._lc_amount_slider.setRange(0, 100)
        self._lc_amount_slider.setValue(100)
        self._lc_amount_label = QLabel("1.00")
        self._lc_amount_slider.valueChanged.connect(
            lambda v: self._lc_amount_label.setText(f"{v / 100:.2f}")
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Amount:"))
        row.addWidget(self._lc_amount_slider)
        row.addWidget(self._lc_amount_label)
        lc_layout.addLayout(row)

        self._add_preview_checkbox(lc_layout, "local_contrast")
        self._btn_local_contrast = QPushButton("Apply Local Contrast")
        self._btn_local_contrast.clicked.connect(self.run_local_contrast.emit)
        lc_layout.addWidget(self._btn_local_contrast)
        layout.addWidget(lc_group)

        # --- Morphology ---
        morph_group = QGroupBox("Morphological Operations")
        morph_layout = QVBoxLayout(morph_group)
        morph_layout.addWidget(
            _info_label("Apply morphological operations for star shaping and mask refinement.")
        )

        self._morph_op_combo = QComboBox()
        self._morph_op_combo.addItems(["Dilate", "Erode", "Open", "Close"])
        morph_layout.addLayout(_h_row("Operation:", self._morph_op_combo))

        self._morph_element_combo = QComboBox()
        self._morph_element_combo.addItems(["Circle", "Square", "Diamond"])
        morph_layout.addLayout(_h_row("Element:", self._morph_element_combo))

        self._morph_kernel_spin = QSpinBox()
        self._morph_kernel_spin.setRange(3, 21)
        self._morph_kernel_spin.setValue(3)
        self._morph_kernel_spin.setSingleStep(2)
        morph_layout.addLayout(_h_row("Kernel:", self._morph_kernel_spin))

        self._morph_iter_spin = QSpinBox()
        self._morph_iter_spin.setRange(1, 10)
        self._morph_iter_spin.setValue(1)
        morph_layout.addLayout(_h_row("Iterations:", self._morph_iter_spin))

        self._btn_morphology = QPushButton("Apply Morphology")
        self._btn_morphology.clicked.connect(self.run_morphology.emit)
        morph_layout.addWidget(self._btn_morphology)
        layout.addWidget(morph_group)

        # --- Unsharp Mask ---
        usm_group = QGroupBox("Unsharp Mask")
        usm_layout = QVBoxLayout(usm_group)
        usm_layout.addWidget(_info_label("Classic sharpening with radius, amount, and threshold."))

        self._usm_radius_spin = QDoubleSpinBox()
        self._usm_radius_spin.setRange(0.5, 20.0)
        self._usm_radius_spin.setValue(2.0)
        self._usm_radius_spin.setSingleStep(0.5)
        usm_layout.addLayout(_h_row("Radius:", self._usm_radius_spin))

        self._usm_amount_slider = QSlider(Qt.Orientation.Horizontal)
        self._usm_amount_slider.setRange(0, 200)
        self._usm_amount_slider.setValue(50)
        self._usm_amount_label = QLabel("0.50")
        self._usm_amount_slider.valueChanged.connect(
            lambda v: self._usm_amount_label.setText(f"{v / 100:.2f}")
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Amount:"))
        row.addWidget(self._usm_amount_slider)
        row.addWidget(self._usm_amount_label)
        usm_layout.addLayout(row)

        self._usm_threshold_spin = QDoubleSpinBox()
        self._usm_threshold_spin.setRange(0.0, 0.1)
        self._usm_threshold_spin.setValue(0.0)
        self._usm_threshold_spin.setSingleStep(0.005)
        self._usm_threshold_spin.setDecimals(3)
        usm_layout.addLayout(_h_row("Threshold:", self._usm_threshold_spin))

        self._add_preview_checkbox(usm_layout, "unsharp_mask")
        btn_usm = QPushButton("Apply Unsharp Mask")
        btn_usm.clicked.connect(self.run_unsharp_mask.emit)
        usm_layout.addWidget(btn_usm)
        layout.addWidget(usm_group)

        # --- Median Filter ---
        mf_group = QGroupBox("Median Filter")
        mf_layout = QVBoxLayout(mf_group)
        mf_layout.addWidget(_info_label("Noise reduction via median filtering."))

        self._mf_kernel_spin = QSpinBox()
        self._mf_kernel_spin.setRange(3, 15)
        self._mf_kernel_spin.setValue(3)
        self._mf_kernel_spin.setSingleStep(2)
        mf_layout.addLayout(_h_row("Kernel size:", self._mf_kernel_spin))

        self._add_preview_checkbox(mf_layout, "median_filter")
        btn_mf = QPushButton("Apply Median Filter")
        btn_mf.clicked.connect(self.run_median_filter.emit)
        mf_layout.addWidget(btn_mf)
        layout.addWidget(mf_group)

        layout.addStretch()
        self._tabs.addTab(_scrollable_tab(layout), "Detail")

    # ================================================================
    # TAB 6: Utility
    # ================================================================
    # ================================================================
    # TAB 6: AI Tools
    # ================================================================
    def _build_ai_pro_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- AI Denoise ---
        aid_group = QGroupBox("AI Denoise")
        aid_layout = QVBoxLayout(aid_group)
        aid_layout.addWidget(
            _info_label(
                "Deep learning noise reduction using a trained U-Net model. Requires Pro license."
            )
        )

        self._aid_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self._aid_strength_slider.setRange(0, 100)
        self._aid_strength_slider.setValue(100)
        self._aid_strength_label = QLabel("1.00")
        self._aid_strength_slider.valueChanged.connect(
            lambda v: self._aid_strength_label.setText(f"{v / 100:.2f}")
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Strength:"))
        row.addWidget(self._aid_strength_slider)
        row.addWidget(self._aid_strength_label)
        aid_layout.addLayout(row)

        self._btn_ai_denoise = QPushButton("Run AI Denoise")
        self._btn_ai_denoise.setToolTip("GPU-accelerated deep learning denoising")
        self._btn_ai_denoise.clicked.connect(self.run_ai_denoise.emit)
        aid_layout.addWidget(self._btn_ai_denoise)
        layout.addWidget(aid_group)

        # --- AI Sharpen ---
        ais_group = QGroupBox("AI Sharpen")
        ais_layout = QVBoxLayout(ais_group)
        ais_layout.addWidget(
            _info_label(
                "Deep learning sharpening using a trained U-Net model. Requires Pro license."
            )
        )

        self._ais_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self._ais_strength_slider.setRange(0, 100)
        self._ais_strength_slider.setValue(100)
        self._ais_strength_label = QLabel("1.00")
        self._ais_strength_slider.valueChanged.connect(
            lambda v: self._ais_strength_label.setText(f"{v / 100:.2f}")
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Strength:"))
        row.addWidget(self._ais_strength_slider)
        row.addWidget(self._ais_strength_label)
        ais_layout.addLayout(row)

        self._btn_ai_sharpen = QPushButton("Run AI Sharpen")
        self._btn_ai_sharpen.setToolTip("GPU-accelerated deep learning sharpening")
        self._btn_ai_sharpen.clicked.connect(self.run_ai_sharpen.emit)
        ais_layout.addWidget(self._btn_ai_sharpen)
        layout.addWidget(ais_group)

        # --- StarNet Star Removal ---
        sn_group = QGroupBox("StarNet Star Removal")
        sn_layout = QVBoxLayout(sn_group)
        sn_layout.addWidget(
            _info_label(
                "Remove stars using StarNet++ (must be installed separately). "
                "Runs as isolated subprocess for GPL compliance."
            )
        )

        self._sn_extract_stars = QCheckBox("Extract stars-only image")
        self._sn_extract_stars.setChecked(True)
        self._sn_extract_stars.setToolTip("Also compute original - starless = stars only")
        sn_layout.addWidget(self._sn_extract_stars)

        self._btn_starnet = QPushButton("Run StarNet")
        self._btn_starnet.setToolTip("Requires StarNet++ binary in PATH")
        self._btn_starnet.clicked.connect(self.run_starnet.emit)
        sn_layout.addWidget(self._btn_starnet)
        layout.addWidget(sn_group)

        # --- Smart Processor ---
        sp_group = QGroupBox("Smart Processor")
        sp_layout = QVBoxLayout(sp_group)
        sp_layout.addWidget(
            _info_label(
                "AI-driven adaptive processing. Plate-solves your image, identifies "
                "the target, measures PSF, and builds an optimal processing pipeline "
                "with per-channel tuning and quality checks."
            )
        )

        self._btn_smart_processor = QPushButton("Open Smart Processor...")
        self._btn_smart_processor.setToolTip("Launch the Smart Processor dialog")
        self._btn_smart_processor.clicked.connect(self.open_smart_processor.emit)
        sp_layout.addWidget(self._btn_smart_processor)

        self._btn_equipment = QPushButton("Equipment Profile...")
        self._btn_equipment.setToolTip("Configure camera, telescope, and filters")
        self._btn_equipment.clicked.connect(self.open_equipment_dialog.emit)
        sp_layout.addWidget(self._btn_equipment)

        layout.addWidget(sp_group)

        layout.addStretch()
        self._tabs.addTab(_scrollable_tab(layout), "AI Tools")

    # ================================================================
    # TAB 7: Utility
    # ================================================================
    def _build_utility_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- Chromatic Aberration ---
        ca_group = QGroupBox("Chromatic Aberration")
        ca_layout = QVBoxLayout(ca_group)
        ca_layout.addWidget(_info_label("Detect and correct lateral color fringing."))

        self._ca_auto_check = QCheckBox("Auto-detect from stars")
        self._ca_auto_check.setChecked(True)
        ca_layout.addWidget(self._ca_auto_check)

        self._ca_red_x_spin = QDoubleSpinBox()
        self._ca_red_x_spin.setRange(-3.0, 3.0)
        self._ca_red_x_spin.setValue(0.0)
        self._ca_red_x_spin.setSingleStep(0.1)
        self._ca_red_x_spin.setDecimals(2)
        ca_layout.addLayout(_h_row("Red shift X:", self._ca_red_x_spin))

        self._ca_red_y_spin = QDoubleSpinBox()
        self._ca_red_y_spin.setRange(-3.0, 3.0)
        self._ca_red_y_spin.setValue(0.0)
        self._ca_red_y_spin.setSingleStep(0.1)
        self._ca_red_y_spin.setDecimals(2)
        ca_layout.addLayout(_h_row("Red shift Y:", self._ca_red_y_spin))

        self._ca_blue_x_spin = QDoubleSpinBox()
        self._ca_blue_x_spin.setRange(-3.0, 3.0)
        self._ca_blue_x_spin.setValue(0.0)
        self._ca_blue_x_spin.setSingleStep(0.1)
        self._ca_blue_x_spin.setDecimals(2)
        ca_layout.addLayout(_h_row("Blue shift X:", self._ca_blue_x_spin))

        self._ca_blue_y_spin = QDoubleSpinBox()
        self._ca_blue_y_spin.setRange(-3.0, 3.0)
        self._ca_blue_y_spin.setValue(0.0)
        self._ca_blue_y_spin.setSingleStep(0.1)
        self._ca_blue_y_spin.setDecimals(2)
        ca_layout.addLayout(_h_row("Blue shift Y:", self._ca_blue_y_spin))

        btn_ca = QPushButton("Correct CA")
        btn_ca.clicked.connect(self.run_chromatic_aberration.emit)
        ca_layout.addWidget(btn_ca)
        layout.addWidget(ca_group)

        # --- Quick tools row ---
        quick_group = QGroupBox("Tools")
        quick_layout = QVBoxLayout(quick_group)
        quick_row1 = QHBoxLayout()
        btn_stats = QPushButton("Image Statistics...")
        btn_stats.clicked.connect(self.show_image_statistics.emit)
        quick_row1.addWidget(btn_stats)
        btn_starmask = QPushButton("Star Mask...")
        btn_starmask.clicked.connect(self.open_star_mask_dialog.emit)
        quick_row1.addWidget(btn_starmask)
        quick_layout.addLayout(quick_row1)

        quick_row2 = QHBoxLayout()
        btn_console = QPushButton("Python Console...")
        btn_console.setToolTip("Open embedded Python scripting console")
        btn_console.clicked.connect(self.open_python_console.emit)
        quick_row2.addWidget(btn_console)
        quick_layout.addLayout(quick_row2)
        layout.addWidget(quick_group)

        # --- Narrowband ---
        nb_group = QGroupBox("Narrowband Combining")
        nb_layout = QVBoxLayout(nb_group)
        nb_layout.addWidget(
            _info_label(
                "Combine Ha, OIII, and SII images into a color composite using palette mapping."
            )
        )
        self._btn_narrowband = QPushButton("Open Narrowband Dialog...")
        self._btn_narrowband.clicked.connect(self.open_narrowband_dialog.emit)
        nb_layout.addWidget(self._btn_narrowband)
        layout.addWidget(nb_group)

        # --- Continuum Subtraction ---
        cont_group = QGroupBox("Continuum Subtraction")
        cont_layout = QVBoxLayout(cont_group)
        cont_layout.addWidget(
            _info_label(
                "Subtract broadband continuum from narrowband to isolate emission lines (Ha, OIII, SII)."
            )
        )

        self._cont_nb_combo = QComboBox()
        self._cont_nb_combo.addItems(["Current image as narrowband"])
        self._cont_nb_combo.setToolTip("Source for narrowband channel")
        cont_layout.addLayout(_h_row("Narrowband:", self._cont_nb_combo))

        self._cont_bb_combo = QComboBox()
        self._cont_bb_combo.addItems(["Open file…"])
        self._cont_bb_combo.setToolTip("Source for broadband (continuum) channel")
        cont_layout.addLayout(_h_row("Broadband:", self._cont_bb_combo))

        self._cont_scale_spin = QDoubleSpinBox()
        self._cont_scale_spin.setRange(0.01, 5.0)
        self._cont_scale_spin.setValue(1.0)
        self._cont_scale_spin.setSingleStep(0.05)
        self._cont_scale_spin.setDecimals(3)
        self._cont_scale_spin.setToolTip(
            "Scale factor applied to broadband before subtraction. "
            "Adjust to fully suppress stellar continuum."
        )
        cont_layout.addLayout(_h_row("Scale factor:", self._cont_scale_spin))

        self._btn_cont_subtract = QPushButton("Subtract Continuum…")
        self._btn_cont_subtract.setToolTip("Load broadband file and subtract from current image")
        self._btn_cont_subtract.clicked.connect(self.run_continuum_subtraction.emit)
        cont_layout.addWidget(self._btn_cont_subtract)
        layout.addWidget(cont_group)

        # --- Pixel Math ---
        pm_group = QGroupBox("Pixel Math")
        pm_layout = QVBoxLayout(pm_group)
        pm_layout.addWidget(
            _info_label("Apply mathematical expressions to pixel data (T, R, G, B, L variables).")
        )
        self._btn_pixelmath = QPushButton("Open Pixel Math...")
        self._btn_pixelmath.clicked.connect(self.open_pixelmath_dialog.emit)
        pm_layout.addWidget(self._btn_pixelmath)
        layout.addWidget(pm_group)

        # --- Channels ---
        ch_group = QGroupBox("Channel Operations")
        ch_layout = QVBoxLayout(ch_group)
        ch_layout.addWidget(
            _info_label("Split, extract, and manipulate individual image channels.")
        )

        btn_row = QHBoxLayout()
        self._btn_split_ch = QPushButton("Split Channels")
        self._btn_split_ch.setToolTip("Split color image into separate R, G, B mono images")
        self._btn_split_ch.clicked.connect(self.run_split_channels.emit)
        btn_row.addWidget(self._btn_split_ch)

        self._btn_extract_lum = QPushButton("Extract Luminance")
        self._btn_extract_lum.setToolTip("Extract weighted luminance from color image")
        self._btn_extract_lum.clicked.connect(self.run_extract_luminance.emit)
        btn_row.addWidget(self._btn_extract_lum)
        ch_layout.addLayout(btn_row)
        layout.addWidget(ch_group)

        # --- HDR Composition ---
        hdr_group = QGroupBox("HDR Composition")
        hdr_layout = QVBoxLayout(hdr_group)
        hdr_layout.addWidget(
            _info_label("Merge multiple exposures into a single high dynamic range image.")
        )
        self._btn_hdr = QPushButton("Open HDR Dialog...")
        self._btn_hdr.clicked.connect(self.open_hdr_dialog.emit)
        hdr_layout.addWidget(self._btn_hdr)
        layout.addWidget(hdr_group)

        # --- Macros ---
        macro_group = QGroupBox("Macros")
        macro_layout = QVBoxLayout(macro_group)
        macro_layout.addWidget(_info_label("Record processing steps and replay them on any image."))

        rec_row = QHBoxLayout()
        self._btn_macro_start = QPushButton("Record")
        self._btn_macro_start.setToolTip("Start recording processing steps")
        self._btn_macro_start.clicked.connect(self.start_macro_recording.emit)
        rec_row.addWidget(self._btn_macro_start)

        self._btn_macro_stop = QPushButton("Stop")
        self._btn_macro_stop.setToolTip("Stop recording and save macro")
        self._btn_macro_stop.clicked.connect(self.stop_macro_recording.emit)
        self._btn_macro_stop.setEnabled(False)
        rec_row.addWidget(self._btn_macro_stop)
        macro_layout.addLayout(rec_row)

        play_row = QHBoxLayout()
        self._btn_macro_play = QPushButton("Play")
        self._btn_macro_play.setToolTip("Replay the last recorded macro")
        self._btn_macro_play.clicked.connect(self.play_macro.emit)
        play_row.addWidget(self._btn_macro_play)

        self._btn_macro_save = QPushButton("Save...")
        self._btn_macro_save.clicked.connect(self.save_macro.emit)
        play_row.addWidget(self._btn_macro_save)

        self._btn_macro_load = QPushButton("Load...")
        self._btn_macro_load.clicked.connect(self.load_macro.emit)
        play_row.addWidget(self._btn_macro_load)
        macro_layout.addLayout(play_row)

        self._macro_status_label = QLabel("Not recording")
        self._macro_status_label.setStyleSheet("color: #969696; font-size: 11px;")
        macro_layout.addWidget(self._macro_status_label)
        layout.addWidget(macro_group)

        layout.addStretch()
        self._tabs.addTab(_scrollable_tab(layout), "Utility")

    # ================================================================
    # Parameter getters
    # ================================================================

    def get_drizzle_params(self):
        """Return (enabled, DrizzleParams) or (False, None)."""
        from cosmica.core.drizzle import DrizzleParams
        if not self._drizzle_check.isChecked():
            return False, None
        scale = 2 if self._drizzle_scale_combo.currentIndex() == 0 else 3
        return True, DrizzleParams(
            scale=scale,
            drop_shrink=self._drizzle_drop_spin.value(),
        )

    def get_stacking_params(self) -> StackingParams:
        rejection_map = {
            0: RejectionMethod.SIGMA_CLIP,
            1: RejectionMethod.WINSORIZED_SIGMA,
            2: RejectionMethod.LINEAR_FIT,
            3: RejectionMethod.PERCENTILE_CLIP,
            4: RejectionMethod.ESD,
            5: RejectionMethod.MIN_MAX,
            6: RejectionMethod.NONE,
        }
        integration_map = {
            0: IntegrationMethod.AVERAGE,
            1: IntegrationMethod.MEDIAN,
        }
        registration_map = {
            0: RegistrationMode.STAR_1_PASS,
            1: RegistrationMode.STAR_2_PASS,
            2: RegistrationMode.FFT_TRANSLATION,
        }
        kappa = self._kappa_spin.value()
        return StackingParams(
            rejection=rejection_map.get(
                self._rejection_combo.currentIndex(), RejectionMethod.SIGMA_CLIP
            ),
            integration=integration_map.get(
                self._integration_combo.currentIndex(), IntegrationMethod.AVERAGE
            ),
            registration_mode=registration_map.get(
                self._reg_mode_combo.currentIndex(), RegistrationMode.STAR_1_PASS
            ),
            kappa_low=kappa,
            kappa_high=kappa,
        )

    def get_quality_filter_params(self) -> dict | None:
        """Return quality filter config, or None if filter is disabled."""
        if not self._quality_filter_check.isChecked():
            return None
        metric_map = {0: "fwhm", 1: "snr", 2: "quality_score"}
        mode_map = {0: "top_n", 1: "top_percent", 2: "sigma"}
        return {
            "metric": metric_map.get(self._quality_metric_combo.currentIndex(), "quality_score"),
            "mode": mode_map.get(self._quality_mode_combo.currentIndex(), "sigma"),
            "top_n": self._quality_n_spin.value(),
            "top_percent": self._quality_percent_spin.value(),
            "rejection_sigma": self._quality_sigma_spin.value(),
        }

    def _on_quality_mode_changed(self, index: int):
        """Handle quality filter mode change - show/hide relevant spin boxes."""
        self._quality_n_spin.setEnabled(index == 0)
        self._quality_percent_spin.setEnabled(index == 1)
        self._quality_sigma_spin.setEnabled(index == 2)

    def _on_quality_filter_toggled(self, checked: bool):
        """Handle quality filter checkbox toggle."""
        self._quality_n_spin.setEnabled(checked and self._quality_mode_combo.currentIndex() == 0)
        self._quality_percent_spin.setEnabled(
            checked and self._quality_mode_combo.currentIndex() == 1
        )
        self._quality_sigma_spin.setEnabled(
            checked and self._quality_mode_combo.currentIndex() == 2
        )

    def get_alignment_params(self) -> dict:
        mode_map = {
            0: RegistrationMode.STAR_1_PASS,
            1: RegistrationMode.STAR_2_PASS,
            2: RegistrationMode.FFT_TRANSLATION,
        }
        return {
            "mode": mode_map.get(self._reg_mode_combo.currentIndex(), RegistrationMode.STAR_1_PASS),
            "star_sensitivity": self._star_sens_spin.value(),
            "max_match_distance": self._max_shift_spin.value(),
            "ransac_threshold": self._ransac_thresh_spin.value(),
        }

    def get_stretch_params(self) -> StretchParams:
        return StretchParams(
            shadow_clip=self._shadow_spin.value(),
            midtone=self._midtone_slider.value() / 100.0,
            linked=self._linked_check.isChecked(),
        )

    def reset_stretch_params(self):
        """Reset auto-stretch controls to defaults."""
        self._midtone_slider.blockSignals(True)
        self._shadow_spin.blockSignals(True)
        self._midtone_slider.setValue(25)
        self._shadow_spin.setValue(-2.8)
        self._midtone_slider.blockSignals(False)
        self._shadow_spin.blockSignals(False)

    def get_ghs_params(self) -> GHSParams:
        return GHSParams(
            D=self._ghs_d_spin.value(),
            b=self._ghs_b_spin.value(),
            SP=self._ghs_sp_spin.value(),
            shadow_protection=self._ghs_shadow_slider.value() / 100.0,
            highlight_protection=self._ghs_highlight_slider.value() / 100.0,
        )

    def reset_ghs_params(self):
        """Reset GHS controls to their defaults."""
        for widget in (
            self._ghs_d_spin,
            self._ghs_b_spin,
            self._ghs_sp_spin,
            self._ghs_shadow_slider,
            self._ghs_highlight_slider,
        ):
            widget.blockSignals(True)
        self._ghs_d_spin.setValue(5.0)
        self._ghs_b_spin.setValue(0.0)
        self._ghs_sp_spin.setValue(0.0)
        self._ghs_shadow_slider.setValue(0)
        self._ghs_highlight_slider.setValue(0)
        for widget in (
            self._ghs_d_spin,
            self._ghs_b_spin,
            self._ghs_sp_spin,
            self._ghs_shadow_slider,
            self._ghs_highlight_slider,
        ):
            widget.blockSignals(False)

    def get_curves_params(self) -> CurvesParams:
        return self._curve_editor.get_params()

    def get_background_params(self, manual_points: list | None = None) -> BackgroundParams:
        return BackgroundParams(
            grid_size=self._bg_grid_spin.value(),
            polynomial_order=self._bg_order_spin.value(),
            manual_points=manual_points or [],
        )

    def set_bg_sample_count(self, n: int):
        self._bg_sample_label.setText(f"{n} manual sample{'s' if n != 1 else ''}")

    def get_cosmetic_params(self) -> CosmeticParams:
        return CosmeticParams(
            hot_sigma=self._hot_sigma_spin.value(),
            cold_sigma=self._cold_sigma_spin.value(),
            detect_dead=self._dead_pixel_check.isChecked(),
        )

    def get_banding_params(self) -> BandingParams:
        return BandingParams(
            horizontal=self._band_h_check.isChecked(),
            vertical=self._band_v_check.isChecked(),
            amount=self._band_amount_slider.value() / 100.0,
            protection_sigma=self._band_sigma_spin.value(),
        )

    def get_histogram_transform_params(self) -> HistogramTransformParams:
        return HistogramTransformParams(
            black_point=self._ht_black_spin.value(),
            midtone=self._ht_midtone_slider.value() / 100.0,
            white_point=self._ht_white_spin.value(),
        )

    def reset_histogram_transform_params(self):
        """Reset HT controls to their defaults (black=0, midtone=0.5, white=1)."""
        self._ht_black_spin.blockSignals(True)
        self._ht_midtone_slider.blockSignals(True)
        self._ht_white_spin.blockSignals(True)
        self._ht_black_spin.setValue(0.0)
        self._ht_midtone_slider.setValue(50)
        self._ht_white_spin.setValue(1.0)
        self._ht_black_spin.blockSignals(False)
        self._ht_midtone_slider.blockSignals(False)
        self._ht_white_spin.blockSignals(False)

    def get_scnr_params(self) -> SCNRParams:
        method_map = {0: SCNRMethod.AVERAGE_NEUTRAL, 1: SCNRMethod.MAXIMUM_NEUTRAL}
        return SCNRParams(
            method=method_map.get(
                self._scnr_method_combo.currentIndex(), SCNRMethod.AVERAGE_NEUTRAL
            ),
            amount=self._scnr_amount_slider.value() / 100.0,
            preserve_luminance=self._scnr_preserve_lum.isChecked(),
        )

    def get_color_adjust_params(self) -> ColorAdjustParams:
        return ColorAdjustParams(
            saturation=self._saturation_slider.value() / 100.0,
            hue_shift=float(self._hue_slider.value()),
            vibrance=self._vibrance_slider.value() / 100.0,
        )

    def get_color_calibration_params(self) -> ColorCalibrationParams:
        ref_map = {0: "average", 1: "G2V", 2: "custom"}
        return ColorCalibrationParams(
            white_reference=ref_map.get(self._cc_reference_combo.currentIndex(), "average"),
            neutralize_background=self._cc_neutralize_bg.isChecked(),
            background_percentile=self._cc_bg_percentile.value(),
        )

    def get_pcc_params(self) -> dict:
        solver_map = {0: "auto", 1: "astap", 2: "astrometry_net"}
        return {
            "ra_hint": self._pcc_ra_spin.value() if self._pcc_ra_spin.value() != 0 else None,
            "dec_hint": self._pcc_dec_spin.value() if self._pcc_dec_spin.value() != 0 else None,
            "solver": solver_map.get(self._pcc_solver_combo.currentIndex(), "auto"),
        }

    def set_psf_measurement(self, fwhm: float, ellipticity: float, n_stars: int) -> None:
        """Update FWHM spin from a PSF measurement result."""
        self._decon_fwhm_spin.setValue(round(fwhm, 2))
        self._psf_result_label.setText(
            f"Measured: FWHM={fwhm:.2f}px  ellip={ellipticity:.2f}  n={n_stars}"
        )

    def get_deconvolution_params(self) -> DeconvolutionParams | SpatialDeconvParams:
        if self._decon_spatial_check.isChecked():
            return SpatialDeconvParams(
                grid_zones=3,
                iterations=self._decon_iter_spin.value(),
                regularization=self._decon_reg_slider.value() / 10000.0,
                deringing=self._decon_dering_check.isChecked(),
                deringing_amount=self._decon_dering_amount.value(),
                fallback_fwhm=self._decon_fwhm_spin.value(),
            )
        return DeconvolutionParams(
            psf_fwhm=self._decon_fwhm_spin.value(),
            iterations=self._decon_iter_spin.value(),
            regularization=self._decon_reg_slider.value() / 10000.0,
            deringing=self._decon_dering_check.isChecked(),
            deringing_amount=self._decon_dering_amount.value(),
        )

    def get_denoise_params(self) -> DenoiseParams:
        method_map = {0: DenoiseMethod.WAVELET, 1: DenoiseMethod.NLM}
        return DenoiseParams(
            method=method_map.get(self._nr_method_combo.currentIndex(), DenoiseMethod.WAVELET),
            strength=self._nr_strength_slider.value() / 100.0,
            detail_preservation=self._nr_detail_slider.value() / 100.0,
            chrominance_only=self._nr_chrom_check.isChecked(),
        )

    def get_star_reduction_params(self) -> StarReductionParams:
        return StarReductionParams(
            amount=self._sr_amount_slider.value() / 100.0,
            iterations=self._sr_iterations_spin.value(),
            kernel_size=self._sr_kernel_spin.value(),
        )

    def get_mlt_params(self) -> WaveletParams:
        """Return WaveletParams from the MLT panel (6-band with thresholds)."""
        weights = []
        thresholds = []
        for boost_sl, _, thr_sl, _ in self._mlt_sliders:
            weights.append(boost_sl.value() / 100.0)
            thresholds.append(thr_sl.value() / 10000.0)
        return WaveletParams(
            n_scales=6,
            scale_weights=weights,
            residual_weight=self._mlt_residual_spin.value(),
            noise_thresholds=thresholds,
        )

    def get_wavelet_params(self) -> WaveletParams:
        n_scales = self._wav_scales_spin.value()
        weights = []
        fine = self._wav_fine_slider.value() / 100.0
        medium = self._wav_medium_slider.value() / 100.0
        coarse = self._wav_coarse_slider.value() / 100.0
        # Distribute 3 sliders across n_scales: fine for scale 0, coarse for last
        for i in range(n_scales):
            t = i / max(n_scales - 1, 1)
            if t < 0.5:
                w = fine * (1 - 2 * t) + medium * (2 * t)
            else:
                w = medium * (2 - 2 * t) + coarse * (2 * t - 1)
            weights.append(w)
        return WaveletParams(n_scales=n_scales, scale_weights=weights)

    def get_local_contrast_params(self) -> LocalContrastParams:
        return LocalContrastParams(
            clip_limit=self._lc_clip_spin.value(),
            tile_size=self._lc_tile_spin.value(),
            amount=self._lc_amount_slider.value() / 100.0,
        )

    def get_morphology_params(self) -> MorphologyParams:
        op_map = {0: MorphOp.DILATE, 1: MorphOp.ERODE, 2: MorphOp.OPEN, 3: MorphOp.CLOSE}
        el_map = {
            0: StructuringElement.CIRCLE,
            1: StructuringElement.SQUARE,
            2: StructuringElement.DIAMOND,
        }
        return MorphologyParams(
            operation=op_map.get(self._morph_op_combo.currentIndex(), MorphOp.DILATE),
            element=el_map.get(self._morph_element_combo.currentIndex(), StructuringElement.CIRCLE),
            kernel_size=self._morph_kernel_spin.value(),
            iterations=self._morph_iter_spin.value(),
        )

    def get_ai_denoise_params(self) -> AIDenoiseParams:
        return AIDenoiseParams(
            strength=self._aid_strength_slider.value() / 100.0,
        )

    def get_ai_sharpen_params(self) -> AISharpenParams:
        return AISharpenParams(
            strength=self._ais_strength_slider.value() / 100.0,
        )

    def get_crop_params(self) -> CropParams:
        return CropParams(
            x=self._crop_x_spin.value(),
            y=self._crop_y_spin.value(),
            width=self._crop_w_spin.value(),
            height=self._crop_h_spin.value(),
        )

    def get_rotate_params(self) -> RotateParams:
        angle_map = {
            0: RotateAngle.CW_90,
            1: RotateAngle.CW_180,
            2: RotateAngle.CW_270,
            3: RotateAngle.ARBITRARY,
        }
        return RotateParams(
            angle=angle_map.get(self._rotate_combo.currentIndex(), RotateAngle.CW_90),
            arbitrary_degrees=self._rotate_angle_spin.value(),
            expand=self._rotate_expand_check.isChecked(),
        )

    def get_flip_params(self) -> FlipParams:
        axis_map = {0: FlipAxis.HORIZONTAL, 1: FlipAxis.VERTICAL, 2: FlipAxis.BOTH}
        return FlipParams(
            axis=axis_map.get(self._flip_combo.currentIndex(), FlipAxis.HORIZONTAL),
        )

    def get_resize_params(self) -> ResizeParams:
        interp_map = {
            0: InterpolationMethod.LANCZOS,
            1: InterpolationMethod.BICUBIC,
            2: InterpolationMethod.BILINEAR,
            3: InterpolationMethod.NEAREST,
        }
        return ResizeParams(
            scale=self._resize_scale_spin.value(),
            interpolation=interp_map.get(
                self._resize_interp_combo.currentIndex(), InterpolationMethod.LANCZOS
            ),
        )

    def get_bin_params(self) -> BinParams:
        factor_map = {0: 2, 1: 3, 2: 4}
        mode_map = {0: BinMode.AVERAGE, 1: BinMode.SUM}
        return BinParams(
            factor=factor_map.get(self._bin_factor_combo.currentIndex(), 2),
            mode=mode_map.get(self._bin_mode_combo.currentIndex(), BinMode.AVERAGE),
        )

    def get_unsharp_mask_params(self) -> UnsharpMaskParams:
        return UnsharpMaskParams(
            radius=self._usm_radius_spin.value(),
            amount=self._usm_amount_slider.value() / 100.0,
            threshold=self._usm_threshold_spin.value(),
        )

    def get_median_filter_params(self) -> MedianFilterParams:
        k = self._mf_kernel_spin.value()
        if k % 2 == 0:
            k += 1
        return MedianFilterParams(kernel_size=k)

    def get_abe_params(self) -> ABEParams:
        kernel_map = {0: "thin_plate_spline", 1: "multiquadric", 2: "gaussian"}
        mode_map = {0: "subtraction", 1: "division"}
        return ABEParams(
            grid_size=self._abe_grid_spin.value(),
            rbf_kernel=kernel_map.get(self._abe_kernel_combo.currentIndex(), "thin_plate_spline"),
            rbf_smoothing=self._abe_smoothing_spin.value(),
            correction_mode=mode_map.get(self._abe_mode_combo.currentIndex(), "subtraction"),
        )

    def get_vignette_params(self) -> VignetteParams:
        return VignetteParams(
            strength=self._vig_strength_spin.value(),
            falloff=self._vig_falloff_spin.value(),
        )

    def get_ca_params(self) -> CAParams:
        return CAParams(
            auto_detect=self._ca_auto_check.isChecked(),
            red_shift_x=self._ca_red_x_spin.value(),
            red_shift_y=self._ca_red_y_spin.value(),
            blue_shift_x=self._ca_blue_x_spin.value(),
            blue_shift_y=self._ca_blue_y_spin.value(),
        )

    def get_continuum_scale(self) -> float:
        return self._cont_scale_spin.value()

    @property
    def starnet_extract_stars(self) -> bool:
        return self._sn_extract_stars.isChecked()

    @property
    def curve_editor(self) -> CurveEditor:
        return self._curve_editor

    @property
    def split_preview_enabled(self) -> bool:
        return self._split_check.isChecked()

    def set_macro_recording(self, recording: bool):
        """Update macro UI state when recording starts/stops."""
        self._btn_macro_start.setEnabled(not recording)
        self._btn_macro_stop.setEnabled(recording)
        self._macro_status_label.setText("Recording..." if recording else "Not recording")
        self._macro_status_label.setStyleSheet(
            "color: #ff4444; font-size: 11px; font-weight: bold;"
            if recording
            else "color: #969696; font-size: 11px;"
        )

    def _on_curve_channel_changed(self, index: int):
        """Switch curve editor to show the selected channel's curve."""
        # This will be used by the main window to swap curve data
        pass

    def _on_curves_changed(self):
        """Handle curve editor changes."""
        self._emit_if_preview_enabled("curves")

    @property
    def curves_histogram_visible(self) -> bool:
        return self._curves_histogram_check.isChecked()

    @property
    def current_curve_channel(self) -> int:
        return self._curve_channel_combo.currentIndex()
