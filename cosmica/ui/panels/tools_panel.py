"""tools_panel.py — Cosmica Tools Panel (PyQt6 redesign).

Drop-in replacement for the existing cosmica/ui/panels/tools_panel.py.
All signals and getter/setter methods are identical to the original.
Visual style matches the HTML prototype exactly using ui_kit widgets.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox,
    QHBoxLayout, QLabel, QListWidget, QPushButton,
    QSpinBox, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from cosmica.ai.inference.denoise import AIDenoiseParams
from cosmica.ai.inference.sharpen import AISharpenParams
from cosmica.core.abe import ABEParams
from cosmica.core.background import BackgroundParams
from cosmica.core.background_neutralization import BackgroundNeutralizationParams
from cosmica.core.banding import BandingParams
from cosmica.core.chromatic_aberration import CAParams
from cosmica.core.color_calibration import ColorCalibrationParams
from cosmica.core.color_tools import ColorAdjustParams, SCNRParams
from cosmica.core.cosmetic import CosmeticParams
from cosmica.core.curves import CurvesParams
from cosmica.core.deconvolution import DeconvolutionParams, SpatialDeconvParams
from cosmica.core.denoise import DenoiseParams
from cosmica.core.filters import UnsharpMaskParams
from cosmica.core.histogram_transform import HistogramTransformParams
from cosmica.core.local_contrast import LocalContrastParams
from cosmica.core.morphology import MorphologyParams
from cosmica.core.stacking import IntegrationMethod, RejectionMethod, StackingParams
from cosmica.core.star_reduction import StarReductionParams
from cosmica.core.stretch import ArcsinhStretchParams, GHSParams, StretchParams
from cosmica.core.transforms import (
    BinParams, CropParams, FlipParams, ResizeParams, RotateParams,
)
from cosmica.core.vignette import VignetteParams
from cosmica.core.wavelets import WaveletParams
from cosmica.ui.widgets.curves_widget import CurveEditor
from cosmica.ui.widgets.ui_kit import (
    ACCENT, ACCENT_DARK, ACCENT_HOVER, ACCENT_PURPLE, BG_HOVER, BG_PRIMARY, BG_SECONDARY,
    BG_TERTIARY, BLUE, BORDER, FONT_MONO, ORANGE, RED,
    TEXT_PRIMARY, TEXT_SECONDARY,
    CollapsibleSection, InfoLabel, RunBtn, SliderRow,
    divider, field_row, make_label, scrollable_tab,
    styled_check, styled_combo, styled_spin,
)

# Tab-bar stylesheet
_TAB_SS = f"""
QTabWidget::pane {{
    border: none; background: {BG_PRIMARY};
}}
QTabBar {{
    background: {BG_PRIMARY};
}}
QTabBar::tab {{
    background: {BG_PRIMARY}; color: {TEXT_SECONDARY};
    padding: 7px 10px; font-size: 10px; font-weight: 600;
    border: none; border-bottom: 2px solid transparent;
    min-width: 0;
}}
QTabBar::tab:selected {{
    color: {ACCENT}; border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover:!selected {{
    color: {TEXT_PRIMARY};
}}
QTabBar::scroller {{
    width: 24px;
}}
QTabBar QToolButton {{
    background-color: {BG_TERTIARY}; border: 1px solid {BORDER};
    border-radius: 3px; color: {TEXT_PRIMARY};
    width: 20px; height: 20px;
    padding: 0px; margin: 2px 2px;
    font-size: 11px;
}}
QTabBar QToolButton:hover {{
    background-color: {BG_HOVER}; color: #ffffff;
}}
"""

_BOTTOM_SS = f"""
QWidget#tools_bottom {{
    background: {BG_SECONDARY};
    border-top: 1px solid {BORDER};
}}
"""


class ToolsPanel(QWidget):
    """Right-side tabbed processing controls."""

    # ── Signals (identical to original) ──────────────────
    run_calibration          = pyqtSignal()
    run_stacking             = pyqtSignal()
    run_alignment            = pyqtSignal()
    run_stretch              = pyqtSignal()
    run_background           = pyqtSignal()
    stretch_params_changed   = pyqtSignal()
    run_cosmetic             = pyqtSignal()
    run_banding              = pyqtSignal()
    run_histogram_transform  = pyqtSignal()
    run_curves               = pyqtSignal()
    run_scnr                 = pyqtSignal()
    run_color_adjust         = pyqtSignal()
    run_deconvolution        = pyqtSignal()
    run_ghs                  = pyqtSignal()
    run_arcsinh_stretch      = pyqtSignal()
    run_color_calibration    = pyqtSignal()
    run_pcc                  = pyqtSignal()
    run_denoise              = pyqtSignal()
    run_star_reduction       = pyqtSignal()
    open_narrowband_dialog   = pyqtSignal()
    open_pixelmath_dialog    = pyqtSignal()
    run_split_channels       = pyqtSignal()
    run_extract_luminance    = pyqtSignal()
    run_wavelet_sharpen      = pyqtSignal()
    run_local_contrast       = pyqtSignal()
    run_morphology           = pyqtSignal()
    open_hdr_dialog          = pyqtSignal()
    run_ai_denoise           = pyqtSignal()
    run_ai_sharpen           = pyqtSignal()
    run_starnet              = pyqtSignal()
    open_batch_dialog        = pyqtSignal()
    start_macro_recording    = pyqtSignal()
    stop_macro_recording     = pyqtSignal()
    play_macro               = pyqtSignal()
    save_macro               = pyqtSignal()
    load_macro               = pyqtSignal()
    run_unsharp_mask         = pyqtSignal()
    run_median_filter        = pyqtSignal()
    run_abe                  = pyqtSignal()
    run_vignette_correction  = pyqtSignal()
    run_chromatic_aberration = pyqtSignal()
    show_image_statistics    = pyqtSignal()
    curves_histogram_changed = pyqtSignal()
    measure_psf              = pyqtSignal()
    run_continuum_subtraction= pyqtSignal()
    toggle_sample_mode       = pyqtSignal(bool)
    clear_bg_samples         = pyqtSignal()
    add_bg_grid              = pyqtSignal(int, int, int)
    toggle_wcs_overlay       = pyqtSignal(bool)
    run_background_neutralization = pyqtSignal()
    open_python_console      = pyqtSignal()
    run_mlt                  = pyqtSignal()
    run_lrgb_combine         = pyqtSignal()
    run_spcc                 = pyqtSignal()
    toggle_dso_overlay       = pyqtSignal(bool)
    open_star_mask_dialog    = pyqtSignal()
    open_subframe_selector   = pyqtSignal()
    blink_load_a             = pyqtSignal()
    blink_load_b             = pyqtSignal()
    blink_use_current_as_a   = pyqtSignal()
    blink_use_current_as_b   = pyqtSignal()
    blink_toggle             = pyqtSignal(bool)
    blink_fps_changed        = pyqtSignal(int)
    start_crop_draw          = pyqtSignal()
    run_crop                 = pyqtSignal()
    run_rotate               = pyqtSignal()
    run_flip                 = pyqtSignal()
    run_resize               = pyqtSignal()
    run_bin                  = pyqtSignal()
    run_invert               = pyqtSignal()
    preview_requested        = pyqtSignal(str)
    preview_cancelled        = pyqtSignal()
    run_multi_session        = pyqtSignal()
    multi_session_add_folder = pyqtSignal()
    multi_session_clear      = pyqtSignal()
    open_channel_combine_dialog = pyqtSignal()
    run_debayer              = pyqtSignal()
    clip_points_changed      = pyqtSignal(float, float)
    open_smart_processor     = pyqtSignal()
    open_equipment_dialog    = pyqtSignal()

    # ── Init ─────────────────────────────────────────────
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(260)
        self.setMaximumWidth(420)
        self.setStyleSheet(f"background: {BG_SECONDARY};")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)
        self._tabs.setUsesScrollButtons(True)
        self._tabs.tabBar().setExpanding(False)
        self._tabs.setStyleSheet(_TAB_SS)
        outer.addWidget(self._tabs)

        # bottom preset/undo bar
        bottom = QWidget()
        bottom.setObjectName("tools_bottom")
        bottom.setFixedHeight(34)
        bottom.setStyleSheet(_BOTTOM_SS)
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(8, 4, 8, 4)
        bl.setSpacing(4)

        for label in ("↙ Load Preset", "↗ Save Preset"):
            b = RunBtn(label, flat=True)
            b.setFixedHeight(24)
            bl.addWidget(b)
        bl.addStretch()
        self._btn_undo = RunBtn("↩", flat=True)
        self._btn_undo.setFixedWidth(32)
        self._btn_undo.setFixedHeight(24)
        self._btn_undo.setToolTip("Undo (Ctrl+Z)")
        self._btn_redo = RunBtn("↪", flat=True)
        self._btn_redo.setFixedWidth(32)
        self._btn_redo.setFixedHeight(24)
        self._btn_redo.setToolTip("Redo (Ctrl+Y)")
        bl.addWidget(self._btn_undo)
        bl.addWidget(self._btn_redo)
        outer.addWidget(bottom)

        # build tabs
        self._build_preprocess_tab()
        self._build_stacking_tab()
        self._build_background_tab()
        self._build_stretch_tab()
        self._build_transform_tab()
        self._build_color_tab()
        self._build_detail_tab()
        self._build_ai_tab()
        self._build_utility_tab()

    # ── TAB 1: Pre-Process ────────────────────────────────
    def _build_preprocess_tab(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Calibration
        cal = CollapsibleSection("Calibration", accent=True)
        cal.add_info(
            "Create masters from raw frame folders or use pre-made masters."
        )
        self._cal_bias_label = cal.add_status_label("Bias: none")
        bl, _ = self._cal_frame_row(cal, "bias")
        self._cal_dark_label = cal.add_status_label("Dark: none")
        self._cal_frame_row(cal, "dark")
        self._cal_flat_label = cal.add_status_label("Flat: none")
        self._cal_frame_row(cal, "flat")
        cal.add_info("Light frames: add via Project panel → Import Lights")
        cal.add_run("▶ Run Calibration", self.run_calibration.emit)
        lay.addWidget(cal)

        # Cosmetic
        cos = CollapsibleSection("Cosmetic Correction")
        cos.add_info("Detect and remove hot, cold, and dead pixels.")
        self._hot_sigma = cos.add_slider("Hot sigma", 5.0, 1.0, 20.0, 0.5, 1)
        self._cold_sigma = cos.add_slider("Cold sigma", 5.0, 1.0, 20.0, 0.5, 1)
        self._dead_pixel_check = cos.add_check("Detect dead pixels (value=0)", True)
        cos.add_run("▶ Apply Cosmetic Correction", self.run_cosmetic.emit)
        lay.addWidget(cos)

        # Debayer
        deb = CollapsibleSection("Debayer (OSC / Color Camera)")
        deb.add_info("Convert raw Bayer mosaic to color image.")
        self._debayer_pattern_combo = deb.add_combo(
            "Pattern",
            ["Auto-detect", "RGGB", "BGGR", "GRBG", "GBRG"],
        )
        self._debayer_method_combo = deb.add_combo(
            "Method",
            ["VNG (best quality)", "Edge-Aware (EA)",
             "Superpixel (2× bin)", "Bilinear (fastest)"],
        )
        deb.add_run("▶ Apply Debayer", self.run_debayer.emit)
        lay.addWidget(deb)

        self._tabs.addTab(scrollable_tab(lay), "⬡  Pre-Process")

    def _cal_frame_row(self, sec: CollapsibleSection, frame_type: str):
        rl = QHBoxLayout()
        rl.setSpacing(4)
        bf = RunBtn(f"Folder…", flat=True)
        bm = RunBtn(f"Master…", flat=True)
        bf.setFixedHeight(26)
        bm.setFixedHeight(26)
        rl.addWidget(bf)
        rl.addWidget(bm)
        sec.add_layout(rl)
        return rl, (bf, bm)

    # ── TAB 2: Stacking ───────────────────────────────────
    def _build_stacking_tab(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Subframe Selector (moved here from Pre-Process)
        sub = CollapsibleSection("Subframe Selector")
        sub.add_info("Score and reject frames by FWHM, eccentricity, SNR, star count.")
        self._subframe_count_label = sub.add_status_label("No subframe selection active")
        sub.add_run("⊞ Open Subframe Selector…", self.open_subframe_selector.emit, flat=True)
        lay.addWidget(sub)

        # Registration
        reg = CollapsibleSection("Registration (Alignment)", accent=True)
        reg.add_info("Detect stars and align frames to a reference.")
        self._reg_mode_combo = reg.add_combo(
            "Mode",
            ["Star (1-Pass)", "Star (2-Pass)", "Triangle Match",
             "FFT Translation", "Comet"],
            "Star (2-Pass)",
        )
        self._star_sens_spin = reg.add_slider("Star sensitivity", 5.0, 1.0, 20.0, 0.5, 1)
        self._max_shift_spin = reg.add_spin("Max distance (px)", 10, 500, 50, 10)
        self._ransac_thresh_spin = reg.add_spin("RANSAC threshold", 1.0, 10.0, 3.0, 0.5, 1)
        self._ref_frame_combo = reg.add_combo(
            "Reference frame",
            ["Auto (best quality)", "First frame", "Last frame", "Specific frame #"],
        )
        reg.add_run("▶ Align Frames", self.run_alignment.emit, flat=True)
        lay.addWidget(reg)

        # Integration
        integ = CollapsibleSection("Integration (Stacking)", accent=True)
        integ.add_info("Combine aligned frames using rejection to increase SNR.")
        self._rejection_combo = integ.add_combo(
            "Rejection",
            ["Sigma Clipping", "Winsorized Sigma", "Linear Fit",
             "Percentile Clip", "ESD (Generalized)", "Min/Max", "None"],
        )
        self._integration_combo = integ.add_combo(
            "Integration",
            ["Average", "Median", "Weighted Average"],
        )
        self._kappa_spin = integ.add_slider("Kappa (σ)", 3.0, 0.5, 10.0, 0.1, 1)
        integ.add_run("▶ Stack Images", self.run_stacking.emit)
        lay.addWidget(integ)

        # Drizzle
        drz = CollapsibleSection("Drizzle Integration")
        self._drizzle_check = drz.add_check("Enable Drizzle")
        self._drizzle_scale_combo = drz.add_combo("Output scale", ["2× (recommended)", "3×"])
        self._drizzle_drop_spin = drz.add_slider("Drop shrink", 0.7, 0.5, 1.0, 0.05, 2)
        lay.addWidget(drz)

        # Multi-session
        ms = CollapsibleSection("Multi-Session Integration")
        ms.add_info("Stack frames from different telescopes, cameras, or nights.")
        self._ms_session_list = QListWidget()
        self._ms_session_list.setFixedHeight(80)
        self._ms_session_list.setStyleSheet(f"""
            QListWidget {{
                background: {BG_TERTIARY}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER}; border-radius: 4px;
                font-size: 11px;
            }}
        """)
        ms.add_widget(self._ms_session_list)
        btns = ms.add_btn_row([("+ Add Session…", True), ("Clear All", True)])
        btns[0].clicked.connect(self.multi_session_add_folder.emit)
        btns[1].clicked.connect(self.multi_session_clear.emit)
        self._ms_weight_combo = ms.add_combo(
            "Weighting",
            ["SNR (recommended)", "Integration time", "Equal weight"],
        )
        self._ms_normalize_check = ms.add_check("Normalize background", True)
        self._ms_align_check     = ms.add_check("Align sub-stacks", True)
        self._btn_ms_stack = ms.add_run("▶ Stack All Sessions", self.run_multi_session.emit)
        self._btn_ms_stack.setEnabled(False)
        lay.addWidget(ms)

        # Batch
        batch = CollapsibleSection("Batch Processing")
        batch.add_info("Apply a pipeline to multiple images at once.")
        batch.add_run("⊞ Open Batch Dialog…", self.open_batch_dialog.emit, flat=True)
        lay.addWidget(batch)

        self._tabs.addTab(scrollable_tab(lay), "⧉  Stacking")

    # ── TAB 3: Background ─────────────────────────────────
    def _build_background_tab(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Background Extraction
        bg = CollapsibleSection("Background Extraction", accent=True)
        bg.add_info("Remove light pollution gradients.")
        self._bg_grid_spin  = bg.add_spin("Grid size", 4, 32, 8)
        self._bg_order_spin = bg.add_spin("Poly order", 1, 6, 3)

        # auto-grid row
        grid_row = QHBoxLayout()
        grid_row.setSpacing(6)
        for lbl_text, attr in [("Rows", "_bg_grid_rows_spin"), ("Cols", "_bg_grid_cols_spin")]:
            sub_lay = QVBoxLayout()
            sub_lay.setSpacing(2)
            sub_lay.addWidget(make_label(lbl_text, TEXT_SECONDARY, 10))
            spin = styled_spin(2, 20, 5)
            setattr(self, attr, spin)
            sub_lay.addWidget(spin)
            grid_row.addLayout(sub_lay)
        bg.add_layout(grid_row)

        self._bg_box_size_spin = bg.add_spin("Box size (px)", 8, 256, 64, 8)
        btn_grid = bg.add_run(
            "⊞ Add Auto-Grid Samples",
            lambda: self.add_bg_grid.emit(
                int(self._bg_grid_rows_spin.value()),
                int(self._bg_grid_cols_spin.value()),
                int(self._bg_box_size_spin.value()),
            ),
            flat=True,
        )

        self._btn_place_samples = RunBtn("Place Samples", flat=True)
        self._btn_place_samples.setCheckable(True)
        self._btn_place_samples.toggled.connect(self.toggle_sample_mode.emit)
        bg.add_widget(self._btn_place_samples)

        self._bg_sample_label = bg.add_status_label("0 manual samples")
        btn_clear = bg.add_run("Clear Samples", self.clear_bg_samples.emit, flat=True)
        bg.add_run("▶ Extract Background", self.run_background.emit)
        lay.addWidget(bg)

        # ABE
        abe = CollapsibleSection("ABE (Advanced)")
        abe.add_info("Background extraction using polynomial or RBF surface fitting.")
        self._abe_grid_spin   = abe.add_spin("Grid size", 5, 30, 10)
        self._abe_model_combo = abe.add_combo("Model", ["Polynomial (recommended)", "RBF"])
        self._abe_degree_spin = abe.add_spin("Poly degree", 1, 5, 2)
        self._abe_kernel_combo = abe.add_combo(
            "RBF kernel", ["Thin Plate Spline", "Multiquadric", "Gaussian"]
        )
        self._abe_mode_combo  = abe.add_combo("Mode", ["Subtraction", "Division"])
        abe.add_run("▶ Run ABE", self.run_abe.emit)
        lay.addWidget(abe)

        # Background Neutralization (NEW)
        bn = CollapsibleSection("Background Neutralization")
        bn.add_info(
            "Shift sky background to neutral zero per-channel. "
            "Equivalent to PixInsight BackgroundNeutralization (statistical mode)."
        )
        self._bn_percentile = bn.add_slider("Percentile", 2.0, 0.5, 10.0, 0.5, 1)
        self._bn_amount     = bn.add_slider("Amount", 1.0, 0.0, 1.0, 0.05, 2)
        self._bn_protect    = bn.add_slider("Protect bright", 0.5, 0.0, 1.0, 0.05, 2)
        bn.add_run("▶ Apply Background Neutralization",
                   self.run_background_neutralization.emit)
        lay.addWidget(bn)

        # Vignette
        vig = CollapsibleSection("Vignette Correction")
        vig.add_info("Remove optical vignetting toward image edges.")
        self._vignette_amount = vig.add_slider("Amount", 0.3, 0.0, 1.0, 0.05, 2)
        self._vignette_radius = vig.add_slider("Radius", 0.8, 0.3, 1.0, 0.05, 2)
        vig.add_run("▶ Correct Vignette", self.run_vignette_correction.emit)
        lay.addWidget(vig)

        # Banding
        band = CollapsibleSection("Banding Reduction")
        band.add_info("Remove horizontal/vertical banding from CMOS sensors.")
        self._banding_amount = band.add_slider("Amount", 1.0, 0.1, 3.0, 0.1, 1)
        self._banding_dir_combo = band.add_combo(
            "Direction", ["Horizontal", "Vertical", "Both"]
        )
        band.add_run("▶ Reduce Banding", self.run_banding.emit)
        lay.addWidget(band)

        self._tabs.addTab(scrollable_tab(lay), "◫  Background")

    # ── TAB 4: Stretch ────────────────────────────────────
    def _build_stretch_tab(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Auto-stretch
        aut = CollapsibleSection("Auto-Stretch", accent=True)
        aut.add_info("Statistical midtone stretch.")
        self._midtone_slider = aut.add_slider("Midtone", 0.25, 0.01, 0.99, 0.01, 2, 0.25)
        self._midtone_slider.value_changed.connect(lambda _: self.stretch_params_changed.emit())
        self._shadow_spin = aut.add_spin("Shadow clip", -10.0, 0.0, -2.8, 0.1, 1)
        self._linked_check = aut.add_check("Link RGB channels", True)
        self._split_check  = aut.add_check("Before/After split preview")
        self._split_check.toggled.connect(lambda _: self.stretch_params_changed.emit())
        btns = aut.add_btn_row([("▶ Apply Stretch", False), ("Reset", True)])
        btns[0].clicked.connect(self.run_stretch.emit)
        btns[1].clicked.connect(lambda: self._midtone_slider.setValue(0.25))
        lay.addWidget(aut)

        # Arcsinh (NEW)
        arc = CollapsibleSection("Arcsinh Stretch")
        arc.add_info(
            "Lupton et al. 2004 — linear-to-arcsinh ramp. Preserves star colours "
            "better than log; reveals faint nebulosity without blowing out stars."
        )
        self._arcsinh_factor_spin = arc.add_spin(
            "Stretch factor β", 0.1, 1000.0, 10.0, 1.0, 1
        )
        self._arcsinh_bp_spin = arc.add_spin("Black point", 0.0, 0.5, 0.0, 0.001, 4)
        self._arcsinh_linked_check = arc.add_check("Linked RGB", True)
        self._arcsinh_preview_check = arc.add_check("Live split preview")
        self._arcsinh_factor_spin.valueChanged.connect(
            lambda _: self._fire_preview("arcsinh_stretch", self._arcsinh_preview_check)
        )
        self._arcsinh_bp_spin.valueChanged.connect(
            lambda _: self._fire_preview("arcsinh_stretch", self._arcsinh_preview_check)
        )
        self._arcsinh_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("arcsinh_stretch") if on
            else self.preview_cancelled.emit()
        )
        btns = arc.add_btn_row([("▶ Apply Arcsinh Stretch", False), ("Reset", True)])
        btns[0].clicked.connect(self.run_arcsinh_stretch.emit)
        btns[1].clicked.connect(lambda: (
            self._arcsinh_factor_spin.setValue(10.0),
            self._arcsinh_bp_spin.setValue(0.0),
        ))
        lay.addWidget(arc)

        # GHS
        ghs = CollapsibleSection("Generalized Hyperbolic Stretch")
        ghs.add_info("Advanced non-linear stretch.")
        self._ghs_d_spin  = ghs.add_spin("Stretch (D)",   0.0, 20.0, 5.0, 0.5, 1)
        self._ghs_b_spin  = ghs.add_spin("Asymmetry (b)", -5.0, 5.0, 0.0, 0.1, 1)
        self._ghs_sp_spin = ghs.add_spin("Sym. point",    0.0,  1.0, 0.0, 0.05, 3)
        self._ghs_shadow_slider    = ghs.add_slider("Shadow prot.",    0.0, 0.0, 1.0, 0.01, 2)
        self._ghs_highlight_slider = ghs.add_slider("Highlight prot.", 0.0, 0.0, 1.0, 0.01, 2)
        self._ghs_preview_check = ghs.add_check("Live split preview")
        for _ghs_spin in (self._ghs_d_spin, self._ghs_b_spin, self._ghs_sp_spin):
            _ghs_spin.valueChanged.connect(
                lambda _, s=self._ghs_preview_check: self._fire_preview("ghs", s)
            )
        for _ghs_sl in (self._ghs_shadow_slider, self._ghs_highlight_slider):
            _ghs_sl.value_changed.connect(
                lambda _, s=self._ghs_preview_check: self._fire_preview("ghs", s)
            )
        self._ghs_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("ghs") if on
            else self.preview_cancelled.emit()
        )
        btns = ghs.add_btn_row([("▶ Apply GHS", False), ("Reset", True)])
        btns[0].clicked.connect(self.run_ghs.emit)
        lay.addWidget(ghs)

        # Histogram Transform
        ht = CollapsibleSection("Histogram Transform")
        ht.add_info("Black point, midtone, and white point adjustment.")
        self._ht_black_spin  = ht.add_spin("Black point", 0.0, 0.99, 0.0, 0.01, 3)
        self._ht_midtone_slider = ht.add_slider("Midtone",  0.5, 0.01, 0.99, 0.01, 2, 0.5)
        self._ht_white_spin  = ht.add_spin("White point", 0.01, 1.0, 1.0, 0.01, 3)
        self._ht_black_spin.valueChanged.connect(lambda _: self._emit_clip_points())
        self._ht_white_spin.valueChanged.connect(lambda _: self._emit_clip_points())
        self._ht_preview_check = ht.add_check("Live split preview")
        self._ht_black_spin.valueChanged.connect(
            lambda _: self._fire_preview("histogram_transform", self._ht_preview_check)
        )
        self._ht_midtone_slider.value_changed.connect(
            lambda _: self._fire_preview("histogram_transform", self._ht_preview_check)
        )
        self._ht_white_spin.valueChanged.connect(
            lambda _: self._fire_preview("histogram_transform", self._ht_preview_check)
        )
        self._ht_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("histogram_transform") if on
            else self.preview_cancelled.emit()
        )
        btns = ht.add_btn_row([("▶ Apply HT", False), ("Reset", True)])
        btns[0].clicked.connect(self.run_histogram_transform.emit)
        lay.addWidget(ht)

        # Curves
        crv = CollapsibleSection("Curves")
        crv.add_info("Click to add points, drag to adjust. Right-click to remove.")
        self._curve_channel_combo = crv.add_combo(
            "Channel", ["Master (L)", "Red", "Green", "Blue"]
        )
        self._curve_channel_combo.currentIndexChanged.connect(
            lambda _: self.curves_histogram_changed.emit()
        )
        self._curve_editor = CurveEditor()
        self._curve_editor.setMinimumHeight(180)
        crv.add_widget(self._curve_editor)
        self._curves_histogram_check = crv.add_check("Show histogram")
        self._curves_histogram_check.stateChanged.connect(
            lambda _: self.curves_histogram_changed.emit()
        )
        btns = crv.add_btn_row([("▶ Apply Curves", False), ("Reset", True)])
        btns[0].clicked.connect(self.run_curves.emit)
        btns[1].clicked.connect(self._curve_editor.reset)
        lay.addWidget(crv)

        self._tabs.addTab(scrollable_tab(lay), "◑  Stretch")

    # ── TAB 5: Transform ──────────────────────────────────
    def _build_transform_tab(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Crop
        crop = CollapsibleSection("Crop", accent=True)
        crop.add_info("Crop image to a rectangular region.")
        self._btn_crop_draw = RunBtn("✏ Draw on Image…", flat=True)
        self._btn_crop_draw.setCheckable(True)
        self._btn_crop_draw.toggled.connect(
            lambda on: self.start_crop_draw.emit() if on else None
        )
        crop.add_widget(self._btn_crop_draw)
        self._crop_x_spin = crop.add_spin("X offset", 0, 99999, 0)
        self._crop_y_spin = crop.add_spin("Y offset", 0, 99999, 0)
        self._crop_w_spin = crop.add_spin("Width",    0, 99999, 0)
        self._crop_h_spin = crop.add_spin("Height",   0, 99999, 0)
        crop.add_run("▶ Apply Crop", self.run_crop.emit)
        lay.addWidget(crop)

        # Rotate
        rot = CollapsibleSection("Rotate")
        self._rotate_combo = rot.add_combo(
            "Preset", ["90° CW", "180°", "270° CW", "Custom angle"]
        )
        self._rotate_angle_spin = rot.add_spin("Custom angle", -360, 360, 0, 0.1, 1, "°")
        self._rotate_expand_check = rot.add_check("Expand canvas", True)
        rot.add_run("▶ Apply Rotation", self.run_rotate.emit)
        lay.addWidget(rot)

        # Flip
        flp = CollapsibleSection("Flip")
        self._flip_combo = flp.add_combo("Axis", ["Horizontal", "Vertical", "Both"])
        flp.add_run("▶ Apply Flip", self.run_flip.emit)
        lay.addWidget(flp)

        # Resize
        rsz = CollapsibleSection("Resize / Resample")
        self._resize_scale_spin = rsz.add_spin("Scale", 0.1, 10.0, 1.0, 0.1, 2)
        self._resize_interp_combo = rsz.add_combo(
            "Interpolation", ["Lanczos", "Bicubic", "Bilinear", "Nearest"]
        )
        rsz.add_run("▶ Apply Resize", self.run_resize.emit)
        lay.addWidget(rsz)

        # Bin
        bn = CollapsibleSection("Bin")
        bn.add_info("Combine pixels to increase SNR at lower resolution.")
        self._bin_factor_combo = bn.add_combo("Factor", ["2x2", "3x3", "4x4"])
        self._bin_mode_combo   = bn.add_combo("Mode",   ["Average", "Sum"])
        bn.add_run("▶ Apply Bin", self.run_bin.emit)
        lay.addWidget(bn)

        # Invert
        inv = CollapsibleSection("Invert")
        inv.add_info("Invert all pixel values (1 − image).")
        inv.add_run("▶ Invert Image", self.run_invert.emit)
        lay.addWidget(inv)

        self._tabs.addTab(scrollable_tab(lay), "⟳  Transform")

    # ── TAB 6: Color ──────────────────────────────────────
    def _build_color_tab(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # SCNR
        scnr = CollapsibleSection("SCNR (Green Noise Removal)", accent=True)
        scnr.add_info("Remove color noise, typically excess green channel.")
        self._scnr_target_combo  = scnr.add_combo("Target",  ["Green", "Red", "Blue"])
        self._scnr_method_combo  = scnr.add_combo(
            "Method",
            ["Average Neutral", "Maximum Neutral", "Additive-Subtractive Mask"],
        )
        self._scnr_amount = scnr.add_slider("Amount", 0.5, 0.0, 1.0, 0.01, 2)
        self._scnr_preview_check = scnr.add_check("Live split preview")
        self._scnr_amount.value_changed.connect(
            lambda _: self._fire_preview("scnr", self._scnr_preview_check)
        )
        self._scnr_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("scnr") if on
            else self.preview_cancelled.emit()
        )
        scnr.add_run("▶ Apply SCNR", self.run_scnr.emit)
        lay.addWidget(scnr)

        # Color Adjust
        ca = CollapsibleSection("Color Adjustment")
        self._hue_slider        = ca.add_slider("Hue shift",   0, -180, 180, 1, 0)
        self._sat_slider        = ca.add_slider("Saturation",  0, -100, 100, 1, 0)
        self._vibrance_slider   = ca.add_slider("Vibrance",    0, -100, 100, 1, 0)
        self._lightness_slider  = ca.add_slider("Lightness",   0, -100, 100, 1, 0)
        ca.add_run("▶ Apply Color Adjust", self.run_color_adjust.emit)
        lay.addWidget(ca)

        # Color Calibration
        cc = CollapsibleSection("Color Calibration")
        cc.add_info("White balance using background reference or star colours.")
        self._cc_method_combo = cc.add_combo(
            "Method",
            ["Background reference", "Photometric (SPCC)", "Manual RGB"],
        )
        btns = cc.add_btn_row([("Pick BG Reference", True), ("Calibrate", False)])
        btns[1].clicked.connect(self.run_color_calibration.emit)
        lay.addWidget(cc)

        # SPCC
        spcc = CollapsibleSection("SPCC (Photometric)")
        spcc.add_info("Spectrophotometric calibration using star spectra + filter database.")
        self._spcc_filter_combo  = spcc.add_combo(
            "Filter set", ["Broadband (L/R/G/B)", "Narrowband Ha/OIII/SII", "Custom"]
        )
        self._spcc_camera_combo  = spcc.add_combo(
            "Camera", ["ZWO ASI2600MM", "QHY268M", "ZWO ASI533MC", "Other…"]
        )
        spcc.add_run("▶ Run SPCC", self.run_spcc.emit)
        lay.addWidget(spcc)

        # Narrowband
        nb = CollapsibleSection("Narrowband Tools")
        nb.add_info("SHO/HOO/HaRGB palette mapping, continuum subtraction, blending.")
        nb.add_run("⊞ Open Narrowband Dialog…", self.open_narrowband_dialog.emit, flat=True)
        nb.add_run("▶ Continuum Subtraction", self.run_continuum_subtraction.emit, flat=True)
        lay.addWidget(nb)

        # LRGB / Channels
        lc = CollapsibleSection("LRGB / Channel Combine")
        lc.add_info("Combine luminance and colour channels.")
        btns = lc.add_btn_row([("LRGB Combine…", True), ("Channel Combine…", True)])
        btns[0].clicked.connect(self.run_lrgb_combine.emit)
        btns[1].clicked.connect(self.open_channel_combine_dialog.emit)
        lc.add_divider()
        lc.add_run("▶ Extract Luminance", self.run_extract_luminance.emit, flat=True)
        lc.add_run("▶ Split Channels", self.run_split_channels.emit, flat=True)
        lay.addWidget(lc)

        self._tabs.addTab(scrollable_tab(lay), "◈  Color")

    # ── TAB 7: Detail ─────────────────────────────────────
    def _build_detail_tab(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Deconvolution
        dec = CollapsibleSection("Deconvolution", accent=True)
        dec.add_info("Restore fine detail lost to seeing or tracking.")
        self._deconv_method_combo = dec.add_combo(
            "Method", ["Richardson-Lucy", "Blind (Spatial)", "Wiener"]
        )
        self._deconv_psf_spin   = dec.add_spin("PSF FWHM (px)", 0.5, 20.0, 3.0, 0.1, 1)
        self._deconv_iter       = dec.add_slider("Iterations", 50, 5, 200, 1, 0)
        self._deconv_reg        = dec.add_spin("Regularization", 0.0, 0.1, 0.001, 0.001, 4)
        self._deconv_deringing  = dec.add_check("Deringing protection", True)
        self._deconv_dering_amt = dec.add_slider("Deringing amount", 0.5, 0.0, 1.0, 0.05, 2)
        btns = dec.add_btn_row([("Measure PSF", True), ("Star Mask", True)])
        btns[0].clicked.connect(self.measure_psf.emit)
        btns[1].clicked.connect(self.open_star_mask_dialog.emit)
        self._deconv_preview_check = dec.add_check("Live split preview")
        for _w in (self._deconv_iter, self._deconv_dering_amt):
            _w.value_changed.connect(
                lambda _: self._fire_preview("deconvolution", self._deconv_preview_check)
            )
        self._deconv_psf_spin.valueChanged.connect(
            lambda _: self._fire_preview("deconvolution", self._deconv_preview_check)
        )
        self._deconv_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("deconvolution") if on
            else self.preview_cancelled.emit()
        )
        dec.add_run("▶ Apply Deconvolution", self.run_deconvolution.emit)
        lay.addWidget(dec)

        # PSF Measurement (expanded results)
        psf = CollapsibleSection("PSF Measurement")
        psf.add_info("Detect stars, fit 2D Gaussians, report FWHM and ellipticity.")
        # results grid
        self._psf_result_labels: dict[str, QLabel] = {}
        metrics = [
            ("FWHM", "—"), ("FWHM X", "—"), ("FWHM Y", "—"),
            ("Ellipticity", "—"), ("Rotation", "—"), ("Stars used", "—"),
            ("FWHM σ", "—"),
        ]
        from PyQt6.QtWidgets import QGridLayout
        grid_w = QWidget()
        grid_w.setStyleSheet(
            f"background: {BG_TERTIARY}; border-radius: 5px; border: 1px solid {BORDER};"
        )
        grid_lay = QGridLayout(grid_w)
        grid_lay.setContentsMargins(8, 8, 8, 8)
        grid_lay.setSpacing(4)
        for i, (name, default) in enumerate(metrics):
            col = (i % 2) * 2
            row = i // 2
            grid_lay.addWidget(make_label(name, TEXT_SECONDARY, 9), row * 2, col)
            val_lbl = make_label(default, ACCENT, 11, mono=True)
            self._psf_result_labels[name] = val_lbl
            grid_lay.addWidget(val_lbl, row * 2 + 1, col)
        psf.add_widget(grid_w)
        self._psf_cutout_spin = psf.add_spin("Cutout radius", 6, 32, 12)
        self._psf_force_cpu   = psf.add_check("Force CPU (for parallel use)")
        psf.add_run("▶ Measure PSF", self.measure_psf.emit)
        lay.addWidget(psf)

        # Noise Reduction
        dnz = CollapsibleSection("Noise Reduction")
        self._denoise_method_combo = dnz.add_combo(
            "Method",
            ["TGV Denoise", "NLM (Non-Local Means)", "Wavelet Denoise", "Median Filter"],
        )
        self._denoise_amount     = dnz.add_slider("Amount",     0.5, 0.0, 1.0, 0.05, 2)
        self._denoise_lum        = dnz.add_slider("Luminance",  0.7, 0.0, 1.0, 0.05, 2)
        self._denoise_chrom      = dnz.add_slider("Chrominance",0.5, 0.0, 1.0, 0.05, 2)
        self._denoise_preview_check = dnz.add_check("Live split preview")
        for _sl in (self._denoise_amount, self._denoise_lum, self._denoise_chrom):
            _sl.value_changed.connect(
                lambda _, s=self._denoise_preview_check: self._fire_preview("denoise", s)
            )
        self._denoise_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("denoise") if on
            else self.preview_cancelled.emit()
        )
        dnz.add_run("▶ Apply Denoise", self.run_denoise.emit)
        lay.addWidget(dnz)

        # Star Reduction
        sr = CollapsibleSection("Star Reduction")
        sr.add_info("Reduce star halos to reveal faint nebula details.")
        self._star_reduction_amount = sr.add_slider("Amount (%)", 50, 0, 100, 1, 0)
        self._star_reduction_method = sr.add_combo(
            "Method", ["Morphological", "Halo only", "Full star"]
        )
        sr.add_run("▶ Reduce Stars", self.run_star_reduction.emit)
        lay.addWidget(sr)

        # Wavelets / MLT
        wav = CollapsibleSection("Wavelets / MLT")
        wav.add_info("Multi-scale sharpening with per-layer control.")
        self._wavelet_layers = wav.add_slider("Layers", 5, 2, 8, 1, 0)
        self._wavelet_layer_sliders: list[SliderRow] = []
        defaults = [0.3, 0.3, 0.0, 0.0, 0.0]
        for i in range(5):
            s = wav.add_slider(f"Layer {i+1}", defaults[i], 0.0, 2.0, 0.1, 1)
            self._wavelet_layer_sliders.append(s)
        self._wav_preview_check = wav.add_check("Live split preview")
        for _sl in self._wavelet_layer_sliders:
            _sl.value_changed.connect(
                lambda _, s=self._wav_preview_check: self._fire_preview("wavelet", s)
            )
        self._wav_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("wavelet") if on
            else self.preview_cancelled.emit()
        )
        btns = wav.add_btn_row([("▶ Wavelets", False), ("▶ MLT", False)])
        btns[0].clicked.connect(self.run_wavelet_sharpen.emit)
        btns[1].clicked.connect(self.run_mlt.emit)
        lay.addWidget(wav)

        # CLAHE
        clh = CollapsibleSection("Local Contrast / CLAHE")
        self._clahe_clip  = clh.add_slider("Clip limit", 2.0, 0.5, 10.0, 0.5, 1)
        self._clahe_tiles = clh.add_slider("Tile size",  8,   4,   32,   1,   0)
        self._clahe_preview_check = clh.add_check("Live split preview")
        for _sl in (self._clahe_clip, self._clahe_tiles):
            _sl.value_changed.connect(
                lambda _, s=self._clahe_preview_check: self._fire_preview("local_contrast", s)
            )
        self._clahe_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("local_contrast") if on
            else self.preview_cancelled.emit()
        )
        clh.add_run("▶ Apply CLAHE", self.run_local_contrast.emit)
        lay.addWidget(clh)

        # Unsharp Mask
        um = CollapsibleSection("Unsharp Mask")
        self._um_radius    = um.add_slider("Radius (px)", 1.5, 0.5, 10.0, 0.5, 1)
        self._um_amount    = um.add_slider("Amount",      0.5, 0.0,  2.0, 0.05, 2)
        self._um_threshold = um.add_slider("Threshold",   0.0, 0.0,  0.1, 0.005, 3)
        self._um_preview_check = um.add_check("Live split preview")
        for _sl in (self._um_radius, self._um_amount, self._um_threshold):
            _sl.value_changed.connect(
                lambda _, s=self._um_preview_check: self._fire_preview("unsharp_mask", s)
            )
        self._um_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("unsharp_mask") if on
            else self.preview_cancelled.emit()
        )
        um.add_run("▶ Apply Unsharp Mask", self.run_unsharp_mask.emit)
        lay.addWidget(um)

        # Morphology
        mor = CollapsibleSection("Morphology")
        self._morph_op     = mor.add_combo(
            "Operation", ["Erosion", "Dilation", "Opening", "Closing", "Gradient"]
        )
        self._morph_kernel = mor.add_combo("Kernel", ["Disk", "Square", "Diamond"])
        self._morph_iters  = mor.add_slider("Iterations", 1, 1, 10, 1, 0)
        mor.add_run("▶ Apply Morphology", self.run_morphology.emit)
        lay.addWidget(mor)

        # Chromatic Aberration
        ca = CollapsibleSection("Chromatic Aberration")
        ca.add_info("Correct lateral colour fringing at image edges.")
        ca.add_run("▶ Correct CA", self.run_chromatic_aberration.emit)
        lay.addWidget(ca)

        self._tabs.addTab(scrollable_tab(lay), "◎  Detail")

    # ── TAB 8: AI Tools ───────────────────────────────────
    def _build_ai_tab(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Status card
        status_w = QWidget()
        status_w.setStyleSheet(
            f"background: {BG_SECONDARY}; border: 1px solid {BORDER}; border-radius: 6px;"
        )
        sl = QVBoxLayout(status_w)
        sl.setContentsMargins(10, 10, 10, 10)
        sl.setSpacing(6)
        hdr_row = QHBoxLayout()
        hdr_row.addWidget(make_label("AI Model Status", TEXT_SECONDARY, 10, bold=True))
        hdr_row.addStretch()
        gpu_lbl = QLabel("GPU")
        gpu_lbl.setStyleSheet(
            f"background: {ACCENT_PURPLE}; color: #fff; font-size: 10px; "
            "font-weight: 700; border-radius: 8px; padding: 1px 8px;"
        )
        hdr_row.addWidget(gpu_lbl)
        sl.addLayout(hdr_row)

        _model_statuses = [
            ("AI Denoise (Noise2Self)",   "ready"),
            ("AI Sharpen (Neural Deconv)", "training"),
            ("StarNet (Star Removal)",     "planned"),
        ]
        _status_style = {
            "ready":    (ACCENT_DARK, ACCENT, "Ready"),
            "training": ("#2d1f00", ORANGE,  "Training…"),
            "planned":  ("#1c1c2e", ACCENT_PURPLE, "Planned"),
        }
        for name, status in _model_statuses:
            row = QHBoxLayout()
            row.addWidget(make_label(name, TEXT_PRIMARY, 11))
            row.addStretch()
            bg, col, lbl_text = _status_style[status]
            badge = QLabel(lbl_text)
            badge.setStyleSheet(
                f"background: {bg}; color: {col}; font-size: 10px; "
                "font-weight: 600; border-radius: 8px; padding: 1px 8px;"
            )
            row.addWidget(badge)
            sl.addLayout(row)
        lay.addWidget(status_w)

        # AI Denoise
        den = CollapsibleSection("AI Denoise", accent=True)
        den.add_info(
            "Noise2Self — self-supervised denoising trained on real astro images. 7.7M params."
        )
        self._ai_denoise_strength = den.add_slider("Strength", 0.7, 0.0, 1.0, 0.05, 2)
        self._ai_tile_combo       = den.add_combo("Tile size", ["128", "256", "512", "Full"])
        self._ai_star_protect     = den.add_check("Protect stars (star mask)", True)
        self._ai_tiled_check      = den.add_check("Tiled inference (reduces VRAM)", True)
        den.add_run("▶ Apply AI Denoise", self.run_ai_denoise.emit)
        lay.addWidget(den)

        # AI Sharpen
        shr = CollapsibleSection("AI Sharpen")
        # warning banner
        warn = QLabel("Model training in progress (epoch 16/30). Traditional fallback active.")
        warn.setWordWrap(True)
        warn.setStyleSheet(
            f"background: #2d1f00; color: {ORANGE}; border: 1px solid rgba(210,153,34,0.19); "
            "border-radius: 5px; padding: 6px 8px; font-size: 10px;"
        )
        shr.add_widget(warn)
        self._ai_sharpen_strength = shr.add_slider("Strength", 0.5, 0.0, 1.0, 0.05, 2)
        shr.add_run("▶ Apply AI Sharpen", self.run_ai_sharpen.emit)
        lay.addWidget(shr)

        # StarNet
        star = CollapsibleSection("StarNet (Star Removal)")
        info_banner = QLabel("Architecture planned. Use Star Reduction in Detail tab for now.")
        info_banner.setWordWrap(True)
        info_banner.setStyleSheet(
            f"background: #1c1c2e; color: {ACCENT_PURPLE}; "
            "border: 1px solid rgba(137,87,229,0.19); "
            "border-radius: 5px; padding: 6px 8px; font-size: 10px;"
        )
        star.add_widget(info_banner)
        btn_sn = star.add_run("▶ Run StarNet", self.run_starnet.emit)
        btn_sn.setEnabled(False)
        lay.addWidget(star)

        # Train
        train = CollapsibleSection("Train Your Own Models")
        train.add_info("Self-supervised training on your own astro images.")
        train.add_code_block(
            "poetry run python scripts/\ntrain_denoise_model.py\n--input astro_data --epochs 30"
        )
        train.add_run("Open Training Guide…", flat=True)
        lay.addWidget(train)

        self._tabs.addTab(scrollable_tab(lay), "✦  AI Tools")

    # ── TAB 9: Utility ────────────────────────────────────
    def _build_utility_tab(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Pixel Math
        pm = CollapsibleSection("Pixel Math", accent=True)
        pm.add_info("Custom per-pixel math. Variables: R, G, B, L, img1, img2.")
        self._pixelmath_expr = QTextEdit()
        self._pixelmath_expr.setPlaceholderText("R * 0.5 + B * 0.5")
        self._pixelmath_expr.setFixedHeight(54)
        self._pixelmath_expr.setStyleSheet(
            f"background: {BG_TERTIARY}; color: {ACCENT}; "
            f"border: 1px solid {BORDER}; border-radius: 5px; "
            f"padding: 4px 8px; font-family: {FONT_MONO}; font-size: 11px;"
        )
        pm.add_widget(self._pixelmath_expr)
        pm.add_run("⊞ Open Pixel Math Dialog…", self.open_pixelmath_dialog.emit, flat=True)
        lay.addWidget(pm)

        # HDR
        hdr = CollapsibleSection("HDR Composition")
        hdr.add_info("Merge differently-exposed images for extended dynamic range.")
        hdr.add_run("⊞ Open HDR Dialog…", self.open_hdr_dialog.emit, flat=True)
        lay.addWidget(hdr)

        # Blink
        blink = CollapsibleSection("Blink Comparator")
        blink.add_info("Rapidly alternate between two images to spot differences.")
        btns = blink.add_btn_row([("Load A…", True), ("Load B…", True)])
        btns[0].clicked.connect(self.blink_load_a.emit)
        btns[1].clicked.connect(self.blink_load_b.emit)
        btns2 = blink.add_btn_row([("Current → A", True), ("Current → B", True)])
        btns2[0].clicked.connect(self.blink_use_current_as_a.emit)
        btns2[1].clicked.connect(self.blink_use_current_as_b.emit)
        self._blink_fps = blink.add_slider("FPS", 2, 1, 10, 1, 0)
        self._blink_fps.value_changed.connect(lambda v: self.blink_fps_changed.emit(int(v)))
        self._btn_blink_toggle = blink.add_run("▶ Start Blinking")
        self._blink_active = False
        self._btn_blink_toggle.clicked.connect(self._on_blink_toggle)
        lay.addWidget(blink)

        # Macros
        mac = CollapsibleSection("Macros / Scripting")
        mac.add_info("Record, edit, and replay processing sequences.")
        btns = mac.add_btn_row([("⏺ Record", True), ("⏹ Stop", True), ("▶ Play", True)])
        btns[0].clicked.connect(self.start_macro_recording.emit)
        btns[1].clicked.connect(self.stop_macro_recording.emit)
        btns[2].clicked.connect(self.play_macro.emit)
        btns2 = mac.add_btn_row([("Save Macro…", True), ("Load Macro…", True)])
        btns2[0].clicked.connect(self.save_macro.emit)
        btns2[1].clicked.connect(self.load_macro.emit)
        lay.addWidget(mac)

        # Python Console
        con = CollapsibleSection("Python Console")
        con.add_info("Full Python access to the Cosmica core API.")
        con.add_run("⊞ Open Python Console…", self.open_python_console.emit, flat=True)
        lay.addWidget(con)

        # Statistics
        stats = CollapsibleSection("Image Statistics")
        stats.add_info("Mean, median, SD, min, max, histogram percentiles.")
        stats.add_run("⊞ Show Statistics…", self.show_image_statistics.emit, flat=True)
        lay.addWidget(stats)

        self._tabs.addTab(scrollable_tab(lay), "⚙  Utility")

    # ── Internal helpers ──────────────────────────────────

    def _fire_preview(self, tool_name: str, check: "QCheckBox") -> None:
        if check.isChecked():
            self.preview_requested.emit(tool_name)

    def _emit_clip_points(self):
        self.clip_points_changed.emit(
            float(self._ht_black_spin.value()),
            float(self._ht_white_spin.value()),
        )

    def _on_blink_toggle(self):
        self._blink_active = not self._blink_active
        self.blink_toggle.emit(self._blink_active)
        self._btn_blink_toggle.setText(
            "⏹ Stop Blinking" if self._blink_active else "▶ Start Blinking"
        )
        self._btn_blink_toggle.setStyleSheet(
            self._btn_blink_toggle.styleSheet().replace(
                ACCENT, "#c93030" if self._blink_active else ACCENT
            )
        )

    # ── Public setters (called from main_window) ──────────

    def set_calibration_status(
        self,
        bias: str | None,
        dark: str | None,
        flat: str | None,
    ) -> None:
        self._cal_bias_label.setText(f"Bias: {bias or 'none'}")
        self._cal_dark_label.setText(f"Dark: {dark or 'none'}")
        self._cal_flat_label.setText(f"Flat: {flat or 'none'}")

    def set_bg_sample_count(self, n: int) -> None:
        self._bg_sample_label.setText(f"{n} manual sample{'s' if n != 1 else ''}")

    def set_psf_result(self, fwhm: float, fwhm_x: float, fwhm_y: float,
                       ellipticity: float, theta: float,
                       n_stars: int, fwhm_std: float) -> None:
        updates = {
            "FWHM": f"{fwhm:.2f} px",
            "FWHM X": f"{fwhm_x:.2f} px",
            "FWHM Y": f"{fwhm_y:.2f} px",
            "Ellipticity": f"{ellipticity:.3f}",
            "Rotation": f"{theta:.1f}°",
            "Stars used": str(n_stars),
            "FWHM σ": f"{fwhm_std:.2f} px",
        }
        for key, val in updates.items():
            if key in self._psf_result_labels:
                self._psf_result_labels[key].setText(val)

    def set_subframe_count(self, n_selected: int, n_total: int) -> None:
        if n_selected == 0:
            self._subframe_count_label.setText("No subframe selection active")
        else:
            self._subframe_count_label.setText(
                f"{n_selected} / {n_total} frames selected"
            )

    def add_multi_session(self, label: str) -> None:
        self._ms_session_list.addItem(label)
        self._btn_ms_stack.setEnabled(self._ms_session_list.count() > 0)

    def clear_multi_sessions(self) -> None:
        self._ms_session_list.clear()
        self._btn_ms_stack.setEnabled(False)

    # ── Public getters (called from main_window) ──────────

    def get_stacking_params(self) -> StackingParams:
        rejection_map = {
            "Sigma Clipping":      RejectionMethod.SIGMA_CLIP,
            "Winsorized Sigma":    RejectionMethod.WINSORIZED,
            "Linear Fit":         RejectionMethod.LINEAR_FIT,
            "Percentile Clip":    RejectionMethod.PERCENTILE,
            "ESD (Generalized)":  RejectionMethod.ESD,
            "Min/Max":            RejectionMethod.MINMAX,
            "None":               RejectionMethod.NONE,
        }
        integ_map = {
            "Average":          IntegrationMethod.AVERAGE,
            "Median":           IntegrationMethod.MEDIAN,
            "Weighted Average": IntegrationMethod.WEIGHTED_AVERAGE,
        }
        return StackingParams(
            rejection=rejection_map.get(
                self._rejection_combo.currentText(), RejectionMethod.SIGMA_CLIP
            ),
            integration=integ_map.get(
                self._integration_combo.currentText(), IntegrationMethod.AVERAGE
            ),
            kappa=self._kappa_spin.value(),
        )

    def get_stretch_params(self) -> StretchParams:
        return StretchParams(
            midtone=self._midtone_slider.value(),
            shadow_clip=float(self._shadow_spin.value()),
            linked=self._linked_check.isChecked(),
        )

    def get_ghs_params(self) -> GHSParams:
        return GHSParams(
            D=float(self._ghs_d_spin.value()),
            b=float(self._ghs_b_spin.value()),
            SP=float(self._ghs_sp_spin.value()),
            shadow_protection=self._ghs_shadow_slider.value(),
            highlight_protection=self._ghs_highlight_slider.value(),
        )

    def get_arcsinh_params(self) -> ArcsinhStretchParams:
        return ArcsinhStretchParams(
            stretch_factor=float(self._arcsinh_factor_spin.value()),
            black_point=float(self._arcsinh_bp_spin.value()),
            linked=self._arcsinh_linked_check.isChecked(),
        )

    def get_histogram_transform_params(self) -> HistogramTransformParams:
        return HistogramTransformParams(
            black_point=float(self._ht_black_spin.value()),
            midtone=self._ht_midtone_slider.value(),
            white_point=float(self._ht_white_spin.value()),
        )

    def reset_histogram_transform_params(self) -> None:
        self._ht_black_spin.setValue(0.0)
        self._ht_midtone_slider.setValue(0.5)
        self._ht_white_spin.setValue(1.0)

    def get_background_params(self) -> BackgroundParams:
        return BackgroundParams(
            grid_size=int(self._bg_grid_spin.value()),
            poly_order=int(self._bg_order_spin.value()),
        )

    def get_background_neutralization_params(self) -> BackgroundNeutralizationParams:
        return BackgroundNeutralizationParams(
            percentile=self._bn_percentile.value(),
            amount=self._bn_amount.value(),
            protect_bright=self._bn_protect.value(),
        )

    def get_cosmetic_params(self) -> CosmeticParams:
        return CosmeticParams(
            hot_sigma=self._hot_sigma.value(),
            cold_sigma=self._cold_sigma.value(),
            fix_dead=self._dead_pixel_check.isChecked(),
        )

    def get_vignette_params(self) -> VignetteParams:
        return VignetteParams(
            amount=self._vignette_amount.value(),
            radius=self._vignette_radius.value(),
        )

    def get_banding_params(self) -> BandingParams:
        return BandingParams(
            amount=self._banding_amount.value(),
            direction=self._banding_dir_combo.currentText().lower(),
        )

    def get_ai_denoise_params(self) -> AIDenoiseParams:
        tile_map = {"128": 128, "256": 256, "512": 512, "Full": 0}
        return AIDenoiseParams(
            strength=self._ai_denoise_strength.value(),
            tile_size=tile_map.get(self._ai_tile_combo.currentText(), 256),
            protect_stars=self._ai_star_protect.isChecked(),
            tiled=self._ai_tiled_check.isChecked(),
        )

    def get_deconvolution_params(self) -> "DeconvolutionParams | SpatialDeconvParams":
        method = self._deconv_method_combo.currentText()
        iters = int(self._deconv_iter.value())
        reg = float(self._deconv_reg.value())
        if method == "Blind (Spatial)":
            return SpatialDeconvParams(
                iterations=iters,
                regularization=reg,
            )
        return DeconvolutionParams(
            psf_fwhm=float(self._deconv_psf_spin.value()),
            iterations=iters,
            regularization=reg,
            deringing=self._deconv_deringing.isChecked(),
            deringing_amount=self._deconv_dering_amt.value(),
        )

    def set_psf_fwhm(self, fwhm: float) -> None:
        """Auto-populate the PSF FWHM field from a Measure PSF result."""
        self._deconv_psf_spin.setValue(round(fwhm, 1))

    def get_denoise_params(self) -> DenoiseParams:
        return DenoiseParams(
            method=self._denoise_method_combo.currentText(),
            amount=self._denoise_amount.value(),
            luminance=self._denoise_lum.value(),
            chrominance=self._denoise_chrom.value(),
        )

    def get_star_reduction_params(self) -> StarReductionParams:
        return StarReductionParams(
            amount=self._star_reduction_amount.value() / 100.0,
            method=self._star_reduction_method.currentText(),
        )

    def get_unsharp_mask_params(self) -> UnsharpMaskParams:
        return UnsharpMaskParams(
            radius=self._um_radius.value(),
            amount=self._um_amount.value(),
            threshold=self._um_threshold.value(),
        )

    def get_local_contrast_params(self) -> LocalContrastParams:
        return LocalContrastParams(
            clip_limit=self._clahe_clip.value(),
            tile_size=int(self._clahe_tiles.value()),
        )

    def get_scnr_params(self) -> SCNRParams:
        return SCNRParams(
            target=self._scnr_target_combo.currentText().lower(),
            method=self._scnr_method_combo.currentText(),
            amount=self._scnr_amount.value(),
        )

    def get_color_adjust_params(self) -> ColorAdjustParams:
        return ColorAdjustParams(
            hue=self._hue_slider.value(),
            saturation=self._sat_slider.value() / 100.0,
            vibrance=self._vibrance_slider.value() / 100.0,
            lightness=self._lightness_slider.value() / 100.0,
        )

    def get_curves_params(self) -> CurvesParams:
        channel_map = {
            "Master (L)": "luminance",
            "Red": "red", "Green": "green", "Blue": "blue",
        }
        return CurvesParams(
            channel=channel_map.get(
                self._curve_channel_combo.currentText(), "luminance"
            ),
            points=self._curve_editor.get_points(),
        )

    def get_wavelet_params(self) -> WaveletParams:
        return WaveletParams(
            n_layers=int(self._wavelet_layers.value()),
            layer_amounts=[s.value() for s in self._wavelet_layer_sliders],
        )

    def get_crop_params(self) -> CropParams:
        return CropParams(
            x=int(self._crop_x_spin.value()),
            y=int(self._crop_y_spin.value()),
            width=int(self._crop_w_spin.value()) or None,
            height=int(self._crop_h_spin.value()) or None,
        )

    def get_rotate_params(self) -> RotateParams:
        preset_map = {
            "90° CW": 90, "180°": 180, "270° CW": 270
        }
        text = self._rotate_combo.currentText()
        angle = preset_map.get(text, float(self._rotate_angle_spin.value()))
        return RotateParams(
            angle=angle,
            expand=self._rotate_expand_check.isChecked(),
        )

    def get_flip_params(self) -> FlipParams:
        return FlipParams(axis=self._flip_combo.currentText().lower())

    def get_resize_params(self) -> ResizeParams:
        return ResizeParams(
            scale=float(self._resize_scale_spin.value()),
            interpolation=self._resize_interp_combo.currentText(),
        )

    def get_bin_params(self) -> BinParams:
        factor_map = {"2x2": 2, "3x3": 3, "4x4": 4}
        return BinParams(
            factor=factor_map.get(self._bin_factor_combo.currentText(), 2),
            mode=self._bin_mode_combo.currentText().lower(),
        )

    def get_abe_params(self) -> ABEParams:
        return ABEParams(
            grid_size=int(self._abe_grid_spin.value()),
            model=self._abe_model_combo.currentText(),
            degree=int(self._abe_degree_spin.value()),
            kernel=self._abe_kernel_combo.currentText(),
            mode=self._abe_mode_combo.currentText().lower(),
        )

    def get_morphology_params(self) -> MorphologyParams:
        return MorphologyParams(
            operation=self._morph_op.currentText(),
            kernel=self._morph_kernel.currentText(),
            iterations=int(self._morph_iters.value()),
        )

    def get_color_calibration_params(self) -> ColorCalibrationParams:
        return ColorCalibrationParams(
            method=self._cc_method_combo.currentText(),
        )

    # ── Compatibility properties used by main_window ──────
    @property
    def split_preview_enabled(self) -> bool:
        return self._split_check.isChecked()

    @property
    def curve_editor(self) -> "CurveEditor":
        return self._curve_editor

    @property
    def curves_histogram_visible(self) -> bool:
        return self._curves_histogram_check.isChecked()

    @property
    def current_curve_channel(self) -> int:
        return self._curve_channel_combo.currentIndex()
