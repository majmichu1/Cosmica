"""Main Window — 4-panel layout: project(left), canvas(center), tools(right), log(bottom)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialogButtonBox,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import cosmica
from cosmica.core.abe import abe_extract
from cosmica.core.background import extract_background
from cosmica.core.banding import banding_reduction
from cosmica.core.calibration import (
    calibrate_lights_batch,
    create_master_bias,
    create_master_dark,
    create_master_flat,
)
from cosmica.core.channels import extract_luminance, split_channels
from cosmica.core.chromatic_aberration import correct_chromatic_aberration
from cosmica.core.color_calibration import color_calibrate
from cosmica.core.color_tools import color_adjust, scnr
from cosmica.core.cosmetic import cosmetic_correction
from cosmica.core.curves import CurvesParams, curves_transform
from cosmica.core.deconvolution import (
    SpatialDeconvParams,
    richardson_lucy,
    richardson_lucy_spatial,
)
from cosmica.core.denoise import denoise
from cosmica.core.device_manager import get_device_manager
from cosmica.core.equipment import EquipmentProfile
from cosmica.core.filters import median_filter, unsharp_mask
from cosmica.core.histogram_transform import histogram_transform
from cosmica.core.image_io import (
    FrameType,
    ImageData,
    auto_stretch_for_display_ref,
    load_image,
    save_image,
)
from cosmica.core.local_contrast import local_contrast_enhance
from cosmica.core.masks import Mask
from cosmica.core.morphology import morphology_transform
from cosmica.core.presets import load_default_presets
from cosmica.core.project import Project
from cosmica.core.scripting import MacroRecorder, load_macro, play_macro, save_macro
from cosmica.core.stacking import StackingParams, align_frames, stack_images
from cosmica.core.star_reduction import reduce_stars
from cosmica.core.statistics import compute_image_statistics
from cosmica.core.stretch import (
    auto_stretch,
    compute_histogram,
    generalized_hyperbolic_stretch,
)
from cosmica.core.subframe_selector import SubframeSelectorParams
from cosmica.core.transforms import (
    bin_image,
    crop,
    flip,
    invert,
    resize,
    rotate,
)
from cosmica.core.undo import CosmicaUndoStack
from cosmica.core.vignette import correct_vignette
from cosmica.core.wavelets import wavelet_sharpen
from cosmica.ui.panels.project_panel import ProjectPanel
from cosmica.ui.panels.tools_panel import ToolsPanel
from cosmica.ui.widgets.histogram import HistogramWidget
from cosmica.ui.widgets.image_canvas import ImageCanvas
from cosmica.ui.widgets.log_panel import LogPanel, QtLogHandler

log = logging.getLogger(__name__)


class ProcessingWorker(QThread):
    """Runs processing tasks off the main thread."""

    progress = pyqtSignal(float, str)
    finished = pyqtSignal(object)  # result
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            self._kwargs["progress"] = self._emit_progress
            result = self._func(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as e:
            log.exception("Processing error")
            self.error.emit(str(e))

    def _emit_progress(self, fraction: float, message: str):
        self.progress.emit(fraction, message)


class MainWindow(QMainWindow):
    """The main Cosmica application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{cosmica.__app_name__} v{cosmica.__version__}")
        self.setMinimumSize(1200, 750)
        self.resize(1600, 950)

        self._project: Project | None = None
        self._current_image: ImageData | None = None
        self._worker: ProcessingWorker | None = None
        self._master_bias: ImageData | None = None
        self._master_dark: ImageData | None = None
        self._master_flat: ImageData | None = None
        self._calibrated_lights: list[ImageData] = []

        # Mask registry
        self._masks: list[Mask] = []

        # Macro recording
        self._macro_recorder = MacroRecorder()
        self._current_macro = None  # last recorded/loaded Pipeline

        # Equipment profile for Smart Processor
        self._equipment_profile: EquipmentProfile | None = None

        # Undo/Redo stack
        self._undo_stack = CosmicaUndoStack()
        # Single-element list as mutable reference for undo commands
        self._image_ref: list[ImageData | None] = [None]
        self._undo_stack.set_target(self._image_ref)

        # Preview debounce timers
        self._stretch_preview_timer = QTimer()
        self._stretch_preview_timer.setSingleShot(True)
        self._stretch_preview_timer.setInterval(100)
        self._stretch_preview_timer.timeout.connect(self._do_stretch_preview)
        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(100)
        self._preview_timer.timeout.connect(self._do_preview_requested)
        self._pending_preview_tool: str | None = None

        # Dynamic background extraction samples (image-space coords)
        self._bg_samples: list[tuple[float, float]] = []

        # WCS overlay data (image-space x, y, magnitude)
        self._wcs_overlay_stars: list[tuple[float, float, float]] = []
        self._current_wcs: dict = {}   # last solved WCS dict

        # Python console dock (lazy init)
        self._python_console_dock = None

        # Multi-session stacking
        self._ms_sessions: list = []  # list of SessionGroup objects

        # Blink comparator
        self._blink_images: list = [None, None]  # [A, B] — display RGB uint8 arrays (H,W,3)
        self._blink_names: list[str] = ["", ""]
        self._blink_index = 0
        self._blink_timer = QTimer()
        self._blink_timer.timeout.connect(self._blink_tick)

        # Register tools for preset system
        load_default_presets()

        self.setAcceptDrops(True)
        self._setup_menu()
        self._setup_ui()
        self._setup_logging()
        self._setup_statusbar()

    def _setup_menu(self):
        menu = self.menuBar()

        # File menu
        file_menu = menu.addMenu("&File")

        new_proj = QAction("&New Project...", self)
        new_proj.setShortcut("Ctrl+N")
        new_proj.triggered.connect(self._new_project)
        file_menu.addAction(new_proj)

        open_proj = QAction("&Open Project...", self)
        open_proj.setShortcut("Ctrl+O")
        open_proj.triggered.connect(self._open_project)
        file_menu.addAction(open_proj)

        save_proj = QAction("&Save Project", self)
        save_proj.setShortcut("Ctrl+S")
        save_proj.triggered.connect(self._save_project)
        file_menu.addAction(save_proj)

        file_menu.addSeparator()

        open_img = QAction("Open &Image...", self)
        open_img.setShortcut("Ctrl+I")
        open_img.triggered.connect(self._open_image)
        file_menu.addAction(open_img)

        save_img = QAction("Save Image &As...", self)
        save_img.setShortcut("Ctrl+Shift+S")
        save_img.triggered.connect(self._save_image)
        file_menu.addAction(save_img)

        fits_hdr = QAction("Edit FITS &Header...", self)
        fits_hdr.setShortcut("Ctrl+H")
        fits_hdr.triggered.connect(self._show_fits_header)
        file_menu.addAction(fits_hdr)

        file_menu.addSeparator()

        prefs_act = QAction("&Preferences...", self)
        prefs_act.setShortcut("Ctrl+,")
        prefs_act.triggered.connect(self._show_preferences)
        file_menu.addAction(prefs_act)

        file_menu.addSeparator()

        quit_act = QAction("E&xit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # View menu
        view_menu = menu.addMenu("&View")

        fit_act = QAction("&Fit to Window", self)
        fit_act.setShortcut("Ctrl+0")
        fit_act.triggered.connect(lambda: self._canvas.fit_to_window())
        view_menu.addAction(fit_act)

        zoom100 = QAction("Zoom &100%", self)
        zoom100.setShortcut("Ctrl+1")
        zoom100.triggered.connect(lambda: self._canvas.zoom_to(1.0))
        view_menu.addAction(zoom100)

        zoom200 = QAction("Zoom &200%", self)
        zoom200.setShortcut("Ctrl+2")
        zoom200.triggered.connect(lambda: self._canvas.zoom_to(2.0))
        view_menu.addAction(zoom200)

        # Edit menu
        edit_menu = menu.addMenu("&Edit")

        self._undo_act = QAction("&Undo", self)
        self._undo_act.setShortcut("Ctrl+Z")
        self._undo_act.triggered.connect(self._undo)
        self._undo_act.setEnabled(False)
        edit_menu.addAction(self._undo_act)

        self._redo_act = QAction("Re&do", self)
        self._redo_act.setShortcut("Ctrl+Shift+Z")
        self._redo_act.triggered.connect(self._redo)
        self._redo_act.setEnabled(False)
        edit_menu.addAction(self._redo_act)

        edit_menu.addSeparator()

        clear_undo = QAction("&Clear History", self)
        clear_undo.triggered.connect(self._clear_undo_history)
        edit_menu.addAction(clear_undo)

        # Process menu
        process_menu = menu.addMenu("&Process")

        pm_act = QAction("&Pixel Math...", self)
        pm_act.setShortcut("Ctrl+P")
        pm_act.triggered.connect(self._show_pixelmath_dialog)
        process_menu.addAction(pm_act)

        nb_act = QAction("&Narrowband Combine...", self)
        nb_act.triggered.connect(self._show_narrowband_dialog)
        process_menu.addAction(nb_act)

        hdr_act = QAction("&HDR Composition...", self)
        hdr_act.triggered.connect(self._show_hdr_dialog)
        process_menu.addAction(hdr_act)

        batch_act = QAction("&Batch Processing...", self)
        batch_act.setShortcut("Ctrl+B")
        batch_act.triggered.connect(self._show_batch_dialog)
        process_menu.addAction(batch_act)

        process_menu.addSeparator()

        smart_act = QAction("&Smart Processor...", self)
        smart_act.setShortcut("Ctrl+Shift+P")
        smart_act.triggered.connect(self._show_smart_processor_dialog)
        process_menu.addAction(smart_act)

        equip_act = QAction("&Equipment Profile...", self)
        equip_act.triggered.connect(self._show_equipment_dialog)
        process_menu.addAction(equip_act)

        process_menu.addSeparator()

        macro_start = QAction("Start &Recording", self)
        macro_start.triggered.connect(self._on_start_macro)
        process_menu.addAction(macro_start)

        macro_stop = QAction("Sto&p Recording", self)
        macro_stop.triggered.connect(self._on_stop_macro)
        process_menu.addAction(macro_stop)

        macro_play = QAction("P&lay Macro", self)
        macro_play.triggered.connect(self._on_play_macro)
        process_menu.addAction(macro_play)

        # Masks menu
        masks_menu = menu.addMenu("&Masks")

        create_mask = QAction("Create &Mask...", self)
        create_mask.triggered.connect(self._show_mask_dialog)
        masks_menu.addAction(create_mask)

        # Help menu
        help_menu = menu.addMenu("&Help")

        about_act = QAction("&About Cosmica", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

        dm = get_device_manager()
        device_act = QAction(f"GPU: {dm.info.name}", self)
        device_act.setEnabled(False)
        help_menu.addAction(device_act)

    def _setup_ui(self):
        # Central widget with splitters
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top: horizontal splitter (project | canvas | tools)
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Project panel
        self._project_panel = ProjectPanel()
        self._project_panel.frame_selected.connect(self._load_frame)
        self._project_panel.frames_imported.connect(self._on_frames_imported)
        top_splitter.addWidget(self._project_panel)

        # Center: Canvas + histogram
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self._canvas = ImageCanvas()
        self._canvas.cursor_position.connect(self._update_pixel_readout)
        center_layout.addWidget(self._canvas, 1)

        self._histogram = HistogramWidget()
        center_layout.addWidget(self._histogram)

        top_splitter.addWidget(center_widget)

        # Right: Tools panel
        self._tools_panel = ToolsPanel()
        self._connect_tool_signals()
        top_splitter.addWidget(self._tools_panel)

        # Set initial splitter sizes
        top_splitter.setSizes([250, 900, 320])

        # Vertical splitter (top panels | bottom log)
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.addWidget(top_splitter)

        self._log_panel = LogPanel()
        v_splitter.addWidget(self._log_panel)
        v_splitter.setSizes([750, 120])

        main_layout.addWidget(v_splitter)

    def _connect_tool_signals(self):
        """Wire all tool panel signals to processing handlers."""
        tp = self._tools_panel

        # Existing signals
        tp.run_calibration.connect(self._on_run_calibration)
        tp.run_stacking.connect(self._on_run_stacking)
        tp.run_alignment.connect(self._on_run_alignment)
        tp.run_stretch.connect(self._on_run_stretch)
        tp.run_background.connect(self._on_run_background)
        tp.stretch_params_changed.connect(self._on_stretch_preview)

        # Phase A signals
        tp.run_cosmetic.connect(self._on_run_cosmetic)
        tp.run_banding.connect(self._on_run_banding)
        tp.run_histogram_transform.connect(self._on_run_histogram_transform)
        tp.run_curves.connect(self._on_run_curves)
        tp.run_scnr.connect(self._on_run_scnr)
        tp.run_color_adjust.connect(self._on_run_color_adjust)
        tp.run_deconvolution.connect(self._on_run_deconvolution)

        # Phase B signals
        tp.run_ghs.connect(self._on_run_ghs)
        tp.run_color_calibration.connect(self._on_run_color_calibration)
        tp.run_pcc.connect(self._on_run_pcc)
        tp.run_denoise.connect(self._on_run_denoise)
        tp.run_star_reduction.connect(self._on_run_star_reduction)
        tp.open_narrowband_dialog.connect(self._show_narrowband_dialog)
        tp.open_pixelmath_dialog.connect(self._show_pixelmath_dialog)
        tp.run_split_channels.connect(self._on_run_split_channels)
        tp.run_extract_luminance.connect(self._on_run_extract_luminance)

        # Phase C signals
        tp.run_wavelet_sharpen.connect(self._on_run_wavelet_sharpen)
        tp.run_local_contrast.connect(self._on_run_local_contrast)
        tp.run_morphology.connect(self._on_run_morphology)
        tp.open_hdr_dialog.connect(self._show_hdr_dialog)

        # Phase D signals
        tp.run_ai_denoise.connect(self._on_run_ai_denoise)
        tp.run_ai_sharpen.connect(self._on_run_ai_sharpen)
        tp.run_starnet.connect(self._on_run_starnet)
        tp.open_batch_dialog.connect(self._show_batch_dialog)
        tp.start_macro_recording.connect(self._on_start_macro)
        tp.stop_macro_recording.connect(self._on_stop_macro)
        tp.play_macro.connect(self._on_play_macro)
        tp.save_macro.connect(self._on_save_macro)
        tp.load_macro.connect(self._on_load_macro)

        # Transform signals
        tp.start_crop_draw.connect(self._on_start_crop_draw)
        tp.run_crop.connect(self._on_run_crop)
        tp.run_rotate.connect(self._on_run_rotate)
        tp.run_flip.connect(self._on_run_flip)
        tp.run_resize.connect(self._on_run_resize)
        tp.run_bin.connect(self._on_run_bin)
        tp.run_invert.connect(self._on_run_invert)

        # New tool signals
        tp.run_unsharp_mask.connect(self._on_run_unsharp_mask)
        tp.run_median_filter.connect(self._on_run_median_filter)
        tp.run_abe.connect(self._on_run_abe)
        tp.run_vignette_correction.connect(self._on_run_vignette)
        tp.run_chromatic_aberration.connect(self._on_run_ca)
        tp.show_image_statistics.connect(self._on_show_statistics)
        tp.open_star_mask_dialog.connect(self._on_open_star_mask)
        tp.open_subframe_selector.connect(self._on_open_subframe_selector)
        tp.measure_psf.connect(self._on_measure_psf)
        tp.run_continuum_subtraction.connect(self._on_run_continuum_subtraction)
        tp.toggle_sample_mode.connect(self._on_toggle_sample_mode)
        tp.clear_bg_samples.connect(self._on_clear_bg_samples)
        tp.toggle_wcs_overlay.connect(self._on_toggle_wcs_overlay)
        tp.toggle_dso_overlay.connect(self._on_toggle_dso_overlay)
        tp.open_python_console.connect(self._on_open_python_console)
        tp.run_mlt.connect(self._on_run_mlt)
        tp.run_lrgb_combine.connect(self._on_run_lrgb_combine)
        tp.run_spcc.connect(self._on_run_spcc)

        # Multi-session stacking
        tp.run_multi_session.connect(self._on_run_multi_session)
        tp.multi_session_add_folder.connect(self._on_ms_add_folder)
        tp.multi_session_clear.connect(self._on_ms_clear)

        # Blink comparator
        tp.blink_load_a.connect(lambda: self._blink_load_from_file(slot=0))
        tp.blink_load_b.connect(lambda: self._blink_load_from_file(slot=1))
        tp.blink_use_current_as_a.connect(lambda: self._blink_use_current(slot=0))
        tp.blink_use_current_as_b.connect(lambda: self._blink_use_current(slot=1))
        tp.blink_toggle.connect(self._on_blink_toggle)
        tp.blink_fps_changed.connect(self._on_blink_fps_changed)

        # Canvas sample signals
        self._canvas.sample_placed.connect(self._on_sample_placed)
        self._canvas.sample_removed.connect(self._on_sample_removed)
        self._canvas.crop_rect_selected.connect(self._on_crop_rect_selected)

        # Preview signals
        tp.preview_requested.connect(self._on_preview_requested)
        tp.preview_cancelled.connect(self._on_preview_cancelled)
        tp.curves_histogram_changed.connect(self._update_curves_histogram)

        # Smart Processor signals
        tp.open_smart_processor.connect(self._show_smart_processor_dialog)
        tp.open_equipment_dialog.connect(self._show_equipment_dialog)

    def _setup_logging(self):
        handler = QtLogHandler(self._log_panel)
        handler.setLevel(logging.INFO)
        logging.getLogger("cosmica").addHandler(handler)
        logging.getLogger("cosmica").setLevel(logging.INFO)
        self._log_panel.log("Cosmica started", "success")

        dm = get_device_manager()
        self._log_panel.log(f"Device: {dm.info.name} ({dm.backend.name})", "info")

    def _setup_statusbar(self):
        from PyQt6.QtWidgets import QLabel as _QLabel
        self._pixel_label = self.statusBar()
        self._preview_indicator = _QLabel("")
        self._preview_indicator.setStyleSheet(
            "color: #00cc44; font-weight: bold; padding: 0 8px;"
        )
        self.statusBar().addPermanentWidget(self._preview_indicator)
        self.statusBar().showMessage("Ready")

    # ---------- File operations ----------

    def _new_project(self):
        name, ok = QInputDialog.getText(self, "New Project", "Project name:")
        if not ok or not name.strip():
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose Project Location")
        if not directory:
            return
        self._project = Project.create(name.strip(), Path(directory))
        self._project_panel.set_project(self._project)
        self._log_panel.log(f"Created project: {name}", "success")

    def _open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "Cosmica Project (cosmica_project.json)"
        )
        if path:
            self._project = Project.load(Path(path))
            self._project_panel.set_project(self._project)
            self._log_panel.log(f"Opened project: {self._project.name}", "success")

    def _save_project(self):
        """Helper to save current project safely."""
        if self._project:
            try:
                self._project.save()
            except Exception as e:
                log.debug("Failed to save project: %s", e)

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "All Supported (*.fit *.fits *.fts *.xisf *.tif *.tiff *.png);;FITS (*.fit *.fits *.fts);;XISF (*.xisf);;All (*)",
        )
        if path:
            self._load_frame(path)

    def _save_image(self):
        if self._current_image is None:
            return
        from cosmica.ui.dialogs.export_dialog import ExportDialog

        dialog = ExportDialog(self)
        if dialog.exec() != QDialogButtonBox.StandardButton.Ok:
            return
        params = dialog.get_export_params()
        try:
            save_image(
                self._current_image,
                path=params["path"],
                bit_depth=params["bit_depth"],
                jpeg_quality=params["jpeg_quality"],
            )
            self._log_panel.log(f"Image exported: {params['path']}", "success")
        except Exception as e:
            log.exception("Export failed")
            self._log_panel.log(f"Export failed: {e}", "error")
            QMessageBox.critical(self, "Export Error", f"Failed to export image:\n{e}")

    def _show_fits_header(self):
        if self._current_image is None:
            self._log_panel.log("Load an image first", "warning")
            return
        from cosmica.ui.dialogs.fits_header_dialog import FITSHeaderDialog
        path = getattr(self._current_image, "file_path", None)
        dlg = FITSHeaderDialog(self._current_image.header, file_path=path, parent=self)
        if dlg.exec():
            self._current_image.header.update(dlg.get_header())
            self._log_panel.log("FITS header updated", "success")

    def _show_preferences(self):
        from cosmica.ui.dialogs.preferences_dialog import PreferencesDialog

        dialog = PreferencesDialog(self)
        if dialog.exec() == QDialogButtonBox.StandardButton.Ok:
            dialog.save()
            self._log_panel.log("Preferences saved", "success")
            self._apply_preferences(dialog.get_prefs())

    def _apply_preferences(self, prefs: dict):
        """Apply preference changes to running application."""
        # GPU device
        if prefs["processing"]["use_gpu"]:
            dev = get_device_manager()
            if dev.has_gpu():
                self._log_panel.log(f"GPU device: {dev.info.name}", "info")
            else:
                self._log_panel.log("GPU requested but not available, using CPU", "warning")

        # Pixel readout format
        fmt = prefs["appearance"]["pixel_readout_format"]
        self._pixel_format = fmt

        # Histogram log scale
        self._histogram.set_log_scale(prefs["appearance"]["histogram_log_scale"])

    def _show_mask_dialog(self):
        if self._current_image is None:
            self._log_panel.log("Load an image first to create masks", "warning")
            return
        from cosmica.ui.dialogs.mask_dialog import MaskDialog

        dialog = MaskDialog(self._current_image.data, self)
        dialog.mask_created.connect(self._on_mask_created)
        dialog.exec()

    def _on_mask_created(self, mask: Mask):
        self._masks.append(mask)
        self._log_panel.log(f"Created mask: {mask.name}", "success")

    def _show_narrowband_dialog(self):
        from cosmica.ui.dialogs.narrowband_dialog import NarrowbandDialog

        dialog = NarrowbandDialog(self)
        dialog.result_ready.connect(self._on_narrowband_result)
        dialog.exec()

    def _on_narrowband_result(self, data):
        self._update_current_image(data, "Narrowband combine complete")
        if self._project:
            self._project.add_history("Narrowband Combine", {})

    def _show_pixelmath_dialog(self):
        if self._current_image is None:
            self._log_panel.log("Load an image first", "warning")
            return
        from cosmica.ui.dialogs.pixelmath_dialog import PixelMathDialog

        dialog = PixelMathDialog(self._current_image.data, self)
        dialog.result_ready.connect(self._on_pixelmath_result)
        dialog.exec()

    def _on_pixelmath_result(self, data):
        self._update_current_image(data, "Pixel Math applied")
        if self._project:
            self._project.add_history("Pixel Math", {})

    def _show_about(self):
        dm = get_device_manager()
        QMessageBox.about(
            self,
            "About Cosmica",
            f"<h2>Cosmica v{cosmica.__version__}</h2>"
            f"<p>Professional astrophotography image processing</p>"
            f"<p>GPU: {dm.info.name} ({dm.backend.name})</p>"
            f"<p>&copy; 2024 Cosmica Team</p>",
        )

    # ---------- Image display ----------

    @pyqtSlot(str)
    def _load_frame(self, path: str):
        self._log_panel.log(f"Loading {Path(path).name}...", "info")

        def _load_work(path, progress=None):
            return load_image(path)

        def _on_loaded(image):
            self._display_image(image)
            self._log_panel.log(f"Loaded: {Path(path).name} ({image.shape_str})", "info")

        self._start_worker(_load_work, path, on_done=_on_loaded)

    def _display_image(self, image: ImageData):
        self._current_image = image
        self._image_ref[0] = image  # sync with undo ref
        rgb = image.to_display(stretch=True)
        self._canvas.set_image(rgb, image.data)

        hist_data = compute_histogram(image.data)
        self._histogram.set_histogram_data(hist_data)
        self._update_curves_histogram(hist_data)
        self._sync_console_image()

    def _update_curves_histogram(self, hist_data: dict | None = None):
        """Push histogram data into the curve editor if the show-histogram checkbox is on."""
        tp = self._tools_panel
        if not tp.curves_histogram_visible:
            tp.curve_editor.set_histogram(None)
            return
        if hist_data is None:
            if self._current_image is None:
                return
            hist_data = compute_histogram(self._current_image.data)
        # Pick the channel matching the current curve selector
        channel_map = {0: "luminance", 1: "red", 2: "green", 3: "blue"}
        key = channel_map.get(tp.current_curve_channel, "luminance")
        counts = hist_data.get(key) or hist_data.get("gray")
        if counts is not None:
            import numpy as np
            tp.curve_editor.set_histogram(np.asarray(counts, dtype=np.float32))
        else:
            tp.curve_editor.set_histogram(None)

    def _push_undo(self, before: ImageData, after: ImageData, description: str):
        """Push an undo command and update the Undo/Redo action states."""
        self._undo_stack.push(before, after, description)
        self._update_undo_actions()

    def _update_undo_actions(self):
        """Update the enabled state and text of undo/redo actions."""
        self._undo_act.setEnabled(self._undo_stack.can_undo())
        self._redo_act.setEnabled(self._undo_stack.can_redo())
        undo_text = self._undo_stack.undo_text()
        redo_text = self._undo_stack.redo_text()
        self._undo_act.setText(f"&Undo ({undo_text})" if undo_text else "&Undo")
        self._redo_act.setText(f"Re&do ({redo_text})" if redo_text else "Re&do")

    def _undo(self):
        if self._undo_stack.undo():
            self._display_image(self._image_ref[0])
            self._log_panel.log(f"Undid: {self._undo_stack.redo_text()}", "info")

    def _redo(self):
        if self._undo_stack.redo():
            self._display_image(self._image_ref[0])
            self._log_panel.log(f"Redid: {self._undo_stack.undo_text()}", "info")

    def _clear_undo_history(self):
        self._undo_stack.clear()
        self._update_undo_actions()
        self._log_panel.log("Undo history cleared", "info")

    def _update_current_image(self, data, message: str, undo_desc: str | None = None):
        """Replace current image data and update display, recording undo."""
        before = self._current_image
        image = ImageData(
            data=data,
            header=self._current_image.header.copy() if self._current_image else {},
            frame_type=self._current_image.frame_type if self._current_image else FrameType.RESULT,
        )
        if before is not None:
            desc = undo_desc if undo_desc else message
            self._push_undo(before, image, desc)
        self._display_image(image)
        self._log_panel.log(message, "success")

    @pyqtSlot(int, int, list)
    def _update_pixel_readout(self, x: int, y: int, values: list):
        if len(values) == 1:
            self.statusBar().showMessage(f"x={x} y={y}  |  L={values[0]:.5f}")
        elif len(values) >= 3:
            self.statusBar().showMessage(
                f"x={x} y={y}  |  R={values[0]:.5f}  G={values[1]:.5f}  B={values[2]:.5f}"
            )

    # ---------- Drag and drop ----------

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
        supported = [
            p
            for p in paths
            if p.suffix.lower() in (".fit", ".fits", ".fts", ".xisf", ".tif", ".tiff", ".png")
        ]
        if not supported:
            return

        if self._project:
            # Add to project as lights by default
            self._on_frames_imported(supported, FrameType.LIGHT)
        else:
            # Just display the first one
            self._load_frame(str(supported[0]))

    def _on_frames_imported(self, paths: list[Path], frame_type: FrameType):
        if self._project is None:
            self._log_panel.log("Create a project first (File > New Project)", "warning")
            return
        added = self._project.add_frames(paths, frame_type)
        self._project_panel.refresh()
        self._log_panel.log(f"Imported {added} {frame_type.name.lower()} frame(s)", "success")
        self._save_project()

    # ---------- Worker management ----------

    def _start_worker(self, func, *args, on_done=None, **kwargs):
        if self._worker is not None and self._worker.isRunning():
            self._log_panel.log("A processing task is already running", "warning")
            return

        self._worker = ProcessingWorker(func, *args, **kwargs)
        self._worker.progress.connect(self._log_panel.update_progress)
        self._worker.error.connect(lambda msg: self._log_panel.log(f"Error: {msg}", "error"))
        if on_done:
            self._worker.finished.connect(on_done)
        self._worker.finished.connect(lambda: self._log_panel.reset_progress())
        self._worker.start()

    # ---------- Processing operations ----------

    @pyqtSlot()
    def _on_run_calibration(self):
        if self._project is None:
            self._log_panel.log("No project loaded", "warning")
            return

        light_frames = self._project.frames_by_type(FrameType.LIGHT)
        if not light_frames:
            self._log_panel.log("No light frames to calibrate", "warning")
            return

        # Get calibration sources from panel (raw folders OR pre-made masters)
        cal_sources = self._tools_panel.get_calibration_sources()

        # Fall back to project-imported frames if panel has nothing configured
        bias_paths  = (cal_sources["bias_paths"]  or
                       [e.path for e in self._project.frames_by_type(FrameType.BIAS)])
        dark_paths  = (cal_sources["dark_paths"]  or
                       [e.path for e in self._project.frames_by_type(FrameType.DARK)])
        flat_paths  = (cal_sources["flat_paths"]  or
                       [e.path for e in self._project.frames_by_type(FrameType.FLAT)])

        bias_master_path  = cal_sources["bias_master"]
        dark_master_path  = cal_sources["dark_master"]
        flat_master_path  = cal_sources["flat_master"]

        n_bias = len(bias_paths) or (1 if bias_master_path else 0)
        n_dark = len(dark_paths) or (1 if dark_master_path else 0)
        n_flat = len(flat_paths) or (1 if flat_master_path else 0)
        self._log_panel.log(
            f"Starting calibration — {len(light_frames)} lights, "
            f"{n_bias} bias, {n_dark} dark, {n_flat} flat", "info"
        )
        self._start_worker(
            self._calibration_pipeline,
            bias_paths, dark_paths, flat_paths,
            [e.path for e in light_frames],
            bias_master_path, dark_master_path, flat_master_path,
            on_done=self._on_calibration_done,
        )

    @staticmethod
    def _calibration_pipeline(
        bias_paths, dark_paths, flat_paths, light_paths,
        bias_master_path=None, dark_master_path=None, flat_master_path=None,
        progress=None,
    ):
        results = {}
        prog = progress or (lambda f, m: None)

        # ── Master bias ───────────────────────────────────────────────────────
        master_bias = None
        if bias_master_path:
            from cosmica.core.image_io import load_image
            master_bias = load_image(bias_master_path)
            prog(0.05, f"Loaded master bias: {Path(bias_master_path).name}")
            results["master_bias"] = master_bias
        elif bias_paths:
            prog(0.0, f"Creating master bias from {len(bias_paths)} frames…")
            r = create_master_bias(bias_paths, progress=lambda f, m: prog(f * 0.15, m))
            master_bias = r.master
            results["master_bias"] = master_bias

        # ── Master dark ───────────────────────────────────────────────────────
        master_dark = None
        if dark_master_path:
            from cosmica.core.image_io import load_image
            master_dark = load_image(dark_master_path)
            prog(0.15, f"Loaded master dark: {Path(dark_master_path).name}")
            results["master_dark"] = master_dark
        elif dark_paths:
            prog(0.15, f"Creating master dark from {len(dark_paths)} frames…")
            r = create_master_dark(
                dark_paths, master_bias=master_bias,
                progress=lambda f, m: prog(0.15 + f * 0.2, m),
            )
            master_dark = r.master
            results["master_dark"] = master_dark

        # ── Master flat ───────────────────────────────────────────────────────
        master_flat = None
        if flat_master_path:
            from cosmica.core.image_io import load_image
            master_flat = load_image(flat_master_path)
            prog(0.35, f"Loaded master flat: {Path(flat_master_path).name}")
            results["master_flat"] = master_flat
        elif flat_paths:
            prog(0.35, f"Creating master flat from {len(flat_paths)} frames…")
            r = create_master_flat(
                flat_paths, master_bias=master_bias, master_dark=master_dark,
                progress=lambda f, m: prog(0.35 + f * 0.2, m),
            )
            master_flat = r.master
            results["master_flat"] = master_flat

        # ── Calibrate lights ──────────────────────────────────────────────────
        prog(0.55, f"Calibrating {len(light_paths)} light frames…")
        calibrated = calibrate_lights_batch(
            light_paths,
            master_bias=master_bias,
            master_dark=master_dark,
            master_flat=master_flat,
            progress=lambda f, m: prog(0.55 + f * 0.45, m),
        )
        results["calibrated"] = calibrated
        return results

    @pyqtSlot(object)
    def _on_calibration_done(self, results: dict):
        self._master_bias = results.get("master_bias")
        self._master_dark = results.get("master_dark")
        self._master_flat = results.get("master_flat")
        self._calibrated_lights = results.get("calibrated", [])

        n = len(self._calibrated_lights)
        self._log_panel.log(f"Calibration complete: {n} lights calibrated", "success")

        if self._calibrated_lights:
            self._display_image(self._calibrated_lights[0])

        if self._project:
            self._project.add_history("Calibration", {"n_lights": n})
            self._save_project()

    def _get_lights_for_stacking(self) -> list[ImageData] | None:
        """Helper to get lights (calibrated or raw) with a warning dialog."""
        if self._calibrated_lights:
            return self._calibrated_lights

        # Fallback to raw lights from project
        if not self._project:
            self._log_panel.log("No project loaded", "error")
            return None

        light_frames = [f for f in self._project.frames if f.frame_type == FrameType.LIGHT]
        if not light_frames:
            self._log_panel.log("No light frames in project", "error")
            return None

        settings = QSettings("Cosmica", "Cosmica")
        if not settings.value("stacking/raw_warning_acknowledged", False, type=bool):
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Uncalibrated Frames")
            msg.setText("No calibrated frames found. Do you want to align/stack raw frames?")
            msg.setInformativeText(
                "Stacking raw frames without calibration (bias/dark/flat correction) "
                "may result in artifacts. It is recommended to calibrate frames first."
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msg.setDefaultButton(QMessageBox.StandardButton.No)

            cb = QCheckBox("Do not show this warning again")
            msg.setCheckBox(cb)

            if msg.exec() == QMessageBox.StandardButton.No:
                return None

            if cb.isChecked():
                settings.setValue("stacking/raw_warning_acknowledged", True)

        # Load raw lights
        lights = []
        for frame in light_frames:
            try:
                lights.append(load_image(frame.path))
            except Exception as e:
                log.warning("Failed to load %s: %s", frame.path, e)

        if not lights:
            self._log_panel.log("Failed to load any raw frames", "error")
            return None

        self._log_panel.log(f"Loaded {len(lights)} raw light frames (uncalibrated)", "warning")
        return lights

    def _apply_quality_filter(self, lights: list) -> list:
        """Filter frames by quality score if quality filter is enabled. Returns filtered list."""
        from cosmica.core.subframe_selector import filter_by_metric, score_subframes

        qf = self._tools_panel.get_quality_filter_params()
        if qf is None:
            return lights

        paths = [str(img.file_path) for img in lights if img.file_path is not None]
        if len(paths) != len(lights):
            self._log_panel.log("Quality filter skipped: some frames have no file path", "warning")
            return lights

        self._log_panel.log(f"Running quality filter on {len(paths)} frames...", "info")
        sf_params = SubframeSelectorParams(rejection_sigma=qf.get("rejection_sigma", 1.5))
        scores = score_subframes(paths, sf_params)

        filtered_scores = filter_by_metric(
            scores,
            metric=qf.get("metric", "quality_score"),
            mode=qf.get("mode", "sigma"),
            top_n=qf.get("top_n"),
            top_percent=qf.get("top_percent", 80.0),
            sigma=qf.get("rejection_sigma", 1.5),
        )

        accepted = [img for img, sc in zip(lights, filtered_scores, strict=True) if sc.accepted]
        n_rejected = len(lights) - len(accepted)
        if n_rejected > 0:
            self._log_panel.log(
                f"Quality filter: rejected {n_rejected} frame(s) "
                f"(metric={qf.get('metric')}, mode={qf.get('mode')})",
                "warning",
            )
        else:
            self._log_panel.log("Quality filter: all frames accepted", "info")
        return accepted if accepted else lights  # never drop everything

    @pyqtSlot()
    def _on_run_alignment(self):
        lights = self._get_lights_for_stacking()
        if not lights:
            return
        lights = self._apply_quality_filter(lights)

        params = self._tools_panel.get_alignment_params()

        self._log_panel.log(f"Starting alignment ({params['mode'].name})...", "info")

        self._start_worker(
            align_frames,
            lights,
            params=StackingParams(registration_mode=params["mode"]),
            on_done=self._on_alignment_done,
        )

    def _on_alignment_done(self, aligned_lights: list):
        """Callback when alignment finishes."""
        if not aligned_lights:
            self._log_panel.log("Alignment failed: no frames aligned", "error")
            return

        # Store aligned lights for immediate stacking
        self._aligned_lights = aligned_lights

        # Save aligned frames to project folder
        project_dir = self._project.directory if self._project else None
        if project_dir:
            aligned_dir = os.path.join(project_dir, "aligned")
            os.makedirs(aligned_dir, exist_ok=True)

            for i, img in enumerate(aligned_lights):
                filename = f"aligned_{i + 1:03d}.fits"
                filepath = os.path.join(aligned_dir, filename)
                try:
                    from cosmica.core.image_io import save_image

                    save_image(img, filepath)
                except Exception as e:
                    log.warning(f"Failed to save aligned frame {filename}: {e}")

            self._log_panel.log(
                f"Saved {len(aligned_lights)} aligned frames to 'aligned/' folder", "info"
            )

            # Register aligned frames in the project (REGISTERED section)
            aligned_paths = [
                Path(os.path.join(aligned_dir, f"aligned_{i + 1:03d}.fits"))
                for i in range(len(aligned_lights))
            ]
            # Clear old aligned entries first, then add fresh ones
            for p in aligned_paths:
                if p.exists():
                    self._project.remove_frame(p)
            self._project.add_frames([p for p in aligned_paths if p.exists()], FrameType.ALIGNED)

            # Refresh file tree
            if hasattr(self, "_project_panel"):
                self._project_panel.refresh()

        # Display the aligned reference frame
        self._display_image(aligned_lights[0])

        self._log_panel.log(
            f"Alignment complete: {len(aligned_lights)} frames aligned. Ready to stack.",
            "success",
        )
        if self._project:
            self._project.add_history("Alignment", {"n_frames": len(aligned_lights)})
            self._save_project()

    def _on_run_stacking(self):
        # If we have aligned frames, use them (skip alignment)
        if hasattr(self, "_aligned_lights") and self._aligned_lights:
            lights = self._aligned_lights
            align = False
            self._log_panel.log("Stacking previously aligned frames...", "info")
        else:
            lights = self._get_lights_for_stacking()
            if not lights:
                return
            lights = self._apply_quality_filter(lights)
            align = True
            self._log_panel.log("Aligning and stacking frames...", "info")

        drizzle_enabled, drizzle_params = self._tools_panel.get_drizzle_params()

        if drizzle_enabled:
            self._log_panel.log(
                f"Drizzle {drizzle_params.scale}× enabled (drop={drizzle_params.drop_shrink:.2f})…",
                "info",
            )
            from cosmica.core.drizzle import drizzle_integrate

            def _drizzle_work(light_frames, d_params, progress=None):
                import numpy as _np
                # Collect raw arrays; if aligned frames exist they're already ndarray
                arrays = [
                    f if isinstance(f, _np.ndarray) else f.data
                    for f in light_frames
                ]
                result = drizzle_integrate(arrays, params=d_params, progress=progress)
                return result

            def _on_drizzle_done(result):
                import numpy as _np
                from cosmica.core.image_io import ImageData
                img = ImageData(data=result.data.astype(_np.float32), header={})
                self._display_image(img)
                self._log_panel.log(
                    f"Drizzle complete: {result.n_frames} frames, "
                    f"{result.output_scale}× scale ({img.shape_str})",
                    "success",
                )
                if self._project:
                    self._project.add_history(
                        "Drizzle",
                        {"n_frames": result.n_frames, "scale": result.output_scale},
                    )
                    self._save_project()

            self._start_worker(_drizzle_work, lights, drizzle_params, on_done=_on_drizzle_done)
            return

        params = self._tools_panel.get_stacking_params()

        self._start_worker(
            stack_images,
            lights,
            params=params,
            align=align,
            on_done=self._on_stacking_done,
        )

    @pyqtSlot(object)
    def _on_stacking_done(self, result):
        self._display_image(result.image)
        self._log_panel.log(
            f"Stacking complete: {result.n_frames} frames, {result.total_rejected} pixels rejected",
            "success",
        )
        if self._project:
            self._project.add_history("Stacking", {"n_frames": result.n_frames})
            self._save_project()

    # ── Multi-session stacking ────────────────────────────────────────────────

    @pyqtSlot()
    def _on_ms_add_folder(self):
        """Let the user pick a folder of light frames as a new session."""
        from PyQt6.QtWidgets import QFileDialog, QInputDialog
        folder = QFileDialog.getExistingDirectory(self, "Select Session Folder", "")
        if not folder:
            return
        from pathlib import Path
        import glob as _glob
        extensions = ("*.fits", "*.fit", "*.fts", "*.xisf", "*.FITS", "*.FIT", "*.FTS")
        files = []
        for ext in extensions:
            files.extend(_glob.glob(str(Path(folder) / ext)))
        if not files:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No frames", f"No FITS/XISF files found in:\n{folder}")
            return

        # Ask user for a session name (default = folder name)
        default_name = Path(folder).name
        name, ok = QInputDialog.getText(self, "Session Name", "Name for this session:", text=default_name)
        if not ok or not name.strip():
            name = default_name

        # Load frames
        from cosmica.core.image_io import load_image
        loaded = []
        for fpath in sorted(files):
            try:
                img = load_image(fpath)
                if img is not None:
                    loaded.append(img)
            except Exception as exc:
                self._log_panel.log(f"Skipped {Path(fpath).name}: {exc}", "warning")

        if not loaded:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Load failed", "Could not load any frames from the folder.")
            return

        from cosmica.core.multi_session import SessionGroup
        # Estimate total integration time from headers
        total_time = 0.0
        for img in loaded:
            exp = (img.header or {}).get("EXPTIME", (img.header or {}).get("EXPOSURE", None))
            if exp is not None:
                try:
                    total_time += float(exp)
                except (ValueError, TypeError):
                    pass
        session = SessionGroup(
            frames=loaded,
            name=name.strip(),
            integration_time=total_time if total_time > 0 else None,
        )
        self._ms_sessions.append(session)
        self._tools_panel.ms_add_session(name.strip(), len(loaded))
        self._log_panel.log(f"Session '{name}': {len(loaded)} frames loaded", "success")

    @pyqtSlot()
    def _on_ms_clear(self):
        self._ms_sessions.clear()

    @pyqtSlot()
    def _on_run_multi_session(self):
        if len(self._ms_sessions) < 2:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Multi-Session",
                "Please add at least 2 sessions using 'Add Session…' before stacking."
            )
            return

        ms_params_dict = self._tools_panel.get_multi_session_params()
        stacking_params = self._tools_panel.get_stacking_params()

        from cosmica.core.multi_session import MultiSessionParams, stack_multi_session
        from cosmica.core.stacking import RejectionMethod, IntegrationMethod

        ms_params = MultiSessionParams(
            per_session_params=stacking_params,
            weight_mode=ms_params_dict["weight_mode"],
            normalize_background=ms_params_dict["normalize_background"],
            align_sub_stacks=ms_params_dict["align_sub_stacks"],
        )

        sessions_snapshot = list(self._ms_sessions)
        n = len(sessions_snapshot)
        names = [s.name for s in sessions_snapshot]
        self._log_panel.log(
            f"Multi-session stacking: {n} sessions ({', '.join(names)})…", "info"
        )

        def _ms_work(sessions, params, progress=None):
            from cosmica.core.multi_session import stack_multi_session as _ms
            return _ms(sessions, params, progress=progress or (lambda f, m: None))

        self._start_worker(_ms_work, sessions_snapshot, ms_params,
                           on_done=self._on_multi_session_done)

    @pyqtSlot(object)
    def _on_multi_session_done(self, result):
        self._display_image(result.image)
        total_frames = sum(r.n_frames for r in result.sub_stacks)
        weight_info = ", ".join(
            f"{name}: {w:.2f}"
            for name, w in zip(result.session_names, result.weights)
        )
        self._log_panel.log(
            f"Multi-session complete: {result.n_sessions} sessions, "
            f"{total_frames} total frames  [weights: {weight_info}]",
            "success",
        )
        if self._project:
            self._project.add_history(
                "Multi-Session Stack",
                {"n_sessions": result.n_sessions, "n_frames": total_frames},
            )
            self._save_project()

    @pyqtSlot()
    def _on_run_stretch(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_stretch_params()
        stretched = auto_stretch(self._current_image.data, params)
        self._update_current_image(stretched, "Stretch applied")
        if self._project:
            self._project.add_history(
                "Auto-Stretch",
                {
                    "midtone": params.midtone,
                    "shadow_clip": params.shadow_clip,
                },
            )
        self._macro_recorder.record_step(
            "auto_stretch",
            {
                "midtone": params.midtone,
                "shadow_clip": params.shadow_clip,
            },
        )
        self._tools_panel.reset_stretch_params()

    def _on_stretch_preview(self):
        if self._current_image is None:
            return
        if not self._tools_panel.split_preview_enabled:
            self._canvas.clear_after_image()
            self._preview_indicator.setText("")
            return
        self._preview_indicator.setText("● Live Preview: Auto Stretch")
        self._stretch_preview_timer.start()

    def _do_stretch_preview(self):
        if self._current_image is None:
            return
        small, _scale = self._downscale_for_preview(self._current_image.data)
        params = self._tools_panel.get_stretch_params()
        stretched = auto_stretch(small, params)
        if stretched.size == 0:
            return
        # auto_stretch output is already a final display-ready stretch;
        # re-apply display stretch the same way _display_image does so the
        # preview matches exactly what Apply would show.
        tmp = ImageData(data=stretched, header={})
        after_rgb = tmp.to_display(stretch=True)
        self._canvas.set_after_image(after_rgb)
        self._canvas.set_split_mode(True)

    def _downscale_for_preview(self, data):
        """Downscale image data so longest side is at most 1024 px."""
        import cv2
        import numpy as np

        if data.ndim == 2:
            h, w = data.shape
        else:
            c, h, w = data.shape
        longest = max(h, w)
        if longest <= 1024:
            return data, 1.0
        scale = 1024.0 / longest
        new_w, new_h = int(w * scale), int(h * scale)
        if data.ndim == 2:
            return cv2.resize(data, (new_w, new_h), interpolation=cv2.INTER_AREA), scale
        channels = [
            cv2.resize(data[ch], (new_w, new_h), interpolation=cv2.INTER_AREA) for ch in range(c)
        ]
        return np.stack(channels, axis=0), scale

    @pyqtSlot(str)
    def _on_preview_requested(self, tool_name: str):
        """Run the named tool on a downscaled copy and show split preview."""
        self._pending_preview_tool = tool_name
        self._preview_timer.start()
        label = tool_name.replace("_", " ").title()
        self._preview_indicator.setText(f"● Live Preview: {label}")

    def _do_preview_requested(self):
        import numpy as np
        tool_name = self._pending_preview_tool
        if tool_name is None or self._current_image is None:
            return

        small, _scale = self._downscale_for_preview(self._current_image.data)

        try:
            result = self._run_tool_preview(tool_name, small)
        except Exception as e:
            self._log_panel.log(f"Preview failed for {tool_name}: {e}", "warning")
            return

        if result is None or result.size == 0:
            return

        # Convert to HWC, use reference-based stretch so before/after brightness matches
        if result.ndim == 2:
            after_hwc = np.stack([result, result, result], axis=-1)
            ref_hwc = np.stack([small, small, small], axis=-1)
        else:
            after_hwc = np.transpose(result, (1, 2, 0))
            ref_hwc = np.transpose(small, (1, 2, 0))
        after_disp = auto_stretch_for_display_ref(after_hwc, ref_hwc)
        after_rgb = np.clip(after_disp * 255, 0, 255).astype(np.uint8)
        self._canvas.set_after_image(after_rgb)
        self._canvas.set_split_mode(True)

    def _run_tool_preview(self, tool_name: str, data):
        """Execute a tool on the given data and return the result array."""
        tp = self._tools_panel
        if tool_name == "cosmetic":
            r = cosmetic_correction(data, tp.get_cosmetic_params())
            return r.data
        elif tool_name == "banding":
            return banding_reduction(data, tp.get_banding_params())
        elif tool_name == "deconvolution":
            params = tp.get_deconvolution_params()
            if isinstance(params, SpatialDeconvParams):
                return richardson_lucy_spatial(data, params=params)
            return richardson_lucy(data, params=params)
        elif tool_name == "denoise":
            return denoise(data, tp.get_denoise_params())
        elif tool_name == "scnr":
            return scnr(data, tp.get_scnr_params())
        elif tool_name == "color_adjust":
            return color_adjust(data, tp.get_color_adjust_params())
        elif tool_name == "wavelet":
            return wavelet_sharpen(data, tp.get_wavelet_params())
        elif tool_name == "mlt":
            return wavelet_sharpen(data, tp.get_mlt_params())
        elif tool_name == "local_contrast":
            return local_contrast_enhance(data, tp.get_local_contrast_params())
        elif tool_name == "unsharp_mask":
            return unsharp_mask(data, tp.get_unsharp_mask_params())
        elif tool_name == "median_filter":
            return median_filter(data, tp.get_median_filter_params())
        elif tool_name == "histogram_transform":
            return histogram_transform(data, tp.get_histogram_transform_params())
        elif tool_name == "ghs":
            return generalized_hyperbolic_stretch(data, tp.get_ghs_params())
        elif tool_name == "curves":
            return curves_transform(data, tp.get_curves_params())
        return None

    @pyqtSlot()
    def _on_preview_cancelled(self):
        """Clear the split preview."""
        self._canvas.clear_after_image()
        self._preview_indicator.setText("")

    # ---------- Transform operations ----------

    @pyqtSlot()
    @pyqtSlot()
    def _on_start_crop_draw(self):
        """Toggle interactive crop-draw mode on the canvas."""
        currently_active = getattr(self._canvas, '_crop_mode', False)
        self._canvas.set_crop_mode(not currently_active)

    @pyqtSlot(int, int, int, int)
    def _on_crop_rect_selected(self, x: int, y: int, w: int, h: int):
        """Called when user finishes drawing a crop rectangle on the canvas."""
        self._tools_panel.set_crop_from_rect(x, y, w, h)
        self._log_panel.log(f"Crop region set: x={x}, y={y}, w={w}, h={h}", "info")

    def _on_run_crop(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_crop_params()
        result = crop(self._current_image.data, params)
        self._update_current_image(result, f"Cropped to {result.shape[-1]}x{result.shape[-2]}")
        if self._project:
            self._project.add_history(
                "Crop",
                {"x": params.x, "y": params.y, "width": params.width, "height": params.height},
            )
        self._macro_recorder.record_step(
            "crop", {"x": params.x, "y": params.y, "width": params.width, "height": params.height}
        )

    @pyqtSlot()
    def _on_run_rotate(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_rotate_params()
        result = rotate(self._current_image.data, params)
        self._update_current_image(result, f"Rotated ({params.angle.name})")
        if self._project:
            self._project.add_history("Rotate", {"angle": params.angle.name})
        self._macro_recorder.record_step("rotate", {"angle": params.angle.name})

    @pyqtSlot()
    def _on_run_flip(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_flip_params()
        result = flip(self._current_image.data, params)
        self._update_current_image(result, f"Flipped ({params.axis.name})")
        if self._project:
            self._project.add_history("Flip", {"axis": params.axis.name})
        self._macro_recorder.record_step("flip", {"axis": params.axis.name})

    @pyqtSlot()
    def _on_run_resize(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_resize_params()
        result = resize(self._current_image.data, params)
        self._update_current_image(result, f"Resized to {result.shape[-1]}x{result.shape[-2]}")
        if self._project:
            self._project.add_history("Resize", {"scale": params.scale})
        self._macro_recorder.record_step("resize", {"scale": params.scale})

    @pyqtSlot()
    def _on_run_bin(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_bin_params()
        result = bin_image(self._current_image.data, params)
        self._update_current_image(
            result, f"Binned {params.factor}x{params.factor} ({params.mode.name})"
        )
        if self._project:
            self._project.add_history("Bin", {"factor": params.factor, "mode": params.mode.name})
        self._macro_recorder.record_step("bin", {"factor": params.factor})

    @pyqtSlot()
    def _on_run_invert(self):
        if self._current_image is None:
            return
        result = invert(self._current_image.data)
        self._update_current_image(result, "Image inverted")
        if self._project:
            self._project.add_history("Invert", {})
        self._macro_recorder.record_step("invert")

    # ---------- New tool operations ----------

    @pyqtSlot()
    def _on_run_unsharp_mask(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_unsharp_mask_params()
        self._log_panel.log(
            f"Applying Unsharp Mask (r={params.radius}, a={params.amount})...", "info"
        )
        result = unsharp_mask(self._current_image.data, params)
        self._update_current_image(result, "Unsharp mask applied")
        if self._project:
            self._project.add_history(
                "Unsharp Mask",
                {
                    "radius": params.radius,
                    "amount": params.amount,
                },
            )
        self._macro_recorder.record_step(
            "unsharp_mask",
            {
                "radius": params.radius,
                "amount": params.amount,
            },
        )

    @pyqtSlot()
    def _on_run_median_filter(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_median_filter_params()
        self._log_panel.log(f"Applying Median Filter (k={params.kernel_size})...", "info")
        result = median_filter(self._current_image.data, params)
        self._update_current_image(result, "Median filter applied")
        if self._project:
            self._project.add_history("Median Filter", {"kernel_size": params.kernel_size})

    @pyqtSlot()
    def _on_run_abe(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_abe_params()
        self._log_panel.log("Running ABE (RBF background extraction)...", "info")
        self._start_worker(
            abe_extract,
            self._current_image.data,
            params=params,
            on_done=self._on_abe_done,
        )

    @pyqtSlot(object)
    def _on_abe_done(self, result):
        corrected, bg_model = result
        self._update_current_image(corrected, "ABE background extraction complete")
        if self._project:
            self._project.add_history("ABE", {})

    @pyqtSlot()
    def _on_run_vignette(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_vignette_params()
        self._log_panel.log("Applying vignette correction...", "info")
        result = correct_vignette(self._current_image.data, params)
        self._update_current_image(result, "Vignette correction applied")
        if self._project:
            self._project.add_history(
                "Vignette Correction",
                {
                    "strength": params.strength,
                    "falloff": params.falloff,
                },
            )

    @pyqtSlot()
    def _on_run_ca(self):
        if self._current_image is None:
            return
        if self._current_image.data.ndim != 3 or self._current_image.data.shape[0] < 3:
            self._log_panel.log("Chromatic aberration requires a color image", "error")
            return
        params = self._tools_panel.get_ca_params()
        self._log_panel.log("Correcting chromatic aberration...", "info")
        result = correct_chromatic_aberration(self._current_image.data, params)
        self._update_current_image(result, "Chromatic aberration corrected")
        if self._project:
            self._project.add_history("CA Correction", {"auto": params.auto_detect})

    @pyqtSlot()
    def _on_show_statistics(self):
        if self._current_image is None:
            return
        from cosmica.ui.dialogs.statistics_dialog import StatisticsDialog

        stats = compute_image_statistics(self._current_image.data)
        dialog = StatisticsDialog(stats, self)
        dialog.exec()

    @pyqtSlot()
    def _on_measure_psf(self):
        if self._current_image is None:
            return
        from cosmica.core.psf import measure_psf

        self._log_panel.log("Measuring PSF from stars…", "info")

        def _psf_work(data, progress=None):
            return measure_psf(data)

        def _on_psf_done(result):
            if result is None or result.n_stars_used == 0:
                self._log_panel.log("PSF measurement failed: no stars found", "warning")
                return
            self._tools_panel.set_psf_measurement(
                result.fwhm, result.ellipticity, result.n_stars_used
            )
            self._log_panel.log(
                f"PSF: FWHM={result.fwhm:.2f}px  ellipticity={result.ellipticity:.2f}"
                f"  ({result.n_stars_used} stars)",
                "success",
            )

        self._start_worker(_psf_work, self._current_image.data, on_done=_on_psf_done)

    @pyqtSlot()
    def _on_run_continuum_subtraction(self):
        if self._current_image is None:
            return
        from pathlib import Path as _Path

        from PyQt6.QtWidgets import QFileDialog

        from cosmica.core.image_io import ImageData, load_image
        from cosmica.core.narrowband import continuum_subtraction

        bb_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Broadband (Continuum) Image",
            "",
            "Images (*.fits *.fit *.fts *.xisf *.tif *.tiff *.png *.jpg)",
        )
        if not bb_path:
            return

        scale = self._tools_panel.get_continuum_scale()
        self._log_panel.log(
            f"Continuum subtraction: scale={scale:.3f}, broadband={_Path(bb_path).name}", "info"
        )

        def _cont_work(nb_data, bb_path_str, scale, progress=None):
            bb_img = load_image(bb_path_str)
            # Use first channel (or luminance) as broadband
            if bb_img.data.ndim == 3:
                import numpy as _np
                bb_ch = _np.mean(bb_img.data, axis=0)
            else:
                bb_ch = bb_img.data
            # Narrowband: if color, use luminance; if mono, use as-is
            import numpy as _np
            if nb_data.ndim == 3:
                nb_ch = _np.mean(nb_data, axis=0)
            else:
                nb_ch = nb_data
            result_ch = continuum_subtraction(nb_ch, bb_ch, scale)
            return result_ch

        def _on_cont_done(result_ch):
            import numpy as _np
            from cosmica.core.image_io import ImageData
            self._update_current_image(result_ch, "Continuum subtraction applied")

        self._start_worker(
            _cont_work,
            self._current_image.data,
            bb_path,
            scale,
            on_done=_on_cont_done,
        )

    # ── Dynamic background sample placement ──────────────────────────────────

    @pyqtSlot(bool)
    def _on_toggle_sample_mode(self, enabled: bool):
        self._canvas.set_sample_mode(enabled)
        if not enabled:
            self._canvas.set_sample_points(self._bg_samples)

    @pyqtSlot(float, float)
    def _on_sample_placed(self, x: float, y: float):
        self._bg_samples.append((x, y))
        self._canvas.set_sample_points(self._bg_samples)
        self._tools_panel.set_bg_sample_count(len(self._bg_samples))

    @pyqtSlot(float, float)
    def _on_sample_removed(self, x: float, y: float):
        if not self._bg_samples:
            return
        import math as _math
        nearest_idx = min(
            range(len(self._bg_samples)),
            key=lambda i: _math.hypot(self._bg_samples[i][0] - x, self._bg_samples[i][1] - y),
        )
        self._bg_samples.pop(nearest_idx)
        self._canvas.set_sample_points(self._bg_samples)
        self._tools_panel.set_bg_sample_count(len(self._bg_samples))

    @pyqtSlot()
    def _on_clear_bg_samples(self):
        self._bg_samples.clear()
        self._canvas.clear_sample_points()
        self._tools_panel.set_bg_sample_count(0)

    # ── WCS overlay ──────────────────────────────────────────────────────────

    @pyqtSlot(bool)
    def _on_toggle_wcs_overlay(self, enabled: bool):
        if enabled and not self._wcs_overlay_stars:
            self._log_panel.log(
                "No WCS data available — run Solve & Calibrate (PCC) first", "warning"
            )
            self._tools_panel._btn_wcs_overlay.setChecked(False)
            return
        self._canvas.set_wcs_overlay_visible(enabled)

    def _update_wcs_overlay(self, wcs: dict, catalog_stars: list):
        """Project catalog stars to image pixel coordinates and store for overlay."""
        if not wcs or not catalog_stars:
            return
        if self._current_image is None:
            return

        import numpy as _np
        from cosmica.core.color_calibration import _make_pixel_to_sky

        h = self._current_image.data.shape[-2] if self._current_image.data.ndim == 3 else self._current_image.data.shape[0]
        w = self._current_image.data.shape[-1] if self._current_image.data.ndim == 3 else self._current_image.data.shape[1]

        sky_fn = _make_pixel_to_sky(wcs, w, h)
        scale_deg = (wcs.get("scale") or 1.0) / 3600.0
        match_radius = max(scale_deg * 10, 0.005)

        overlay = []
        for cat in catalog_stars:
            if cat.g_mag is None:
                continue
            # Invert sky_fn: try to find pixel coords for this star
            # Use linear approximation: center + offset
            ra0, dec0 = sky_fn(w / 2, h / 2)
            cos_dec = _np.cos(_np.radians(dec0))
            dra = (cat.ra_deg - ra0) * cos_dec
            ddec = cat.dec_deg - dec0
            px = w / 2 + dra / max(scale_deg, 1e-10)
            py = h / 2 - ddec / max(scale_deg, 1e-10)
            if 0 <= px < w and 0 <= py < h:
                overlay.append((float(px), float(py), float(cat.g_mag)))

        self._wcs_overlay_stars = overlay
        self._current_wcs = wcs
        self._canvas.set_overlay_stars(overlay)
        # Auto-populate DSO annotations
        self._update_dso_annotations(wcs)

    def _update_dso_annotations(self, wcs: dict):
        """Project DSO catalog entries to image pixel coordinates and push to canvas."""
        if not wcs or self._current_image is None:
            return
        from cosmica.core.dso_catalog import query_dso_in_field
        import numpy as _np

        h = self._current_image.data.shape[-2] if self._current_image.data.ndim == 3 else self._current_image.data.shape[0]
        w = self._current_image.data.shape[-1] if self._current_image.data.ndim == 3 else self._current_image.data.shape[1]

        ra_center = wcs.get("ra_center", 0.0)
        dec_center = wcs.get("dec_center", 0.0)
        scale_deg = (wcs.get("scale") or 1.0) / 3600.0
        fov_deg = max(w, h) * scale_deg * 1.5

        dsos = query_dso_in_field(ra_center, dec_center, fov_deg)
        cos_dec = _np.cos(_np.radians(dec_center))
        annotations = []
        for dso in dsos:
            dra = (dso.ra_deg - ra_center) * cos_dec
            ddec = dso.dec_deg - dec_center
            px = w / 2 + dra / max(scale_deg, 1e-10)
            py = h / 2 - ddec / max(scale_deg, 1e-10)
            if -50 <= px < w + 50 and -50 <= py < h + 50:
                annotations.append((float(px), float(py), dso.name, dso.type_code))

        self._canvas.set_dso_annotations(annotations)

    @pyqtSlot()
    def _on_toggle_dso_overlay(self):
        """Toggle DSO annotation overlay visibility on the canvas."""
        current = getattr(self._canvas, '_show_dso_overlay', False)
        self._canvas.set_dso_overlay_visible(not current)

    # ── Python console ────────────────────────────────────────────────────────

    @pyqtSlot()
    def _on_open_python_console(self):
        from PyQt6.QtWidgets import QDockWidget
        from cosmica.ui.widgets.python_console import PythonConsoleWidget

        if self._python_console_dock is None:
            console = PythonConsoleWidget()
            console.image_updated.connect(self._on_console_image_updated)
            dock = QDockWidget("Python Console", self)
            dock.setWidget(console)
            dock.setMinimumWidth(500)
            dock.setMinimumHeight(300)
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
            self._python_console_dock = dock

        self._python_console_dock.show()
        self._python_console_dock.raise_()
        # Inject current image
        self._sync_console_image()

    def _sync_console_image(self):
        if self._python_console_dock is None:
            return
        console = self._python_console_dock.widget()
        if console and self._current_image is not None:
            console.set_image(self._current_image)

    @pyqtSlot(object)
    def _on_console_image_updated(self, arr):
        import numpy as _np
        if not isinstance(arr, _np.ndarray):
            self._log_panel.log("apply() requires a numpy ndarray", "error")
            return
        self._update_current_image(arr, "Image updated from Python console")

    @pyqtSlot()
    def _on_open_star_mask(self):
        if self._current_image is None:
            return
        from cosmica.ui.dialogs.star_mask_dialog import StarMaskDialog

        dialog = StarMaskDialog(self._current_image.data, self)
        dialog.mask_ready.connect(self._on_star_mask_ready)
        dialog.exec()

    @pyqtSlot(object)
    def _on_star_mask_ready(self, mask):
        self._log_panel.log("Star mask generated", "success")

    @pyqtSlot()
    def _on_open_subframe_selector(self):
        from cosmica.ui.dialogs.subframe_dialog import SubframeDialog

        dialog = SubframeDialog(self)
        dialog.exec()

    @pyqtSlot()
    def _on_run_background(self):
        if self._current_image is None:
            return
        # Convert image-space sample coords to integer (row, col) tuples
        manual_pts = [(int(round(y)), int(round(x))) for x, y in self._bg_samples]
        params = self._tools_panel.get_background_params(manual_points=manual_pts)
        n = len(manual_pts)
        msg = f"Extracting background ({n} manual sample{'s' if n != 1 else ''})..." if n else "Extracting background..."
        self._log_panel.log(msg, "info")

        def _bg_work(data, progress=None):
            return extract_background(data, params)

        self._start_worker(_bg_work, self._current_image.data, on_done=self._on_background_done)

    @pyqtSlot(object)
    def _on_background_done(self, result):
        corrected, bg_model = result
        self._update_current_image(corrected, "Background extraction complete")
        params = self._tools_panel.get_background_params()
        if self._project:
            self._project.add_history(
                "Background Extraction",
                {
                    "grid_size": params.grid_size,
                    "polynomial_order": params.polynomial_order,
                },
            )
        self._macro_recorder.record_step("background_extraction")

    # ---------- Phase A processing operations ----------

    @pyqtSlot()
    def _on_run_cosmetic(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_cosmetic_params()
        self._log_panel.log("Running cosmetic correction...", "info")
        result = cosmetic_correction(self._current_image.data, params)
        self._update_current_image(
            result.data,
            f"Cosmetic correction: {result.total_corrected} pixels fixed "
            f"({result.hot_pixels} hot, {result.cold_pixels} cold, {result.dead_pixels} dead)",
        )
        if self._project:
            self._project.add_history(
                "Cosmetic Correction",
                {
                    "hot": result.hot_pixels,
                    "cold": result.cold_pixels,
                    "dead": result.dead_pixels,
                },
            )

    @pyqtSlot()
    def _on_run_banding(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_banding_params()
        self._log_panel.log("Running banding reduction...", "info")
        result = banding_reduction(self._current_image.data, params)
        self._update_current_image(result, "Banding reduction complete")
        if self._project:
            self._project.add_history(
                "Banding Reduction",
                {
                    "horizontal": params.horizontal,
                    "vertical": params.vertical,
                },
            )

    @pyqtSlot()
    def _on_run_histogram_transform(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_histogram_transform_params()
        result = histogram_transform(self._current_image.data, params)
        self._update_current_image(result, "Histogram transform applied")
        if self._project:
            self._project.add_history(
                "Histogram Transform",
                {
                    "black_point": params.black_point,
                    "midtone": params.midtone,
                    "white_point": params.white_point,
                },
            )
        self._macro_recorder.record_step(
            "histogram_transform",
            {
                "black_point": params.black_point,
                "midtone": params.midtone,
                "white_point": params.white_point,
            },
        )
        self._tools_panel.reset_histogram_transform_params()

    @pyqtSlot()
    def _on_run_curves(self):
        if self._current_image is None:
            return
        params = CurvesParams()
        # Get curve from the editor widget
        params.master = self._tools_panel.curve_editor.curve
        result = curves_transform(self._current_image.data, params)
        self._update_current_image(result, "Curves applied")
        if self._project:
            self._project.add_history("Curves", {})

    @pyqtSlot()
    def _on_run_scnr(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_scnr_params()
        result = scnr(self._current_image.data, params)
        self._update_current_image(result, "SCNR applied")
        if self._project:
            self._project.add_history(
                "SCNR",
                {
                    "method": params.method.name,
                    "amount": params.amount,
                },
            )
        self._macro_recorder.record_step("scnr", {"amount": params.amount})

    @pyqtSlot()
    def _on_run_color_adjust(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_color_adjust_params()
        result = color_adjust(self._current_image.data, params)
        self._update_current_image(result, "Color adjustment applied")
        if self._project:
            self._project.add_history(
                "Color Adjustment",
                {
                    "saturation": params.saturation,
                    "hue_shift": params.hue_shift,
                    "vibrance": params.vibrance,
                },
            )

    @pyqtSlot()
    def _on_run_deconvolution(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_deconvolution_params()
        if isinstance(params, SpatialDeconvParams):
            self._log_panel.log(
                f"Running spatial deconvolution (3x3 zones, "
                f"fallback FWHM={params.fallback_fwhm}, "
                f"{params.iterations} iterations)...",
                "info",
            )
            self._start_worker(
                richardson_lucy_spatial,
                self._current_image.data,
                params=params,
                on_done=self._on_deconvolution_done,
            )
        else:
            self._log_panel.log(
                f"Running deconvolution (FWHM={params.psf_fwhm}, "
                f"{params.iterations} iterations)...",
                "info",
            )
            self._start_worker(
                richardson_lucy,
                self._current_image.data,
                params=params,
                on_done=self._on_deconvolution_done,
            )

    @pyqtSlot(object)
    def _on_deconvolution_done(self, result):
        self._update_current_image(result, "Deconvolution complete")
        if self._project:
            self._project.add_history("Deconvolution", {})

    # ---------- Phase B processing operations ----------

    @pyqtSlot()
    def _on_run_ghs(self):
        import numpy as np
        if self._current_image is None:
            return
        params = self._tools_panel.get_ghs_params()
        data = self._current_image.data
        # Warn if image appears already stretched (GHS is designed for linear data)
        sample = data[0] if data.ndim == 3 else data
        median_val = float(np.median(sample[sample > 0])) if np.any(sample > 0) else 0.0
        if median_val > 0.1:
            self._log_panel.log(
                f"Warning: image median={median_val:.3f} — GHS works best on linear (unstretched) data",
                "warning",
            )
        self._log_panel.log(f"Applying GHS (D={params.D})...", "info")
        result = generalized_hyperbolic_stretch(data, params)
        self._update_current_image(result, "GHS applied")
        if self._project:
            self._project.add_history("GHS", {"D": params.D, "b": params.b, "SP": params.SP})
        self._tools_panel.reset_ghs_params()

    @pyqtSlot()
    def _on_run_color_calibration(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_color_calibration_params()
        self._log_panel.log("Running color calibration...", "info")
        result = color_calibrate(self._current_image.data, params)
        factors = result.correction_factors
        self._update_current_image(
            result.data,
            f"Color calibration complete (R={factors[0]:.3f}, G={factors[1]:.3f}, B={factors[2]:.3f})",
        )
        if self._project:
            self._project.add_history(
                "Color Calibration",
                {
                    "factors": list(factors),
                },
            )

    @pyqtSlot()
    def _on_run_pcc(self):
        if self._current_image is None:
            return

        if self._current_image.data.ndim != 3 or self._current_image.data.shape[0] < 3:
            self._log_panel.log("PCC requires a color (RGB) image", "error")
            return

        params = self._tools_panel.get_pcc_params()
        self._log_panel.log("Starting Photometric Color Calibration (PCC)...", "info")

        # Get astrometry.net key from QSettings
        from PyQt6.QtCore import QSettings
        settings = QSettings("Cosmica", "Cosmica")
        api_key = settings.value("platesolver/astrometry_api_key", "") or None

        image_data = self._current_image.data
        file_path = self._current_image.file_path
        ra_hint = params.get("ra_hint")
        dec_hint = params.get("dec_hint")
        solver = params.get("solver", "auto")

        def _pcc_work(data, progress=None):
            import tempfile
            from pathlib import Path as _Path
            from cosmica.core.color_calibration import photometric_color_calibrate, ColorCalibrationParams
            from cosmica.core.star_catalog import plate_solve_auto, plate_solve_astap, plate_solve_astrometry_net, query_gaia_dr3

            # Determine the FITS path to give to the plate solver.
            # If the image has no saved file, write a temp FITS.
            solve_path = file_path
            tmp_fits = None
            if solve_path is None or not _Path(str(solve_path)).exists():
                try:
                    from cosmica.core.image_io import save_image, ImageData, FrameType
                    tmp_file = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
                    tmp_fits = _Path(tmp_file.name)
                    tmp_file.close()
                    save_image(ImageData(data=data, header={}), tmp_fits)
                    solve_path = tmp_fits
                except Exception as e:
                    log.warning("Could not save temp FITS for plate solving: %s", e)
                    solve_path = None

            wcs = None
            if solve_path is not None:
                if progress:
                    progress(0.1, "Plate solving...")
                if solver == "astap":
                    wcs = plate_solve_astap(solve_path, ra_hint, dec_hint)
                elif solver == "astrometry_net":
                    if api_key:
                        wcs = plate_solve_astrometry_net(solve_path, api_key, ra_hint, dec_hint)
                    else:
                        log.warning("No astrometry.net API key set in Preferences")
                else:  # auto
                    wcs = plate_solve_auto(solve_path, api_key, ra_hint, dec_hint)

            if tmp_fits is not None:
                tmp_fits.unlink(missing_ok=True)
                # Also clean up ASTAP output files
                for ext in (".wcs", ".ini"):
                    tmp_fits.with_suffix(ext).unlink(missing_ok=True)

            if progress:
                progress(0.5, "Querying Gaia DR3 catalog...")

            catalog_stars = []
            effective_ra = ra_hint
            effective_dec = dec_hint
            if wcs:
                effective_ra = effective_ra or wcs.get("ra")
                effective_dec = effective_dec or wcs.get("dec")

            if effective_ra is not None and effective_dec is not None:
                catalog_stars = query_gaia_dr3(effective_ra, effective_dec, radius_deg=0.5)
                log.info("Gaia DR3: %d stars retrieved", len(catalog_stars))

            if progress:
                progress(0.75, "Computing color correction...")

            if wcs is None and not catalog_stars:
                log.warning("No plate solution and no catalog — falling back to statistical calibration")
                result = __import__("cosmica.core.color_calibration", fromlist=["color_calibrate"]).color_calibrate(
                    data, ColorCalibrationParams()
                )
            else:
                result = photometric_color_calibrate(
                    data,
                    catalog_stars=catalog_stars if catalog_stars else None,
                    wcs=wcs,
                )

            if progress:
                progress(1.0, "Done")
            return result, wcs, catalog_stars

        def _on_pcc_done(payload):
            result, solved_wcs, stars = payload
            factors = result.correction_factors
            self._update_current_image(
                result.data,
                f"PCC complete (R={factors[0]:.3f}, G={factors[1]:.3f}, B={factors[2]:.3f})",
            )
            if solved_wcs or stars:
                self._update_wcs_overlay(solved_wcs or {}, stars or [])
            if self._project:
                self._project.add_history(
                    "Photometric Color Calibration", {"factors": list(factors)}
                )

        self._start_worker(_pcc_work, image_data, on_done=_on_pcc_done)

    @pyqtSlot()
    def _on_run_spcc(self):
        if self._current_image is None:
            return
        if not self._wcs_overlay_stars:
            self._log_panel.log(
                "SPCC requires plate solve data — run Solve & Calibrate (PCC) first", "warning"
            )
            return
        if self._current_image.data.ndim != 3 or self._current_image.data.shape[0] != 3:
            self._log_panel.log("SPCC requires a 3-channel RGB image", "warning")
            return

        from cosmica.core.spcc import spcc_calibrate
        params = self._tools_panel.get_spcc_params()
        catalog = self._wcs_overlay_stars          # list of (x_img, y_img, mag) — need bp_rp
        # _wcs_overlay_stars stores (x, y, magnitude); we use magnitude as proxy for bp_rp
        # In a future version, store actual BP-RP from catalog query.
        # For now use G magnitude as a rough temperature proxy (brighter G-type stars)
        catalog_with_color = [(x, y, max(0.0, min(3.0, (m - 5.0) / 3.0)))
                              for x, y, m in catalog]

        self._log_panel.log(
            f"Running SPCC ({params.filter_name}, {len(catalog_with_color)} catalog stars)…", "info"
        )

        def _spcc_work(data, progress=None):
            return spcc_calibrate(data, catalog_with_color, params=params,
                                  progress=progress or (lambda f, m: None))

        self._start_worker(_spcc_work, self._current_image.data,
                           on_done=lambda r: self._update_current_image(r, "SPCC complete"))

    @pyqtSlot()
    def _on_run_denoise(self):
        if self._current_image is None:
            return

        if self._tools_panel.is_tgv_denoise_selected():
            from cosmica.core.tgv_denoise import tgv_denoise
            tgv_params = self._tools_panel.get_tgv_params()
            self._log_panel.log(
                f"Running TGV denoising (strength={tgv_params.strength:.2f}, "
                f"{tgv_params.n_iter} iters)...", "info"
            )
            self._start_worker(tgv_denoise, self._current_image.data, tgv_params,
                               on_done=lambda r: self._update_current_image(r, "TGV denoising complete"))
            return

        params = self._tools_panel.get_denoise_params()
        self._log_panel.log(f"Running noise reduction ({params.method.name})...", "info")
        result = denoise(self._current_image.data, params)
        self._update_current_image(result, f"Noise reduction complete ({params.method.name})")
        if self._project:
            self._project.add_history(
                "Denoise",
                {
                    "method": params.method.name,
                    "strength": params.strength,
                },
            )

    @pyqtSlot()
    def _on_run_star_reduction(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_star_reduction_params()
        self._log_panel.log("Running star reduction...", "info")
        result = reduce_stars(self._current_image.data, params=params)
        self._update_current_image(result, "Star reduction complete")
        if self._project:
            self._project.add_history(
                "Star Reduction",
                {
                    "amount": params.amount,
                    "iterations": params.iterations,
                },
            )

    @pyqtSlot()
    def _on_run_split_channels(self):
        if self._current_image is None:
            return
        data = self._current_image.data
        if data.ndim != 3:
            self._log_panel.log("Split channels requires a color image", "warning")
            return
        channels = split_channels(data)
        # Display the first channel (R), log that others are available
        self._update_current_image(
            channels[0], f"Channel split: showing Red ({len(channels)} channels)"
        )
        if self._project:
            self._project.add_history("Split Channels", {"n_channels": len(channels)})

    @pyqtSlot()
    def _on_run_extract_luminance(self):
        if self._current_image is None:
            return
        lum = extract_luminance(self._current_image.data)
        self._update_current_image(lum, "Luminance extracted")
        if self._project:
            self._project.add_history("Extract Luminance", {})

    # ---------- Phase C processing operations ----------

    @pyqtSlot()
    def _on_run_wavelet_sharpen(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_wavelet_params()
        self._log_panel.log(f"Running wavelet sharpening ({params.n_scales} scales)...", "info")
        result = wavelet_sharpen(self._current_image.data, params)
        self._update_current_image(result, "Wavelet sharpening complete")
        if self._project:
            self._project.add_history(
                "Wavelet Sharpen",
                {
                    "n_scales": params.n_scales,
                    "scale_weights": params.scale_weights,
                },
            )
        self._macro_recorder.record_step(
            "wavelet_sharpen",
            {
                "n_scales": params.n_scales,
                "scale_weights": params.scale_weights,
            },
        )

    @pyqtSlot()
    def _on_run_mlt(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_mlt_params()
        n_thresh = sum(1 for t in params.noise_thresholds if t > 0)
        self._log_panel.log(
            f"Running MLT ({params.n_scales} scales, {n_thresh} denoise bands)…", "info"
        )
        result = wavelet_sharpen(self._current_image.data, params)
        self._update_current_image(result, "MLT complete")
        if self._project:
            self._project.add_history("MLT", {"n_scales": params.n_scales})

    @pyqtSlot()
    def _on_run_lrgb_combine(self):
        if self._current_image is None:
            return
        from PyQt6.QtWidgets import QFileDialog
        from cosmica.core.lrgb import lrgb_combine
        from cosmica.core.image_io import load_image

        # Validate current image is mono (luminance)
        data = self._current_image.data
        if data.ndim == 3 and data.shape[0] == 3:
            self._log_panel.log(
                "LRGB Combine: current image should be the Luminance (mono) image. "
                "Load your L image first.", "warning"
            )

        rgb_path, _ = QFileDialog.getOpenFileName(
            self, "Select RGB Color Image", "",
            "Images (*.fits *.fit *.fts *.xisf *.tif *.tiff *.png *.jpg)"
        )
        if not rgb_path:
            return

        params = self._tools_panel.get_lrgb_params()
        lum_data = data
        self._log_panel.log(
            f"LRGB Combine: L weight={params.luminance_weight}, "
            f"sat boost={params.saturation_boost}…", "info"
        )

        def _lrgb_work(lum, rgb_path_str, p, progress=None):
            import numpy as _np
            rgb_img = load_image(rgb_path_str)
            rgb = rgb_img.data
            if rgb.ndim == 2:
                rgb = _np.stack([rgb, rgb, rgb], axis=0)
            return lrgb_combine(lum, rgb, params=p,
                                progress=progress or (lambda f, m: None))

        def _on_lrgb_done(result):
            self._update_current_image(result, "LRGB combine complete")
            if self._project:
                self._project.add_history("LRGB Combine", {
                    "lum_weight": params.luminance_weight,
                    "sat_boost": params.saturation_boost,
                })

        self._start_worker(_lrgb_work, lum_data, rgb_path, params, on_done=_on_lrgb_done)

    @pyqtSlot()
    def _on_run_local_contrast(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_local_contrast_params()
        self._log_panel.log("Running local contrast enhancement...", "info")
        result = local_contrast_enhance(self._current_image.data, params)
        self._update_current_image(result, "Local contrast enhancement complete")
        if self._project:
            self._project.add_history(
                "Local Contrast",
                {
                    "clip_limit": params.clip_limit,
                    "amount": params.amount,
                },
            )
        self._macro_recorder.record_step(
            "local_contrast",
            {
                "clip_limit": params.clip_limit,
                "tile_size": params.tile_size,
                "amount": params.amount,
            },
        )

    @pyqtSlot()
    def _on_run_morphology(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_morphology_params()
        self._log_panel.log(f"Running morphology ({params.operation.name})...", "info")
        result = morphology_transform(self._current_image.data, params)
        self._update_current_image(result, f"Morphology {params.operation.name} complete")
        if self._project:
            self._project.add_history(
                "Morphology",
                {
                    "operation": params.operation.name,
                    "kernel_size": params.kernel_size,
                },
            )
        self._macro_recorder.record_step(
            "morphology",
            {
                "operation": params.operation.name,
                "kernel_size": params.kernel_size,
                "iterations": params.iterations,
            },
        )

    def _show_hdr_dialog(self):
        from cosmica.ui.dialogs.hdr_dialog import HDRDialog

        dialog = HDRDialog(self)
        dialog.result_ready.connect(self._on_hdr_result)
        dialog.exec()

    def _on_hdr_result(self, data):
        self._update_current_image(data, "HDR composition complete")
        if self._project:
            self._project.add_history("HDR Composition", {})

    # ---------- Phase D processing operations ----------

    @pyqtSlot()
    def _on_run_ai_denoise(self):
        if self._current_image is None:
            return
            self._log_panel.log(
                "warning",
            )
            return
        params = self._tools_panel.get_ai_denoise_params()
        self._log_panel.log("Running AI Denoise (GPU)...", "info")

        from cosmica.ai.inference.denoise import ai_denoise

        self._start_worker(
            ai_denoise,
            self._current_image.data,
            params=params,
            on_done=self._on_ai_denoise_done,
        )

    @pyqtSlot(object)
    def _on_ai_denoise_done(self, result):
        self._update_current_image(result, "AI Denoise complete")
        if self._project:
            self._project.add_history("AI Denoise", {})
        self._macro_recorder.record_step("ai_denoise")

    @pyqtSlot()
    def _on_run_ai_sharpen(self):
        if self._current_image is None:
            return
            self._log_panel.log(
                "warning",
            )
            return
        params = self._tools_panel.get_ai_sharpen_params()
        self._log_panel.log("Running AI Sharpen (GPU)...", "info")

        from cosmica.ai.inference.sharpen import ai_sharpen

        self._start_worker(
            ai_sharpen,
            self._current_image.data,
            params=params,
            on_done=self._on_ai_sharpen_done,
        )

    @pyqtSlot(object)
    def _on_ai_sharpen_done(self, result):
        self._update_current_image(result, "AI Sharpen complete")
        if self._project:
            self._project.add_history("AI Sharpen", {})
        self._macro_recorder.record_step("ai_sharpen")

    @pyqtSlot()
    def _on_run_starnet(self):
        if self._current_image is None:
            return
        extract = self._tools_panel.starnet_extract_stars
        self._log_panel.log("Running StarNet star removal (subprocess)...", "info")

        from cosmica.ai.inference.starnet import run_starnet

        self._start_worker(
            lambda data, progress=None: run_starnet(data, extract_stars=extract),
            self._current_image.data,
            on_done=self._on_starnet_done,
        )

    @pyqtSlot(object)
    def _on_starnet_done(self, result):
        if not result.success:
            self._log_panel.log(f"StarNet failed: {result.message}", "error")
            return
        self._update_current_image(result.starless, "StarNet complete: stars removed")
        if self._project:
            self._project.add_history("StarNet", {})
        self._macro_recorder.record_step("starnet")

    def _show_batch_dialog(self):
        from cosmica.ui.dialogs.batch_dialog import BatchDialog

        dialog = BatchDialog(self)
        dialog.exec()

    # ---------- Macro operations ----------

    @pyqtSlot()
    def _on_start_macro(self):
        self._macro_recorder.start("User Macro")
        self._tools_panel.set_macro_recording(True)
        self._log_panel.log("Macro recording started", "info")

    @pyqtSlot()
    def _on_stop_macro(self):
        self._current_macro = self._macro_recorder.stop()
        self._tools_panel.set_macro_recording(False)
        n = len(self._current_macro.steps)
        self._log_panel.log(f"Macro recording stopped: {n} steps captured", "success")

    @pyqtSlot()
    def _on_play_macro(self):
        if self._current_image is None:
            self._log_panel.log("Load an image first", "warning")
            return
        if self._current_macro is None or len(self._current_macro.steps) == 0:
            self._log_panel.log("No macro recorded or loaded", "warning")
            return
        self._log_panel.log(
            f"Playing macro: {self._current_macro.name} ({len(self._current_macro.steps)} steps)...",
            "info",
        )
        result = play_macro(self._current_image.data, self._current_macro)
        self._update_current_image(result, "Macro playback complete")
        if self._project:
            self._project.add_history("Play Macro", {"name": self._current_macro.name})

    @pyqtSlot()
    def _on_save_macro(self):
        if self._current_macro is None:
            self._log_panel.log("No macro to save", "warning")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Macro", "", "Cosmica Macro (*.json)")
        if path:
            save_macro(self._current_macro, Path(path))
            self._log_panel.log(f"Macro saved: {path}", "success")

    @pyqtSlot()
    def _on_load_macro(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Macro", "", "Cosmica Macro (*.json)")
        if path:
            self._current_macro = load_macro(Path(path))
            self._log_panel.log(
                f"Macro loaded: {self._current_macro.name} ({len(self._current_macro.steps)} steps)",
                "success",
            )

    # ---------- Smart Processor ----------

    def _show_equipment_dialog(self):
        from cosmica.ui.dialogs.equipment_dialog import EquipmentDialog

        dlg = EquipmentDialog(self, self._equipment_profile)
        dlg.profile_ready.connect(self._on_equipment_set)
        dlg.exec()

    def _on_equipment_set(self, profile: EquipmentProfile):
        self._equipment_profile = profile
        self._log_panel.log(
            f"Equipment set: {profile.camera.name} + {profile.telescope.name} "
            f"({profile.plate_scale():.2f} arcsec/px)",
            "success",
        )

    def _show_smart_processor_dialog(self):
        from cosmica.ui.dialogs.smart_process_dialog import SmartProcessDialog

        if self._current_image is None:
            self._log_panel.log("Load an image first", "warning")
            return

        dlg = SmartProcessDialog(self, equipment=self._equipment_profile)
        dlg.set_image_data(
            self._current_image.data,
            fits_header=getattr(self._current_image, "header", None),
        )
        dlg.result_ready.connect(self._on_smart_processor_result)
        dlg.exec()

    def _on_smart_processor_result(self, result):
        n_checks = len(result.quality_checks)
        n_passed = sum(1 for q in result.quality_checks if q.passed)
        target_info = ""
        if result.analysis.primary_target:
            target_info = f" Target: {result.analysis.primary_target.id}."

        self._update_current_image(
            result.image,
            f"Smart Processor complete.{target_info} Quality: {n_passed}/{n_checks} checks passed.",
        )
        if self._project:
            self._project.add_history(
                "Smart Processor",
                {
                    "quality_checks": f"{n_passed}/{n_checks}",
                },
            )

    # ── Blink Comparator ──────────────────────────────────────────────────────

    def _blink_load_from_file(self, slot: int):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Load Blink Image {'A' if slot == 0 else 'B'}", "",
            "Images (*.fits *.fit *.fts *.xisf *.tif *.tiff *.png *.jpg)"
        )
        if not path:
            return
        import numpy as _np
        img = load_image(path)
        self._blink_images[slot] = self._make_display_rgb(img.data)
        name = Path(path).name
        self._blink_names[slot] = name
        self._tools_panel.set_blink_slot_label("a" if slot == 0 else "b", name)
        self._log_panel.log(f"Blink {'A' if slot == 0 else 'B'}: loaded {name}", "info")

    def _blink_use_current(self, slot: int):
        if self._current_image is None:
            self._log_panel.log("No image loaded", "warning")
            return
        self._blink_images[slot] = self._make_display_rgb(self._current_image.data)
        name = getattr(self._current_image, "path", None)
        name = Path(name).name if name else "current image"
        self._blink_names[slot] = name
        self._tools_panel.set_blink_slot_label("a" if slot == 0 else "b", name)
        self._log_panel.log(f"Blink {'A' if slot == 0 else 'B'}: set to {name}", "info")

    def _make_display_rgb(self, data) -> "np.ndarray":
        """Convert image data to display-ready uint8 RGB (H,W,3) array."""
        import numpy as _np
        from cosmica.core.stretch import auto_stretch, StretchParams
        stretched = auto_stretch(data, StretchParams())
        if stretched.ndim == 2:
            rgb = _np.stack([stretched] * 3, axis=-1)
        else:
            # (3, H, W) → (H, W, 3)
            rgb = _np.transpose(stretched, (1, 2, 0))
        return (_np.clip(rgb, 0, 1) * 255).astype(_np.uint8)

    @pyqtSlot(bool)
    def _on_blink_toggle(self, enabled: bool):
        if enabled:
            if self._blink_images[0] is None or self._blink_images[1] is None:
                self._log_panel.log(
                    "Blink Comparator: load both Image A and Image B first", "warning"
                )
                self._tools_panel.reset_blink_toggle()
                return
            fps = self._tools_panel._blink_fps_spin.value()
            self._blink_index = 0
            self._blink_timer.start(1000 // fps)
            self._log_panel.log(f"Blink Comparator started ({fps} fps)", "info")
        else:
            self._blink_timer.stop()
            # Restore the original current image on canvas
            if self._current_image is not None:
                self._display_image(self._current_image)
            self._log_panel.log("Blink Comparator stopped", "info")

    @pyqtSlot(int)
    def _on_blink_fps_changed(self, fps: int):
        if self._blink_timer.isActive():
            self._blink_timer.setInterval(1000 // fps)

    def _blink_tick(self):
        img = self._blink_images[self._blink_index]
        if img is not None:
            self._canvas.set_image(img)
            slot_name = "A" if self._blink_index == 0 else "B"
            self.statusBar().showMessage(
                f"Blink: {slot_name} — {self._blink_names[self._blink_index]}"
            )
        self._blink_index = 1 - self._blink_index

    def keyPressEvent(self, event):
        """Global keyboard shortcuts."""
        from PyQt6.QtCore import Qt as _Qt
        if event.key() == _Qt.Key.Key_B and not event.isAutoRepeat():
            # Toggle blink comparator
            btn = self._tools_panel._blink_toggle_btn
            btn.setChecked(not btn.isChecked())
            return
        super().keyPressEvent(event)
