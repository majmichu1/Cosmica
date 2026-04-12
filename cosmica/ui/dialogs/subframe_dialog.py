"""Subframe Selector Dialog — score and reject light frames by quality.

Shows a thumbnail preview alongside per-frame statistics so the user can
visually verify which frames are sharp, round-star, and high-SNR before
passing them to stacking.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSize, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from cosmica.core.subframe_selector import (
    SubframeScore,
    SubframeSelectorParams,
    score_subframes,
)

_THUMB_SIZE = 80   # px — thumbnail column width/height


def _make_thumbnail(path: str, size: int = _THUMB_SIZE) -> QPixmap:
    """Load an image, auto-stretch, and return a square QPixmap thumbnail."""
    try:
        from cosmica.core.image_io import load_image
        from cosmica.core.stretch import auto_stretch, StretchParams
        img = load_image(path)
        data = img.data
        stretched = auto_stretch(data, StretchParams())
        if stretched.ndim == 2:
            rgb = np.stack([stretched] * 3, axis=-1)
        else:
            rgb = np.transpose(stretched, (1, 2, 0))
        rgb8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        h, w = rgb8.shape[:2]
        # Centre-crop to square
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        crop = np.ascontiguousarray(rgb8[y0:y0+s, x0:x0+s])
        qimg = QImage(crop.tobytes(), s, s, s * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        return pix.scaled(size, size,
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
    except Exception:
        pix = QPixmap(size, size)
        pix.fill(QColor(40, 40, 40))
        return pix


class _ScoreWorker(QThread):
    """Run subframe scoring off the main thread."""

    progress = pyqtSignal(float, str)
    finished = pyqtSignal(list)      # list[SubframeScore]
    thumbnail = pyqtSignal(int, object)  # row, QPixmap
    error = pyqtSignal(str)

    def __init__(self, paths: list[str], params: SubframeSelectorParams):
        super().__init__()
        self._paths = paths
        self._params = params

    def run(self):
        try:
            scores = score_subframes(
                self._paths,
                self._params,
                progress=lambda f, m: self.progress.emit(f, m),
            )
            # Emit thumbnails after scoring (scoring already loaded the images)
            for row, s in enumerate(scores):
                pix = _make_thumbnail(s.file_path)
                self.thumbnail.emit(row, pix)
            self.finished.emit(scores)
        except Exception as exc:
            self.error.emit(str(exc))


class SubframeDialog(QDialog):
    """Dialog for scoring and selecting subframes with thumbnail previews."""

    accepted_frames = pyqtSignal(list)  # list[str] of accepted file paths

    # Column indices
    _COL_THUMB  = 0
    _COL_FILE   = 1
    _COL_FWHM   = 2
    _COL_ECC    = 3
    _COL_SNR    = 4
    _COL_BG     = 5
    _COL_STARS  = 6
    _COL_SCORE  = 7
    _COL_STATUS = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Subframe Selector")
        self.setMinimumSize(950, 580)

        self._scores: list[SubframeScore] = []
        self._worker = None
        self._paths: list[str] = []

        layout = QVBoxLayout(self)

        # ── Scoring weights ───────────────────────────────────────────────────
        params_group = QGroupBox("Scoring Weights")
        p_layout = QHBoxLayout(params_group)

        self._fwhm_w = QDoubleSpinBox()
        self._fwhm_w.setRange(0, 1); self._fwhm_w.setValue(0.3); self._fwhm_w.setSingleStep(0.05)
        p_layout.addWidget(QLabel("FWHM:")); p_layout.addWidget(self._fwhm_w)

        self._ecc_w = QDoubleSpinBox()
        self._ecc_w.setRange(0, 1); self._ecc_w.setValue(0.2); self._ecc_w.setSingleStep(0.05)
        p_layout.addWidget(QLabel("Ecc:")); p_layout.addWidget(self._ecc_w)

        self._snr_w = QDoubleSpinBox()
        self._snr_w.setRange(0, 1); self._snr_w.setValue(0.3); self._snr_w.setSingleStep(0.05)
        p_layout.addWidget(QLabel("SNR:")); p_layout.addWidget(self._snr_w)

        self._stars_w = QDoubleSpinBox()
        self._stars_w.setRange(0, 1); self._stars_w.setValue(0.2); self._stars_w.setSingleStep(0.05)
        p_layout.addWidget(QLabel("Stars:")); p_layout.addWidget(self._stars_w)

        p_layout.addStretch()
        layout.addWidget(params_group)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load Frames...")
        load_btn.clicked.connect(self._load_frames)
        btn_row.addWidget(load_btn)

        load_dir_btn = QPushButton("Load Folder...")
        load_dir_btn.setToolTip("Load all FITS files from a folder")
        load_dir_btn.clicked.connect(self._load_folder)
        btn_row.addWidget(load_dir_btn)

        self._score_btn = QPushButton("Score All Frames")
        self._score_btn.setEnabled(False)
        self._score_btn.clicked.connect(self._run_scoring)
        btn_row.addWidget(self._score_btn)

        btn_row.addStretch()

        self._frame_count_label = QLabel("No frames loaded")
        self._frame_count_label.setStyleSheet("color: #aaa; font-size: 11px;")
        btn_row.addWidget(self._frame_count_label)

        btn_row.addStretch()

        accept_btn = QPushButton("Use Accepted Frames")
        accept_btn.setToolTip("Send accepted frames to the stacking pipeline")
        accept_btn.clicked.connect(self._emit_accepted)
        btn_row.addWidget(accept_btn)

        layout.addLayout(btn_row)

        # ── Progress ──────────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # ── Results table ─────────────────────────────────────────────────────
        self._table = QTableWidget(0, 9)
        self._table.setHorizontalHeaderLabels([
            "Preview", "File", "FWHM", "Eccentricity", "SNR",
            "Background", "Stars", "Score", "Status",
        ])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(self._COL_THUMB,  QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(self._COL_FILE,   QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(self._COL_STATUS, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(self._COL_THUMB,  _THUMB_SIZE + 8)
        self._table.setColumnWidth(self._COL_STATUS, 80)
        self._table.verticalHeader().setDefaultSectionSize(_THUMB_SIZE + 8)

        layout.addWidget(self._table, 1)

        # ── Summary label ─────────────────────────────────────────────────────
        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self._summary_label)

    # ── Load ──────────────────────────────────────────────────────────────────

    def _load_frames(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Light Frames", "",
            "Images (*.fits *.fit *.fts *.FTS *.xisf *.tif *.tiff *.png)"
        )
        if paths:
            self._set_paths(paths)

    def _load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder of Light Frames")
        if not folder:
            return
        p = Path(folder)
        paths = []
        for ext in ("*.fits", "*.fit", "*.fts", "*.FTS", "*.xisf", "*.tif", "*.tiff"):
            paths.extend(str(f) for f in sorted(p.glob(ext)))
        if paths:
            self._set_paths(paths)

    def _set_paths(self, paths: list[str]):
        self._paths = paths
        self._score_btn.setEnabled(True)
        self._table.setRowCount(0)
        self._summary_label.setText("")
        n = len(paths)
        self._frame_count_label.setText(f"{n} frame{'s' if n != 1 else ''} loaded")

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _run_scoring(self):
        if not self._paths:
            return
        params = SubframeSelectorParams(
            fwhm_weight=self._fwhm_w.value(),
            eccentricity_weight=self._ecc_w.value(),
            snr_weight=self._snr_w.value(),
            stars_weight=self._stars_w.value(),
        )

        # Pre-populate rows with placeholder thumbnails
        self._table.setRowCount(len(self._paths))
        for row, path in enumerate(self._paths):
            ph = QPixmap(_THUMB_SIZE, _THUMB_SIZE)
            ph.fill(QColor(30, 30, 30))
            thumb_label = QLabel()
            thumb_label.setPixmap(ph)
            thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setCellWidget(row, self._COL_THUMB, thumb_label)

            name_item = QTableWidgetItem(Path(path).name)
            name_item.setForeground(QColor(160, 160, 160))
            self._table.setItem(row, self._COL_FILE, name_item)

        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._score_btn.setEnabled(False)

        self._worker = _ScoreWorker(self._paths, params)
        self._worker.progress.connect(lambda f, _m: self._progress.setValue(int(f * 100)))
        self._worker.thumbnail.connect(self._on_thumbnail)
        self._worker.finished.connect(self._on_scored)
        self._worker.error.connect(lambda msg: self._progress.setFormat(f"Error: {msg}"))
        self._worker.start()

    def _on_thumbnail(self, row: int, pix: QPixmap):
        if row >= self._table.rowCount():
            return
        label = QLabel()
        label.setPixmap(pix)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setCellWidget(row, self._COL_THUMB, label)

    def _on_scored(self, scores: list[SubframeScore]):
        self._scores = scores
        self._progress.setVisible(False)
        self._score_btn.setEnabled(True)
        self._table.setRowCount(len(scores))

        n_accepted = 0
        for row, s in enumerate(scores):
            accepted = s.accepted
            if accepted:
                n_accepted += 1

            color = QColor(200, 255, 200) if accepted else QColor(255, 150, 150)
            bg    = QColor(30, 50, 30)   if accepted else QColor(50, 30, 30)

            def _item(text, align=Qt.AlignmentFlag.AlignCenter):
                it = QTableWidgetItem(text)
                it.setForeground(color)
                it.setBackground(bg)
                it.setTextAlignment(align)
                return it

            self._table.setItem(row, self._COL_FILE,   _item(Path(s.file_path).name,
                                                              Qt.AlignmentFlag.AlignLeft |
                                                              Qt.AlignmentFlag.AlignVCenter))
            self._table.setItem(row, self._COL_FWHM,   _item(f"{s.fwhm:.2f}"))
            self._table.setItem(row, self._COL_ECC,    _item(f"{s.eccentricity:.3f}"))
            self._table.setItem(row, self._COL_SNR,    _item(f"{s.snr:.1f}"))
            self._table.setItem(row, self._COL_BG,     _item(f"{s.background:.4f}"))
            self._table.setItem(row, self._COL_STARS,  _item(str(s.n_stars)))
            self._table.setItem(row, self._COL_SCORE,  _item(f"{s.quality_score:.3f}"))
            self._table.setItem(row, self._COL_STATUS, _item("Accepted" if accepted else "Rejected"))

            # Colour the thumbnail cell background too
            if self._table.cellWidget(row, self._COL_THUMB):
                self._table.cellWidget(row, self._COL_THUMB).setStyleSheet(
                    f"background: {'#1a3a1a' if accepted else '#3a1a1a'};"
                )

        pct = 100 * n_accepted / max(len(scores), 1)
        self._summary_label.setText(
            f"{n_accepted}/{len(scores)} frames accepted ({pct:.0f}%)  —  "
            f"green = accepted, red = rejected"
        )

    # ── Emit ──────────────────────────────────────────────────────────────────

    def _emit_accepted(self):
        accepted = [s.file_path for s in self._scores if s.accepted]
        if accepted:
            self.accepted_frames.emit(accepted)
            self.accept()
