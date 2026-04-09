"""Subframe Selector Dialog — score and reject light frames by quality."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, Qt, pyqtSignal
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


class _ScoreWorker(QThread):
    """Run subframe scoring off the main thread."""

    progress = pyqtSignal(float, str)
    finished = pyqtSignal(list)  # list[SubframeScore]
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
            self.finished.emit(scores)
        except Exception as exc:
            self.error.emit(str(exc))


class SubframeDialog(QDialog):
    """Dialog for scoring and selecting subframes."""

    accepted_frames = pyqtSignal(list)  # list[str] of accepted file paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Subframe Selector")
        self.setMinimumSize(750, 500)

        self._scores: list[SubframeScore] = []
        self._worker = None

        layout = QVBoxLayout(self)

        # Parameters
        params_group = QGroupBox("Scoring Weights")
        p_layout = QHBoxLayout(params_group)

        self._fwhm_w = QDoubleSpinBox()
        self._fwhm_w.setRange(0, 1)
        self._fwhm_w.setValue(0.3)
        self._fwhm_w.setSingleStep(0.05)
        p_layout.addWidget(QLabel("FWHM:"))
        p_layout.addWidget(self._fwhm_w)

        self._ecc_w = QDoubleSpinBox()
        self._ecc_w.setRange(0, 1)
        self._ecc_w.setValue(0.2)
        self._ecc_w.setSingleStep(0.05)
        p_layout.addWidget(QLabel("Ecc:"))
        p_layout.addWidget(self._ecc_w)

        self._snr_w = QDoubleSpinBox()
        self._snr_w.setRange(0, 1)
        self._snr_w.setValue(0.3)
        self._snr_w.setSingleStep(0.05)
        p_layout.addWidget(QLabel("SNR:"))
        p_layout.addWidget(self._snr_w)

        self._stars_w = QDoubleSpinBox()
        self._stars_w.setRange(0, 1)
        self._stars_w.setValue(0.2)
        self._stars_w.setSingleStep(0.05)
        p_layout.addWidget(QLabel("Stars:"))
        p_layout.addWidget(self._stars_w)

        layout.addWidget(params_group)

        # Buttons
        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load Frames...")
        load_btn.clicked.connect(self._load_frames)
        btn_row.addWidget(load_btn)

        self._score_btn = QPushButton("Score")
        self._score_btn.setEnabled(False)
        self._score_btn.clicked.connect(self._run_scoring)
        btn_row.addWidget(self._score_btn)

        btn_row.addStretch()

        accept_btn = QPushButton("Use Accepted")
        accept_btn.clicked.connect(self._emit_accepted)
        btn_row.addWidget(accept_btn)

        layout.addLayout(btn_row)

        # Progress
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # Results table
        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels([
            "File", "FWHM", "Eccentricity", "SNR", "Background",
            "Stars", "Score", "Status",
        ])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self._table)

        self._paths: list[str] = []

    def _load_frames(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Light Frames", "",
            "Images (*.fits *.fit *.fts *.xisf *.tif *.tiff *.png)"
        )
        if paths:
            self._paths = paths
            self._score_btn.setEnabled(True)
            self._table.setRowCount(0)

    def _run_scoring(self):
        if not self._paths:
            return
        params = SubframeSelectorParams(
            fwhm_weight=self._fwhm_w.value(),
            eccentricity_weight=self._ecc_w.value(),
            snr_weight=self._snr_w.value(),
            stars_weight=self._stars_w.value(),
        )
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._score_btn.setEnabled(False)

        self._worker = _ScoreWorker(self._paths, params)
        self._worker.progress.connect(
            lambda f, _m: self._progress.setValue(int(f * 100))
        )
        self._worker.finished.connect(self._on_scored)
        self._worker.error.connect(
            lambda msg: self._progress.setFormat(f"Error: {msg}")
        )
        self._worker.start()

    def _on_scored(self, scores: list[SubframeScore]):
        self._scores = scores
        self._progress.setVisible(False)
        self._score_btn.setEnabled(True)

        self._table.setRowCount(len(scores))
        for row, s in enumerate(scores):
            name = Path(s.file_path).name
            values = [
                name,
                f"{s.fwhm:.2f}",
                f"{s.eccentricity:.3f}",
                f"{s.snr:.1f}",
                f"{s.background:.4f}",
                str(s.n_stars),
                f"{s.quality_score:.3f}",
                "Accepted" if s.accepted else "Rejected",
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                if col == 7:
                    item.setForeground(
                        Qt.GlobalColor.green if s.accepted else Qt.GlobalColor.red
                    )
                self._table.setItem(row, col, item)

    def _emit_accepted(self):
        accepted = [s.file_path for s in self._scores if s.accepted]
        if accepted:
            self.accepted_frames.emit(accepted)
            self.accept()
