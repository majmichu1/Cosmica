"""Processing Log Panel — displays operation logs and progress."""

from __future__ import annotations

import logging
from datetime import datetime

from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QTextEdit, QVBoxLayout, QWidget


class LogPanel(QWidget):
    """Bottom panel showing processing log and progress bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Progress bar row
        progress_row = QHBoxLayout()
        self._progress_label = QLabel("Ready")
        self._progress_label.setStyleSheet("color: #969696; font-size: 12px;")
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setMaximumHeight(16)
        progress_row.addWidget(self._progress_label, 1)
        progress_row.addWidget(self._progress_bar, 2)
        layout.addLayout(progress_row)

        # Log text
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(120)
        self._log_text.setStyleSheet(
            "QTextEdit { font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 11px; }"
        )
        layout.addWidget(self._log_text)

    @pyqtSlot(float, str)
    def update_progress(self, fraction: float, message: str):
        self._progress_bar.setValue(int(fraction * 1000))
        self._progress_label.setText(message)

    def log(self, message: str, level: str = "info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {
            "info": "#d4d4d4",
            "warning": "#e5c07b",
            "error": "#e06c75",
            "success": "#98c379",
        }
        color = colors.get(level, "#d4d4d4")
        self._log_text.append(
            f'<span style="color:#6e6e6e">[{timestamp}]</span> '
            f'<span style="color:{color}">{message}</span>'
        )
        self._log_text.verticalScrollBar().setValue(
            self._log_text.verticalScrollBar().maximum()
        )

    def clear_log(self):
        self._log_text.clear()

    def reset_progress(self):
        self._progress_bar.setValue(0)
        self._progress_label.setText("Ready")


class QtLogHandler(logging.Handler):
    """Route Python logging to the LogPanel."""

    def __init__(self, log_panel: LogPanel):
        super().__init__()
        self._panel = log_panel

    def emit(self, record: logging.LogRecord):
        level_map = {
            logging.DEBUG: "info",
            logging.INFO: "info",
            logging.WARNING: "warning",
            logging.ERROR: "error",
            logging.CRITICAL: "error",
        }
        level = level_map.get(record.levelno, "info")
        self._panel.log(record.getMessage(), level)
