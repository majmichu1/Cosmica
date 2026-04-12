"""Embedded Python REPL console for interactive scripting."""

from __future__ import annotations

import code
import io
import sys
import traceback

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QKeyEvent, QTextCursor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

_BANNER = (
    "Cosmica Python Console\n"
    "  image      — current ImageData (read-only reference)\n"
    "  apply(arr) — update current image with a new ndarray\n"
    "  np         — numpy\n"
    "Type help() for Python help.\n"
)


class _HistoryLineEdit(QPlainTextEdit):
    """Single-line-ish input with up/down history navigation."""

    execute_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumHeight(80)
        self.setPlaceholderText("Enter Python code — Shift+Enter for multi-line, Enter to run")
        self._history: list[str] = []
        self._hist_idx = -1
        font = QFont("Monospace", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Return and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            text = self.toPlainText().strip()
            if text:
                self._history.append(text)
                self._hist_idx = -1
                self.execute_requested.emit(text)
                self.clear()
            return
        if event.key() == Qt.Key.Key_Up:
            if self._history:
                self._hist_idx = min(self._hist_idx + 1, len(self._history) - 1)
                self.setPlainText(self._history[-(self._hist_idx + 1)])
            return
        if event.key() == Qt.Key.Key_Down:
            if self._hist_idx > 0:
                self._hist_idx -= 1
                self.setPlainText(self._history[-(self._hist_idx + 1)])
            else:
                self._hist_idx = -1
                self.clear()
            return
        super().keyPressEvent(event)


class PythonConsoleWidget(QWidget):
    """Embedded Python console dockable widget."""

    #: Emitted when apply(arr) is called — main window connects this
    image_updated = pyqtSignal(object)  # np.ndarray

    def __init__(self, parent=None):
        super().__init__(parent)
        self._namespace: dict = {}
        self._setup_namespace()
        self._build_ui()
        self._write(_BANNER)

    def _setup_namespace(self):
        import numpy as np
        import cosmica

        console_ref = self

        def apply(arr):
            """Push an ndarray back as the current image."""
            console_ref.image_updated.emit(arr)
            console_ref._write("Image updated.\n")

        self._namespace = {
            "__name__": "__console__",
            "np": np,
            "cosmica": cosmica,
            "image": None,
            "apply": apply,
        }

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Output area
        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        font = QFont("Monospace", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._output.setFont(font)
        self._output.setStyleSheet(
            "background: #1a1a2e; color: #e0e0e0; border: 1px solid #444;"
        )
        layout.addWidget(self._output, stretch=1)

        # Input row
        self._input = _HistoryLineEdit()
        self._input.setStyleSheet(
            "background: #16213e; color: #e0e0e0; border: 1px solid #555;"
        )
        self._input.execute_requested.connect(self._execute)
        layout.addWidget(self._input)

        btn_row = QHBoxLayout()
        btn_run = QPushButton("Run (Enter)")
        btn_run.clicked.connect(lambda: self._execute(self._input.toPlainText().strip()))
        btn_row.addWidget(btn_run)

        btn_clear = QPushButton("Clear output")
        btn_clear.clicked.connect(self._output.clear)
        btn_row.addWidget(btn_clear)

        btn_row.addStretch()
        layout.addLayout(btn_row)

    def set_image(self, image_data):
        """Update the 'image' variable in the console namespace."""
        self._namespace["image"] = image_data

    def _execute(self, source: str):
        if not source.strip():
            return
        self._write(f">>> {source}\n")

        # Redirect stdout/stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = buf = io.StringIO()

        try:
            try:
                result = eval(compile(source, "<console>", "eval"), self._namespace)
                output = buf.getvalue()
                if output:
                    self._write(output)
                if result is not None:
                    self._write(repr(result) + "\n")
            except SyntaxError:
                exec(compile(source, "<console>", "exec"), self._namespace)
                output = buf.getvalue()
                if output:
                    self._write(output)
        except Exception:
            output = buf.getvalue()
            if output:
                self._write(output)
            self._write(traceback.format_exc())
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

        # Scroll to bottom
        self._output.moveCursor(QTextCursor.MoveOperation.End)

    def _write(self, text: str):
        self._output.moveCursor(QTextCursor.MoveOperation.End)
        self._output.insertPlainText(text)
        self._output.moveCursor(QTextCursor.MoveOperation.End)
