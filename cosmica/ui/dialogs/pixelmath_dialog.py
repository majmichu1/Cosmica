"""Pixel Math Dialog — evaluate mathematical expressions on image pixels."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from cosmica.core.pixel_math import (
    PixelMathError,
    evaluate,
    prepare_variables,
    validate_expression,
)


class PixelMathDialog(QDialog):
    """Dialog for evaluating pixel math expressions on the current image."""

    result_ready = pyqtSignal(np.ndarray)  # emits processed image

    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pixel Math")
        self.setMinimumWidth(500)
        self.setMinimumHeight(350)

        self._image = image
        self._variables = prepare_variables(image)

        layout = QVBoxLayout(self)

        # Variable reference
        ref_group = QGroupBox("Available Variables")
        ref_layout = QVBoxLayout(ref_group)
        var_list = sorted(self._variables.keys())
        ref_text = QLabel(
            f"<b>Variables:</b> {', '.join(var_list)}<br>"
            "<b>Functions:</b> min, max, abs, sqrt, log, exp, clip, normalize, "
            "mean, median, mtf, pow, iif<br>"
            "<b>Operators:</b> +, -, *, /, **, >, <, >=, <=, ==, !="
        )
        ref_text.setWordWrap(True)
        ref_text.setStyleSheet("color: #b8b8b8; font-size: 11px;")
        ref_layout.addWidget(ref_text)
        layout.addWidget(ref_group)

        # Expression input
        expr_group = QGroupBox("Expression")
        expr_layout = QVBoxLayout(expr_group)

        self._expr_input = QLineEdit()
        self._expr_input.setPlaceholderText("e.g.  T * 2  or  clip(R - G, 0, 1)  or  sqrt(L)")
        self._expr_input.setStyleSheet("font-family: monospace; font-size: 13px; padding: 4px;")
        self._expr_input.textChanged.connect(self._on_expression_changed)
        expr_layout.addWidget(self._expr_input)

        self._validation_label = QLabel("")
        self._validation_label.setStyleSheet("font-size: 11px;")
        expr_layout.addWidget(self._validation_label)

        layout.addWidget(expr_group)

        # Output log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(100)
        self._log.setStyleSheet("font-family: monospace; font-size: 11px; background: #1e1e1e;")
        layout.addWidget(self._log)

        # Buttons
        btn_row = QHBoxLayout()

        self._btn_evaluate = QPushButton("Evaluate")
        self._btn_evaluate.setEnabled(False)
        self._btn_evaluate.clicked.connect(self._evaluate)
        btn_row.addWidget(self._btn_evaluate)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self._btn_cancel)

        layout.addLayout(btn_row)

    def _on_expression_changed(self, text: str):
        if not text.strip():
            self._validation_label.setText("")
            self._btn_evaluate.setEnabled(False)
            return

        error = validate_expression(text)
        if error is None:
            self._validation_label.setText("Valid expression")
            self._validation_label.setStyleSheet("color: #69db7c; font-size: 11px;")
            self._btn_evaluate.setEnabled(True)
        else:
            self._validation_label.setText(f"Error: {error}")
            self._validation_label.setStyleSheet("color: #ff6b6b; font-size: 11px;")
            self._btn_evaluate.setEnabled(False)

    def _evaluate(self):
        expr = self._expr_input.text().strip()
        if not expr:
            return

        self._log.append(f"> {expr}")
        try:
            result = evaluate(expr, self._variables)
            shape_str = "x".join(str(d) for d in result.shape)
            self._log.append(
                f"  Result: {shape_str}, "
                f"min={result.min():.4f}, max={result.max():.4f}, "
                f"mean={result.mean():.4f}"
            )
            self.result_ready.emit(result)
            self.accept()
        except PixelMathError as e:
            self._log.append(f"  Error: {e}")
        except Exception as e:
            self._log.append(f"  Unexpected error: {e}")
