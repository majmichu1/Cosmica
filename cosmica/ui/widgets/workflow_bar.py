"""Workflow Pipeline Bar — 7-step astrophotography processing pipeline indicator."""

from __future__ import annotations

from enum import Enum, auto

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

from cosmica.ui.theme import (
    ACCENT,
    BG_HOVER,
    BG_SECONDARY,
    BORDER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

_STEPS = [
    ("Pre-Process", "Calibrate · Cosmetic"),
    ("Stacking", "Align · Integrate"),
    ("Background", "Extract · ABE"),
    ("Stretch", "GHS · Curves"),
    ("Color", "SCNR · Cal"),
    ("Detail", "Decon · Denoise"),
    ("Export", "FITS · TIFF · PNG"),
]


class StepState(Enum):
    DONE = auto()
    ACTIVE = auto()
    NEXT = auto()
    DISABLED = auto()


class _StepButton(QPushButton):
    """A single workflow step button with two lines of text."""

    def __init__(self, title: str, subtitle: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._subtitle = subtitle
        self._state = StepState.DISABLED
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(44)
        self._update_style()

    def set_state(self, state: StepState):
        self._state = state
        self._update_style()
        self.setEnabled(state in (StepState.ACTIVE, StepState.NEXT, StepState.DONE))

    def _update_style(self):
        s = self._state
        if s == StepState.DONE:
            title_html = f'<span style="color:{ACCENT}">✓ {self._title}</span>'
            sub_color = TEXT_SECONDARY
            border_bottom = "none"
            bg = "transparent"
        elif s == StepState.ACTIVE:
            title_html = f'<span style="color:{TEXT_PRIMARY};font-weight:700">{self._title}</span>'
            sub_color = TEXT_PRIMARY
            border_bottom = f"2px solid {ACCENT}"
            bg = BG_HOVER
        elif s == StepState.NEXT:
            title_html = f'<span style="color:{TEXT_PRIMARY}">{self._title}</span>'
            sub_color = TEXT_SECONDARY
            border_bottom = "none"
            bg = "transparent"
        else:  # DISABLED
            title_html = f'<span style="color:{TEXT_SECONDARY}">{self._title}</span>'
            sub_color = TEXT_SECONDARY
            border_bottom = "none"
            bg = "transparent"

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg};
                border: none;
                border-bottom: {border_bottom};
                border-radius: 0;
                padding: 2px 8px;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: {BG_HOVER};
            }}
        """)
        self.setText(
            f'<html><body style="text-align:center">'
            f'{title_html}<br>'
            f'<span style="color:{sub_color};font-size:10px">{self._subtitle}</span>'
            f'</body></html>'
        )

    def setText(self, text: str):
        # We abuse QAbstractButton's text for rich text by using a child label instead
        if not hasattr(self, "_label"):
            from PyQt6.QtWidgets import QVBoxLayout as _VL
            layout = _VL(self)
            layout.setContentsMargins(4, 2, 4, 2)
            layout.setSpacing(0)
            self._label = QLabel()
            self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            layout.addWidget(self._label)
        self._label.setText(text)


class WorkflowBar(QWidget):
    """Horizontal 7-step pipeline progress bar."""

    step_clicked = pyqtSignal(int)  # step index 0-6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("WorkflowBar")
        self.setFixedHeight(44)
        self.setStyleSheet(f"background-color: {BG_SECONDARY}; border-bottom: 1px solid {BORDER};")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(0)

        self._steps: list[_StepButton] = []

        for i, (title, subtitle) in enumerate(_STEPS):
            btn = _StepButton(title, subtitle)
            btn.clicked.connect(lambda checked, idx=i: self.step_clicked.emit(idx))
            self._steps.append(btn)
            layout.addWidget(btn)

            if i < len(_STEPS) - 1:
                sep = QLabel("›")
                sep.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 16px; padding: 0 2px;")
                sep.setFixedWidth(18)
                sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(sep)

        # Default state: first step active, rest next, last two disabled
        self._states = [StepState.ACTIVE] + [StepState.NEXT] * 6
        self._apply_states()

    def _apply_states(self):
        for btn, state in zip(self._steps, self._states):
            btn.set_state(state)

    def set_step_state(self, step_idx: int, state: StepState):
        """Set the visual state of a single step."""
        if 0 <= step_idx < len(self._steps):
            self._states[step_idx] = state
            self._steps[step_idx].set_state(state)
