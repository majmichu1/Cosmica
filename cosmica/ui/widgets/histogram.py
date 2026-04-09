"""Histogram Widget — interactive histogram display for image channels."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget


class HistogramWidget(QWidget):
    """Renders R/G/B/Luminance histograms with log scale."""

    CHANNEL_COLORS = {
        "red": QColor(220, 50, 50, 180),
        "green": QColor(50, 200, 50, 180),
        "blue": QColor(50, 100, 220, 180),
        "luminance": QColor(200, 200, 200, 120),
        "gray": QColor(200, 200, 200, 180),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 100)
        self.setMaximumHeight(160)
        self._data: dict | None = None
        self._log_scale = True

    def set_histogram_data(self, data: dict):
        """Set histogram data from core.stretch.compute_histogram()."""
        self._data = data
        self.update()

    def clear(self):
        self._data = None
        self.update()

    def set_log_scale(self, enabled: bool):
        self._log_scale = enabled
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self._data is None:
            painter.setPen(QPen(QColor(100, 100, 100)))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No histogram data")
            painter.end()
            return

        margin = 4
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin

        # Draw order: luminance first (behind), then R, G, B
        draw_order = ["luminance", "gray", "red", "green", "blue"]

        for channel_name in draw_order:
            if channel_name not in self._data:
                continue
            counts = self._data[channel_name].astype(np.float64)
            if counts.max() == 0:
                continue

            if self._log_scale:
                counts = np.log1p(counts)

            max_val = counts.max()
            if max_val > 0:
                counts = counts / max_val

            color = self.CHANNEL_COLORS.get(channel_name, QColor(200, 200, 200, 180))
            n_bins = len(counts)

            path = QPainterPath()
            path.moveTo(margin, margin + h)

            for i in range(n_bins):
                x = margin + (i / n_bins) * w
                y = margin + h - counts[i] * h
                path.lineTo(x, y)

            path.lineTo(margin + w, margin + h)
            path.closeSubpath()

            fill_color = QColor(color)
            fill_color.setAlpha(60)
            painter.fillPath(path, fill_color)

            painter.setPen(QPen(color, 1.0))
            painter.drawPath(path)

        # Draw border
        painter.setPen(QPen(QColor(60, 60, 60)))
        painter.drawRect(QRectF(margin, margin, w, h))

        painter.end()
