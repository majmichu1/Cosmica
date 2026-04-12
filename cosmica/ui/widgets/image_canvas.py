"""Image Canvas Widget — zoomable, pannable image display with before/after split."""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QWidget


class ImageCanvas(QWidget):
    """High-performance image display with zoom, pan, split preview, and overlays."""

    zoom_changed = pyqtSignal(float)
    cursor_position = pyqtSignal(int, int, list)  # x, y, pixel values
    sample_placed = pyqtSignal(float, float)       # image-space x, y
    sample_removed = pyqtSignal(float, float)      # image-space x, y (nearest)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._pixmap: QPixmap | None = None
        self._pixmap_after: QPixmap | None = None
        self._image_data: np.ndarray | None = None

        self._zoom = 1.0
        self._pan_offset = QPointF(0, 0)
        self._dragging = False
        self._drag_start = QPoint()

        self._split_mode = False
        self._split_position = 0.5

        self._fit_to_window = True

        # Sample placement mode (for dynamic background extraction)
        self._sample_mode = False
        self._sample_points: list[tuple[float, float]] = []  # image-space coords

        # WCS/catalog star overlay
        self._overlay_stars: list[tuple[float, float, float]] = []  # (x, y, mag) image-space
        self._show_wcs_overlay = False

    # ── Public API ───────────────────────────────────────────────────────────

    def set_image(self, rgb_array: np.ndarray, image_data: np.ndarray | None = None):
        rgb_array = np.ascontiguousarray(rgb_array)
        h, w, ch = rgb_array.shape
        bytes_per_line = w * ch
        qimg = QImage(rgb_array.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._image_data = image_data
        if self._fit_to_window:
            self._fit_zoom()
        self.update()

    def set_after_image(self, rgb_array: np.ndarray):
        rgb_array = np.ascontiguousarray(rgb_array)
        h, w, ch = rgb_array.shape
        bytes_per_line = w * ch
        qimg = QImage(rgb_array.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._pixmap_after = QPixmap.fromImage(qimg)
        self.update()

    def clear_after_image(self):
        self._pixmap_after = None
        self._split_mode = False
        self.update()

    def set_split_mode(self, enabled: bool):
        self._split_mode = enabled
        self.update()

    def set_split_position(self, pos: float):
        self._split_position = max(0.0, min(1.0, pos))
        self.update()

    def fit_to_window(self):
        self._fit_to_window = True
        self._fit_zoom()
        self.update()

    def zoom_to(self, level: float):
        self._fit_to_window = False
        self._zoom = max(0.01, min(50.0, level))
        self.zoom_changed.emit(self._zoom)
        self.update()

    # ── Sample mode ──────────────────────────────────────────────────────────

    def set_sample_mode(self, enabled: bool):
        self._sample_mode = enabled
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def set_sample_points(self, points: list[tuple[float, float]]):
        self._sample_points = list(points)
        self.update()

    def clear_sample_points(self):
        self._sample_points = []
        self.update()

    # ── WCS overlay ──────────────────────────────────────────────────────────

    def set_overlay_stars(self, stars: list[tuple[float, float, float]]):
        """Set catalog stars for WCS overlay. Each entry: (x_img, y_img, magnitude)."""
        self._overlay_stars = stars
        self.update()

    def set_wcs_overlay_visible(self, visible: bool):
        self._show_wcs_overlay = visible
        self.update()

    # ── Paint ────────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        if self._pixmap is None:
            painter = QPainter(self)
            painter.fillRect(self.rect(), Qt.GlobalColor.black)
            painter.setPen(QPen(Qt.GlobalColor.gray))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No image loaded\nDrag & drop FITS/XISF files here")
            painter.end()
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        pw, ph = self._pixmap.width(), self._pixmap.height()
        dst = QRectF(
            self._pan_offset.x(), self._pan_offset.y(),
            pw * self._zoom, ph * self._zoom,
        )
        src = QRectF(0, 0, pw, ph)

        if self._split_mode and self._pixmap_after is not None:
            split_x = dst.left() + dst.width() * self._split_position
            painter.setClipRect(QRectF(dst.left(), dst.top(), split_x - dst.left(), dst.height()))
            painter.drawPixmap(dst, self._pixmap, src)
            aw, ah = self._pixmap_after.width(), self._pixmap_after.height()
            src_after = QRectF(0, 0, aw, ah)
            painter.setClipRect(QRectF(split_x, dst.top(), dst.right() - split_x, dst.height()))
            painter.drawPixmap(dst, self._pixmap_after, src_after)
            painter.setClipping(False)
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawLine(QPointF(split_x, dst.top()), QPointF(split_x, dst.bottom()))
        else:
            painter.drawPixmap(dst, self._pixmap, src)

        # Draw WCS overlay
        if self._show_wcs_overlay and self._overlay_stars:
            self._draw_wcs_overlay(painter, dst, pw, ph)

        # Draw background sample points
        if self._sample_points or self._sample_mode:
            self._draw_sample_points(painter, dst, pw, ph)

        painter.end()

    def _draw_wcs_overlay(self, painter: QPainter, dst: QRectF, pw: int, ph: int):
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)

        for x_img, y_img, mag in self._overlay_stars:
            wx = dst.left() + (x_img / pw) * dst.width()
            wy = dst.top() + (y_img / ph) * dst.height()

            # Circle radius scaled by brightness (brighter = larger)
            r = max(4.0, 12.0 - mag)

            painter.setPen(QPen(QColor(80, 200, 255, 200), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(wx, wy), r, r)

            if mag < 10.0:
                painter.setPen(QPen(QColor(150, 220, 255, 180)))
                painter.drawText(QPointF(wx + r + 2, wy + 4), f"{mag:.1f}")

    def _draw_sample_points(self, painter: QPainter, dst: QRectF, pw: int, ph: int):
        for idx, (x_img, y_img) in enumerate(self._sample_points):
            wx = dst.left() + (x_img / pw) * dst.width()
            wy = dst.top() + (y_img / ph) * dst.height()
            r = 8.0
            painter.setPen(QPen(QColor(255, 200, 0, 230), 2))
            painter.setBrush(QColor(255, 200, 0, 40))
            painter.drawEllipse(QPointF(wx, wy), r, r)
            painter.setPen(QPen(QColor(255, 200, 0, 200)))
            font = QFont()
            font.setPointSize(7)
            painter.setFont(font)
            painter.drawText(QPointF(wx + r + 2, wy + 4), str(idx + 1))

        if self._sample_mode:
            # Draw mode indicator
            painter.setPen(QPen(QColor(255, 200, 0)))
            font = QFont()
            font.setPointSize(9)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(QPointF(8, 20), f"Sample mode — {len(self._sample_points)} pts  [RClick=remove]")

    # ── Mouse events ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent):
        img_pos = self._widget_to_image(event.position())

        if self._sample_mode and img_pos is not None:
            if event.button() == Qt.MouseButton.LeftButton:
                self.sample_placed.emit(img_pos.x(), img_pos.y())
                return
            if event.button() == Qt.MouseButton.RightButton:
                self.sample_removed.emit(img_pos.x(), img_pos.y())
                return

        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._dragging = True
            self._drag_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            delta = event.pos() - self._drag_start
            self._pan_offset += QPointF(delta.x(), delta.y())
            self._drag_start = event.pos()
            self._fit_to_window = False
            self.update()

        img_pos = self._widget_to_image(event.position())
        if img_pos is not None and self._image_data is not None:
            ix, iy = int(img_pos.x()), int(img_pos.y())
            data = self._image_data
            if data.ndim == 2:
                if 0 <= iy < data.shape[0] and 0 <= ix < data.shape[1]:
                    self.cursor_position.emit(ix, iy, [float(data[iy, ix])])
            elif data.ndim == 3:
                if 0 <= iy < data.shape[1] and 0 <= ix < data.shape[2]:
                    vals = [float(data[ch, iy, ix]) for ch in range(data.shape[0])]
                    self.cursor_position.emit(ix, iy, vals)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._dragging:
            self._dragging = False
            cursor = Qt.CursorShape.CrossCursor if self._sample_mode else Qt.CursorShape.ArrowCursor
            self.setCursor(cursor)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15
        mouse_pos = event.position()
        old_scene = self._widget_to_image(mouse_pos)
        self._fit_to_window = False
        self._zoom = max(0.01, min(50.0, self._zoom * factor))
        new_scene = self._widget_to_image(mouse_pos)
        if old_scene is not None and new_scene is not None:
            dx = (new_scene.x() - old_scene.x()) * self._zoom
            dy = (new_scene.y() - old_scene.y()) * self._zoom
            self._pan_offset += QPointF(dx, dy)
        self.zoom_changed.emit(self._zoom)
        self.update()

    def resizeEvent(self, event):
        if self._fit_to_window:
            self._fit_zoom()
        super().resizeEvent(event)

    def _fit_zoom(self):
        if self._pixmap is None:
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        if pw == 0 or ph == 0:
            return
        self._zoom = min(ww / pw, wh / ph)
        self._pan_offset = QPointF(
            (ww - pw * self._zoom) / 2,
            (wh - ph * self._zoom) / 2,
        )
        self.zoom_changed.emit(self._zoom)

    def _widget_to_image(self, pos: QPointF) -> QPointF | None:
        if self._pixmap is None or self._zoom == 0:
            return None
        x = (pos.x() - self._pan_offset.x()) / self._zoom
        y = (pos.y() - self._pan_offset.y()) / self._zoom
        return QPointF(x, y)
