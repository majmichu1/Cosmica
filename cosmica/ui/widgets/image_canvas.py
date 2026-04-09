"""Image Canvas Widget — zoomable, pannable image display with before/after split."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QWidget


class ImageCanvas(QWidget):
    """High-performance image display with zoom, pan, and split preview."""

    zoom_changed = pyqtSignal(float)
    cursor_position = pyqtSignal(int, int, list)  # x, y, pixel values

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._pixmap: QPixmap | None = None
        self._pixmap_after: QPixmap | None = None  # for split preview
        self._image_data: np.ndarray | None = None  # original float data for readout

        self._zoom = 1.0
        self._pan_offset = QPointF(0, 0)
        self._dragging = False
        self._drag_start = QPoint()

        # Split preview
        self._split_mode = False
        self._split_position = 0.5  # 0-1, fraction from left

        self._fit_to_window = True

    def set_image(self, rgb_array: np.ndarray, image_data: np.ndarray | None = None):
        """Set the display image from a uint8 RGB array (H, W, 3)."""
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
        """Set the 'after' image for split preview."""
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

        # Compute destination rectangle
        pw, ph = self._pixmap.width(), self._pixmap.height()
        dst = QRectF(
            self._pan_offset.x(),
            self._pan_offset.y(),
            pw * self._zoom,
            ph * self._zoom,
        )
        src = QRectF(0, 0, pw, ph)

        if self._split_mode and self._pixmap_after is not None:
            # Draw before on left, after on right
            split_x = dst.left() + dst.width() * self._split_position

            # Left: before — use before pixmap's own source rect
            painter.setClipRect(QRectF(dst.left(), dst.top(), split_x - dst.left(), dst.height()))
            painter.drawPixmap(dst, self._pixmap, src)

            # Right: after — use after pixmap's own source rect (may differ in size)
            aw, ah = self._pixmap_after.width(), self._pixmap_after.height()
            src_after = QRectF(0, 0, aw, ah)
            painter.setClipRect(QRectF(split_x, dst.top(), dst.right() - split_x, dst.height()))
            painter.drawPixmap(dst, self._pixmap_after, src_after)

            painter.setClipping(False)

            # Draw split line
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawLine(QPointF(split_x, dst.top()), QPointF(split_x, dst.bottom()))
        else:
            painter.drawPixmap(dst, self._pixmap, src)

        painter.end()

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15

        # Zoom centered on cursor
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

    def mousePressEvent(self, event: QMouseEvent):
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

        # Emit cursor position for pixel readout
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
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def resizeEvent(self, event):
        if self._fit_to_window:
            self._fit_zoom()
        super().resizeEvent(event)

    def _widget_to_image(self, pos: QPointF) -> QPointF | None:
        if self._pixmap is None or self._zoom == 0:
            return None
        x = (pos.x() - self._pan_offset.x()) / self._zoom
        y = (pos.y() - self._pan_offset.y()) / self._zoom
        return QPointF(x, y)
