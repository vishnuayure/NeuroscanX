"""One orthogonal slice view.

Mouse:  left click/drag = move crosshair · wheel = scroll slices · Ctrl+wheel = zoom (about the cursor)
        middle-drag = pan · right-drag = window/level (↔ level, ↕ width) · double-click = reset zoom/pan
Keys:   ↑/↓ or PageUp/PageDown = scroll slices
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from src.visualization.viewer_2d import ORIENT_LABELS
from . import theme


class SliceView(QWidget):
    clicked = Signal(int, int)           # image row, col
    scrolled = Signal(int)
    wlDragged = Signal(float, float)     # Δx, Δy in pixels
    zoomChanged = Signal(float)

    def __init__(self, plane: str):
        super().__init__()
        self.plane = plane
        self.rgba: Optional[np.ndarray] = None
        self._qimg: Optional[QImage] = None
        self.aspect = 1.0
        self.cross: Optional[tuple] = None
        self.show_cross = True
        self.title, self.info = plane.upper(), ""
        self.zoom, self.pan = 1.0, QPointF(0, 0)
        self._drag = None
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(False)

    # ---------------------------------------------------------------- data
    def set_image(self, rgba: np.ndarray, aspect: float, cross, info: str):
        self.rgba = np.ascontiguousarray(rgba)
        h, w = self.rgba.shape[:2]
        self._qimg = QImage(self.rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
        self.aspect, self.cross, self.info = aspect, cross, info
        self.update()

    def reset_view(self):
        self.zoom, self.pan = 1.0, QPointF(0, 0)
        self.update()

    # ---------------------------------------------------------------- geometry
    def _geom(self):
        h, w = self.rgba.shape[:2]
        s = min((self.width() - 10) / w, (self.height() - 10) / (h * self.aspect)) * self.zoom
        return s, w, h, self.width() / 2 + self.pan.x(), self.height() / 2 + self.pan.y()

    def _to_widget(self, col, row):
        s, w, h, cx, cy = self._geom()
        return cx + (col - w / 2) * s, cy + (row - h / 2) * s * self.aspect

    def _to_image(self, x, y):
        s, w, h, cx, cy = self._geom()
        return (x - cx) / s + w / 2, (y - cy) / (s * self.aspect) + h / 2

    # ---------------------------------------------------------------- painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#000000"))
        if self._qimg is None:
            p.setPen(QColor(theme.DIM)); p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "NO DATA"); return
        s, w, h, cx, cy = self._geom()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, s < 4)
        p.save()
        p.translate(cx, cy); p.scale(s, s * self.aspect); p.translate(-w / 2, -h / 2)
        p.drawImage(0, 0, self._qimg)
        p.restore()
        if self.show_cross and self.cross is not None:
            x, y = self._to_widget(self.cross[1] + 0.5, self.cross[0] + 0.5)
            pen = QPen(QColor(41, 211, 255, 170), 1); pen.setCosmetic(True)
            p.setPen(pen)
            gap = 7
            p.drawLine(0, int(y), int(x - gap), int(y)); p.drawLine(int(x + gap), int(y), self.width(), int(y))
            p.drawLine(int(x), 0, int(x), int(y - gap)); p.drawLine(int(x), int(y + gap), int(x), self.height())
            p.setPen(QPen(QColor(255, 255, 255, 220), 1)); p.drawEllipse(QPointF(x, y), 3, 3)
        # HUD annotations
        f = QFont("Consolas", 9); p.setFont(f)
        p.setPen(QColor(theme.CYAN)); p.drawText(10, 18, self.title)
        p.setPen(QColor(theme.DIM)); p.drawText(10, 34, self.info)
        left, right, top, bottom = ORIENT_LABELS[self.plane]
        p.setPen(QColor(255, 200, 87))
        p.drawText(6, self.height() // 2 + 4, left); p.drawText(self.width() - 14, self.height() // 2 + 4, right)
        p.drawText(self.width() // 2 - 4, 14, top); p.drawText(self.width() // 2 - 4, self.height() - 4, bottom)
        if self.zoom != 1.0:
            p.setPen(QColor(theme.DIM)); p.drawText(self.width() - 60, self.height() - 10, f"{self.zoom:.1f}×")
        pen = QPen(QColor(theme.BORDER), 1); p.setPen(pen); p.drawRect(0, 0, self.width() - 1, self.height() - 1)
        p.end()

    # ---------------------------------------------------------------- interaction
    def _emit_click(self, pos):
        if self.rgba is None:
            return
        col, row = self._to_image(pos.x(), pos.y())
        h, w = self.rgba.shape[:2]
        self.clicked.emit(int(np.clip(np.floor(row), 0, h - 1)), int(np.clip(np.floor(col), 0, w - 1)))

    def mousePressEvent(self, e):
        self.setFocus()
        b = e.button()
        self._drag = (b, e.position())
        if b == Qt.MouseButton.LeftButton:
            self._emit_click(e.position())

    def mouseMoveEvent(self, e):
        if not self._drag:
            return
        b, last = self._drag
        d = e.position() - last
        if b == Qt.MouseButton.LeftButton:
            self._emit_click(e.position())
        elif b == Qt.MouseButton.MiddleButton:
            self.pan += d; self.update()
        elif b == Qt.MouseButton.RightButton:
            self.wlDragged.emit(d.x(), d.y())
        self._drag = (b, e.position())

    def mouseReleaseEvent(self, e):
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        self.reset_view()

    def wheelEvent(self, e):
        dy = e.angleDelta().y()
        if dy == 0:
            return
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier and self.rgba is not None:
            pos = e.position()
            col, row = self._to_image(pos.x(), pos.y())
            self.zoom = float(np.clip(self.zoom * (1.15 if dy > 0 else 1 / 1.15), 0.5, 40))
            s, w, h, _, _ = self._geom()
            self.pan = QPointF(pos.x() - (col - w / 2) * s - self.width() / 2,
                               pos.y() - (row - h / 2) * s * self.aspect - self.height() / 2)
            self.zoomChanged.emit(self.zoom); self.update()
        else:
            self.scrolled.emit(1 if dy > 0 else -1)

    def keyPressEvent(self, e):
        k = e.key()
        if k in (Qt.Key.Key_Up, Qt.Key.Key_PageUp):
            self.scrolled.emit(1)
        elif k in (Qt.Key.Key_Down, Qt.Key.Key_PageDown):
            self.scrolled.emit(-1)
        else:
            super().keyPressEvent(e)
