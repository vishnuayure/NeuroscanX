"""Small custom widgets: glowing HUD bars and section headers."""
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QLabel, QWidget

from . import theme


class HudBar(QWidget):
    """Thin segmented progress bar with a soft glow; value ∈ [0,1]."""

    def __init__(self, value=0.0, color=theme.CYAN, segments=20, height=10, parent=None):
        super().__init__(parent)
        self._v, self._color, self._seg = float(value), QColor(color), segments
        self.setMinimumHeight(height)
        self.setMaximumHeight(height + 2)

    def set_value(self, v, color=None):
        self._v = max(0.0, min(1.0, float(v)))
        if color:
            self._color = QColor(color)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height() - 2
        gap = 2.0
        sw = (w - gap * (self._seg - 1)) / self._seg
        lit = self._v * self._seg
        for i in range(self._seg):
            x = i * (sw + gap)
            frac = max(0.0, min(1.0, lit - i))
            base = QColor(theme.BORDER)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(base)
            p.drawRoundedRect(QRectF(x, 1, sw, h), 1.5, 1.5)
            if frac > 0:
                c = QColor(self._color); c.setAlphaF(0.35 + 0.65 * frac)
                p.setBrush(c)
                p.drawRoundedRect(QRectF(x, 1, sw * frac, h), 1.5, 1.5)
        p.end()


def header(text: str) -> QLabel:
    l = QLabel(text.upper()); l.setObjectName("h")
    return l


class Corners(QWidget):
    """Transparent overlay painting HUD corner brackets (used on the slice views)."""

    def __init__(self, parent=None, color=theme.CYAN_DIM, size=14):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._c, self._s = QColor(color), size

    def paintEvent(self, _):
        p = QPainter(self)
        p.setPen(QPen(self._c, 1.5))
        w, h, s = self.width() - 1, self.height() - 1, self._s
        for (x, y, dx, dy) in ((0, 0, 1, 1), (w, 0, -1, 1), (0, h, 1, -1), (w, h, -1, -1)):
            p.drawLine(x, y, x + dx * s, y); p.drawLine(x, y, x, y + dy * s)
        p.end()
