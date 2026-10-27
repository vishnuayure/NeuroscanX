"""NeuroScanX – launch the desktop application:   python main.py"""
import logging
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:               # makes `src`, `app`, `agents` importable from any working directory
    sys.path.insert(0, ROOT)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    try:
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("PySide6 is required:  pip install -r requirements.txt")
        return 1
    from app.ui import theme
    from app.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("NeuroScanX")
    app.setStyleSheet(theme.QSS)
    f = QFont(); f.setPointSize(10); app.setFont(f)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
