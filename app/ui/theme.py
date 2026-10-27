"""Dark medical-imaging workstation theme (black, restrained cyan accents)."""
BG = "#04070a"
PANEL = "#0a1016"
PANEL_2 = "#0e1720"
BORDER = "#16303d"
CYAN = "#29d3ff"
CYAN_DIM = "#177a96"
TEXT = "#cfe8f1"
DIM = "#6f8a97"
OK = "#3ddc97"
WARN = "#ffc857"
DANGER = "#ff5d7a"
MONO = "Consolas, 'JetBrains Mono', 'DejaVu Sans Mono', monospace"
SANS = "'Segoe UI', 'Inter', 'Helvetica Neue', Arial, sans-serif"

LOW_COLOR = "#5f8aa3"   # LOW evidence is informational, not an error → neutral steel-blue
LEVEL_COLORS = {"HIGH": OK, "MODERATE": WARN, "LOW": LOW_COLOR}

QSS = f"""
* {{ font-family: {SANS}; color: {TEXT}; font-size: 12px; }}
QMainWindow, QWidget#root, QStackedWidget {{ background: {BG}; }}
QWidget {{ background: transparent; }}
QFrame#card, QFrame#panel {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; }}
QFrame#card:hover {{ border: 1px solid {CYAN_DIM}; }}
QFrame#card[state="ok"] {{ border: 1px solid #1d6b52; }}
QFrame#card[state="warning"] {{ border: 1px solid #8a6b1d; }}
QFrame#card[state="error"] {{ border: 1px solid #8a2b3d; }}
QFrame#card[dragover="true"] {{ border: 1px dashed {CYAN}; background: {PANEL_2}; }}
QLabel#title {{ font-size: 40px; font-weight: 300; letter-spacing: 14px; color: #ffffff; }}
QLabel#subtitle {{ font-size: 13px; letter-spacing: 3px; color: {CYAN}; }}
QLabel#h {{ font-size: 10px; letter-spacing: 2px; color: {DIM}; font-weight: 600; }}
QLabel#cardTitle {{ font-size: 15px; font-weight: 600; letter-spacing: 3px; color: #ffffff; }}
QLabel#mono, QLabel#info {{ font-family: {MONO}; font-size: 11px; color: {TEXT}; }}
QLabel#dim {{ color: {DIM}; }}
QLabel#disclaimer {{ color: {WARN}; font-size: 10px; padding: 4px 8px; background: #120f05; border-top: 1px solid #3a2f0c; }}
QLabel#big {{ font-size: 26px; font-weight: 300; color: #ffffff; }}
QPushButton {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 4px; padding: 7px 16px;
  letter-spacing: 1px; font-weight: 600; }}
QPushButton:hover {{ border-color: {CYAN}; color: #ffffff; }}
QPushButton:pressed {{ background: #0b2230; }}
QPushButton:disabled {{ color: #3d525c; border-color: #10222b; }}
QPushButton#primary {{ background: #0a3446; border: 1px solid {CYAN}; color: #ffffff; padding: 10px 28px; font-size: 13px; }}
QPushButton#primary:hover {{ background: #0e4a63; }}
QPushButton#primary:disabled {{ background: #0a161d; border-color: #16303d; color: #3d525c; }}
QComboBox {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 4px; padding: 5px 10px; min-width: 120px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {CYAN_DIM}; selection-background-color: #0e4a63; }}
QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {BORDER}; border-radius: 3px; background: {PANEL_2}; }}
QCheckBox::indicator:checked {{ background: {CYAN}; border-color: {CYAN}; }}
QRadioButton::indicator {{ width: 12px; height: 12px; border: 1px solid {BORDER}; border-radius: 7px; background: {PANEL_2}; }}
QRadioButton::indicator:checked {{ background: {CYAN}; border-color: {CYAN}; }}
QSlider::groove:horizontal {{ height: 3px; background: {BORDER}; border-radius: 1px; }}
QSlider::sub-page:horizontal {{ background: {CYAN_DIM}; border-radius: 1px; }}
QSlider::handle:horizontal {{ background: {CYAN}; width: 12px; margin: -5px 0; border-radius: 6px; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; background: {BG}; top: -1px; }}
QTabBar::tab {{ background: {PANEL}; border: 1px solid {BORDER}; padding: 8px 22px; letter-spacing: 2px; font-size: 11px; color: {DIM}; }}
QTabBar::tab:selected {{ color: #ffffff; border-bottom: 2px solid {CYAN}; background: {PANEL_2}; }}
QProgressBar {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 3px; height: 8px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {CYAN_DIM}, stop:1 {CYAN}); border-radius: 2px; }}
QPlainTextEdit, QTextEdit {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 4px; font-family: {MONO}; font-size: 11px; }}
QTableWidget {{ background: {PANEL}; gridline-color: {BORDER}; border: 1px solid {BORDER}; font-family: {MONO}; }}
QHeaderView::section {{ background: {PANEL_2}; color: {DIM}; border: 0; border-bottom: 1px solid {BORDER}; padding: 6px; letter-spacing: 1px; }}
QScrollBar:vertical {{ background: {BG}; width: 8px; }} QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QToolTip {{ background: {PANEL}; color: {TEXT}; border: 1px solid {CYAN_DIM}; }}
"""
