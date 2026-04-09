"""Dark theme stylesheet for Cosmica — professional-grade dark theme."""

from pathlib import Path

_ICONS_DIR = Path(__file__).resolve().parent.parent / "resources" / "icons"

# Accent palette — GitHub-inspired green with refined tones
ACCENT = "#2ea043"
ACCENT_HOVER = "#3fb950"
ACCENT_DARK = "#1a4d2e"
BG_PRIMARY = "#0d1117"    # Deepest background
BG_SECONDARY = "#161b22"   # Panels, sidebars
BG_TERTIARY = "#21262d"    # Inputs, buttons
BG_HOVER = "#30363d"       # Hover state
BORDER = "#30363d"         # Borders
TEXT_PRIMARY = "#e6edf3"   # Primary text
TEXT_SECONDARY = "#8b949e" # Secondary text, hints


def _icon(name: str) -> str:
    """Resolve icon path."""
    return (_ICONS_DIR / f"{name}.svg").as_posix()


def get_dark_theme() -> str:
    """Return the complete dark theme stylesheet."""
    return f"""
/* ═══════════════════════════════════════════════════════
   Cosmica Dark Theme — Professional Astrophotography UI
   ═══════════════════════════════════════════════════════ */

/* ── Global ─────────────────────────────────────────── */
QMainWindow, QDialog {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
}}

QWidget {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
    font-family: "Inter", "Segoe UI", "Roboto", "Ubuntu", sans-serif;
    font-size: 13px;
}}

/* ── Menu Bar ───────────────────────────────────────── */
QMenuBar {{
    background-color: {BG_SECONDARY};
    color: {TEXT_PRIMARY};
    border-bottom: 1px solid {BORDER};
    padding: 2px 8px;
    spacing: 4px;
}}

QMenuBar::item {{
    padding: 4px 8px;
    border-radius: 4px;
}}

QMenuBar::item:selected {{
    background-color: {BG_HOVER};
}}

QMenuBar::item:pressed {{
    background-color: {ACCENT_DARK};
}}

/* ── Dropdown Menus ─────────────────────────────────── */
QMenu {{
    background-color: {BG_SECONDARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px;
}}

QMenu::item {{
    padding: 6px 24px 6px 12px;
    border-radius: 4px;
}}

QMenu::item:selected {{
    background-color: {BG_HOVER};
}}

QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 8px;
}}

QMenu::icon {{
    margin-right: 6px;
}}

QMenu::shortcut {{
    color: {TEXT_SECONDARY};
    margin-left: 24px;
}}

/* ── Toolbars ───────────────────────────────────────── */
QToolBar {{
    background-color: {BG_SECONDARY};
    border: none;
    border-bottom: 1px solid {BORDER};
    spacing: 2px;
    padding: 4px;
}}

QToolButton {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 6px 10px;
    color: {TEXT_PRIMARY};
}}

QToolButton:hover {{
    background-color: {BG_HOVER};
    border-color: {BORDER};
}}

QToolButton:pressed {{
    background-color: {ACCENT_DARK};
}}

QToolButton:checked {{
    background-color: {ACCENT_DARK};
    border-color: {ACCENT};
}}

/* ── Status Bar ─────────────────────────────────────── */
QStatusBar {{
    background-color: {BG_SECONDARY};
    color: {TEXT_SECONDARY};
    font-size: 12px;
    border-top: 1px solid {BORDER};
    padding: 2px 8px;
}}

/* ── Dock Widgets ───────────────────────────────────── */
QDockWidget {{
    color: {TEXT_PRIMARY};
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}

QDockWidget::title {{
    background-color: {BG_SECONDARY};
    padding: 6px 10px;
    border-bottom: 1px solid {BORDER};
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* ── Tabs ───────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    background-color: {BG_PRIMARY};
    top: -1px;
}}

QTabBar::tab {{
    background-color: {BG_SECONDARY};
    color: {TEXT_SECONDARY};
    padding: 8px 16px;
    border: 1px solid {BORDER};
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    margin-right: 2px;
    font-weight: 500;
}}

QTabBar::tab:selected {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
    border-bottom: 2px solid {ACCENT};
}}

QTabBar::tab:hover:!selected {{
    background-color: {BG_HOVER};
    color: {TEXT_PRIMARY};
}}

QTabBar::scroller {{
    width: 24px;
}}

QTabBar QToolButton {{
    background-color: {BG_SECONDARY};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 2px;
}}

QTabBar QToolButton:hover {{
    background-color: {BG_HOVER};
}}

/* ── Trees, Lists, Tables ───────────────────────────── */
QTreeWidget, QListWidget, QTableWidget {{
    background-color: {BG_SECONDARY};
    border: 1px solid {BORDER};
    border-radius: 4px;
    alternate-background-color: {BG_PRIMARY};
    selection-background-color: {ACCENT_DARK};
    selection-color: {TEXT_PRIMARY};
    outline: none;
    padding: 2px;
}}

QTreeWidget::item, QListWidget::item {{
    padding: 4px 6px;
    border-radius: 3px;
}}

QTreeWidget::item:hover, QListWidget::item:hover {{
    background-color: {BG_HOVER};
}}

QHeaderView::section {{
    background-color: {BG_SECONDARY};
    color: {TEXT_SECONDARY};
    padding: 6px 8px;
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
}}

/* ── Buttons ────────────────────────────────────────── */
QPushButton {{
    background-color: {ACCENT};
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 13px;
}}

QPushButton:hover {{
    background-color: {ACCENT_HOVER};
}}

QPushButton:pressed {{
    background-color: {ACCENT_DARK};
}}

QPushButton:disabled {{
    background-color: {BG_TERTIARY};
    color: {TEXT_SECONDARY};
}}

QPushButton[flat="true"], QPushButton#flatButton {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 4px;
}}

QPushButton[flat="true"]:hover {{
    background-color: {BG_HOVER};
}}

/* ── Inputs ─────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {BG_TERTIARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    selection-background-color: {ACCENT_DARK};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
    outline: none;
}}

QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}

QComboBox QAbstractItemView {{
    background-color: {BG_SECONDARY};
    color: {TEXT_PRIMARY};
    selection-background-color: {ACCENT_DARK};
    border: 1px solid {BORDER};
    border-radius: 4px;
}}

/* ── Sliders ────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 4px;
    background: {BG_TERTIARY};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
    border: 2px solid {BG_PRIMARY};
}}

QSlider::handle:horizontal:hover {{
    background: {ACCENT_HOVER};
}}

QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}

/* ── Progress Bar ───────────────────────────────────── */
QProgressBar {{
    background-color: {BG_TERTIARY};
    border: none;
    border-radius: 4px;
    text-align: center;
    color: {TEXT_PRIMARY};
    height: 20px;
    font-weight: 500;
}}

QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 4px;
}}

/* ── Scrollbars ─────────────────────────────────────── */
QScrollBar:vertical {{
    background-color: {BG_PRIMARY};
    width: 12px;
    margin: 0;
    border-radius: 6px;
}}

QScrollBar::handle:vertical {{
    background-color: {BG_HOVER};
    min-height: 20px;
    border-radius: 6px;
    margin: 2px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {TEXT_SECONDARY};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: {BG_PRIMARY};
    height: 12px;
    margin: 0;
    border-radius: 6px;
}}

QScrollBar::handle:horizontal {{
    background-color: {BG_HOVER};
    min-width: 20px;
    border-radius: 6px;
    margin: 2px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {TEXT_SECONDARY};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── Tooltips ───────────────────────────────────────── */
QToolTip {{
    background-color: {BG_SECONDARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
}}

/* ── Splitters ──────────────────────────────────────── */
QSplitter::handle {{
    background-color: {BORDER};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

QSplitter::handle:hover {{
    background-color: {ACCENT};
}}

/* ── Group Boxes ────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 14px;
    font-weight: 600;
    font-size: 12px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {TEXT_SECONDARY};
}}

/* ── Checkboxes ─────────────────────────────────────── */
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 1.5px solid {BORDER};
    border-radius: 4px;
    background-color: {BG_TERTIARY};
}}

QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    image: url({_icon('check')});
}}

QCheckBox::indicator:hover {{
    border-color: {ACCENT};
}}

/* ── Text Edits ─────────────────────────────────────── */
QTextEdit, QPlainTextEdit {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 4px;
    selection-background-color: {ACCENT_DARK};
    font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
}}

/* ── Pro Feature Accent ─────────────────────────────── */
QPushButton#proButton {{
    background-color: #8957e5;
    color: #ffffff;
}}

QPushButton#proButton:hover {{
    background-color: #a371f7;
}}

/* ── Canvas area (image display background) ─────────── */
#ImageCanvas {{
    background-color: #000000;
    border: 1px solid {BORDER};
}}
"""


# Backward compatibility
DARK_THEME = get_dark_theme()
