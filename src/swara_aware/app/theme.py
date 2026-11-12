"""Colours and the Qt style sheet: dark theme with one saffron accent.

Status colours are reserved for tuning state and always appear together with a text
label (the status chip), so the meaning never depends on colour alone.
"""

BG = "#111318"
PANEL = "#1a1d24"
BORDER = "#2b2f3a"
TEXT = "#ecebe8"
MUTED = "#9b9a94"
FAINT = "#3a3f4b"
ACCENT = "#ff9933"  # saffron

# Trace styles per ornament state (colour, width, Qt pen style name).
TRACE_STYLES = {
    "stable": (ACCENT, 3.0, "solid"),
    "glide": ("#8fa8d8", 2.0, "dash"),
    "ornament": ("#c49bf2", 2.0, "dot"),
}

# Status chip: (label, background colour). Text on chips is dark for contrast.
STATUS = {
    "in_tune": ("IN TUNE", "#3fb950"),
    "close": ("CLOSE", "#d6a531"),
    "sharp": ("SHARP", "#f26b5b"),
    "flat": ("FLAT", "#f26b5b"),
    "gliding": ("GLIDING", "#8fa8d8"),
    "ornament": ("ORNAMENT", "#c49bf2"),
    "silent": ("SILENT", "#5b606b"),
}

STYLE_SHEET = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-size: 14px; }}
QFrame#panel, QWidget#bar {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; }}
QLabel {{ background: transparent; border: none; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#caption {{ color: {MUTED}; font-size: 12px; letter-spacing: 1px; }}
QComboBox, QDoubleSpinBox {{
    background: {BG}; border: 1px solid {BORDER}; border-radius: 6px; padding: 4px 8px;
    min-height: 24px;
}}
QComboBox QAbstractItemView {{ background: {PANEL}; selection-background-color: {FAINT}; }}
QPushButton {{
    background: {FAINT}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 6px 14px; min-height: 24px;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {MUTED}; }}
QPushButton#primary {{ background: {ACCENT}; color: #1a1205; font-weight: bold; border: none; }}
QPushButton:checked {{ background: {ACCENT}; color: #1a1205; border: none; }}
QCheckBox {{ background: transparent; spacing: 6px; }}
QSlider::groove:horizontal {{ height: 4px; background: {FAINT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; margin: -6px 0; border-radius: 7px; }}
QStatusBar {{ color: {MUTED}; }}
QDialog {{ background: {BG}; }}
"""
