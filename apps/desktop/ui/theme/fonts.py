"""Registers the app's vendored typefaces with Qt.

Outfit (variable, UI face) and Space Mono (static, data face) are shipped as
TTF files in this package's `fonts/` directory rather than relied on as
system fonts — the app must render identically with no internet access and
no fonts pre-installed on the machine (theme.md section 3).

Must run once, after a QApplication exists and before the stylesheet is
applied, so `tokens.FONT_FAMILY` / `FONT_FAMILY_DATA` resolve to these files
instead of silently falling back to whatever sans-serif Windows has.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFontDatabase

from observability.logging.setup import get_logger

logger = get_logger(__name__)

_FONTS_DIR = Path(__file__).parent / "fonts"
_FONT_FILES = [
    "Outfit[wght].ttf",
    "SpaceMono-Regular.ttf",
    "SpaceMono-Bold.ttf",
    "SpaceMono-Italic.ttf",
]


def load_fonts() -> None:
    for filename in _FONT_FILES:
        path = _FONTS_DIR / filename
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id == -1:
            logger.warning("font_load_failed", file=filename)
