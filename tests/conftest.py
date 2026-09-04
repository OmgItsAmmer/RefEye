"""Shared test configuration.

Qt tests run headless. Setting the platform here (before any PySide6 import)
keeps `pytest` working on a build agent with no display.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
