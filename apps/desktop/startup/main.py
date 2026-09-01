"""Application entry point.

Boot order: load config -> configure logging -> build Qt app -> show window.
Model loading / video start happen after the window is shown so the UI never
blocks on startup (architecture.md section 38).
"""

from __future__ import annotations

import sys
import uuid

from PySide6.QtWidgets import QApplication

from apps.desktop.ui.main_window import MainWindow
from apps.desktop.ui.theme.stylesheet import build_stylesheet
from core.config.loader import load_settings
from observability.logging.setup import configure_logging, get_logger


def main() -> int:
    settings = load_settings()

    session_id = uuid.uuid4().hex[:12]
    configure_logging(settings.logging, session_id=session_id)
    logger = get_logger(__name__)
    logger.info("application_started", session_id=session_id, app_name=settings.application.name)

    app = QApplication(sys.argv)
    app.setStyleSheet(build_stylesheet())

    window = MainWindow()
    window.show()

    exit_code = app.exec()
    logger.info("application_stopped", exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
