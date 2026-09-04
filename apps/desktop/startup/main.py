"""RefEye entry point.

Boot order: config -> logging -> Qt app -> window -> background services.
Services start only after the window is shown, so the UI never blocks on
startup and a failing video source leaves a usable application behind
(architecture.md sections 38, 49).

A splash screen appears before anything else — see
apps/desktop/ui/theme/splash.py for why: the rest of the app's imports
(the full widget tree, transitively pulling in a fair amount of machinery)
are deliberately deferred until after it's on screen, so a slow first
launch shows visible progress instead of a blank window.
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from apps.desktop.ui.theme.fonts import load_fonts
from apps.desktop.ui.theme.splash import build_splash, set_splash_message


def main() -> int:
    app = QApplication(sys.argv)
    load_fonts()

    splash = build_splash()
    splash.show()
    app.processEvents()

    # Deferred past this point so the splash is already visible before any
    # of this (or what it transitively imports) starts running.
    import uuid

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QMessageBox

    from apps.desktop.ui.main_window import MainWindow
    from apps.desktop.ui.theme.app_icon import build_app_icon
    from apps.desktop.ui.theme.stylesheet import build_stylesheet
    from apps.desktop.viewmodels.main_viewmodel import MainViewModel
    from core.config.loader import load_settings
    from core.errors.exceptions import ConfigurationError
    from observability.diagnostics.system_probe import probe
    from observability.logging.setup import configure_logging, get_logger

    app.setStyleSheet(build_stylesheet())
    app.processEvents()

    try:
        settings = load_settings()
    except ConfigurationError as exc:
        # Config is the one thing we cannot degrade around — without it there
        # is nothing to show. Report it in the UI, not just on stderr.
        splash.close()
        QMessageBox.critical(None, "Configuration error", str(exc))
        return 2

    session_id = uuid.uuid4().hex[:12]
    configure_logging(settings.logging, session_id=session_id)
    logger = get_logger(__name__)

    logger.info(
        "application_started",
        app_name=settings.application.name,
        environment=settings.application.environment,
        device=settings.runtime.device,
        input_type=settings.video.input_type,
    )
    # Recorded once so support questions about a slow or failing machine start
    # from facts (architecture.md section 45).
    logger.info("hardware_detected", **probe().as_dict())

    app.setApplicationName(settings.application.name)
    app.setApplicationDisplayName(settings.application.name)
    app.setWindowIcon(build_app_icon())

    set_splash_message(splash, "Preparing interface…")
    app.processEvents()

    viewmodel = MainViewModel(settings)
    window = MainWindow(settings, viewmodel)
    window.show()
    splash.finish(window)

    # Start ingest on the next event-loop turn so the window paints first.
    QTimer.singleShot(0, viewmodel.start)

    exit_code = app.exec()
    viewmodel.shutdown()
    logger.info("application_stopped", exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
