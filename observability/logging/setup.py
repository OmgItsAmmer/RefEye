"""Structured logging setup. See architecture.md sections 45-46.

Two sinks:
  * console  — human-readable, for development
  * file     — JSON lines, size-rotated, for diagnostics and handover

Every log line carries `session_id`. Analysis-related calls additionally bind
`request_id` (see `bind_request`), and model results must include model name
and version.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog

from core.config.schema import LoggingConfig


def configure_logging(config: LoggingConfig, session_id: str) -> None:
    log_dir = Path(config.directory)
    log_dir.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, config.level, logging.INFO)

    # structlog renders the event dict; stdlib handlers only transport the
    # already-formatted string, so each sink gets its own renderer via
    # ProcessorFormatter.
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    root = logging.getLogger()
    root.setLevel(level)
    for existing in list(root.handlers):
        root.removeHandler(existing)

    if config.console:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.dev.ConsoleRenderer(colors=False),
                foreign_pre_chain=shared_processors,
            )
        )
        root.addHandler(console)

    if config.json_file:
        file_handler = RotatingFileHandler(
            log_dir / f"session_{session_id}.jsonl",
            maxBytes=config.rotation_max_bytes,
            backupCount=config.rotation_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=shared_processors,
            )
        )
        root.addHandler(file_handler)

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(session_id=session_id)


def get_logger(module_name: str):
    return structlog.get_logger(module_name)


@contextmanager
def bind_request(request_id: str):
    """Bind `request_id` to every log line emitted inside this block.

    Uses contextvars, so it survives across await points but stays scoped to
    the calling thread/task — never leaks into unrelated requests.
    """
    tokens = structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
