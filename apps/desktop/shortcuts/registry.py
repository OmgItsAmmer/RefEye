"""Configurable shortcut registration.

Every binding comes from config (`shortcuts:` in YAML) — nothing is
hard-coded (architecture.md section 58, rule 11). An unparseable or duplicate
binding is reported rather than silently ignored, because a shortcut that
quietly does nothing is worse than one that fails loudly at startup.

Scope note: these are application-scoped (Qt) shortcuts, active while the
window has focus. A true system-wide global hotkey (working while the
operator is in another application) needs an OS-level hook; that is a
deliberate M1.5 item, not silently assumed to work here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QWidget

from core.config.schema import ShortcutsConfig
from observability.logging.setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ShortcutBinding:
    action: str
    key: str
    description: str


class ShortcutRegistry:
    """Binds configured key sequences to callables on a parent widget."""

    def __init__(self, parent: QWidget, config: ShortcutsConfig):
        self._parent = parent
        self._config = config
        self._shortcuts: dict[str, QShortcut] = {}
        self._bindings: list[ShortcutBinding] = []

    def bind(self, action: str, handler: Callable[[], None], description: str = "") -> bool:
        """Bind the configured key for `action`. Returns False if unusable."""
        key = getattr(self._config, action, None)
        if key is None:
            logger.warning("shortcut_not_configured", action=action)
            return False

        sequence = QKeySequence.fromString(key, QKeySequence.SequenceFormat.PortableText)
        if sequence.isEmpty():
            logger.warning("shortcut_unparseable", action=action, key=key)
            return False

        if action in self._shortcuts:
            logger.warning("shortcut_already_bound", action=action)
            return False

        shortcut = QShortcut(sequence, self._parent)
        shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        shortcut.activated.connect(handler)

        self._shortcuts[action] = shortcut
        self._bindings.append(ShortcutBinding(action=action, key=key, description=description))

        logger.debug("shortcut_bound", action=action, key=key)
        return True

    def bindings(self) -> list[ShortcutBinding]:
        return list(self._bindings)

    def key_for(self, action: str) -> str:
        return getattr(self._config, action, "")
