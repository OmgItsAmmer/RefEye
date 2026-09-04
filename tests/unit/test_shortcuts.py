"""Shortcut binding: everything comes from config, nothing is hard-coded."""

from __future__ import annotations

import pytest

from core.config.schema import ShortcutsConfig

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QWidget  # noqa: E402

from apps.desktop.shortcuts.registry import ShortcutRegistry  # noqa: E402


def make_config(**overrides) -> ShortcutsConfig:
    base = {
        "analyze": "F8",
        "previous_frame": "Left",
        "next_frame": "Right",
        "previous_candidate": "Up",
        "next_candidate": "Down",
        "jump_to_best": "Home",
        "confirm_frame": "Return",
        "select_camera_1": "1",
        "select_camera_2": "2",
        "select_camera_3": "3",
        "select_camera_4": "4",
        "select_best": "B",
    }
    base.update(overrides)
    return ShortcutsConfig(**base)


def test_binds_configured_key(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    registry = ShortcutRegistry(widget, make_config())

    assert registry.bind("analyze", lambda: None, "Analyse")
    assert registry.key_for("analyze") == "F8"


def test_key_change_in_config_is_honoured(qtbot):
    """Rebinding is a config edit, never a code change."""
    widget = QWidget()
    qtbot.addWidget(widget)
    registry = ShortcutRegistry(widget, make_config(analyze="Ctrl+Shift+A"))

    assert registry.bind("analyze", lambda: None)
    assert registry.bindings()[0].key == "Ctrl+Shift+A"


def test_unparseable_key_is_reported_not_silently_ignored(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    registry = ShortcutRegistry(widget, make_config(analyze=""))

    assert registry.bind("analyze", lambda: None) is False
    assert registry.bindings() == []


def test_duplicate_binding_is_refused(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    registry = ShortcutRegistry(widget, make_config())

    assert registry.bind("analyze", lambda: None)
    assert registry.bind("analyze", lambda: None) is False


def test_unknown_action_is_refused(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    registry = ShortcutRegistry(widget, make_config())

    assert registry.bind("not_an_action", lambda: None) is False


def test_all_configured_actions_can_bind(qtbot):
    """Every key the UI advertises must actually resolve to a shortcut."""
    widget = QWidget()
    qtbot.addWidget(widget)
    registry = ShortcutRegistry(widget, make_config())

    actions = [
        "analyze",
        "previous_frame",
        "next_frame",
        "previous_candidate",
        "next_candidate",
        "jump_to_best",
        "confirm_frame",
    ]
    assert all(registry.bind(action, lambda: None) for action in actions)
    assert len(registry.bindings()) == len(actions)
