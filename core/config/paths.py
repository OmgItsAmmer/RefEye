"""Filesystem layout, in development and inside a PyInstaller bundle.

Running from source, everything is relative to the repository root. Running
from a packaged build the layout differs in a way that matters:

    RefEye/
    ├── RefEye.exe
    ├── _internal/          <- PyInstaller puts bundled data here
    │   └── config/         <-   ...including the shipped default config
    ├── config/             <- the operator-editable copy lives HERE
    ├── models/
    └── logs/

Config must sit beside the executable, not inside `_internal/`: the operator
guide tells people to open it in Notepad, and `_internal` is an implementation
detail they should never have to know about. On first run the bundled default
is copied out to that editable location.

Relative paths in the config file (`./models/...`, `./logs`) are resolved
against the application root, not the process working directory — otherwise
launching from a shortcut or another folder silently breaks them.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def app_root() -> Path:
    """Directory the application treats as its base for relative paths."""
    if is_frozen():
        return Path(sys.executable).parent
    # core/config/paths.py -> core/config -> core -> repo root
    return Path(__file__).resolve().parents[2]


def bundled_root() -> Path:
    """Where PyInstaller unpacked read-only bundled data."""
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass) if meipass else app_root()


def resolve(path: str | Path) -> Path:
    """Resolve a config-supplied path against the application root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (app_root() / candidate).resolve()


def ensure_editable_config(filename: str = "default.yaml") -> Path:
    """Return the operator-editable config path, seeding it on first run.

    In a packaged build the shipped config lives inside `_internal/`, which is
    replaced wholesale on every upgrade. Copying it out once gives the operator
    a stable file they can edit and that survives being looked at.
    """
    target = app_root() / "config" / filename
    if target.exists():
        return target

    source = bundled_root() / "config" / filename
    if source.exists() and source != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    return target
