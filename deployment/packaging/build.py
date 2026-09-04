"""Build the RefEye Windows folder distribution.

    python -m deployment.packaging.build

Produces dist/RefEye/ containing RefEye.exe plus config, models and a logs
directory — the layout in architecture.md section 53. The result runs on a
machine with no Python and no development environment.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGING_DIR = PROJECT_ROOT / "deployment" / "packaging"
SPEC = PACKAGING_DIR / "RefEye.spec"
ICON = PACKAGING_DIR / "refeye.ico"
DIST = PROJECT_ROOT / "dist" / "RefEye"


def write_icon() -> None:
    """Render the app icon to .ico — PyInstaller needs a real file."""
    # Offscreen so the build works on a headless CI agent.
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from apps.desktop.ui.theme.app_icon import save_icon

    owns_app = QApplication.instance() is None
    app = QApplication([]) if owns_app else QApplication.instance()
    save_icon(str(ICON))
    if owns_app:
        app.quit()
    print(f"  icon      {ICON.relative_to(PROJECT_ROOT)}")


def run_pyinstaller(clean: bool) -> None:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(SPEC),
        "--distpath",
        str(PROJECT_ROOT / "dist"),
        "--workpath",
        str(PROJECT_ROOT / "build"),
        "--noconfirm",
    ]
    if clean:
        command.append("--clean")

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def finalize() -> None:
    """Create the runtime directories the app expects beside the exe."""
    (DIST / "logs").mkdir(parents=True, exist_ok=True)
    (DIST / "models").mkdir(parents=True, exist_ok=True)

    guide = PROJECT_ROOT / "docs" / "OPERATOR_GUIDE.md"
    if guide.exists():
        shutil.copy2(guide, DIST / "OPERATOR_GUIDE.md")


def report() -> None:
    exe = DIST / "RefEye.exe"
    if not exe.exists():
        print("\nBuild finished but RefEye.exe is missing — check the log above.")
        return

    total = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file())
    print("\nBuild complete")
    print(f"  output    {DIST}")
    print(f"  size      {total / 1_048_576:.0f} MB")
    print(f"  run       {exe}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="discard cached build state")
    args = parser.parse_args()

    print("Building RefEye…")
    write_icon()
    run_pyinstaller(clean=args.clean)
    finalize()
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
