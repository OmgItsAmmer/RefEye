# PyInstaller spec for RefEye — Windows folder build.
#
# --onedir, not --onefile (STACK.md section 8): startup stays fast and the
# model weights stay external and inspectable rather than being unpacked into
# a temp directory on every launch.
#
# Produced layout matches architecture.md section 53:
#
#     dist/RefEye/
#     ├── RefEye.exe
#     ├── _internal/        (PyInstaller runtime + libraries)
#     ├── config/
#     ├── models/
#     ├── data/videos/      (the four demo camera clips)
#     └── logs/
#
# Build:  python -m deployment.packaging.build

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

PROJECT_ROOT = Path(SPECPATH).resolve().parents[1]

datas = [
    # Only the versioned default — config/local.yaml is documented as a
    # gitignored, per-machine dev override (core/config/loader.py) and must
    # not ship; the client gets ensure_editable_config()'s fresh copy of
    # default.yaml as their one editable file, not a leftover dev override.
    (str(PROJECT_ROOT / "config" / "default.yaml"), "config"),
    # Vendored typefaces (theme.md section 3) — fonts.py loads these via
    # Path(__file__).parent / "fonts", a data lookup PyInstaller's static
    # analysis cannot discover on its own; the destination mirrors the
    # source package path so that lookup still resolves once frozen.
    (
        str(PROJECT_ROOT / "apps" / "desktop" / "ui" / "theme" / "fonts"),
        "apps/desktop/ui/theme/fonts",
    ),
]

# Model weights ship alongside the executable, not inside it. Large binaries
# in the bundle would slow every build and every launch for no benefit.
models_dir = PROJECT_ROOT / "models"
if models_dir.exists():
    datas.append((str(models_dir), "models"))

# The four demo camera feeds (config's video.local_file / video.preview_cameras
# point here) — without these the packaged build opens to "No signal" on
# every tile, since there is no other configured video source.
data_dir = PROJECT_ROOT / "data"
if data_dir.exists():
    datas.append((str(data_dir), "data"))

# The vendored T-DEED training config (SoccerNetBall_challenge1.json) is data
# read at runtime by adapter.py, not a Python import — PyInstaller's static
# analysis cannot discover it on its own.
tdeed_vendor = PROJECT_ROOT / "ai" / "action_spotting" / "tdeed" / "_vendor"
for json_file in tdeed_vendor.glob("*.json"):
    datas.append((str(json_file), "ai/action_spotting/tdeed/_vendor"))

hiddenimports = [
    # Adapters are resolved by name from YAML, so static analysis cannot see
    # them. Without these the packaged build fails at model-load time only.
    "vision.detection.yolo_detector",
    "vision.detection.fixture_detector",
    "ai.action_spotting.kinematic.spotter",
    "ai.action_spotting.tdeed.adapter",
    "offside.body_keypoints.estimator",
    "ai.action_spotting.tdeed._vendor.model.model",
    "ai.action_spotting.tdeed._vendor.model.modules",
    "ai.action_spotting.tdeed._vendor.model.shift",
    "ai.action_spotting.tdeed._vendor.model.impl.gsm",
    "ai.action_spotting.tdeed._vendor.model.impl.gsf",
    # timm resolves its model registry dynamically by string name
    # (timm.create_model("regnety_002")); PyInstaller cannot trace that call.
    "timm.models.regnet",
]

a = Analysis(
    [str(PROJECT_ROOT / "apps" / "desktop" / "startup" / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=collect_dynamic_libs("av"),
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Test/plotting stacks pulled in transitively; none are used at runtime
        # and they add hundreds of megabytes to the build.
        "pytest",
        "matplotlib",
        "tkinter",
        "IPython",
        "notebook",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RefEye",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # GUI app: no console window
    icon=str(PROJECT_ROOT / "deployment" / "packaging" / "refeye.ico"),
    # PyInstaller >=6 defaults to nesting everything under _internal/, which
    # breaks core/config/paths.py's app_root()-relative resolution (config/,
    # models/, data/ are expected directly beside the exe — architecture.md
    # section 53). "." restores that flat, pre-6 layout.
    contents_directory=".",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="RefEye",
)
