# Technology Stack — Soccer Analysis MVP

This document is the concrete technology decision record for the project. [architecture.md](architecture.md) §8 gives the *direction*; this file pins the *actual* choices, versions, and rationale so implementation is unambiguous. Treat this as binding unless a decision is explicitly revisited and this file is updated.

Guiding constraints from the architecture doc: models must stay replaceable (§4.3), UI must never block (§4.1), memory must be bounded (§4.2), and everything must be config-driven, not hard-coded (§58).

---

## 1. Language & Runtime

| Item | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Native ecosystem for PyTorch/OpenCV/research models (T-DEED, FOOTPASS, SoccerNet are all Python). Avoids a cross-language IPC boundary for the MVP; keeps velocity high for a trial engagement. |
| Package/env manager | `uv` (fallback: `venv` + `pip`) | Fast, reproducible installs; single lockfile; easy for client handover (`uv sync`). |
| Typing | Full type hints, `dataclasses` for domain models, `Protocol` for interfaces (per architecture doc code samples) | Matches architecture doc exactly; keeps interfaces explicit without an ABC ceremony. |

We are **not** splitting into a .NET/WPF front-end + Python service for M1 (architecture §8 alternative). That adds an IPC boundary, a second build toolchain, and cross-process frame marshalling cost we don't need to pay to win a trial. PySide6 gives a native-feeling, GPU-accelerated Windows UI directly in the same process as the AI stack, while still keeping AI code decoupled via interfaces (never importing Qt inside `ai/`, `vision/`, or `video/`). Revisit only if profiling shows Python UI-thread overhead is a real bottleneck.

---

## 2. Desktop UI

| Item | Choice | Why |
|---|---|---|
| UI framework | **PySide6** (Qt 6, LGPL) | Official Qt-for-Python binding, LGPL (commercially safe, no GPL contamination for client handover). Native widgets, hardware-accelerated video rendering via `QVideoWidget`/custom `QOpenGLWidget` painter. |
| Styling | Hand-authored QSS (Qt Style Sheets) + a small design-token module (`apps/desktop/ui/theme`) | Full control over a professional dark theme without pulling in a heavier third-party Qt theming dependency. Centralized tokens (colors, spacing, radii, typography) avoid ad-hoc inline styling (architecture §58 rule 10 spirit). |
| Icons | Lucide/Feather-style SVG icon set (MIT-licensed), rendered via `QSvgWidget`/`QIcon` | Crisp at any DPI, consistent modern look, no licensing risk. |
| Video rendering | Decoded frame (`np.ndarray`, BGR) → `QImage` → custom `QWidget.paintEvent` (or `QGraphicsView` for overlay support) | Direct numpy→QImage blit is cheap and gives us pixel-level control for drawing candidate overlays (bounding boxes, ball marker) later. |
| Charts/diagnostics widgets | Qt native (`QProgressBar`, custom sparkline widget) | No need for a full charting library at MVP scope; avoids extra dependency weight. |

**Non-functional target for "professional UI":** consistent 4/8px spacing grid, one accent color, dark theme by default, smooth (no visible jank) frame updates at the live-preview frame rate, custom window icon/title, and a single-window layout with clearly separated live/review regions — no default-Qt-grey unstyled widgets anywhere in the shipped build.

---

## 3. Video Ingest & Decoding

| Item | Choice | Why |
|---|---|---|
| Decode backend | **PyAV** (`av`) wrapping FFmpeg | Frame-accurate PTS access, keyframe flags, and packet-level access needed for the encoded rolling buffer (architecture §12.1) — more control than `cv2.VideoCapture`. |
| Raw FFmpeg fallback | `ffmpeg` CLI subprocess adapter, isolated behind the same `VideoInput` interface | Reserved for exotic sources later (NDI, SRT) without touching core code. |
| M1 input adapter | `LocalFileInput` implementing `VideoInput` protocol exactly as specified in architecture §10 | Capture-card/RTSP adapters are stubbed interfaces only for M1; implemented in a later milestone. |
| Color/frame format | BGR `np.ndarray`, `uint8`, analysis resolution configurable (default 960×540 per architecture §39 example) | Matches OpenCV convention used throughout `vision/`. |

---

## 4. Computer Vision / AI

| Item | Choice | Why |
|---|---|---|
| Deep learning framework | **PyTorch** (2.x, CUDA 12.x build where GPU available) | Required to run T-DEED/FOOTPASS/SoccerNet checkpoints natively; largest ecosystem for adapting research models. |
| Array/image ops | **NumPy**, **OpenCV (`opencv-python`)** | Standard for detection pre/post-processing, drawing, resizing, color conversion. |
| Inference acceleration | **ONNX Runime (`onnxruntime-gpu`)** for the object detector (exported once, loaded fast); PyTorch native for the action spotter in M1 | Detector runs continuously in the background loop, so ONNX Runtime's lower overhead matters most there. Action spotter only runs on-trigger, so PyTorch-native is fine for M1; ONNX export considered later only if profiling justifies it (architecture §4.7, §37). |
| TensorRT | Not used in M1 | Explicitly deferred per architecture §37 — "use ONNX/TensorRT only if measured benefit justifies it." Revisit in stabilization milestone if latency requires it. |
| Action spotting (M1 **primary, active**) | **T-DEED** — `SoccerNetBall_challenge1` checkpoint | 1st place, 2024 SoccerNet Ball Action Spotting Challenge. Genuinely wired: the checkpoint loads with a **strict** PyTorch state-dict match against the vendored architecture (`ai/action_spotting/tdeed/_vendor/`, copied from the [T-DEED repo](https://github.com/arturxe2/T-DEED)), so a wrong or mismatched checkpoint fails loudly at load time rather than silently producing bad output. Sliding-window inference (100-frame clips, 796×448, stride 2) mirrors the upstream `inference.py` exactly. Outputs 12 ball-action classes (PASS, SHOT, CROSS, HEADER, GOAL, etc.) mapped to the domain vocabulary in `ai/action_spotting/common/actions.py`. Checkpoint is a ~50MB binary, not committed — see `models/tdeed/README.md` for how to obtain it. Wrapped entirely behind `ai/action_spotting/tdeed/adapter.py` implementing the common `ActionSpotter` protocol (architecture §19) — no code outside this adapter touches T-DEED-specific types. |
| Action spotting (fallback) | **Kinematic baseline** (`ai/action_spotting/kinematic`) | **Not a learned model.** A ball-trajectory analyser: finds moments where the ball's direction/speed changes sharply while a player is within contact range, and types the action from the resulting geometry. Used automatically (via `ai.action_spotter.fallback_provider`) if the T-DEED checkpoint is absent or fails to load — so the app still demonstrates its full loop with zero setup. Reports itself as `kinematic-baseline` in every candidate, log line and diagnostics field, so its output can never be mistaken for T-DEED's. Its thresholds are provisional and its accuracy on real broadcast footage is unmeasured. |
| Action spotting (not implemented) | FOOTPASS/TAAD, SoccerNet-oriented spotters | Interface stubs only (`ai/action_spotting/footpass`, `ai/action_spotting/soccernet`) for the Phase 2 multi-model benchmarking milestone. |
| Fixture detector (development only) | **`fixture` colour-blob detector** (`vision/detection/fixture_detector.py`) | The generated development clip draws players and ball as flat colour blocks, which a COCO detector correctly ignores — leaving the pipeline untestable in an environment with no licensed footage. This adapter detects those exact colours and emits normal `Detection` objects, so tracking → spotting → refinement → ranking all run deterministically. **Selected by default in `config/default.yaml` because that is the only clip available here; switch `ai.detector.provider` to `yolo` for real footage.** |
| Ball/player detection (real footage) | YOLO-family detector (Ultralytics **YOLO11n**, weights at `models/detector/yolo11n.pt`) zero-shot; fine-tuning deferred | Fast enough to run continuously at low resolution; well-documented export path to ONNX Runtime. A generic "sports ball" class is *not* assumed sufficient (architecture §14) — validated against representative footage before being trusted, with confidence thresholds fully configurable. |
| Tracking | **ByteTrack** (lightweight, detection-only, no re-ID network required) | Simple, fast, good default for player tracking; ball tracking gets its own lighter continuity logic (short-gap interpolation) per architecture §15/§25 rather than forcing it through the same multi-object tracker. |
| Model registry/config | YAML-driven `ModelRegistry` per architecture §20 | Swapping `action_spotter.provider` in config swaps the implementation with zero code changes elsewhere. |

---

## 5. Configuration, Persistence, Logging

| Item | Choice | Why |
|---|---|---|
| Config format | **YAML** + **Pydantic v2** models for validation/typing | Human-editable, matches architecture §39 examples exactly; Pydantic gives fail-fast validation with clear error messages instead of silent `dict.get()` bugs. |
| Env overrides | `python-dotenv` + explicit env-var mapping for machine-specific values (device, paths) | Keeps machine-specific values out of versioned YAML. |
| Local persistence | **SQLite** via `sqlalchemy` (only if/when incident save is enabled; not required for M1 core loop) | Matches architecture §43; zero-ops, file-based, sufficient for local settings/incident metadata. No network DB. |
| Logging | **structlog** → JSON-lines file sink (rotated) + human-readable console sink | Structured fields (session id, request id, module, model version) required by architecture §45; structlog gives this without hand-rolled formatting. |
| Metrics/diagnostics | In-process counters/gauges exposed to a diagnostics panel; no external metrics backend for M1 | Matches "no cloud-native distributed inference" non-goal (§5) and keeps M1 fully offline-capable. |

---

## 6. Concurrency

| Item | Choice | Why |
|---|---|---|
| UI thread | Qt event loop only | Never blocked by decode/inference (architecture §4.1, §33). |
| Background workers | Python `threading` for I/O-bound decode/ingest workers; a dedicated single-threaded **inference scheduler** (priority queue: HIGH/MEDIUM/LOW per architecture §34) serializing GPU access | Avoids uncontrolled concurrent GPU access (architecture §33 rule: "do not create arbitrary parallel GPU threads"); Python's GIL makes threading fine for I/O-bound decode, while GPU work is intentionally serialized regardless of GIL. |
| UI↔worker communication | Qt signals/slots (`Signal`/`Slot`) crossing thread boundaries via `QueuedConnection` | Idiomatic, thread-safe, and matches the "UI reacts to events" requirement (architecture §32) without polling. |
| Async orchestration | `asyncio` inside the analysis pipeline (`analysis/pipelines`), bridged to Qt via a dedicated asyncio-thread + signal bridge | The pseudocode in architecture §40 (`async def analyze_recent_play`) is asyncio-shaped; running it on a dedicated loop thread keeps it uncoupled from Qt's own loop. |

---

## 7. Testing

| Item | Choice | Why |
|---|---|---|
| Test runner | **pytest** | Standard, good fixture support for video/model fixtures. |
| Coverage | `pytest-cov` | Visibility into untested pure-logic paths (ranking, refinement, buffers). |
| GUI smoke tests | `pytest-qt` | Allows testing shortcut handling, signal wiring, and navigation commands without full manual QA each build. |
| Fixtures | Short representative labeled clips under `tests/fixtures` (architecture §50 example schema) | Enables model-evaluation and integration tests to run reproducibly. |

---

## 8. Packaging & Deployment

| Item | Choice | Why |
|---|---|---|
| Packaging tool | **PyInstaller** (folder build, `--onedir`, not `--onefile`) | `--onedir` keeps startup fast and keeps large model weights external/inspectable rather than bundled into a single opaque binary — matches architecture §53 example layout (`SoccerAnalysis.exe`, `runtime/`, `models/`, `config/`, `logs/`). |
| Installer (post-M1) | Inno Setup | Deferred past M1 — a zipped folder build is sufficient for a trial demo; a signed installer belongs to the handover milestone. |
| Model assets | Stored external to the executable under `models/`, referenced by a JSON manifest (id, provider, version, sha256, runtime) per architecture §54 | Keeps the executable small, avoids re-bundling multi-hundred-MB checkpoints on every build, and gives traceability for which model produced which result. |
| Target platform | Windows 10/11 x64, NVIDIA GPU preferred (CUDA), CPU fallback for graceful degradation | Matches the stated Windows-workstation deployment target (architecture §52); CPU fallback satisfies §49 graceful degradation. |

---

## 9. Licensing Notes (per architecture §55)

| Component | License | M1 implication |
|---|---|---|
| PySide6 | LGPLv3 | Safe for commercial distribution as dynamically-linked libraries (default PyPI wheels); do not statically link Qt. |
| PyTorch, NumPy, OpenCV (`opencv-python`), PyAV | BSD/Apache-2.0-family | Permissive, safe for commercial use. |
| Ultralytics YOLO | AGPL-3.0 (commercial license available from Ultralytics) | **Flag before commercial distribution.** Fine for the R&D trial build; if the client wants to keep the source private/commercial post-trial, either purchase an Ultralytics commercial license or swap the detector for a permissively-licensed alternative (e.g. a custom-trained lightweight detector). Tracked as an open item, not silently assumed away. |
| ByteTrack | MIT | Safe. |
| T-DEED / FOOTPASS / SoccerNet reference implementations & checkpoints | Vary by repo (check each individually before any redistribution) | Acceptable for R&D evaluation now; license text and any checkpoint redistribution terms must be verified and documented before the checkpoint ships inside a client deliverable. |

---

## 10. Summary Dependency List (M1)

```text
python==3.11
pyside6
av                  # PyAV (FFmpeg bindings)
torch               # CUDA build where available
opencv-python
numpy
onnxruntime-gpu     # falls back to onnxruntime (CPU) if no CUDA
ultralytics         # YOLO detector (license flag — see §9)
pydantic
pyyaml
python-dotenv
sqlalchemy          # only exercised if incident persistence is enabled
structlog
pytest
pytest-cov
pytest-qt
pyinstaller
```

Exact pinned versions live in `pyproject.toml` / `requirements.txt` at the repo root, not duplicated here to avoid drift.

---

## 11. Explicit Deferrals

These are deliberate "not now" decisions, not oversights — each is revisited when its trigger milestone is reached:

- **TensorRT** — after profiling shows PyTorch/ONNX Runtime latency is insufficient (architecture §37).
- **FOOTPASS/SoccerNet spotter adapters (implemented, not just stubbed)** — Phase 2 model-benchmarking milestone.
- **Capture card / RTSP / NDI inputs** — once a live-source requirement is confirmed with the client (architecture §61 item 1).
- **Inno Setup signed installer** — handover milestone (architecture §57 Phase 7).
- **Ultralytics AGPL resolution** — before any commercial (non-trial) distribution.
- **TensorRT/ONNX export of the action spotter** — only if trigger-to-result latency measured in Phase M1.2/M1.3 is unacceptable.
