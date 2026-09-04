# RefEye

AI-assisted soccer broadcast analysis for Windows. The application keeps a rolling window of recent video, and when the operator presses a hotkey it analyses the recent play and proposes ranked candidate frames for important ball-contact actions (pass, shot, cross, header, goalkeeper contact). The operator reviews the candidates, steps frame by frame, and confirms the final frame.

The human stays in the loop — RefEye proposes, the operator decides.

## Documentation

- [docs/architecture/architecture.md](docs/architecture/architecture.md) — full technical architecture (source of truth).
- [docs/architecture/STACK.md](docs/architecture/STACK.md) — technology choices and rationale.
- [docs/architecture/theme.md](docs/architecture/theme.md) — visual design system.
- [docs/milestones/M1_Plan.md](docs/milestones/M1_Plan.md) — Milestone 1 (MVP) phases and scope.

## Status — Milestone 1 complete

The full loop runs end to end: video → continuous background CV → hotkey → action spotting → contact refinement → de-duplication → ranking → operator review → confirmed frame.

| Phase | Scope | State |
|---|---|---|
| M1.1 | Foundation, video pipeline, UI shell, request state machine | ✅ |
| M1.2 | Detector, tracker, feature cache, model registry, inference scheduler | ✅ |
| M1.3 | Contact refinement, de-duplication, ranking, event chain | ✅ |
| M1.4 | Candidate review UI | ✅ |
| M1.5 | Diagnostics, graceful degradation, PyInstaller build | ✅ |

### Read this before demoing

Two components are **not** what the architecture ultimately calls for, because the required assets are not obtainable in this environment. Both are honest stand-ins behind the real interfaces, and both announce themselves in the UI and logs:

- **Action spotting runs on a kinematic baseline, not T-DEED.** T-DEED's checkpoints are not redistributable. The adapter for it is written and wired; drop a licensed checkpoint into `models/tdeed/` and it activates with no code change. Until then the registry falls back to a ball-trajectory analyser (`kinematic-baseline`) and the status bar reads **AI degraded**. It finds real contacts from real motion, but it is not a learned action model and **its accuracy on broadcast footage is unmeasured**.
- **The default detector is a fixture colour-blob detector, not YOLO.** The only clip available here is the generated synthetic one, whose flat-colour figures a COCO detector correctly ignores. YOLO11n is installed and wired — switch `ai.detector.provider` to `yolo` and point `video.local_file.path` at real footage.

Neither substitution changes the architecture: they sit at the adapter seam that exists precisely so models can be swapped by config. But no accuracy claim should be made to the client until the real model and real footage have been run together.

## Getting started

```bash
# 1. Environment (Python 3.11 or 3.12)
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

# 2. A clip to play. Real footage is not committed, so generate a stand-in:
.venv/Scripts/python -m tools.video_sampling.make_sample_clip
#    ...or point config/default.yaml at your own file.

# 3. Run
.venv/Scripts/python -m apps.desktop.startup.main

# 4. Tests
.venv/Scripts/python -m pytest
```

### Hotkeys

All bindings live in `config/default.yaml` under `shortcuts:` — none are hard-coded.

| Key | Action |
|---|---|
| `F8` | Analyse recent play |
| `←` / `→` | Previous / next frame *(active from M1.4)* |
| `↑` / `↓` | Previous / next candidate *(active from M1.4)* |
| `Home` | Jump to best candidate *(active from M1.4)* |
| `Enter` | Confirm current frame *(active from M1.4)* |

## Configuration

`config/default.yaml` holds the versioned defaults. Override without editing it:

- `config/local.yaml` — gitignored, for machine-specific values.
- Environment variables — `SOCCER_<SECTION>__<KEY>`, e.g.

```bash
SOCCER_RUNTIME__DEVICE=cpu
SOCCER_VIDEO__LOCAL_FILE__PATH=D:/clips/match.mp4
SOCCER_LOGGING__LEVEL=DEBUG
```

Unknown keys are rejected at startup, so a typo fails loudly instead of silently falling back to a default.

### Buffer memory

The decoded buffer is bounded three ways — frame count, duration, and total bytes — and whichever binds first wins. The byte ceiling (`buffer.max_decoded_megabytes`, default 512 MB) is the one that actually protects RAM: at 960×540 a single BGR frame is ~1.5 MB, so 20 seconds would otherwise cost ~780 MB. On startup the log states which limit binds and what it will cost:

```
video_source_opened  buffer_limit_reason=max_decoded_megabytes
                     buffer_limit_frames=345 buffer_projected_mb=511.7
                     buffer_projected_seconds=13.8
```

Live values are shown in the Diagnostics panel.

## Repository layout

Logical boundaries follow [architecture.md §7](docs/architecture/architecture.md#7-recommended-repository-structure).

```text
apps/desktop/    PySide6 UI, viewmodels, shortcuts, startup   (the only Qt code)
core/            domain models, Protocol interfaces, config, errors
video/           input adapters, decoding, timestamps, rolling buffers
vision/          detection, tracking, features, scene analysis   (M1.2)
ai/              action spotting adapters, refinement, ranking   (M1.2-M1.3)
analysis/        request manager, pipelines, event chain, results
storage/         settings, incidents, cache
observability/   structured logging, metrics, diagnostics
deployment/      Windows packaging, model assets                 (M1.5)
tests/           unit, integration, model evaluation, performance
tools/           dataset prep, video sampling, model benchmarking
config/          YAML configuration
models/          model checkpoints (not committed — see .gitignore)
```

Two boundaries matter most and are enforced by convention:

- **Qt lives only in `apps/desktop/`.** `video/`, `vision/`, `ai/`, and `analysis/` import no UI framework, which is why the whole pipeline is testable headlessly.
- **Model-specific code lives only in its adapter.** Everything else consumes the common `ActionSpotter` / `ObjectDetector` / `MultiObjectTracker` interfaces in `core/interfaces/`.
