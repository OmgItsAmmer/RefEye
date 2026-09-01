# Soccer Analysis MVP

AI-assisted soccer broadcast analysis desktop application for Windows. Given a live or recorded broadcast, the operator presses a hotkey to trigger AI analysis of the recent play and reviews ranked candidate frames (pass, shot, cross, header, goalkeeper contact, etc.) before manually confirming the correct one.

See:
- [docs/architecture/architecture.md](docs/architecture/architecture.md) — full technical architecture (source of truth).
- [docs/architecture/STACK.md](docs/architecture/STACK.md) — concrete technology stack and rationale.
- [docs/milestones/M1_Plan.md](docs/milestones/M1_Plan.md) — Milestone 1 (MVP) phase breakdown and scope.

## Getting Started (development)

```bash
# 1. Create environment (uv recommended, or plain venv)
uv venv
uv pip install -e ".[dev]"

# 2. Place a sample clip
#    -> tests/fixtures/sample_match.mp4 (path configurable in config/default.yaml)

# 3. Run the app
python -m apps.desktop.startup.main

# 4. Run tests
pytest
```

## Repository Layout

See [architecture.md §7](docs/architecture/architecture.md#7-recommended-repository-structure) for the full rationale. Top level:

```text
apps/        desktop UI, viewmodels, shortcuts, startup
core/        domain models, interfaces (protocols), config, errors
video/       input adapters, decoding, timestamps, rolling buffers
vision/      detection, tracking, features, scene analysis
ai/          action spotting adapters, contact refinement, ranking, model registry
analysis/    request manager, pipelines, event chain, results
storage/     settings, incidents, cache
observability/  logging, metrics, diagnostics
deployment/  Windows packaging, model assets
tests/       unit, integration, model evaluation, performance
tools/       dataset prep, video sampling, model benchmarking
config/      YAML configuration (default.yaml)
models/      local model checkpoints (not committed — see .gitignore)
```

## Status

Milestone 1 (MVP) in progress — see [M1_Plan.md](docs/milestones/M1_Plan.md) for current phase and scope.
# RefEye
