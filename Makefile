# RefEye — dev commands.
#
# RefEye is a single PySide6 desktop app (no separate frontend/backend
# server) — "up" means: venv ready, deps installed, app running.
#
# Works from PowerShell, cmd.exe, or Git Bash — recipes use only commands
# every one of those understands (no bash-only test/&&/rm), because the
# client is expected to run this from PowerShell, not a Unix shell.
#
# Usage:
#   make up        first run: create venv, install deps, launch the app
#   make run       launch the app (assumes venv already set up)
#   make test      run the test suite
#   make fixture   generate the synthetic sample clip tests/fixtures need
#   make clean     remove venv, caches, logs

VENV_DIR := .venv
PYTHON   := $(VENV_DIR)\Scripts\python.exe

.PHONY: up venv install run inspect debug-ui test test-unit test-integration soak lint fixture package clean help

help:
	@echo make up        - create venv, install deps, run the app
	@echo make venv      - create the virtualenv only
	@echo make install   - install/sync dependencies into the venv
	@echo make run       - run the app (venv must already exist)
	@echo make inspect   - open the pipeline inspector (any video; reference clip by default)
	@echo make debug-ui  - alias for make inspect
	@echo make test      - run the full test suite (soak tests excluded)
	@echo make soak      - run the long-running memory/stability tests
	@echo make lint      - run ruff
	@echo make fixture   - generate the synthetic clip + ground-truth labels
	@echo make package   - build the Windows folder distribution into dist/RefEye
	@echo make clean     - remove venv, build output, caches, logs

# Marker file so `install` only reruns pip when requirements.txt changes.
# `if not exist` / batch FOR loops are cmd.exe builtins — chosen deliberately
# so this works with no reliance on a Unix shell being on PATH.
#
# pyvenv.cfg (not python.exe) is the completeness marker: a venv interrupted
# mid-creation (e.g. another process holding python.exe open, as happens when
# an IDE's language server points at it) can leave Scripts/python.exe behind
# with no pyvenv.cfg — checking for python.exe alone would then skip venv
# creation and fail confusingly deep inside pip instead.
# torch/torchvision install separately from requirements.txt so the CUDA
# build is tried first, with an automatic CPU fallback — see
# deployment/packaging/install_torch.cmd for why this can't just be a line in
# requirements.txt. This is what makes AI analysis fast on the client's GPU
# instead of silently running on CPU.
$(VENV_DIR)\.deps-installed: requirements.txt
	@if not exist "$(VENV_DIR)\pyvenv.cfg" deployment\packaging\create_venv.cmd $(VENV_DIR)
	"$(PYTHON)" -m pip install --quiet --upgrade pip
	deployment\packaging\install_torch.cmd "$(PYTHON)"
	"$(PYTHON)" -m pip install --quiet -r requirements.txt
	@type nul > "$(VENV_DIR)\.deps-installed"

venv:
	@if not exist "$(VENV_DIR)\pyvenv.cfg" deployment\packaging\create_venv.cmd $(VENV_DIR)

install: $(VENV_DIR)\.deps-installed

# venv + deps + fixture clip + launch — one command, per the client-facing ask.
up: install fixture
	"$(PYTHON)" -m apps.desktop.startup.main

run:
	"$(PYTHON)" -m apps.desktop.startup.main

# Visual debugger for the offside pipeline. Opens the client reference clip
# by default; pass another with CLIP=path/to/file.mp4
inspect:
	"$(PYTHON)" -m tools.pipeline_debugger $(CLIP)

# Kept so existing notes and muscle memory still work.
debug-ui: inspect

test: install
	"$(PYTHON)" -m pytest tests -q

test-unit: install
	"$(PYTHON)" -m pytest tests/unit -q

test-integration: install
	"$(PYTHON)" -m pytest tests/integration -q

# Minutes-long: bounded memory over a long run, repeated triggers, stream end.
soak: install fixture
	"$(PYTHON)" -m pytest tests/performance -m soak -q

lint: install
	"$(PYTHON)" -m ruff check .

# Real broadcast footage is not committed (licensing, and video does not
# belong in git) — this stands in so `make up`/`make test` work on a clean
# checkout without asking for client footage first.
fixture:
	@if not exist tests\fixtures\sample_match.mp4 "$(PYTHON)" -m tools.video_sampling.make_sample_clip

# Windows folder distribution: runs on a machine with no Python installed.
package: install fixture
	"$(PYTHON)" -m deployment.packaging.build

clean:
	@if exist $(VENV_DIR) rmdir /s /q $(VENV_DIR)
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist .ruff_cache rmdir /s /q .ruff_cache
	@if exist htmlcov rmdir /s /q htmlcov
	@if exist build rmdir /s /q build
	@if exist dist rmdir /s /q dist
	@if exist .coverage del /q .coverage
	@if exist logs\*.jsonl del /q logs\*.jsonl
	@if exist logs\*.log del /q logs\*.log
	@for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
