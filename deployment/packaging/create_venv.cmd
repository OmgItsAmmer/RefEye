@echo off
REM Creates the project venv, trying Python launcher version pins in order.
REM
REM RefEye requires Python >=3.11,<3.13 (pyproject.toml). `py -3` alone is
REM unsafe here: it resolves to whatever the newest installed Python is
REM (e.g. 3.14 on a machine that also has a compatible 3.12), which would
REM silently build a venv on an unsupported interpreter. This tries known-good
REM versions first and only falls back to unpinned resolution as a last resort.
REM
REM Usage: create_venv.cmd <venv_dir>

set VENV_DIR=%1
if "%VENV_DIR%"=="" set VENV_DIR=.venv

if exist "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"

py -3.12 -m venv "%VENV_DIR%" 2>nul
if exist "%VENV_DIR%\pyvenv.cfg" exit /b 0

py -3.11 -m venv "%VENV_DIR%" 2>nul
if exist "%VENV_DIR%\pyvenv.cfg" exit /b 0

python -m venv "%VENV_DIR%" 2>nul
if exist "%VENV_DIR%\pyvenv.cfg" exit /b 0

py -3 -m venv "%VENV_DIR%" 2>nul
if exist "%VENV_DIR%\pyvenv.cfg" exit /b 0

echo.
echo Could not create a virtual environment. RefEye needs Python 3.11 or 3.12.
echo Install one from https://www.python.org/downloads/ and try "make up" again.
exit /b 1
