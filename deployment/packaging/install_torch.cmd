@echo off
REM Installs torch + torchvision with CUDA GPU support, falling back to the
REM CPU-only build if no CUDA-capable wheel can be installed.
REM
REM Why this is not simply in requirements.txt: `pip install -r
REM requirements.txt` resolves against plain PyPI, which only publishes the
REM CPU build of torch under the plain version number (the CUDA build lives
REM under a separate index with a `+cu121`-style local version). Listing torch
REM there would silently install CPU-only torch even on a machine with a
REM working GPU — exactly what happened during development here: an RTX 4070
REM was present and unused because torch had resolved to the CPU wheel.
REM
REM Usage: install_torch.cmd <path-to-venv-python.exe>

set PY=%1
if "%PY%"=="" set PY=python

echo Installing torch with CUDA support...
"%PY%" -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto fallback

REM A successful pip install does not guarantee torch.cuda.is_available() —
REM the wheel can install fine on a machine with no NVIDIA driver at all, and
REM would then silently run CPU inference with no explanation to the client.
"%PY%" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 goto no_gpu_detected

echo   torch installed with CUDA support - GPU acceleration active.
exit /b 0

:no_gpu_detected
echo   torch installed with CUDA support, but no CUDA GPU was detected on this machine.
echo   RefEye will run AI models on CPU, which is significantly slower.
exit /b 0

:fallback
echo   CUDA build could not be installed - falling back to CPU-only torch.
echo   RefEye will still run, but AI analysis will be significantly slower.
"%PY%" -m pip install --quiet torch torchvision
exit /b 0
