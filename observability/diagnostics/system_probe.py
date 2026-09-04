"""Host capability probe.

Recorded at startup (architecture.md section 45: `hardware_detected`) so a
support conversation about a slow or failing machine starts from facts rather
than guesses. Everything here is best-effort: a probe that cannot answer
returns None rather than raising, because diagnostics must never be the reason
the application fails to start.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import asdict, dataclass


@dataclass
class SystemInfo:
    platform: str
    python_version: str
    cpu_count: int | None
    ram_total_gb: float | None
    disk_free_gb: float | None
    torch_version: str | None
    cuda_available: bool
    cuda_device: str | None
    cuda_memory_gb: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def probe() -> SystemInfo:
    return SystemInfo(
        platform=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        cpu_count=os.cpu_count(),
        ram_total_gb=_ram_gb(),
        disk_free_gb=_disk_free_gb(),
        **_torch_info(),
    )


def _ram_gb() -> float | None:
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return round(status.ullTotalPhys / 1024**3, 1)
    except Exception:  # noqa: BLE001 — non-Windows or restricted host
        return None


def _disk_free_gb() -> float | None:
    try:
        return round(shutil.disk_usage(".").free / 1024**3, 1)
    except Exception:  # noqa: BLE001
        return None


def _torch_info() -> dict:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return {
            "torch_version": None,
            "cuda_available": False,
            "cuda_device": None,
            "cuda_memory_gb": None,
        }

    try:
        available = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if available else None
        memory = (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            if available
            else None
        )
    except Exception:  # noqa: BLE001 — driver present but unusable
        available, device, memory = False, None, None

    return {
        "torch_version": torch.__version__,
        "cuda_available": available,
        "cuda_device": device,
        "cuda_memory_gb": memory,
    }
