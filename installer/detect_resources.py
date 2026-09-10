"""Detects the local machine's capacity to run CastoStudio AI in real time
(gRPC server + YOLOv8 + torch) and scores it against recommended specs.

Standalone by design: only stdlib, so it can run before `uv`/deps exist.
Meant to be frozen with PyInstaller into `resource_check.exe` and called
from the Inno Setup installer (see installer/CastoStudioAI.iss).

Usage:
    python detect_resources.py                    # prints JSON to stdout
    python detect_resources.py --out FILE.json     # also writes JSON to FILE.json
    python detect_resources.py --ini FILE.ini      # also writes an INI file
                                                    # (Inno Setup's Pascal Script
                                                    # has no JSON parser, but reads
                                                    # INI natively via GetIniString)

Exit code is always 0 (this script only reports; it never blocks by itself).
The installer UI decides what to do with `tier` / `recommended`.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

# --- Recommended vs minimum specs for real-time YOLOv8 + torch + gRPC ---
CPU_MIN, CPU_REC = 4, 8            # logical cores
RAM_MIN, RAM_REC = 8.0, 16.0       # GB
GPU_VRAM_MIN, GPU_VRAM_REC = 0.0, 6.0  # GB (0 = no dedicated GPU)

WEIGHT_CPU, WEIGHT_RAM, WEIGHT_GPU = 0.30, 0.30, 0.40

TIER_LOW_MAX = 49       # score_percent <= 49 -> "low" (red, needs override)
TIER_OK_MAX = 79        # 50-79 -> "ok" (orange); 80+ -> "excellent" (green)
RECOMMENDED_THRESHOLD = 80


@dataclass
class ResourceReport:
    checked_at: str
    os: str
    cpu_cores: int
    ram_gb: float
    gpu_name: str | None
    gpu_vram_gb: float
    cpu_score: int
    ram_score: int
    gpu_score: int
    score_percent: int
    tier: str          # "excellent" | "ok" | "low"
    recommended: bool  # score_percent >= RECOMMENDED_THRESHOLD


def _sub_score(value: float, minimum: float, recommended: float) -> int:
    if recommended <= minimum:
        return 100 if value >= recommended else 0
    if value <= minimum:
        return 0
    if value >= recommended:
        return 100
    return round((value - minimum) / (recommended - minimum) * 100)


def get_cpu_cores() -> int:
    return os_cpu_count()


def os_cpu_count() -> int:
    import os

    return os.cpu_count() or 1


def get_ram_gb() -> float:
    if platform.system() == "Windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return round(stat.ullTotalPhys / (1024**3), 1)

    # Non-Windows fallback (dev machines running this script for testing).
    try:
        import os as _os

        pages = _os.sysconf("SC_PHYS_PAGES")
        page_size = _os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024**3), 1)
    except (ValueError, AttributeError, OSError):
        return 0.0


def get_gpu_info() -> tuple[str | None, float]:
    """Returns (gpu_name, vram_gb). Only NVIDIA VRAM is measured reliably
    (nvidia-smi); other vendors report a name with vram_gb=0.0 rather than
    a guessed number, since AdapterRAM via WMI is unreliable on modern cards.
    """
    # NVIDIA: authoritative VRAM figure.
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=5,
            text=True,
        )
        first_line = out.strip().splitlines()[0]
        name, mem_mib = [p.strip() for p in first_line.split(",")]
        return name, round(float(mem_mib) / 1024, 1)
    except (subprocess.SubprocessError, OSError, FileNotFoundError, ValueError, IndexError):
        pass

    # Fallback: name only, via WMI (Windows) — no reliable VRAM figure.
    # `wmic` is deprecated and removed on newer Windows builds (11 24H2+),
    # so try it first (older systems) then fall back to PowerShell's
    # Get-CimInstance, which replaces it everywhere wmic is gone.
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                stderr=subprocess.DEVNULL,
                timeout=5,
                text=True,
            )
            lines = [l.strip() for l in out.splitlines() if l.strip() and "Name" not in l]
            if lines:
                return lines[0], 0.0
        except (subprocess.SubprocessError, OSError, FileNotFoundError):
            pass

        try:
            out = subprocess.check_output(
                [
                    "powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_VideoController | Select-Object -First 1 -ExpandProperty Name)",
                ],
                stderr=subprocess.DEVNULL,
                timeout=5,
                text=True,
            )
            name = out.strip()
            if name:
                return name, 0.0
        except (subprocess.SubprocessError, OSError, FileNotFoundError):
            pass

    return None, 0.0


def build_report() -> ResourceReport:
    cpu_cores = get_cpu_cores()
    ram_gb = get_ram_gb()
    gpu_name, gpu_vram_gb = get_gpu_info()

    cpu_score = _sub_score(cpu_cores, CPU_MIN, CPU_REC)
    ram_score = _sub_score(ram_gb, RAM_MIN, RAM_REC)
    gpu_score = _sub_score(gpu_vram_gb, GPU_VRAM_MIN, GPU_VRAM_REC)

    score_percent = round(
        cpu_score * WEIGHT_CPU + ram_score * WEIGHT_RAM + gpu_score * WEIGHT_GPU
    )

    if score_percent <= TIER_LOW_MAX:
        tier = "low"
    elif score_percent <= TIER_OK_MAX:
        tier = "ok"
    else:
        tier = "excellent"

    return ResourceReport(
        checked_at=datetime.now(timezone.utc).isoformat(),
        os=platform.platform(),
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        cpu_score=cpu_score,
        ram_score=ram_score,
        gpu_score=gpu_score,
        score_percent=score_percent,
        tier=tier,
        recommended=score_percent >= RECOMMENDED_THRESHOLD,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="Also write the JSON report to this file.")
    parser.add_argument("--ini", help="Also write an INI report to this file (for Inno Setup).")
    args = parser.parse_args()

    report = asdict(build_report())
    text = json.dumps(report, indent=2)
    print(text)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)

    if args.ini:
        with open(args.ini, "w", encoding="utf-8") as f:
            f.write("[resources]\n")
            for key, value in report.items():
                if value is None:
                    value = ""
                elif isinstance(value, bool):
                    value = "true" if value else "false"
                f.write(f"{key}={value}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
