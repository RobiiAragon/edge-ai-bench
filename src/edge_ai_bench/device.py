"""Collect a small, privacy-friendly description of the host (no hostnames/serials)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Any


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def is_android() -> bool:
    return "ANDROID_ROOT" in os.environ or "com.termux" in os.environ.get("PREFIX", "")


def _cpu_name_linux() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                key, _, value = line.partition(":")
                if key.strip() in {"model name", "Hardware", "Processor"} and value.strip():
                    return value.strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _mem_total_bytes() -> int | None:
    if platform.system() == "Darwin":
        val = _run(["sysctl", "-n", "hw.memsize"])
        return int(val) if val.isdigit() else None
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def collect() -> dict[str, Any]:
    info: dict[str, Any] = {
        "os": platform.system(),
        "os_version": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    system = platform.system()
    if system == "Darwin":
        info["os_version"] = _run(["sw_vers", "-productVersion"]) or platform.mac_ver()[0]
        info["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
        info["model"] = _run(["sysctl", "-n", "hw.model"])
    elif is_android() and shutil.which("getprop"):
        info["os"] = "Android"
        info["os_version"] = _run(["getprop", "ro.build.version.release"])
        info["model"] = " ".join(
            p
            for p in (
                _run(["getprop", "ro.product.manufacturer"]),
                _run(["getprop", "ro.product.model"]),
            )
            if p
        )
        soc = _run(["getprop", "ro.soc.model"]) or _run(["getprop", "ro.board.platform"])
        info["soc"] = soc
        info["cpu"] = _cpu_name_linux()
    else:
        info["cpu"] = _cpu_name_linux()
    mem = _mem_total_bytes()
    if mem:
        info["memory_gb"] = round(mem / 1024**3, 1)
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["available_providers"] = list(ort.get_available_providers())
    except ImportError:
        info["onnxruntime"] = None
    return info


def default_label(info: dict[str, Any]) -> str:
    name = info.get("model") if info.get("os") == "Android" else info.get("cpu")
    return f"{name or info.get('machine')} ({info.get('os')} {info.get('os_version')})"
