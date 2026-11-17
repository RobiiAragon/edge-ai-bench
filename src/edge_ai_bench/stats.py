"""Latency statistics and process memory helpers."""

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np


def latency_summary(samples_s: Sequence[float]) -> dict[str, float]:
    """Summarise per-iteration latencies (seconds in, milliseconds out)."""
    if not samples_s:
        raise ValueError("no latency samples")
    ms = np.asarray(samples_s, dtype=np.float64) * 1000.0
    return {
        "p50_ms": float(np.percentile(ms, 50)),
        "p90_ms": float(np.percentile(ms, 90)),
        "p99_ms": float(np.percentile(ms, 99)),
        "mean_ms": float(ms.mean()),
        "min_ms": float(ms.min()),
        "max_ms": float(ms.max()),
        "std_ms": float(ms.std()),
        "n": len(ms),
    }


def current_rss_bytes() -> int | None:
    """Resident set size of this process right now."""
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        pass
    try:  # Linux / Android without psutil
        with open("/proc/self/statm", encoding="utf-8") as fh:
            pages = int(fh.read().split()[1])
        import os

        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def peak_rss_bytes() -> int | None:
    """Peak RSS of this process (ru_maxrss is bytes on macOS, KiB on Linux)."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)
