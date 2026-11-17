"""Execution-provider discovery and session creation.

CPU is always available. CoreML, QNN (Qualcomm), CUDA, NNAPI and XNNPACK are used only
when the installed onnxruntime build exposes them; otherwise they are skipped with a
reason. (Termux's `python-onnxruntime` is built with NNAPI + XNNPACK.)
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Any

EP_ALIASES: dict[str, str] = {
    "cpu": "CPUExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "qnn": "QNNExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "nnapi": "NnapiExecutionProvider",
    "xnnpack": "XnnpackExecutionProvider",
}

INSTALL_HINTS: dict[str, str] = {
    "coreml": "CoreML ships in the default `onnxruntime` wheel on macOS (Apple Silicon).",
    "qnn": "Install `onnxruntime-qnn` on a Snapdragon device (Windows on Snapdragon / "
    "supported Linux). Android/Termux CPU builds do not include QNN.",
    "cuda": "Install `onnxruntime-gpu` on a machine with an NVIDIA GPU and CUDA runtime.",
    "nnapi": "Android only (Termux `python-onnxruntime` includes it). NNAPI can dispatch "
    "to the Hexagon NPU/GPU depending on the vendor driver.",
    "xnnpack": "Included in Android/Termux builds; optimized Arm CPU kernels.",
    "cpu": "Install `onnxruntime` (or `pkg install python-onnxruntime` in Termux).",
}


@dataclass(frozen=True)
class ProviderChoice:
    alias: str
    name: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResolution:
    selected: list[ProviderChoice]
    skipped: dict[str, str]


def _import_ort() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - exercised only without ORT
        raise SystemExit("onnxruntime is not installed. " + INSTALL_HINTS["cpu"]) from exc
    return ort


def available_providers() -> list[str]:
    return list(_import_ort().get_available_providers())


def default_options(alias: str) -> dict[str, Any]:
    """Provider options that make each EP use its accelerator."""
    if alias == "coreml":
        # MLProgram + ALL lets CoreML schedule on ANE / GPU / CPU.
        return {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}
    if alias == "qnn":
        lib = "QnnHtp.dll" if sys.platform == "win32" else "libQnnHtp.so"
        return {"backend_path": lib, "htp_performance_mode": "burst"}
    return {}


def resolve(requested: list[str], available: list[str] | None = None) -> ProviderResolution:
    """Map aliases to available providers, recording why others were skipped."""
    avail = set(available if available is not None else available_providers())
    selected: list[ProviderChoice] = []
    skipped: dict[str, str] = {}
    for raw in requested:
        alias = raw.strip().lower()
        if not alias:
            continue
        name = EP_ALIASES.get(alias)
        if name is None:
            skipped[alias] = f"unknown provider alias (known: {', '.join(EP_ALIASES)})"
        elif name not in avail:
            skipped[alias] = f"{name} not in this onnxruntime build. {INSTALL_HINTS[alias]}"
        else:
            selected.append(ProviderChoice(alias, name, default_options(alias)))
    return ProviderResolution(selected, skipped)


def make_session(
    model_path: str,
    choice: ProviderChoice,
    threads: int = 0,
    batch: int | None = None,
    log_level: int = 3,
) -> Any:
    """Create an InferenceSession for ``choice`` with CPU as the fallback EP.

    ``batch`` pins the symbolic ``batch`` dimension so every EP sees static shapes,
    which is how models are deployed on NPUs (QNN/HTP and the ANE need static shapes).
    """
    ort = _import_ort()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = log_level
    if batch is not None:
        so.add_free_dimension_override_by_name("batch", batch)
    if threads > 0:
        so.intra_op_num_threads = threads
    providers: list[Any] = [(choice.name, choice.options)]
    if choice.name != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")
    return ort.InferenceSession(model_path, sess_options=so, providers=providers)


def host_summary() -> str:
    return f"{platform.system()} {platform.machine()} / onnxruntime {_import_ort().__version__}"
