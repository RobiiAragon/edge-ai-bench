"""Optional profiling on hosted Snapdragon devices through Qualcomm AI Hub (``qai-hub``).

API used (verified against qai-hub 0.58):
  hub.get_devices(name=..., os=..., attributes=...)
  hub.Device(name=..., attributes=...)
  hub.submit_compile_job(model=<onnx path>, device=..., input_specs=..., options=...)
  compile_job.get_target_model()
  hub.submit_profile_job(model=<target model>, device=..., options=...)
  profile_job.wait() -> JobStatus(.success, .message)
  profile_job.download_profile() -> {"execution_summary": {...}, "execution_detail": [...]}

Credentials come from ``qai-hub configure --api_token <TOKEN>`` which writes
``~/.qai_hub/client.ini`` (or the file pointed to by ``QAIHUB_CLIENT_INI``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .manifest import ModelSpec
from .stats import latency_summary

DEFAULT_DEVICE = "Samsung Galaxy S23 (Family)"  # Snapdragon 8 Gen 2, same SoC as Tab S9 Ultra
RUNTIMES = ("qnn_dlc", "qnn_context_binary", "onnx", "tflite", "precompiled_qnn_onnx")

SETUP_HELP = """\
Qualcomm AI Hub is not configured on this machine.

  1. Create a free account at https://aihub.qualcomm.com/ and sign in.
  2. Copy your API token from Account -> Settings -> API Token.
  3. pip install -e ".[aihub]"
  4. qai-hub configure --api_token <YOUR_TOKEN>
     (writes ~/.qai_hub/client.ini; set QAIHUB_CLIENT_INI to use another path)
  5. Check it works:   edge-ai-bench aihub devices --filter "8 Gen 2"
  6. Profile models:   edge-ai-bench aihub profile --device "Samsung Galaxy S23 (Family)"

Hosted devices are real phones in Qualcomm's lab; jobs usually take a few minutes.
"""


def config_path() -> Path:
    return Path(os.path.expanduser(os.environ.get("QAIHUB_CLIENT_INI", "~/.qai_hub/client.ini")))


def import_hub() -> Any | None:
    try:
        import qai_hub
    except ImportError:
        return None
    return qai_hub


def check_ready() -> tuple[Any | None, str | None]:
    """Return (module, None) when usable, else (None, human-readable reason)."""
    hub = import_hub()
    if hub is None:
        return None, 'qai-hub is not installed: pip install -e ".[aihub]"\n\n' + SETUP_HELP
    if not config_path().exists():
        return None, SETUP_HELP
    return hub, None


def list_devices(hub: Any, name_filter: str = "") -> list[dict[str, Any]]:
    out = []
    for dev in hub.get_devices():
        attrs = list(getattr(dev, "attributes", []) or [])
        chip = next((a.split(":", 1)[1] for a in attrs if a.startswith("chipset:")), "")
        text = f"{dev.name} {chip}".lower()
        if name_filter.lower().replace(" ", "") not in text.replace(" ", "").replace("-", ""):
            continue
        out.append({"name": dev.name, "os": getattr(dev, "os", ""), "chipset": chip})
    return out


def _summary_to_record(summary: dict[str, Any]) -> dict[str, Any]:
    times_us = [t for t in summary.get("all_inference_times") or [] if t]
    est_us = summary.get("estimated_inference_time")
    if times_us:
        latency = latency_summary([t / 1e6 for t in times_us])
    elif est_us:
        latency = latency_summary([est_us / 1e6])
    else:
        raise ValueError("profile has no inference time")
    peak = summary.get("estimated_inference_peak_memory")
    return {
        "latency": latency,
        "load_ms": (summary.get("warm_load_time") or 0) / 1000 or None,
        "memory": {"peak_rss_mb": round(peak / 1024**2, 2) if peak else None},
    }


def profile(
    hub: Any,
    specs: list[ModelSpec],
    root: Path,
    variants: list[str],
    device_name: str,
    runtime: str,
) -> list[dict[str, Any]]:
    """Compile + profile every (model, variant) on one hosted device."""
    device = hub.Device(device_name)
    records = []
    for spec in specs:
        for variant in variants:
            path = spec.path(root, variant)
            base = {
                "model": spec.name,
                "task": spec.task,
                "variant": variant,
                "provider": f"aihub:{runtime}",
                "description": spec.description,
                "size_bytes": path.stat().st_size,
                "throughput": None,
                "quality": None,
            }
            try:
                input_specs = {spec.input_name: ((1, *spec.input_shape), spec.input_dtype)}
                cjob = hub.submit_compile_job(
                    model=str(path),
                    device=device,
                    input_specs=input_specs,
                    name=f"edge-ai-bench {spec.name} {variant}",
                    options=f"--target_runtime {runtime}",
                )
                cstatus = cjob.wait()
                target = cjob.get_target_model() if cstatus.success else None
                if target is None:
                    raise RuntimeError(f"compile failed: {cstatus.message} ({cjob.url})")
                pjob = hub.submit_profile_job(
                    model=target, device=device, name=f"edge-ai-bench {spec.name} {variant}"
                )
                status = pjob.wait()
                if not status.success:
                    raise RuntimeError(f"profile failed: {status.message} ({pjob.url})")
                prof = pjob.download_profile()
                rec = {
                    **base,
                    "status": "ok",
                    **_summary_to_record(prof["execution_summary"]),
                    "aihub": {"compile_job": cjob.url, "profile_job": pjob.url},
                }
                print(
                    f"  {spec.name} {variant}: p50 {rec['latency']['p50_ms']:.3f} ms ({pjob.url})"
                )
            except Exception as exc:  # keep going with the other models
                rec = {**base, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
                print(f"  {spec.name} {variant}: ERROR {exc}")
            records.append(rec)
    return records
