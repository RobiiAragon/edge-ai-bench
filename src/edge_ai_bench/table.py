"""Render result JSON files as a Markdown table (for READMEs, PRs and model cards).

The numbers come straight from the result files, so a README table can be regenerated
instead of being typed by hand.
"""

from __future__ import annotations

from typing import Any

from .coverage import summarize

COLUMNS = (
    "Device",
    "Model",
    "Variant",
    "EP",
    "p50 (ms)",
    "p90 (ms)",
    "p99 (ms)",
    "Throughput (samples/s)",
    "EP coverage",
    "Peak RSS (MB)",
    "Size (KB)",
    "Accuracy",
)


def _num(v: float | None, digits: int) -> str:
    return "—" if v is None else f"{v:,.{digits}f}"


def _speedup(rec: dict[str, Any], base: dict[str, Any] | None) -> str:
    if base is None or rec is base:
        return "1.00x" if rec is base else "—"
    return f"{base['latency']['p50_ms'] / rec['latency']['p50_ms']:.2f}x"


def markdown(runs: list[dict[str, Any]], baseline: str | None = None) -> str:
    """One row per successful (device, model, variant, EP) measurement.

    ``baseline`` (an EP alias such as ``cpu``) adds a column with the p50 speed-up of each
    row over the same device/model/variant on that EP.
    """
    multi = len(runs) > 1
    cols = [c for c in COLUMNS if multi or c != "Device"]
    if baseline:
        cols.insert(cols.index("p90 (ms)"), f"vs {baseline}")
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append(
        "|"
        + "|".join(
            "---" if c in {"Device", "Model", "Variant", "EP", "EP coverage"} else "---:"
            for c in cols
        )
        + "|"
    )
    for run in runs:
        ok = [r for r in run["results"] if r.get("status") == "ok"]
        base_of = {
            (r["model"], r["variant"]): r for r in ok if baseline and r["provider"] == baseline
        }
        for r in ok:
            lat, mem = r["latency"], r.get("memory") or {}
            thr = (r.get("throughput") or {}).get("samples_per_s")
            acc = (r.get("quality") or {}).get("accuracy")
            cov = "all (CPU)" if r["provider"] == "cpu" else summarize(r.get("ep_coverage"))
            cells = {
                "Device": run["label"],
                "Model": r["model"],
                "Variant": r["variant"].upper(),
                "EP": r["provider"],
                "p50 (ms)": _num(lat["p50_ms"], 3),
                "p90 (ms)": _num(lat["p90_ms"], 3),
                "p99 (ms)": _num(lat["p99_ms"], 3),
                "Throughput (samples/s)": _num(thr, 0),
                "EP coverage": cov,
                "Peak RSS (MB)": _num(mem.get("peak_rss_mb"), 0),
                "Size (KB)": _num(r["size_bytes"] / 1024, 0),
                "Accuracy": "n/a" if acc is None else f"{acc * 100:.2f}%",
            }
            if baseline:
                cells[f"vs {baseline}"] = _speedup(r, base_of.get((r["model"], r["variant"])))
            lines.append("| " + " | ".join(cells[c] for c in cols) + " |")
    return "\n".join(lines) + "\n"
