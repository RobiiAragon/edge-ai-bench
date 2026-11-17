"""Merge result JSON files from several devices into one self-contained HTML report."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from .coverage import summarize

PALETTE = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899"]


def load_results(paths: list[Path]) -> list[dict[str, Any]]:
    runs = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        if "results" not in data or "label" not in data:
            raise ValueError(f"{p} is not an edge-ai-bench result file")
        runs.append(data)
    return runs


def _fmt(v: Any, digits: int = 2, suffix: str = "") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.{digits}f}{suffix}"
    return f"{v}{suffix}"


def _rows(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for di, run in enumerate(runs):
        for rec in run["results"]:
            by_model[rec["model"]].append({**rec, "_device": run["label"], "_di": di})
    return by_model


def _acc(rec: dict[str, Any]) -> float | None:
    q = rec.get("quality")
    return q.get("accuracy") if isinstance(q, dict) else None


def _bar_chart(rows: list[dict[str, Any]], title: str) -> str:
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return ""
    ok.sort(key=lambda r: r["latency"]["p50_ms"])
    vmax = max(r["latency"]["p90_ms"] for r in ok) or 1.0
    bar_h, gap, left, width = 18, 8, 300, 420
    height = len(ok) * (bar_h + gap) + 30
    parts = [
        f'<svg class="chart" viewBox="0 0 {left + width + 90} {height}" role="img" '
        f'aria-label="{escape(title)}">'
    ]
    for i, r in enumerate(ok):
        y = 10 + i * (bar_h + gap)
        p50, p90 = r["latency"]["p50_ms"], r["latency"]["p90_ms"]
        w = max(1.0, width * p50 / vmax)
        w90 = max(w, width * p90 / vmax)
        color = PALETTE[r["_di"] % len(PALETTE)]
        label = f"{r['_device']} · {r['variant'].upper()} · {r['provider']}"
        parts.append(
            f'<text x="{left - 8}" y="{y + 13}" text-anchor="end" class="lbl">'
            f"{escape(label)}</text>"
            f'<rect x="{left}" y="{y + 7}" width="{w90:.1f}" height="4" class="whisker"/>'
            f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" '
            f'fill="{color}"><title>p50 {p50:.3f} ms · p90 {p90:.3f} ms</title></rect>'
            f'<text x="{left + w90 + 6}" y="{y + 13}" class="val">{p50:.3f} ms</text>'
        )
    parts.append(
        f'<text x="{left}" y="{height - 4}" class="axis">bar = p50 latency (batch 1), '
        "thin line = p90</text></svg>"
    )
    return "".join(parts)


def _quant_table(rows: list[dict[str, Any]]) -> str:
    groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if r.get("status") == "ok":
            groups[(r["_device"], r["provider"])][r["variant"]] = r
    lines = []
    for (dev, prov), variants in groups.items():
        fp = variants.get("fp32")
        if not fp:
            continue
        for name, q in sorted(variants.items()):
            if name == "fp32":
                continue
            a_fp, a_q = _acc(fp), _acc(q)
            delta = (a_q - a_fp) * 100 if a_fp is not None and a_q is not None else None
            speed = fp["latency"]["p50_ms"] / q["latency"]["p50_ms"]
            shrink = fp["size_bytes"] / q["size_bytes"]
            cls = "neg" if delta is not None and delta < -1.0 else ""
            lines.append(
                f"<tr><td>{escape(dev)}</td><td>{escape(prov)}</td>"
                f"<td>{escape(name.upper())}</td>"
                f"<td>{_fmt(a_fp * 100 if a_fp is not None else None, 2, '%')}</td>"
                f"<td>{_fmt(a_q * 100 if a_q is not None else None, 2, '%')}</td>"
                f'<td class="{cls}">{_fmt(delta, 2, " pp")}</td>'
                f"<td>{speed:.2f}&times;</td><td>{shrink:.2f}&times;</td></tr>"
            )
    if not lines:
        return ""
    return (
        "<h3>Quantized (QDQ) vs FP32</h3><table><thead><tr><th>Device</th><th>EP</th>"
        "<th>Variant</th><th>FP32 acc</th><th>Quant acc</th><th>Δ acc</th><th>p50 speed-up</th>"
        "<th>Size reduction</th></tr></thead><tbody>" + "".join(lines) + "</tbody></table>"
    )


def _detail_table(rows: list[dict[str, Any]]) -> str:
    out = [
        "<table><thead><tr><th>Device</th><th>Variant</th><th>EP</th><th>p50</th>"
        "<th>p90</th><th>p99</th><th>Throughput</th><th>EP coverage</th><th>Load</th>"
        "<th>Peak RSS</th><th>Size</th><th>Accuracy</th></tr></thead><tbody>"
    ]
    for r in sorted(rows, key=lambda r: (r["_di"], r["variant"], r["provider"])):
        head = (
            f"<td>{escape(r['_device'])}</td><td>{escape(r['variant'].upper())}</td>"
            f"<td>{escape(r['provider'])}</td>"
        )
        if r.get("status") != "ok":
            out.append(
                f'<tr>{head}<td colspan="9" class="neg">{escape(str(r.get("error")))}</td></tr>'
            )
            continue
        lat, thr, mem = r["latency"], r.get("throughput") or {}, r.get("memory") or {}
        acc = _acc(r)
        thr_txt = (
            f"{thr['samples_per_s']:,.0f}/s (b={thr['batch']})" if thr.get("samples_per_s") else "—"
        )
        out.append(
            f"<tr>{head}<td>{lat['p50_ms']:.3f} ms</td><td>{lat['p90_ms']:.3f} ms</td>"
            f"<td>{lat['p99_ms']:.3f} ms</td><td>{thr_txt}</td>"
            f"<td>{escape(summarize(r.get('ep_coverage')))}</td>"
            f"<td>{_fmt(r.get('load_ms'), 0, ' ms')}</td>"
            f"<td>{_fmt(mem.get('peak_rss_mb'), 0, ' MB')}</td>"
            f"<td>{r['size_bytes'] / 1024:,.0f} KB</td>"
            f"<td>{_fmt(acc * 100 if acc is not None else None, 2, '%')}</td></tr>"
        )
    out.append("</tbody></table>")
    return "".join(out)


def _device_cards(runs: list[dict[str, Any]]) -> str:
    cards = []
    for i, run in enumerate(runs):
        d = run.get("device", {})
        avail = d.get("available_providers", [])
        eps = ", ".join(p.replace("ExecutionProvider", "") for p in avail)
        rows = [
            ("CPU", d.get("cpu")),
            ("SoC", d.get("soc")),
            ("Model", d.get("model")),
            ("OS", f"{d.get('os', '')} {d.get('os_version', '')}"),
            ("RAM", f"{d['memory_gb']} GB" if d.get("memory_gb") else None),
            ("ONNX Runtime", d.get("onnxruntime")),
            ("EPs available", eps),
            ("Measured", run.get("created")),
        ]
        body = "".join(f"<dt>{k}</dt><dd>{escape(str(v))}</dd>" for k, v in rows if v)
        skipped = run.get("skipped_providers") or {}
        skip = "".join(f"<li><b>{escape(k)}</b>: {escape(v)}</li>" for k, v in skipped.items())
        color = PALETTE[i % len(PALETTE)]
        cards.append(
            f'<div class="card" style="border-top-color:{color}">'
            f"<h3>{escape(run['label'])}</h3><dl>{body}</dl>"
            + (f'<p class="muted">Skipped EPs:</p><ul class="muted">{skip}</ul>' if skip else "")
            + "</div>"
        )
    return '<div class="cards">' + "".join(cards) + "</div>"


CSS = """
:root{--bg:#ffffff;--fg:#111827;--muted:#6b7280;--line:#e5e7eb;--card:#f9fafb;--neg:#b91c1c}
@media (prefers-color-scheme: dark){:root{--bg:#0b0f17;--fg:#e5e7eb;--muted:#9ca3af;
--line:#1f2937;--card:#111827;--neg:#f87171}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px}h1{margin:0 0 4px}
.muted{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fit,
minmax(280px,1fr));gap:12px}.card{background:var(--card);border:1px solid var(--line);
border-top:4px solid;border-radius:8px;padding:12px 14px}.card h3{margin:0 0 8px}
dl{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin:0}dt{color:var(--muted)}
dd{margin:0;overflow-wrap:anywhere}.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;margin:8px 0 20px;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;white-space:nowrap}
th{font-weight:600;color:var(--muted)}.neg{color:var(--neg)}
.chart{width:100%;height:auto;max-width:820px}.chart .lbl,.chart .val,.chart .axis{
fill:var(--fg);font-size:11px}.chart .axis{fill:var(--muted)}.chart .whisker{fill:var(--muted);
opacity:.5}section{margin-top:28px}
"""


def render(runs: list[dict[str, Any]]) -> str:
    by_model = _rows(runs)
    sections = []
    for model, rows in by_model.items():
        desc = next((r.get("description") for r in rows if r.get("description")), "")
        sections.append(
            f"<section><h2>{escape(model)}</h2><p class='muted'>{escape(desc)}</p>"
            + _bar_chart(rows, f"{model} latency")
            + f"<div class='scroll'>{_quant_table(rows)}</div>"
            + f"<h3>All measurements</h3><div class='scroll'>{_detail_table(rows)}</div>"
            + "</section>"
        )
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Edge AI Bench</title><style>{CSS}</style></head><body><main>"
        "<h1>Edge AI Bench</h1>"
        f"<p class='muted'>Same ONNX models, different devices and execution providers. "
        f"Generated {now} from {len(runs)} result file(s).</p>"
        f"<section><h2>Devices</h2>{_device_cards(runs)}</section>"
        + "".join(sections)
        + "</main></body></html>"
    )


def write_report(paths: list[Path], out: Path) -> Path:
    out.write_text(render(load_results(paths)), encoding="utf-8")
    return out
