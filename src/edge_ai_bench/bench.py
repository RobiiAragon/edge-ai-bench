"""Benchmark one (model, variant, provider) per fresh subprocess.

A fresh process per configuration keeps memory numbers honest (no allocator reuse
between runs) and isolates crashes in experimental execution providers.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import coverage
from .coverage import capture_stderr
from .manifest import ModelSpec
from .providers import ProviderChoice, make_session
from .stats import current_rss_bytes, latency_summary, peak_rss_bytes


@dataclass
class BenchConfig:
    warmup: int = 20
    iters: int = 200
    batch: int = 16
    throughput_seconds: float = 3.0
    threads: int = 0
    evaluate: bool = True
    timeout_s: float = 900.0


def _evaluate(sess: Any, spec: ModelSpec, root: Path, batch: int) -> dict[str, Any] | None:
    """Accuracy on the eval set using a fixed-batch session (last batch is padded)."""
    if not spec.eval_data or not (root / spec.eval_data).exists():
        return None
    data = np.load(root / spec.eval_data)
    x, y = data["x"], data["y"]
    preds: list[np.ndarray] = []
    for i in range(0, len(x), batch):
        chunk = x[i : i + batch]
        n = len(chunk)
        if n < batch:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], batch - n, axis=0)])
        out = sess.run(None, {spec.input_name: np.ascontiguousarray(chunk)})[0]
        preds.append(np.asarray(out)[:n].argmax(axis=1))
    pred = np.concatenate(preds)
    return {"accuracy": float((pred == y).mean()), "n": len(y)}


def run_one(
    spec: ModelSpec, root: Path, variant: str, choice: ProviderChoice, cfg: BenchConfig
) -> dict[str, Any]:
    """Run inside the worker process. Returns a JSON-serialisable record."""
    path = spec.path(root, variant)
    rss_before = current_rss_bytes()
    log_level = 3 if choice.alias == "cpu" else 1  # info level prints EP capability

    def _open_latency_session() -> Any:
        return make_session(str(path), choice, cfg.threads, batch=1, log_level=log_level)

    t0 = time.perf_counter()
    sess, log = capture_stderr(_open_latency_session)
    load_s = time.perf_counter() - t0
    ep_coverage = None if choice.alias == "cpu" else coverage.parse(log)
    feed1 = {spec.input_name: spec.sample_input(1, root)}

    t0 = time.perf_counter()
    sess.run(None, feed1)
    first_s = time.perf_counter() - t0
    for _ in range(cfg.warmup):
        sess.run(None, feed1)

    samples: list[float] = []
    for _ in range(cfg.iters):
        t = time.perf_counter()
        sess.run(None, feed1)
        samples.append(time.perf_counter() - t)
    del sess

    sess_b = make_session(str(path), choice, cfg.threads, batch=cfg.batch)
    feedb = {spec.input_name: spec.sample_input(cfg.batch, root)}
    for _ in range(3):
        sess_b.run(None, feedb)
    n_runs, t0 = 0, time.perf_counter()
    while True:
        sess_b.run(None, feedb)
        n_runs += 1
        elapsed = time.perf_counter() - t0
        if elapsed >= cfg.throughput_seconds and n_runs >= 3:
            break

    quality = _evaluate(sess_b, spec, root, cfg.batch) if cfg.evaluate else None
    rss_after = current_rss_bytes()
    return {
        "status": "ok",
        "providers_in_session": list(sess_b.get_providers()),
        "ep_coverage": ep_coverage,
        "size_bytes": path.stat().st_size,
        "load_ms": load_s * 1000,
        "first_inference_ms": first_s * 1000,
        "latency": latency_summary(samples),
        "throughput": {"batch": cfg.batch, "samples_per_s": cfg.batch * n_runs / elapsed},
        "memory": {
            "rss_before_mb": _mb(rss_before),
            "rss_after_mb": _mb(rss_after),
            "rss_delta_mb": _mb(rss_after - rss_before) if rss_after and rss_before else None,
            "peak_rss_mb": _mb(peak_rss_bytes()),
        },
        "quality": quality,
    }


def _mb(v: int | None) -> float | None:
    return round(v / 1024**2, 2) if v is not None else None


def _worker(
    conn: Any,
    spec_d: dict[str, Any],
    root: str,
    variant: str,
    choice_d: dict[str, Any],
    cfg_d: dict[str, Any],
) -> None:
    try:
        rec = run_one(
            ModelSpec(**spec_d),
            Path(root),
            variant,
            ProviderChoice(**choice_d),
            BenchConfig(**cfg_d),
        )
    except Exception as exc:  # report, don't crash the whole suite
        rec = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(limit=3),
        }
    conn.send(json.dumps(rec))
    conn.close()


def run_isolated(
    spec: ModelSpec, root: Path, variant: str, choice: ProviderChoice, cfg: BenchConfig
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_worker, args=(child, asdict(spec), str(root), variant, asdict(choice), asdict(cfg))
    )
    proc.start()
    child.close()
    rec: dict[str, Any]
    if parent.poll(cfg.timeout_s):
        try:
            rec = json.loads(parent.recv())
        except EOFError:
            rec = {"status": "error", "error": "worker exited without a result"}
    else:
        proc.kill()
        rec = {"status": "error", "error": f"timeout after {cfg.timeout_s:.0f}s"}
    proc.join(10)
    if proc.exitcode not in (0, None) and rec.get("status") == "ok":
        rec["warning"] = f"worker exit code {proc.exitcode}"
    if proc.exitcode not in (0, None) and "error" not in rec:
        rec = {"status": "error", "error": f"worker crashed (exit code {proc.exitcode})"}
    rec.update(
        {"model": spec.name, "task": spec.task, "variant": variant, "provider": choice.alias}
    )
    return rec
