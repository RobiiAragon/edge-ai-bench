from __future__ import annotations

import json

import onnx

from edge_ai_bench import report
from edge_ai_bench.bench import BenchConfig, run_isolated
from edge_ai_bench.cli import main
from edge_ai_bench.providers import resolve

FAST = BenchConfig(warmup=2, iters=20, batch=8, throughput_seconds=0.2, timeout_s=120)


def test_int8_model_is_qdq(tiny_models) -> None:
    root, _ = tiny_models
    ops = {n.op_type for n in onnx.load(str(root / "tiny.int8.onnx")).graph.node}
    assert {"QuantizeLinear", "DequantizeLinear"} <= ops


def test_run_isolated_cpu_measures_everything(tiny_models) -> None:
    root, spec = tiny_models
    cpu = resolve(["cpu"]).selected[0]
    for variant in ("fp32", "int8"):
        rec = run_isolated(spec, root, variant, cpu, FAST)
        assert rec["status"] == "ok", rec
        assert rec["latency"]["p50_ms"] > 0 and rec["latency"]["n"] == 20
        assert rec["throughput"]["samples_per_s"] > 0
        assert rec["quality"]["accuracy"] > 0.9
        assert rec["size_bytes"] > 0
        assert rec["providers_in_session"][0] == "CPUExecutionProvider"


def test_run_isolated_reports_errors(tiny_models) -> None:
    root, spec = tiny_models
    (root / "tiny.fp32.onnx").write_bytes(b"not a model")
    rec = run_isolated(spec, root, "fp32", resolve(["cpu"]).selected[0], FAST)
    assert rec["status"] == "error" and rec["model"] == "tiny"


def test_cli_run_and_report(tiny_models, tmp_path) -> None:
    root, _ = tiny_models
    out = tmp_path / "dev.json"
    code = main(
        [
            "run",
            "--models-dir",
            str(root),
            "--providers",
            "cpu,qnn",
            "--iters",
            "10",
            "--warmup",
            "1",
            "--seconds",
            "0.1",
            "--label",
            "Test <Device>",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    data = json.loads(out.read_text())
    assert data["label"] == "Test <Device>" and len(data["results"]) == 2
    assert "qnn" in data["skipped_providers"] or "QNNExecutionProvider" in str(data["device"])

    second = json.loads(out.read_text())
    second["label"] = "Tablet"
    (tmp_path / "tab.json").write_text(json.dumps(second))
    html_path = tmp_path / "r.html"
    assert main(["report", str(out), str(tmp_path / "tab.json"), "-o", str(html_path)]) == 0
    html = html_path.read_text()
    assert "Test &lt;Device&gt;" in html and "Tablet" in html
    assert "Quantized (QDQ) vs FP32" in html and "<svg" in html
    assert "<script" not in html  # self-contained, no external resources
    assert "http://" not in html and "https://" not in html


def test_report_rejects_foreign_json(tmp_path) -> None:
    bad = tmp_path / "x.json"
    bad.write_text("{}")
    try:
        report.load_results([bad])
    except ValueError as exc:
        assert "not an edge-ai-bench" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_w8a16_variant_runs_on_cpu(tiny_models) -> None:
    import numpy as np
    import onnxruntime as ort

    from edge_ai_bench.quantize import quantize_qdq

    root, _ = tiny_models
    calib = np.load(root / "data/calib.npz")["x"]
    for scheme in ("qnn", "generic"):
        out = quantize_qdq(
            root / "tiny.fp32.onnx",
            root / f"t.{scheme}.onnx",
            "x",
            calib,
            scheme,
            activation_bits=16,
        )
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        assert sess.run(None, {"x": calib[:4]})[0].shape == (4, 2)


def test_ok_result_survives_teardown_crash() -> None:
    from edge_ai_bench.bench import _apply_exit_code

    ok = {"status": "ok", "latency": {"p50_ms": 1.0}}
    kept = _apply_exit_code(ok, -11)
    assert kept["status"] == "ok" and kept["warning"] == "worker exit code -11"
    assert _apply_exit_code(ok, 0) is ok
    crashed = _apply_exit_code({}, 1)
    assert crashed["status"] == "error" and "exit code 1" in crashed["error"]
    err = {"status": "error", "error": "boom"}
    assert _apply_exit_code(err, 1) == err


def test_table_renders_committed_results(capsys) -> None:
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    path = repo / "results" / "mac-apple-m5-max.json"
    assert main(["table", str(path), "--baseline", "cpu"]) == 0
    text = capsys.readouterr().out
    lines = text.strip().splitlines()
    data = json.loads(path.read_text())
    assert len(lines) == 2 + sum(r["status"] == "ok" for r in data["results"])
    assert "| Device |" not in lines[0]  # single device -> no device column
    assert "vs cpu" in lines[0]
    first = data["results"][0]
    assert f"{first['latency']['p50_ms']:.3f}" in lines[2]
    assert "1.00x" in lines[2]  # cpu row is its own baseline


def test_table_multi_device_and_file_output(tmp_path) -> None:
    from edge_ai_bench.table import markdown

    rec = {
        "status": "ok",
        "model": "m",
        "variant": "int8",
        "provider": "qnn",
        "latency": {"p50_ms": 0.5, "p90_ms": 0.6, "p99_ms": 0.7},
        "size_bytes": 2048,
        "ep_coverage": {"partitions": 1, "nodes_total": 10, "nodes_on_ep": 10},
    }
    base = {**rec, "provider": "cpu", "latency": {"p50_ms": 1.0, "p90_ms": 1.1, "p99_ms": 1.2}}
    err = {"status": "error", "model": "m", "variant": "fp32", "provider": "qnn"}
    runs = [{"label": "A", "results": [base, rec, err]}, {"label": "B", "results": [rec]}]
    text = markdown(runs, baseline="cpu")
    assert "| Device |" in text.splitlines()[0]
    assert "2.00x" in text  # qnn int8 vs cpu int8 on device A
    assert "10/10 nodes, 1 part." in text and "n/a" in text
    assert len(text.strip().splitlines()) == 2 + 3  # the error row is left out
    src = tmp_path / "a.json"
    src.write_text(json.dumps(runs[0]))
    out = tmp_path / "t.md"
    assert main(["table", str(src), "-o", str(out)]) == 0
    assert out.read_text().startswith("| Model |")
