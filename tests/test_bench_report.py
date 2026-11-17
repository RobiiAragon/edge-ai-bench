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
