from __future__ import annotations

import numpy as np
import pytest

from edge_ai_bench import datasets, manifest
from edge_ai_bench.providers import EP_ALIASES, default_options, resolve
from edge_ai_bench.stats import latency_summary, peak_rss_bytes


def test_latency_summary_percentiles() -> None:
    s = latency_summary([i / 1000 for i in range(1, 101)])  # 1..100 ms
    assert s["p50_ms"] == pytest.approx(50.5)
    assert s["p99_ms"] == pytest.approx(99.01)
    assert s["min_ms"] == pytest.approx(1.0) and s["n"] == 100


def test_latency_summary_rejects_empty() -> None:
    with pytest.raises(ValueError):
        latency_summary([])


def test_peak_rss_is_positive() -> None:
    assert (peak_rss_bytes() or 1) > 0


def test_resolve_skips_missing_providers_with_reason() -> None:
    res = resolve(["cpu", "coreml", "qnn", "bogus"], ["CPUExecutionProvider"])
    assert [c.alias for c in res.selected] == ["cpu"]
    assert "onnxruntime-qnn" in res.skipped["qnn"]
    assert "unknown" in res.skipped["bogus"]
    assert set(res.skipped) == {"coreml", "qnn", "bogus"}


def test_qnn_options_target_htp() -> None:
    opts = default_options("qnn")
    assert "QnnHtp" in opts["backend_path"]
    assert set(EP_ALIASES) >= {"cpu", "coreml", "qnn", "cuda", "nnapi", "xnnpack"}


def test_encode_bytes_pads_and_truncates() -> None:
    ids = datasets.encode_bytes(["ab", "x" * 300], seq_len=4)
    assert ids.tolist() == [[98, 99, 0, 0], [121, 121, 121, 121]]


def test_parse_sms_and_stratified_split() -> None:
    texts, y = datasets.parse_sms("ham\thello\nspam\tWIN NOW\nbad line\nham\tok\n")
    assert texts == ["hello", "WIN NOW", "ok"] and y.tolist() == [0, 1, 0]
    labels = np.array([0] * 80 + [1] * 20)
    tr, te = datasets.stratified_split(labels, 0.25, seed=0)
    assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == 100
    assert labels[te].sum() == 5  # class ratio preserved


def test_manifest_roundtrip_and_select(tmp_path) -> None:
    spec = manifest.ModelSpec(
        name="m",
        task="t",
        description="d",
        input_name="x",
        input_dtype="int64",
        input_shape=[4],
        files={"fp32": "m.onnx"},
    )
    manifest.save(tmp_path, [spec])
    loaded = manifest.load(tmp_path)
    assert loaded == [spec]
    assert manifest.select(loaded, "all") == loaded
    with pytest.raises(SystemExit):
        manifest.select(loaded, "nope")
    x = spec.sample_input(3)
    assert x.shape == (3, 4) and x.dtype == np.int64


def test_committed_manifest_is_consistent() -> None:
    from conftest import REPO

    root = REPO / "models"
    for spec in manifest.load(root):
        for rel in spec.files.values():
            assert (root / rel).exists(), rel
        if spec.eval_data:
            data = np.load(root / spec.eval_data)
            assert data["x"].shape[1:] == tuple(spec.input_shape)
