from __future__ import annotations

import os

from edge_ai_bench import coverage

COREML_LOG = (
    "2026-11-17 [W:onnxruntime:, coreml_execution_provider.cc:141 GetCapability] "
    "CoreMLExecutionProvider::GetCapability, number of partitions supported by CoreML: 21 "
    "number of nodes in the graph: 122 number of nodes supported by CoreML: 102\n"
)
QNN_LOG = (
    "[I:onnxruntime:, qnn_execution_provider.cc:612 GetCapability] Number of partitions "
    "supported by QNN EP: 1, number of nodes in the graph: 60, "
    "number of nodes supported by QNN: 60\n"
)


def test_parse_coreml_line() -> None:
    assert coverage.parse(COREML_LOG) == {"partitions": 21, "nodes_total": 122, "nodes_on_ep": 102}


def test_parse_qnn_style_line_and_last_wins() -> None:
    got = coverage.parse(COREML_LOG + QNN_LOG)
    assert got == {"partitions": 1, "nodes_total": 60, "nodes_on_ep": 60}


def test_parse_none_and_summary() -> None:
    assert coverage.parse("nothing here") is None
    assert coverage.summarize(None) == "—"
    assert "102/122" in coverage.summarize(coverage.parse(COREML_LOG))


def test_capture_stderr_fd_level() -> None:
    result, text = coverage.capture_stderr(lambda: os.write(2, b"from C level\n") and 7)
    assert result == 7 and "from C level" in text
