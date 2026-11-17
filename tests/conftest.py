from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edge_ai_bench.manifest import ModelSpec, save

REPO = Path(__file__).resolve().parents[1]


def _tiny_mlp(path: Path) -> None:
    """A 2-layer MLP classifying whether the sum of 8 features is positive."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    w1 = np.concatenate([np.ones((8, 1)), -np.ones((8, 1)), rng.normal(0, 0.1, (8, 6))], 1)
    w2 = np.zeros((8, 2))
    w2[0, 1], w2[1, 0] = 1.0, 1.0
    inits = [
        numpy_helper.from_array(w1.astype(np.float32), "w1"),
        numpy_helper.from_array(np.zeros(8, np.float32), "b1"),
        numpy_helper.from_array(w2.astype(np.float32), "w2"),
        numpy_helper.from_array(np.zeros(2, np.float32), "b2"),
    ]
    nodes = [
        helper.make_node("MatMul", ["x", "w1"], ["h"]),
        helper.make_node("Add", ["h", "b1"], ["hb"]),
        helper.make_node("Relu", ["hb"], ["r"]),
        helper.make_node("MatMul", ["r", "w2"], ["o"]),
        helper.make_node("Add", ["o", "b2"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 8])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


@pytest.fixture()
def tiny_models(tmp_path: Path) -> tuple[Path, ModelSpec]:
    from edge_ai_bench.quantize import quantize_qdq

    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (200, 8)).astype(np.float32)
    y = (x.sum(1) > 0).astype(np.int64)
    (tmp_path / "data").mkdir()
    np.savez(tmp_path / "data/test.npz", x=x, y=y)
    np.savez(tmp_path / "data/calib.npz", x=x[:64])
    _tiny_mlp(tmp_path / "tiny.fp32.onnx")
    quantize_qdq(tmp_path / "tiny.fp32.onnx", tmp_path / "tiny.int8.onnx", "x", x[:64], "qnn")
    spec = ModelSpec(
        name="tiny",
        task="tabular",
        description="tiny mlp",
        input_name="x",
        input_dtype="float32",
        input_shape=[8],
        files={"fp32": "tiny.fp32.onnx", "int8": "tiny.int8.onnx"},
        eval_data="data/test.npz",
        calib_data="data/calib.npz",
    )
    save(tmp_path, [spec])
    return tmp_path, spec
