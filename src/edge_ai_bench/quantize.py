"""Static INT8 quantization in QDQ format (the layout Qualcomm's QNN HTP consumes)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

SCHEMES = ("qnn", "generic")


class NpzCalibrationReader:
    """Feeds calibration samples one at a time to the ORT calibrator."""

    def __init__(self, input_name: str, samples: np.ndarray, batch: int = 1) -> None:
        self.input_name = input_name
        self._samples = samples
        self._batch = batch
        self._iter: Iterator[dict[str, np.ndarray]] = self._make_iter()

    def _make_iter(self) -> Iterator[dict[str, np.ndarray]]:
        for i in range(0, len(self._samples), self._batch):
            yield {self.input_name: np.ascontiguousarray(self._samples[i : i + self._batch])}

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iter, None)

    def rewind(self) -> None:
        self._iter = self._make_iter()


def quantize_qdq(
    fp32_path: Path,
    int8_path: Path,
    input_name: str,
    calib: np.ndarray,
    scheme: str = "qnn",
    activation_bits: int = 8,
) -> Path:
    """Quantize ``fp32_path`` to INT8 QDQ using ``calib`` samples.

    ``qnn``     -> ORT's QNN-targeted config (uint8 activations + uint8 weights, QDQ),
                   which is what the Hexagon NPU (HTP backend) executes natively.
    ``generic`` -> uint8 activations, per-channel int8 weights (good for CPU EPs).

    ``activation_bits=16`` gives W8A16 (uint16 activations), the usual HTP recipe for
    transformers whose activations have outliers that 8 bits cannot represent.
    """
    try:
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize,
            quantize_static,
        )
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("quantization needs `onnx`: pip install -e '.[quant]'") from exc

    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}")
    if activation_bits not in (8, 16):
        raise ValueError("activation_bits must be 8 or 16")
    act_type = QuantType.QUInt16 if activation_bits == 16 else QuantType.QUInt8
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    pre_path = int8_path.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32_path), str(pre_path), skip_symbolic_shape=False)
    reader = NpzCalibrationReader(input_name, calib)
    try:
        if scheme == "qnn":
            from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config

            cfg: Any = get_qnn_qdq_config(
                str(pre_path),
                reader,
                calibrate_method=CalibrationMethod.MinMax,
                activation_type=act_type,
                weight_type=QuantType.QUInt8,
            )
            quantize(str(pre_path), str(int8_path), cfg)
        else:
            quantize_static(
                str(pre_path),
                str(int8_path),
                reader,
                quant_format=QuantFormat.QDQ,
                activation_type=act_type,
                weight_type=QuantType.QInt8,
                per_channel=True,
                calibrate_method=CalibrationMethod.MinMax,
                extra_options={"UseQDQContribOps": activation_bits == 16},
            )
    finally:
        pre_path.unlink(missing_ok=True)
    return int8_path
