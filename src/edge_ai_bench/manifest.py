"""Model manifest: describes the ONNX files and eval data that ``run`` benchmarks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

MANIFEST_NAME = "manifest.json"


@dataclass
class ModelSpec:
    name: str
    task: str
    description: str
    input_name: str
    input_dtype: str
    input_shape: list[int]  # without the batch dimension
    files: dict[str, str]  # variant -> path relative to the models dir
    eval_data: str | None = None  # npz with arrays x, y
    calib_data: str | None = None  # npz with array x
    source: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    def path(self, root: Path, variant: str) -> Path:
        return root / self.files[variant]

    def sample_input(self, batch: int, root: Path | None = None, seed: int = 0) -> np.ndarray:
        """A batch drawn from the eval/calib set when available, else synthetic data."""
        for rel in (self.eval_data, self.calib_data):
            if root is not None and rel and (root / rel).exists():
                x = np.load(root / rel)["x"]
                reps = int(np.ceil(batch / len(x)))
                return np.ascontiguousarray(np.concatenate([x] * reps)[:batch])
        rng = np.random.default_rng(seed)
        shape = (batch, *self.input_shape)
        if self.input_dtype.startswith("int"):
            return rng.integers(0, 256, size=shape).astype(self.input_dtype)
        return rng.standard_normal(shape).astype(self.input_dtype)


def load(root: Path) -> list[ModelSpec]:
    path = root / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `edge-ai-bench build-models --out {root}` first "
            "(requires the [train] extra) or use the models committed in the repo."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ModelSpec(**m) for m in data["models"]]


def save(root: Path, specs: list[ModelSpec]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / MANIFEST_NAME
    payload = {"models": [asdict(s) for s in specs]}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def select(specs: list[ModelSpec], names: str) -> list[ModelSpec]:
    if names in ("", "all"):
        return specs
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    by_name = {s.name: s for s in specs}
    missing = [n for n in wanted if n not in by_name]
    if missing:
        raise SystemExit(f"unknown model(s): {', '.join(missing)}; available: {', '.join(by_name)}")
    return [by_name[n] for n in wanted]
