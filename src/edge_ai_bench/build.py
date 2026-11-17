"""Train/export the benchmark models (needs the ``[train]`` extra: torch, onnx, sklearn).

Everything here runs in well under a minute on a laptop CPU, and the resulting ONNX
files are small enough to commit, so devices without PyTorch (e.g. Android/Termux)
can benchmark them directly.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from . import datasets
from .manifest import ModelSpec, save
from .quantize import quantize_qdq

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("build-models needs PyTorch: pip install -e '.[train]'") from exc


# --------------------------------------------------------------------------- models
def _dw_block(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
        nn.BatchNorm2d(cin),
        nn.ReLU6(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU6(inplace=True),
    )


class MicroMobileNet(nn.Module):
    """MobileNet-style depthwise-separable CNN for 1x32x32 inputs."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            _dw_block(32, 64, 2),
            _dw_block(64, 64, 1),
            _dw_block(64, 128, 2),
            _dw_block(128, 128, 1),
            _dw_block(128, 256, 2),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x.mean(dim=(2, 3)))


class EncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer written with plain ops for clean ONNX export."""

    def __init__(self, dim: int, heads: int, ff: int) -> None:
        super().__init__()
        self.heads = heads
        # Separate projections (instead of one fused qkv + Split) keep the graph simple
        # for NPU/CoreML compilers, which handle Split on dynamic shapes poorly.
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.heads
        y = self.ln1(x)
        q, k, v = self.q(y), self.k(y), self.v(y)
        q = q.reshape(b, t, h, d // h).transpose(1, 2)
        k = k.reshape(b, t, h, d // h).transpose(1, 2)
        v = v.reshape(b, t, h, d // h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(d // h) + attn_bias
        out = (att.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(out)
        return x + self.ff(self.ln2(x))


class ByteTransformer(nn.Module):
    """Tiny byte-level transformer classifier (no tokenizer needed)."""

    def __init__(
        self,
        seq_len: int = 128,
        dim: int = 96,
        heads: int = 4,
        layers: int = 2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(257, dim, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.layers = nn.ModuleList(EncoderLayer(dim, heads, dim * 2) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids > 0).to(torch.float32)  # B,T
        attn_bias = ((1.0 - mask) * -1e4)[:, None, None, :]
        x = self.embed(ids) + self.pos
        for layer in self.layers:
            x = layer(x, attn_bias)
        x = self.norm(x)
        pooled = (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        return self.head(pooled)


# ------------------------------------------------------------------------ training
def _train(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    batch: int = 64,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(x) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    # Inverse-frequency class weights keep the minority class (spam) from being ignored.
    counts = np.bincount(y)
    weights = torch.tensor(len(y) / (len(counts) * counts), dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), batch):
            idx = perm[i : i + batch]
            loss = F.cross_entropy(model(xt[idx]), yt[idx], weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()


def _accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    with torch.no_grad():
        pred = model(torch.from_numpy(x)).argmax(1).numpy()
    return float((pred == y).mean())


def _export(model: nn.Module, example: torch.Tensor, path: Path, input_name: str) -> None:
    import onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=[input_name],
        output_names=["logits"],
        dynamic_axes={input_name: {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    proto = onnx.load(str(path), load_external_data=True)
    proto.producer_name = "edge-ai-bench"
    onnx.checker.check_model(proto)
    onnx.save_model(proto, str(path), save_as_external_data=False)
    for leftover in path.parent.glob(path.name + ".data"):
        leftover.unlink()


# ---------------------------------------------------------------------------- build
def _quantize_variants(
    out: Path, stem: str, input_name: str, calib: np.ndarray, scheme: str
) -> dict[str, str]:
    """FP32 + W8A8 ("int8") + W8A16 QDQ variants of ``<stem>.fp32.onnx``."""
    fp32 = out / f"{stem}.fp32.onnx"
    quantize_qdq(fp32, out / f"{stem}.int8.onnx", input_name, calib, scheme, 8)
    quantize_qdq(fp32, out / f"{stem}.w8a16.onnx", input_name, calib, scheme, 16)
    return {v: f"{stem}.{v}.onnx" for v in ("fp32", "int8", "w8a16")}


def build_all(out: Path, scheme: str = "qnn", include_mobilenet: bool = True) -> list[ModelSpec]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    specs: list[ModelSpec] = []
    # These nets are tiny: one thread avoids oversubscription and is ~5x faster on M-series.
    torch.set_num_threads(1)

    # Vision: digits
    t0 = time.perf_counter()
    x, y = datasets.load_digits32()
    tr, te = datasets.stratified_split(y, 0.2, seed=0)
    torch.manual_seed(0)
    vision = MicroMobileNet()
    _train(vision, x[tr], y[tr], epochs=20, lr=3e-3)
    print(
        f"[digits-cnn] torch test acc={_accuracy(vision, x[te], y[te]):.4f} "
        f"({time.perf_counter() - t0:.1f}s)"
    )
    np.savez_compressed(out / "data/digits_test.npz", x=x[te], y=y[te])
    np.savez_compressed(out / "data/digits_calib.npz", x=x[tr][:256])
    _export(vision, torch.from_numpy(x[:1]), out / "digits_cnn.fp32.onnx", "image")
    digits_files = _quantize_variants(out, "digits_cnn", "image", x[tr][:256], scheme)
    specs.append(
        ModelSpec(
            name="digits-cnn",
            task="vision",
            description="MobileNet-style depthwise-separable CNN, 1x32x32 digits, 10 classes",
            input_name="image",
            input_dtype="float32",
            input_shape=[1, 32, 32],
            files=digits_files,
            eval_data="data/digits_test.npz",
            calib_data="data/digits_calib.npz",
            source="UCI Optical Recognition of Handwritten Digits (CC BY 4.0) via scikit-learn",
        )
    )

    # Text: SMS spam
    t0 = time.perf_counter()
    xs, ys = datasets.load_sms()
    tr, te = datasets.stratified_split(ys, 0.2, seed=0)
    torch.manual_seed(0)
    text = ByteTransformer(seq_len=datasets.SMS_SEQ_LEN)
    _train(text, xs[tr], ys[tr], epochs=8, lr=2e-3)
    print(
        f"[sms-transformer] torch test acc={_accuracy(text, xs[te], ys[te]):.4f} "
        f"({time.perf_counter() - t0:.1f}s)"
    )
    np.savez_compressed(out / "data/sms_test.npz", x=xs[te], y=ys[te])
    np.savez_compressed(out / "data/sms_calib.npz", x=xs[tr][:256])
    _export(text, torch.from_numpy(xs[:1]), out / "sms_transformer.fp32.onnx", "input_ids")
    sms_files = _quantize_variants(out, "sms_transformer", "input_ids", xs[tr][:256], scheme)
    specs.append(
        ModelSpec(
            name="sms-transformer",
            task="text",
            description="2-layer byte-level transformer encoder (d=96), SMS spam vs ham",
            input_name="input_ids",
            input_dtype="int64",
            input_shape=[datasets.SMS_SEQ_LEN],
            files=sms_files,
            eval_data="data/sms_test.npz",
            calib_data="data/sms_calib.npz",
            source="UCI SMS Spam Collection (CC BY 4.0)",
        )
    )

    if include_mobilenet:
        specs.append(_build_mobilenet(out, scheme))

    save(out, specs)
    return specs


def _build_mobilenet(out: Path, scheme: str) -> ModelSpec:
    """ImageNet MobileNetV3-Small (torchvision weights, BSD-3) for realistic latency.

    No ImageNet data is shipped, so this model is benchmarked for latency, memory and
    size only. INT8 calibration uses smooth synthetic images, which is fine for timing
    but says nothing about accuracy - that is reported as n/a on purpose.
    """
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval()
    rng = np.random.default_rng(0)
    base = torch.from_numpy(rng.standard_normal((96, 3, 14, 14)).astype(np.float32))
    imgs = F.interpolate(base, size=224, mode="bicubic").numpy().astype(np.float32)
    _export(model, torch.from_numpy(imgs[:1]), out / "mobilenetv3s.fp32.onnx", "image")
    files = _quantize_variants(out, "mobilenetv3s", "image", imgs, scheme)
    print("[mobilenetv3-small] exported (latency/size only)")
    return ModelSpec(
        name="mobilenetv3-small",
        task="vision",
        description="torchvision MobileNetV3-Small (ImageNet), 3x224x224 - latency/size only",
        input_name="image",
        input_dtype="float32",
        input_shape=[3, 224, 224],
        files=files,
        eval_data=None,
        calib_data=None,
        source="torchvision MobileNet_V3_Small_Weights.IMAGENET1K_V1 (BSD-3-Clause)",
    )
