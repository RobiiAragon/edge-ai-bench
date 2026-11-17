"""Small public datasets with clear licenses, prepared as numpy arrays.

* digits  - UCI Optical Recognition of Handwritten Digits (bundled with scikit-learn,
            CC BY 4.0). 8x8 images upsampled to 32x32.
* sms     - UCI SMS Spam Collection (CC BY 4.0), ~200 KB download, byte-level encoded.
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

SMS_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SMS_SEQ_LEN = 128


def cache_dir() -> Path:
    base = os.environ.get("EDGE_AI_BENCH_CACHE", Path.home() / ".cache" / "edge-ai-bench")
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def stratified_split(
    y: np.ndarray, test_fraction: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[np.ndarray] = []
    test_idx: list[np.ndarray] = []
    for cls in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == cls))
        n_test = max(1, round(len(idx) * test_fraction))
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    return rng.permutation(np.concatenate(train_idx)), rng.permutation(np.concatenate(test_idx))


def load_digits32() -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import load_digits

    ds = load_digits()
    imgs = (ds.images.astype(np.float32) / 16.0 - 0.5) / 0.5  # [-1, 1]
    up = np.kron(imgs, np.ones((4, 4), dtype=np.float32))  # 8x8 -> 32x32 nearest
    return up[:, None, :, :].astype(np.float32), ds.target.astype(np.int64)


def encode_bytes(texts: list[str], seq_len: int = SMS_SEQ_LEN) -> np.ndarray:
    """UTF-8 byte ids shifted by +1 so that 0 is padding."""
    out = np.zeros((len(texts), seq_len), dtype=np.int64)
    for i, text in enumerate(texts):
        raw = text.encode("utf-8", errors="replace")[:seq_len]
        out[i, : len(raw)] = np.frombuffer(raw, dtype=np.uint8).astype(np.int64) + 1
    return out


def parse_sms(raw: str) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    labels: list[int] = []
    for line in raw.splitlines():
        label, sep, text = line.partition("\t")
        if not sep or label not in {"ham", "spam"}:
            continue
        texts.append(text)
        labels.append(1 if label == "spam" else 0)
    return texts, np.asarray(labels, dtype=np.int64)


def load_sms() -> tuple[np.ndarray, np.ndarray]:
    target = cache_dir() / "SMSSpamCollection"
    if not target.exists():
        print(f"downloading {SMS_URL}")
        with urllib.request.urlopen(SMS_URL, timeout=60) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            target.write_bytes(zf.read("SMSSpamCollection"))
    texts, y = parse_sms(target.read_text(encoding="utf-8", errors="replace"))
    return encode_bytes(texts), y
