"""Command-line interface: ``edge-ai-bench <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__, device, manifest
from .providers import EP_ALIASES, available_providers, resolve

DEFAULT_MODELS_DIR = Path("models")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cmd_providers(_: argparse.Namespace) -> int:
    avail = available_providers()
    res = resolve(list(EP_ALIASES), avail)
    print("onnxruntime providers:", ", ".join(avail))
    for choice in res.selected:
        print(f"  [ok]   {choice.alias:<7} {choice.name}")
    for alias, why in res.skipped.items():
        print(f"  [skip] {alias:<7} {why}")
    return 0


def cmd_info(_: argparse.Namespace) -> int:
    info = device.collect()
    print(json.dumps({"label": device.default_label(info), **info}, indent=2))
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from .build import build_all

    specs = build_all(args.out, scheme=args.scheme, include_mobilenet=not args.no_mobilenet)
    for s in specs:
        sizes = ", ".join(
            f"{v}={s.path(args.out, v).stat().st_size / 1024:,.0f} KB" for v in s.files
        )
        print(f"built {s.name}: {sizes}")
    return 0


def cmd_quantize(args: argparse.Namespace) -> int:
    import numpy as np

    from .quantize import quantize_qdq

    calib = np.load(args.calib)["x"]
    out = quantize_qdq(
        args.model, args.out, args.input_name, calib, args.scheme, args.activation_bits
    )
    print(f"wrote {out} ({out.stat().st_size / 1024:,.0f} KB)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .bench import BenchConfig, run_isolated

    specs = manifest.select(manifest.load(args.models_dir), args.models)
    res = resolve(args.providers.split(","))
    for alias, why in res.skipped.items():
        print(f"[skip] {alias}: {why}")
    if not res.selected:
        print("no usable execution providers", file=sys.stderr)
        return 2
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    cfg = BenchConfig(
        warmup=args.warmup,
        iters=args.iters,
        batch=args.batch,
        throughput_seconds=args.seconds,
        threads=args.threads,
        evaluate=not args.no_eval,
    )
    info = device.collect()
    label = args.label or device.default_label(info)
    records = []
    for spec in specs:
        for variant in variants:
            if variant not in spec.files:
                continue
            for choice in res.selected:
                print(f"> {spec.name:<18} {variant:<5} {choice.alias:<7}", end=" ", flush=True)
                rec = run_isolated(spec, args.models_dir, variant, choice, cfg)
                rec["description"] = spec.description
                records.append(rec)
                if rec["status"] == "ok":
                    lat, q = rec["latency"], rec.get("quality")
                    acc = f" acc={q['accuracy'] * 100:.2f}%" if q else ""
                    print(
                        f"p50={lat['p50_ms']:.3f}ms p99={lat['p99_ms']:.3f}ms "
                        f"thr={rec['throughput']['samples_per_s']:,.0f}/s{acc}"
                    )
                else:
                    print(f"ERROR {rec['error']}")
    out = args.out or Path("results") / f"{_slug(label)}.json"
    _write_json(
        out,
        {
            "schema": 1,
            "tool": f"edge-ai-bench {__version__}",
            "label": label,
            "created": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "device": info,
            "config": vars(cfg),
            "skipped_providers": res.skipped,
            "results": records,
        },
    )
    print(f"wrote {out}")
    return 0 if all(r["status"] == "ok" for r in records) else 1


def cmd_report(args: argparse.Namespace) -> int:
    from .report import write_report

    paths = args.results or sorted(Path("results").glob("*.json"))
    if not paths:
        print("no result files given and none found in results/", file=sys.stderr)
        return 2
    out = write_report(list(paths), args.out)
    print(f"wrote {out} from {len(paths)} result file(s)")
    return 0


def cmd_table(args: argparse.Namespace) -> int:
    from .report import load_results
    from .table import markdown

    paths = args.results or sorted(Path("results").glob("*.json"))
    if not paths:
        print("no result files given and none found in results/", file=sys.stderr)
        return 2
    text = markdown(load_results(list(paths)), baseline=args.baseline or None)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def cmd_aihub(args: argparse.Namespace) -> int:
    from . import aihub

    hub, problem = aihub.check_ready()
    if hub is None:
        print(problem)
        return 0 if args.action == "setup" else 3
    if args.action == "setup":
        print(f"Qualcomm AI Hub looks configured ({aihub.config_path()}).")
        return 0
    if args.action == "devices":
        for d in aihub.list_devices(hub, args.filter):
            print(f"{d['name']:<45} {d['os']:<12} {d['chipset']}")
        return 0
    specs = manifest.select(manifest.load(args.models_dir), args.models)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    print(f"Profiling {len(specs)} model(s) on '{args.device}' with {args.runtime} ...")
    records = aihub.profile(hub, specs, args.models_dir, variants, args.device, args.runtime)
    label = f"{args.device} (AI Hub, {args.runtime})"
    out = args.out or Path("results") / f"{_slug(label)}.json"
    _write_json(
        out,
        {
            "schema": 1,
            "tool": f"edge-ai-bench {__version__}",
            "label": label,
            "created": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "device": {
                "model": args.device,
                "os": "Android (hosted)",
                "onnxruntime": None,
                "available_providers": [f"aihub:{args.runtime}"],
            },
            "config": {"runtime": args.runtime},
            "skipped_providers": {},
            "results": records,
        },
    )
    print(f"wrote {out}")
    return 0 if all(r["status"] == "ok" for r in records) else 1


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in text]
    return "-".join(filter(None, "".join(keep).split("-")))[:80]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="edge-ai-bench", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("providers", help="show which execution providers are usable").set_defaults(
        func=cmd_providers
    )
    sub.add_parser("info", help="print device information").set_defaults(func=cmd_info)

    b = sub.add_parser("build-models", help="train/export/quantize the models ([train] extra)")
    b.add_argument("--out", type=Path, default=DEFAULT_MODELS_DIR)
    b.add_argument("--scheme", choices=["qnn", "generic"], default="qnn")
    b.add_argument("--no-mobilenet", action="store_true")
    b.set_defaults(func=cmd_build)

    q = sub.add_parser("quantize", help="INT8 QDQ-quantize any ONNX model")
    q.add_argument("model", type=Path)
    q.add_argument("--calib", type=Path, required=True, help="npz file with array 'x'")
    q.add_argument("--input-name", required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--scheme", choices=["qnn", "generic"], default="qnn")
    q.add_argument(
        "--activation-bits",
        type=int,
        choices=[8, 16],
        default=8,
        help="8 = W8A8 (INT8), 16 = W8A16 (uint16 activations)",
    )
    q.set_defaults(func=cmd_quantize)

    r = sub.add_parser("run", help="benchmark models across execution providers")
    r.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    r.add_argument("--models", default="all", help="comma-separated names or 'all'")
    r.add_argument("--variants", default="fp32,int8,w8a16")
    r.add_argument("--providers", default="cpu,coreml,qnn,cuda,nnapi,xnnpack")
    r.add_argument("--iters", type=int, default=200)
    r.add_argument("--warmup", type=int, default=20)
    r.add_argument("--batch", type=int, default=16, help="batch size for throughput/eval")
    r.add_argument("--seconds", type=float, default=3.0, help="throughput window")
    r.add_argument("--threads", type=int, default=0, help="intra-op threads (0 = ORT default)")
    r.add_argument("--no-eval", action="store_true", help="skip accuracy evaluation")
    r.add_argument("--label", default="", help="device label shown in reports")
    r.add_argument("--out", type=Path)
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="merge result JSON files into one HTML report")
    rep.add_argument("results", nargs="*", type=Path)
    rep.add_argument("-o", "--out", type=Path, default=Path("report.html"))
    rep.set_defaults(func=cmd_report)

    t = sub.add_parser("table", help="print result JSON files as a Markdown table")
    t.add_argument("results", nargs="*", type=Path)
    t.add_argument("--baseline", default="", help="EP alias for a p50 speed-up column, e.g. cpu")
    t.add_argument("-o", "--out", type=Path, help="write to a file instead of stdout")
    t.set_defaults(func=cmd_table)

    a = sub.add_parser("aihub", help="profile on hosted Snapdragon devices (Qualcomm AI Hub)")
    a.add_argument("action", choices=["setup", "devices", "profile"], nargs="?", default="setup")
    a.add_argument("--filter", default="", help="devices: substring of name/chipset")
    a.add_argument("--device", default="Samsung Galaxy S23 (Family)")
    a.add_argument(
        "--runtime",
        default="qnn_dlc",
        choices=["qnn_dlc", "qnn_context_binary", "onnx", "tflite", "precompiled_qnn_onnx"],
    )
    a.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    a.add_argument("--models", default="all")
    a.add_argument("--variants", default="fp32,int8,w8a16")
    a.add_argument("--out", type=Path)
    a.set_defaults(func=cmd_aihub)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
