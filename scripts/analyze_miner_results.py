#!/usr/bin/env python3
"""Analyze miner forward test JSON output and emit summary stats + PNG charts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LINF_WEIGHT = 0.7
DEFAULT_RMSE_WEIGHT = 0.3


def _load_results(json_path: Path | None, log_path: Path | None) -> list[dict]:
    if json_path is not None:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        return payload.get("results", payload)

    if log_path is None:
        raise ValueError("Provide --json-out from test run or --log-file with pasted output")

    text = log_path.read_text(encoding="utf-8")
    records: list[dict] = []
    current: dict | None = None
    for line in text.splitlines():
        header = re.match(r"^=== (.+?) ===$", line.strip())
        if header:
            if current is not None:
                records.append(current)
            current = {"image": header.group(1), "passed": False}
            continue
        if current is None:
            continue
        if line.startswith("PASS"):
            current["passed"] = True
        elif line.startswith("FAIL"):
            current["passed"] = False
        elif m := re.search(r"true_label='([^']*)'", line):
            current["true_label"] = m.group(1)
        elif m := re.search(r"new_label='([^']*)'", line):
            current["new_label"] = m.group(1)
        elif m := re.search(r"elapsed_ms=(\d+)", line):
            current["elapsed_ms"] = int(m.group(1))
        elif m := re.search(r"linf=([\d.]+)", line):
            current["linf_norm"] = float(m.group(1))
        elif m := re.search(r"rmse=([\d.]+)", line):
            current["rmse"] = float(m.group(1))
        elif m := re.search(r"ssim=([\d.]+)", line):
            current["ssim"] = float(m.group(1))
        elif m := re.search(r"psnr_db=([\d.]+)", line):
            current["psnr_db"] = float(m.group(1))
        elif m := re.search(r"perturbation_score=([\d.]+)", line):
            current["perturbation_score"] = float(m.group(1))
    if current is not None:
        records.append(current)
    return records


def _score_components(linf: float, rmse: float, min_delta: float, effective_max: float) -> tuple[float, float]:
    denom = max(1e-12, effective_max - min_delta)
    linf_ratio = min(max((linf - min_delta) / denom, 0.0), 1.0)
    linf_score = (1.0 - linf_ratio) ** 2
    rmse_ratio = min(max(rmse / max(1e-12, effective_max), 0.0), 1.0)
    rmse_score = (1.0 - rmse_ratio) ** 2
    return linf_score, rmse_score


def _tier(score: float) -> str:
    if score >= 0.905:
        return ">=0.905"
    if score >= 0.900:
        return "0.900-0.905"
    if score >= 0.895:
        return "0.895-0.900"
    return "<0.895"


def _summarize(records: list[dict]) -> dict:
    import numpy as np

    passed = [r for r in records if r.get("passed")]
    scores = np.array([r["perturbation_score"] for r in passed if r.get("perturbation_score") is not None])
    rmses = np.array([r["rmse"] for r in passed if r.get("rmse") is not None])
    ssims = np.array([r["ssim"] for r in passed if r.get("ssim") is not None])
    psnrs = np.array([r["psnr_db"] for r in passed if r.get("psnr_db") is not None])
    elapsed = np.array([r["elapsed_ms"] for r in passed if r.get("elapsed_ms") is not None])

    def stats(arr):
        if len(arr) == 0:
            return {}
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "var": float(arr.var()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
        }

    return {
        "total": len(records),
        "passed": len(passed),
        "perturbation_score": stats(scores),
        "rmse": stats(rmses),
        "ssim": stats(ssims),
        "psnr_db": stats(psnrs),
        "elapsed_ms": stats(elapsed),
    }


def _print_summary(summary: dict) -> None:
    print(f"Total: {summary['total']}  Passed: {summary['passed']}")
    for key in ("perturbation_score", "rmse", "ssim", "psnr_db", "elapsed_ms"):
        s = summary.get(key, {})
        if not s:
            continue
        print(
            f"{key}: mean={s['mean']:.6f} std={s['std']:.6f} var={s['var']:.6f} "
            f"min={s['min']:.6f} max={s['max']:.6f} p50={s['p50']:.6f}"
        )


def _save_charts(records: list[dict], out_dir: Path, min_delta: float, effective_max: float) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    passed = [r for r in records if r.get("passed") and r.get("perturbation_score") is not None]
    scores = np.array([r["perturbation_score"] for r in passed])
    rmses = np.array([r["rmse"] for r in passed])
    names = [r["image"] for r in passed]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=20, color="#4a90d9", edgecolor="white")
    ax.axvline(scores.mean(), color="#e74c3c", linestyle="--", label=f"mean={scores.mean():.4f}")
    ax.set_xlabel("Perturbation score")
    ax.set_ylabel("Image count")
    ax.set_title("Perturbation score distribution (100 images)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "perturbation_score_histogram.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(rmses, scores, c=[r.get("ssim", 0.99) for r in passed], cmap="viridis", alpha=0.8)
    plt.colorbar(sc, ax=ax, label="SSIM")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Perturbation score")
    ax.set_title("RMSE vs perturbation score (color = SSIM)")
    fig.tight_layout()
    fig.savefig(out_dir / "rmse_vs_score_scatter.png", dpi=150)
    plt.close(fig)

    linf_scores = []
    rmse_scores = []
    for r in passed:
        ls, rs = _score_components(r.get("linf_norm", 0.003922), r["rmse"], min_delta, effective_max)
        linf_scores.append(ls)
        rmse_scores.append(rs)
    order = np.argsort(scores)
    decile_idx = np.array_split(order, 10)
    decile_linf = [float(np.mean([linf_scores[i] for i in d])) for d in decile_idx]
    decile_rmse = [float(np.mean([rmse_scores[i] for i in d])) for d in decile_idx]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(10)
    ax.bar(x, decile_linf, label="linf_score", color="#95a5a6")
    ax.bar(x, decile_rmse, bottom=decile_linf, label="rmse_score", color="#3498db")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i+1}" for i in range(10)])
    ax.set_ylabel("Component score (stacked)")
    ax.set_title("Score components by decile (sorted by perturbation_score)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "score_components_by_decile.png", dpi=150)
    plt.close(fig)

    tiers = ["<0.895", "0.895-0.900", "0.900-0.905", ">=0.905"]
    tier_rmse = {t: [] for t in tiers}
    tier_psnr = {t: [] for t in tiers}
    tier_elapsed = {t: [] for t in tiers}
    for r in passed:
        t = _tier(r["perturbation_score"])
        tier_rmse[t].append(r["rmse"])
        tier_psnr[t].append(r["psnr_db"])
        tier_elapsed[t].append(r["elapsed_ms"])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, title, ylabel in [
        (axes[0], tier_rmse, "RMSE by score tier", "RMSE"),
        (axes[1], tier_psnr, "PSNR by score tier", "PSNR (dB)"),
        (axes[2], tier_elapsed, "Elapsed by score tier", "Elapsed (ms)"),
    ]:
        values = [data[t] for t in tiers if data[t]]
        labels = [t for t in tiers if data[t]]
        if values:
            ax.boxplot(values, tick_labels=labels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_by_tier_boxplot.png", dpi=150)
    plt.close(fig)

    sorted_idx = np.argsort(scores)
    bottom = sorted_idx[:10]
    top = sorted_idx[-10:][::-1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    for ax, idxs, title in [
        (axes[0], top, "Top 10 by perturbation score"),
        (axes[1], bottom, "Bottom 10 by perturbation score"),
    ]:
        ax.barh([names[i] for i in idxs], [scores[i] for i in idxs], color="#3498db")
        ax.set_xlabel("Perturbation score")
        ax.set_title(title)
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "top_bottom_leaderboard.png", dpi=150)
    plt.close(fig)

    print(f"Saved charts to {out_dir}/")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze miner forward test results")
    parser.add_argument("--json-out", type=str, default="", help="JSON file from test_miner_forward.py --json-out")
    parser.add_argument("--log-file", type=str, default="", help="Plain-text log file to parse")
    parser.add_argument(
        "--charts-dir",
        type=str,
        default=str(PROJECT_ROOT / "assets" / "analysis"),
        help="Directory for PNG chart output",
    )
    parser.add_argument("--min-delta", type=float, default=0.003)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--no-charts", action="store_true", help="Print summary only, skip PNG generation")
    args = parser.parse_args()

    json_path = Path(args.json_out).expanduser().resolve() if args.json_out.strip() else None
    log_path = Path(args.log_file).expanduser().resolve() if args.log_file.strip() else None

    if json_path is None and log_path is None:
        print("Provide --json-out or --log-file", file=sys.stderr)
        return 1

    records = _load_results(json_path, log_path)
    if not records:
        print("No records found.", file=sys.stderr)
        return 1

    summary = _summarize(records)
    _print_summary(summary)

    if not args.no_charts:
        try:
            _save_charts(
                records,
                Path(args.charts_dir).expanduser().resolve(),
                min_delta=args.min_delta,
                effective_max=min(args.epsilon, 0.03),
            )
        except ImportError:
            print("matplotlib not installed; run: pip install matplotlib", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
