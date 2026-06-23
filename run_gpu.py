#!/usr/bin/env python3
"""Run PerturbMiner forward benchmarks on Modal GPU hardware.

Setup (once):
    pip install modal
    modal setup

Usage:
    modal run run_gpu.py                          # full validation_images benchmark (60s timeout)
    modal run run_gpu.py --no-benchmark           # single default image
    modal run run_gpu.py --image-path assets/validation_images/000_bonnet.jpg
    modal run run_gpu.py --timeout-seconds 45 --post-reserve-ms 2000
    modal run run_gpu.py --image-dir assets/validation_images --json-out assets/analysis/gpu_run.json
    modal run run_gpu.py --list-images

CUDA time limits (mirrors production miner):
    attack_budget_ms = timeout_seconds * 1000 - post_reserve_ms
    Attack search stops early on CUDA and returns the best valid candidate found.
    post_reserve_ms is reserved for PNG shrink, encode, and response overhead.

Change GPU type by editing the ``gpu=`` argument on ``run_forward_test`` (e.g. "A10G").
"""

from __future__ import annotations

import json
import os
import subprocess
import statistics
from pathlib import Path
from typing import Any

import modal

PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = "/root/subnet-26"
REMOTE_JSON = f"{REMOTE_ROOT}/assets/analysis/benchmark_gpu.json"
DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_POST_RESERVE_MS = 2500.0

IGNORE = [
    ".git/**",
    "**/__pycache__/**",
    "**/*.pyc",
    ".venv/**",
    "venv/**",
    "**/.pytest_cache/**",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy>=1.26.0",
        "pillow>=10.0.0",
        "bittensor>=8.0.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_dir(
        str(PROJECT_ROOT),
        remote_path=REMOTE_ROOT,
        ignore=IGNORE,
    )
)

app = modal.App("perturb-gpu-benchmark", image=image)


def _attack_budget_ms(timeout_seconds: float, post_reserve_ms: float) -> float:
    return max(1000.0, float(timeout_seconds) * 1000.0 - float(post_reserve_ms))


def _build_run_env(post_reserve_ms: float) -> dict[str, str]:
    env = os.environ.copy()
    env["PERTURB_MINER_POST_RESERVE_MS"] = str(int(post_reserve_ms))
    env["PERTURB_MINER_TIMEOUT_SECONDS"] = env.get("PERTURB_MINER_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
    return env


def _build_test_cmd(
    image_path: str,
    image_dir: str,
    epsilon: float,
    min_delta: float,
    timeout_seconds: int,
    list_images: bool,
    json_out: str,
    save_output_dir: str,
) -> list[str]:
    cmd = ["python", "scripts/test_miner_forward.py"]
    if list_images:
        cmd.append("--list-images")
    if image_path.strip():
        cmd.extend(["--image-path", image_path.strip()])
    if image_dir.strip():
        cmd.extend(["--image-dir", image_dir.strip()])
    cmd.extend(
        [
            "--epsilon",
            str(epsilon),
            "--min-delta",
            str(min_delta),
            "--timeout-seconds",
            str(timeout_seconds),
        ]
    )
    if json_out.strip():
        cmd.extend(["--json-out", json_out.strip()])
    if save_output_dir.strip():
        cmd.extend(["--save-output-dir", save_output_dir.strip()])
    return cmd


def _print_gpu_info() -> None:
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("nvidia-smi not found")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"torch.cuda.device_count={torch.cuda.device_count()}")
            print(f"torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}")
        else:
            print("torch.cuda.is_available()=False")
    except Exception as exc:
        print(f"Could not query torch CUDA: {exc}")


def _print_time_config(timeout_seconds: float, post_reserve_ms: float, device: str) -> None:
    attack_budget = _attack_budget_ms(timeout_seconds, post_reserve_ms)
    cuda = device.startswith("cuda") or device == "unknown"
    print("\n=== CUDA time limit config ===")
    print(f"timeout_seconds={timeout_seconds}")
    print(f"post_reserve_ms={post_reserve_ms:.0f}")
    print(f"attack_budget_ms={attack_budget:.0f}")
    print(f"cuda_deadline_active={cuda}")


def _speed_score(elapsed_ms: float, timeout_seconds: float) -> float:
    limit_ms = float(timeout_seconds) * 1000.0
    if limit_ms <= 0:
        return 0.0
    return 1.0 - min(elapsed_ms / limit_ms, 1.0)


def _enrich_timeout_summary(
    payload: dict[str, Any],
    timeout_seconds: float,
    post_reserve_ms: float,
) -> dict[str, Any]:
    results = payload.get("results") or []
    limit_ms = float(timeout_seconds) * 1000.0
    attack_budget_ms = _attack_budget_ms(timeout_seconds, post_reserve_ms)
    device = str(payload.get("device", "unknown"))
    cuda_deadline = device.startswith("cuda")

    elapsed_values = [float(r["elapsed_ms"]) for r in results if r.get("elapsed_ms") is not None]
    within_timeout = sum(1 for ms in elapsed_values if ms <= limit_ms)
    over_timeout = len(elapsed_values) - within_timeout
    speed_scores = [_speed_score(ms, timeout_seconds) for ms in elapsed_values]

    passed = [r for r in results if r.get("passed")]
    passed_elapsed = [float(r["elapsed_ms"]) for r in passed if r.get("elapsed_ms") is not None]
    passed_speed = [_speed_score(ms, timeout_seconds) for ms in passed_elapsed]

    payload["time_config"] = {
        "timeout_seconds": timeout_seconds,
        "post_reserve_ms": post_reserve_ms,
        "attack_budget_ms": attack_budget_ms,
        "cuda_deadline_enabled": cuda_deadline,
        "env": {
            "PERTURB_MINER_POST_RESERVE_MS": str(int(post_reserve_ms)),
        },
    }
    payload["timeout_summary"] = {
        "within_timeout": within_timeout,
        "over_timeout": over_timeout,
        "within_timeout_pct": (100.0 * within_timeout / len(elapsed_values)) if elapsed_values else None,
        "elapsed_ms_mean": statistics.mean(elapsed_values) if elapsed_values else None,
        "elapsed_ms_median": statistics.median(elapsed_values) if elapsed_values else None,
        "elapsed_ms_max": max(elapsed_values) if elapsed_values else None,
        "elapsed_ms_p95": (
            sorted(elapsed_values)[max(0, int(len(elapsed_values) * 0.95) - 1)] if elapsed_values else None
        ),
        "speed_score_mean": statistics.mean(speed_scores) if speed_scores else None,
        "speed_score_mean_passed": statistics.mean(passed_speed) if passed_speed else None,
        "attack_budget_exceeded": sum(1 for ms in elapsed_values if ms > attack_budget_ms),
    }
    return payload


def _print_benchmark_summary(payload: dict[str, Any]) -> None:
    summary = payload.get("summary") or {}
    timeout_summary = payload.get("timeout_summary") or {}
    time_config = payload.get("time_config") or {}
    passed = payload.get("passed", 0)
    total = payload.get("total", 0)
    device = payload.get("device", "unknown")
    mean_score = summary.get("perturbation_score_mean")
    min_score = summary.get("perturbation_score_min")
    max_score = summary.get("perturbation_score_max")

    print("\n=== Benchmark summary ===")
    print(f"device={device}")
    print(f"passed={passed}/{total}")
    if mean_score is not None:
        print(f"perturbation_score mean={mean_score:.6f} min={min_score:.6f} max={max_score:.6f}")

    if timeout_summary:
        within = timeout_summary.get("within_timeout", 0)
        over = timeout_summary.get("over_timeout", 0)
        print(
            f"timeout compliance: within={within} over={over} "
            f"pct={timeout_summary.get('within_timeout_pct', 0):.1f}%"
        )
        if timeout_summary.get("elapsed_ms_mean") is not None:
            print(
                f"elapsed_ms mean={timeout_summary['elapsed_ms_mean']:.0f} "
                f"median={timeout_summary['elapsed_ms_median']:.0f} "
                f"p95={timeout_summary['elapsed_ms_p95']:.0f} "
                f"max={timeout_summary['elapsed_ms_max']:.0f}"
            )
        if timeout_summary.get("speed_score_mean_passed") is not None:
            print(f"speed_score_mean_passed={timeout_summary['speed_score_mean_passed']:.4f}")

    if time_config:
        print(
            f"attack_budget_ms={time_config.get('attack_budget_ms')} "
            f"attack_budget_exceeded={timeout_summary.get('attack_budget_exceeded', 0)}"
        )


@app.function(gpu="T4", timeout=3600)
def run_forward_test(
    image_path: str = "",
    image_dir: str = "assets/validation_images",
    epsilon: float = 0.03,
    min_delta: float = 0.003,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    post_reserve_ms: float = DEFAULT_POST_RESERVE_MS,
    list_images: bool = False,
    json_out: str = REMOTE_JSON,
    save_output_dir: str = "",
) -> dict[str, Any] | None:
    _print_gpu_info()
    _print_time_config(timeout_seconds=float(timeout_seconds), post_reserve_ms=post_reserve_ms, device="cuda")

    remote_json = json_out.strip() or REMOTE_JSON
    Path(remote_json).parent.mkdir(parents=True, exist_ok=True)

    cmd = _build_test_cmd(
        image_path=image_path,
        image_dir=image_dir,
        epsilon=epsilon,
        min_delta=min_delta,
        timeout_seconds=timeout_seconds,
        list_images=list_images,
        json_out=remote_json,
        save_output_dir=save_output_dir,
    )
    run_env = _build_run_env(post_reserve_ms=post_reserve_ms)
    run_env["PERTURB_MINER_TIMEOUT_SECONDS"] = str(timeout_seconds)
    print("Running:", " ".join(cmd))
    print(
        "Env:",
        f"PERTURB_MINER_POST_RESERVE_MS={run_env['PERTURB_MINER_POST_RESERVE_MS']}",
        f"PERTURB_MINER_TIMEOUT_SECONDS={run_env['PERTURB_MINER_TIMEOUT_SECONDS']}",
    )
    subprocess.run(cmd, cwd=REMOTE_ROOT, check=True, env=run_env)

    if list_images:
        return None

    payload = json.loads(Path(remote_json).read_text(encoding="utf-8"))
    payload = _enrich_timeout_summary(
        payload,
        timeout_seconds=float(timeout_seconds),
        post_reserve_ms=post_reserve_ms,
    )
    Path(remote_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_benchmark_summary(payload)
    return payload


@app.local_entrypoint()
def main(
    image_path: str = "",
    image_dir: str = "",
    epsilon: float = 0.03,
    min_delta: float = 0.003,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    post_reserve_ms: float = DEFAULT_POST_RESERVE_MS,
    list_images: bool = False,
    benchmark: bool = True,
    json_out: str = "",
    save_output_dir: str = "",
) -> None:
    if benchmark and not list_images:
        if not image_path.strip() and not image_dir.strip():
            image_dir = "assets/validation_images"
        timeout_seconds = max(int(timeout_seconds), DEFAULT_TIMEOUT_SECONDS)

    payload = run_forward_test.remote(
        image_path=image_path,
        image_dir=image_dir,
        epsilon=epsilon,
        min_delta=min_delta,
        timeout_seconds=timeout_seconds,
        post_reserve_ms=post_reserve_ms,
        list_images=list_images,
        json_out=json_out.strip() or REMOTE_JSON,
        save_output_dir=save_output_dir,
    )

    if payload is None:
        return

    local_json = json_out.strip() or (
        str(PROJECT_ROOT / "assets" / "analysis" / "benchmark_gpu.json") if benchmark else ""
    )
    if local_json:
        out_path = Path(local_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote local JSON to {out_path}")
