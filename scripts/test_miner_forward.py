#!/usr/bin/env python3
"""Local smoke test for PerturbMiner.forward() without wallet/subtensor."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image

from neurons.miner import PerturbMiner
from perturbnet.attack import (
    DEFAULT_MAX_LINF_DELTA,
    evaluate_candidate,
    linf_norm,
)
from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import load_efficientnet_v2_l, predict_index, predict_label
from perturbnet.protocol import AttackChallenge

DEFAULT_IMAGE_DIRS = (
    PROJECT_ROOT / "assets" / "test_images",
    PROJECT_ROOT / "assets",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def _build_miner(device: torch.device) -> PerturbMiner:
    """Lightweight miner stub: model + device only, no chain connection."""
    miner = object.__new__(PerturbMiner)
    miner.device = device
    miner.model = load_efficientnet_v2_l(device)
    return miner


def _load_image_chw(image_path: Path, device: torch.device) -> torch.Tensor:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return tensor.contiguous().to(device)


def discover_images(image_path: str | None, image_dir: str | None) -> list[Path]:
    if image_path:
        return [Path(image_path).expanduser().resolve()]

    dirs: list[Path] = []
    if image_dir:
        dirs.append(Path(image_dir).expanduser().resolve())
    else:
        dirs.extend(DEFAULT_IMAGE_DIRS)

    found: list[Path] = []
    for directory in dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                found.append(path.resolve())

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def list_available_images(image_path: str | None, image_dir: str | None) -> int:
    images = discover_images(image_path=image_path, image_dir=image_dir)
    if not images:
        print("No images found.")
        print("Add files under assets/test_images/ or pass --image-path / --image-dir")
        return 1

    print(f"Found {len(images)} image(s):")
    for path in images:
        print(f"  {path}")
    return 0


async def run_forward_test_on_image(
    miner: PerturbMiner,
    clean: torch.Tensor,
    image_path: Path,
    epsilon: float,
    min_delta: float,
    timeout_seconds: int,
    save_output_dir: Path | None,
) -> bool:
    true_label = predict_label(miner.model, clean)
    true_index = predict_index(model=miner.model, image_chw=clean)
    effective_max_delta = min(epsilon, DEFAULT_MAX_LINF_DELTA)

    synapse = AttackChallenge(
        task_id=f"local-forward-test-{image_path.stem}",
        prompt=image_path.stem,
        clean_image_b64=encode_image_b64(clean),
        true_label=true_label,
        epsilon=epsilon,
        norm_type="Linf",
        min_delta=min_delta,
        timeout_seconds=timeout_seconds,
    )

    print(f"\n=== {image_path.name} ===", flush=True)
    print(
        f"shape={tuple(clean.shape)} true_label={true_label!r} "
        f"epsilon={epsilon} min_delta={min_delta} effective_max_delta={effective_max_delta}",
        flush=True,
    )
    print("Running miner.forward()...", flush=True)

    started = time.perf_counter()
    result = await miner.forward(synapse)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if result.perturbed_image_b64 is None:
        print(f"FAIL: forward() returned no perturbed_image_b64 (elapsed_ms={elapsed_ms})")
        return False

    adv = decode_image_b64(result.perturbed_image_b64).to(clean.device)
    new_label = predict_label(miner.model, adv)
    norm = linf_norm(clean, adv)
    changed = result.perturbed_image_b64 != synapse.clean_image_b64

    candidate = evaluate_candidate(
        clean=clean,
        adv=adv,
        source_index=true_index,
        min_delta=min_delta,
        effective_max_delta=effective_max_delta,
        model=miner.model,
        use_png=False,
    )

    print(f"elapsed_ms={elapsed_ms} timeout_seconds={timeout_seconds}")
    print(f"new_label={new_label!r} linf_norm={norm:.6f} changed_image={changed}")

    if candidate is None:
        flipped = predict_index(model=miner.model, image_chw=adv) != true_index
        valid_norm = min_delta <= norm <= effective_max_delta + 1e-6
        print(f"validator_gates=FAIL flipped={flipped} valid_norm={valid_norm}")
        print("FAIL")
        return False

    print(
        f"validator_gates=PASS linf={candidate.linf:.6f} rmse={candidate.rmse:.6f} "
        f"ssim={candidate.ssim:.4f} psnr_db={candidate.psnr_db:.2f} "
        f"perturbation_score={candidate.perturbation_score:.6f}"
    )

    if save_output_dir is not None:
        save_output_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_output_dir / f"{image_path.stem}_perturbed.png"
        arr = (adv.detach().cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(arr, mode="RGB").save(out_path)
        print(f"saved={out_path}")

    if elapsed_ms > timeout_seconds * 1000:
        print(f"WARN: exceeded timeout ({elapsed_ms}ms > {timeout_seconds * 1000}ms)")

    print("PASS")
    return True


async def run_forward_tests(
    image_path: str | None,
    image_dir: str | None,
    epsilon: float,
    min_delta: float,
    timeout_seconds: int,
    save_output_dir: Path | None,
) -> int:
    images = discover_images(image_path=image_path, image_dir=image_dir)
    if not images:
        print("No images found.")
        print("Put files in assets/test_images/ or use --image-path / --image-dir")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    print(f"testing {len(images)} image(s)", flush=True)
    print("Loading EfficientNetV2-L...", flush=True)

    miner = _build_miner(device)
    print("Model ready.", flush=True)
    passed = 0
    for path in images:
        try:
            clean = _load_image_chw(path, device)
        except Exception as exc:
            print(f"\n=== {path.name} ===")
            print(f"FAIL: could not load image: {exc}")
            continue

        ok = await run_forward_test_on_image(
            miner=miner,
            clean=clean,
            image_path=path,
            epsilon=epsilon,
            min_delta=min_delta,
            timeout_seconds=timeout_seconds,
            save_output_dir=save_output_dir,
        )
        if ok:
            passed += 1

    print(f"\nSummary: {passed}/{len(images)} passed")
    return 0 if passed == len(images) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PerturbMiner.forward() locally on image(s) without Bittensor"
    )
    parser.add_argument("--image-path", type=str, default="", help="Run on one image file")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="",
        help="Run on all images in a directory (default: assets/test_images and assets/)",
    )
    parser.add_argument("--list-images", action="store_true", help="List discovered image files and exit")
    parser.add_argument("--epsilon", type=float, default=0.03, help="Linf epsilon bound (default: 0.03)")
    parser.add_argument("--min-delta", type=float, default=0.003, help="Minimum Linf perturbation (default: 0.003)")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=15,
        help="Challenge timeout used for comparison only (default: 15)",
    )
    parser.add_argument(
        "--save-output-dir",
        type=str,
        default="",
        help="If set, write perturbed PNGs to this directory",
    )
    args = parser.parse_args()

    image_path = args.image_path.strip() or None
    image_dir = args.image_dir.strip() or None
    save_output_dir = Path(args.save_output_dir).expanduser().resolve() if args.save_output_dir.strip() else None

    if args.list_images:
        return list_available_images(image_path=image_path, image_dir=image_dir)

    return asyncio.run(
        run_forward_tests(
            image_path=image_path,
            image_dir=image_dir,
            epsilon=args.epsilon,
            min_delta=args.min_delta,
            timeout_seconds=args.timeout_seconds,
            save_output_dir=save_output_dir,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
