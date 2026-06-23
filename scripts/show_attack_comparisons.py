from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from neurons.miner import PerturbMiner
from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import LABELS, load_efficientnet_v2_l, predict_index, predict_label
from perturbnet.constants import VALIDATION_IMAGES_RELATIVE_DIR
from perturbnet.protocol import AttackChallenge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIRS = (
    PROJECT_ROOT / VALIDATION_IMAGES_RELATIVE_DIR,
    PROJECT_ROOT / "assets",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
DEFAULT_OUT_DIR = PROJECT_ROOT / "assets" / "test_outputs" / "comparisons"


def _build_miner(device: torch.device) -> PerturbMiner:
    miner = object.__new__(PerturbMiner)
    miner.device = device
    miner.model = load_efficientnet_v2_l(device)
    return miner


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


def _chw_to_pil(image_chw: torch.Tensor) -> Image.Image:
    arr = (image_chw.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _resize_for_display(image: Image.Image, max_side: int) -> Image.Image:
    if max(image.size) <= max_side:
        return image
    return image.copy().thumbnail((max_side, max_side), Image.Resampling.LANCZOS) or image


def _label_name(index: int) -> str:
    if 0 <= index < len(LABELS):
        return LABELS[index]
    return str(index)


def _make_comparison_panel(
    clean: Image.Image,
    attacked: Image.Image,
    diff: Image.Image,
    caption: str,
) -> Image.Image:
    gap = 8
    header_h = 36
    width = clean.width + attacked.width + diff.width + gap * 2
    height = max(clean.height, attacked.height, diff.height) + header_h
    panel = Image.new("RGB", (width, height), color=(24, 24, 24))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), caption, fill=(240, 240, 240))

    y = header_h
    panel.paste(clean, (0, y))
    panel.paste(attacked, (clean.width + gap, y))
    panel.paste(diff, (clean.width + attacked.width + gap * 2, y))

    labels = ("Original", "Attacked", "Diff x20")
    xs = (0, clean.width + gap, clean.width + attacked.width + gap * 2)
    for text, x in zip(labels, xs):
        draw.text((x + 4, height - 18), text, fill=(200, 200, 200))

    return panel


def _difference_image(clean: torch.Tensor, adv: torch.Tensor, amplify: float = 20.0) -> Image.Image:
    delta = (adv - clean).abs().max(dim=0).values
    arr = (delta * amplify).clamp(0.0, 1.0).cpu().numpy()
    heat = (arr * 255.0).round().astype(np.uint8)
    rgb = np.stack([heat, np.zeros_like(heat), heat], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


async def _attack_image(
    miner: PerturbMiner,
    image_path: Path,
    epsilon: float,
    min_delta: float,
    max_side: int,
) -> tuple[torch.Tensor, torch.Tensor, str, str, float]:
    image = Image.open(image_path).convert("RGB")
    if max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    clean = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).to(miner.device)
    true_label = predict_label(miner.model, clean)
    true_index = predict_index(miner.model, image_chw=clean)

    synapse = AttackChallenge(
        task_id=f"compare-{image_path.stem}",
        prompt=image_path.stem,
        clean_image_b64=encode_image_b64(clean),
        true_label=true_label,
        epsilon=epsilon,
        norm_type="Linf",
        min_delta=min_delta,
        timeout_seconds=15,
    )
    result = await miner.forward(synapse)
    if not result.perturbed_image_b64:
        raise RuntimeError("forward() returned empty perturbed_image_b64")

    adv = decode_image_b64(result.perturbed_image_b64).to(miner.device)
    new_label = predict_label(miner.model, adv)
    norm = float((adv - clean).abs().max().item())
    return clean, adv, true_label, new_label, norm


async def run(args: argparse.Namespace) -> int:
    images = discover_images(image_path=args.image_path or None, image_dir=args.image_dir or None)
    if args.skip_large:
        images = [p for p in images if p.name != "dog_1.jpg"]
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        print("No images found.")
        return 1

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    miner = _build_miner(device)

    print(f"device={device}")
    print(f"output={out_dir}")
    print(f"images={len(images)}")

    for path in images:
        print(f"processing {path.name} ...")
        clean, adv, true_label, new_label, norm = await _attack_image(
            miner=miner,
            image_path=path,
            epsilon=args.epsilon,
            min_delta=args.min_delta,
            max_side=args.max_side,
        )

        clean_pil = _chw_to_pil(clean)
        adv_pil = _chw_to_pil(adv)
        diff_pil = _difference_image(clean, adv, amplify=args.diff_amplify)
        caption = f"{path.name} | {true_label!r} -> {new_label!r} | Linf={norm:.5f}"

        stem = path.stem
        clean_pil.save(out_dir / f"{stem}_original.jpg", quality=92)
        adv_pil.save(out_dir / f"{stem}_attacked.jpg", quality=92)
        diff_pil.save(out_dir / f"{stem}_diff.jpg", quality=92)

        panel = _make_comparison_panel(clean_pil, adv_pil, diff_pil, caption)
        panel.save(out_dir / f"{stem}_compare.jpg", quality=92)
        print(f"  saved {stem}_compare.jpg")

    print(f"Done. Open files in {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Save original vs attacked image comparisons")
    parser.add_argument("--image-path", type=str, default="")
    parser.add_argument("--image-dir", type=str, default="")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--min-delta", type=float, default=0.003)
    parser.add_argument("--max-side", type=int, default=512, help="Resize long side for faster preview")
    parser.add_argument("--diff-amplify", type=float, default=20.0)
    parser.add_argument("--max-images", type=int, default=0, help="0 = all discovered images")
    parser.add_argument("--skip-large", action="store_true", default=True)
    parser.add_argument("--include-large", action="store_true", help="Include dog_1.jpg and other large files")
    args = parser.parse_args()
    if args.include_large:
        args.skip_large = False
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
