from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset
from PIL import Image

from perturbnet.constants import (
    IMAGENET100_HF_DATASET,
    IMAGENET100_MANIFEST_FILENAME,
    VALIDATION_IMAGE_COUNT,
    VALIDATION_IMAGES_RELATIVE_DIR,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / VALIDATION_IMAGES_RELATIVE_DIR


def _sanitize_stem(name: str) -> str:
    primary = name.split(",", 1)[0].strip().lower()
    stem = re.sub(r"[^a-z0-9]+", "_", primary).strip("_")
    return stem or "class"


def _class_names(dataset_name: str, split: str) -> list[str]:
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    feature = dataset.features["label"]
    return list(feature.names)


def fetch_imagenet100_validation_images(
    out_dir: Path,
    *,
    count: int,
    dataset_name: str,
    split: str,
    per_class: int,
    jpeg_quality: int,
    force: bool,
) -> int:
    """Download real ImageNet-100 validation images from Hugging Face (CMC subset)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = _class_names(dataset_name=dataset_name, split=split)
    num_classes = len(class_names)
    if count > num_classes * per_class:
        raise ValueError(
            f"Requested {count} images but ImageNet-100 has only {num_classes} classes "
            f"at {per_class} image(s) per class"
        )

    manifest_path = out_dir / IMAGENET100_MANIFEST_FILENAME
    existing = sorted(out_dir.glob("*.jpg"))
    if existing and not force and manifest_path.is_file():
        if len(existing) >= count:
            print(f"Already have {len(existing)} ImageNet-100 image(s) in {out_dir}")
            return 0

    for path in existing:
        path.unlink()
    if manifest_path.exists():
        manifest_path.unlink()

    print(
        f"Downloading ImageNet-100 validation images from {dataset_name!r} "
        f"({num_classes} classes, target={count}, per_class={per_class})"
    )

    stream = load_dataset(dataset_name, split=split, streaming=True)
    per_label_counts: dict[int, int] = {}
    manifest: list[dict[str, object]] = []
    saved = 0

    for row in stream:
        label = int(row["label"])
        if label < 0 or label >= num_classes:
            continue
        if per_label_counts.get(label, 0) >= per_class:
            continue

        per_label_counts[label] = per_label_counts.get(label, 0) + 1
        class_name = class_names[label]
        stem = _sanitize_stem(class_name)
        suffix = f"_{per_label_counts[label]:02d}" if per_class > 1 else ""
        filename = f"{label:03d}_{stem}{suffix}.jpg"
        dest = out_dir / filename

        image = row["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL image for label={label}, got {type(image)!r}")
        rgb = image.convert("RGB")
        rgb.save(dest, format="JPEG", quality=jpeg_quality)

        manifest.append(
            {
                "filename": filename,
                "label": label,
                "class_name": class_name,
                "split": split,
                "dataset": dataset_name,
                "image_index_in_class": per_label_counts[label],
            }
        )
        saved += 1
        print(f"[{saved}/{count}] label={label:03d} {class_name!r} -> {filename}")

        if saved >= count:
            break

    if saved < count:
        raise RuntimeError(f"Only saved {saved}/{count} ImageNet-100 validation images")

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Done. Saved {saved} ImageNet-100 validation image(s) to {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download ImageNet-100 validation images from Hugging Face (clane9/imagenet-100)"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {VALIDATION_IMAGES_RELATIVE_DIR})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=VALIDATION_IMAGE_COUNT,
        help=f"Number of images to save (default: {VALIDATION_IMAGE_COUNT}, one per class)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=IMAGENET100_HF_DATASET,
        help=f"Hugging Face dataset id (default: {IMAGENET100_HF_DATASET})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=("validation", "train"),
        help="Dataset split to stream from (default: validation)",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=1,
        help="Images to save per class before moving on (default: 1)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing images even if the output directory is already populated",
    )
    parser.add_argument("--jpeg-quality", type=int, default=92)
    args = parser.parse_args()

    if args.count <= 0:
        print("--count must be positive", file=sys.stderr)
        return 1
    if args.per_class <= 0:
        print("--per-class must be positive", file=sys.stderr)
        return 1

    try:
        return fetch_imagenet100_validation_images(
            out_dir=args.out_dir.resolve(),
            count=args.count,
            dataset_name=args.dataset.strip(),
            split=args.split.strip(),
            per_class=args.per_class,
            jpeg_quality=args.jpeg_quality,
            force=args.force,
        )
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
