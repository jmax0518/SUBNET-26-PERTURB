from __future__ import annotations

import argparse
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

from perturbnet.constants import PROMPTS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "assets" / "test_images"
USER_AGENT = "PerturbSubnetTest/1.0 (local miner test; +https://github.com/0xsigurd/Perturb)"
WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"

# Search terms tuned to return photos that match the saved filename.
SEARCH_TERMS: dict[str, str] = {
    "dog": "golden retriever dog",
    "cat": "domestic cat",
    "car": "sports car",
    "banana": "banana fruit",
    "pizza": "pizza food",
    "bird": "cardinal bird",
    "chair": "wooden chair furniture",
}


def _sanitize_stem(name: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return stem or "image"


def _search_wikimedia_jpg_urls(
    query: str,
    width: int = 480,
    limit: int = 10,
    offset: int = 0,
) -> list[tuple[str, str]]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "gsroffset": str(offset),
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "iiurlwidth": str(width),
    }
    response = requests.get(
        WIKIMEDIA_API,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=20,
    )
    response.raise_for_status()
    time.sleep(0.5)
    pages = response.json().get("query", {}).get("pages", {})
    results: list[tuple[str, str]] = []
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        mime = (info.get("mime") or "").lower()
        thumb = info.get("thumburl") or info.get("url")
        title = page.get("title") or "unknown"
        if thumb and mime.startswith("image/"):
            results.append((thumb, title))
    return results


def _search_wikimedia_jpg_url(query: str, width: int = 480) -> tuple[str, str]:
    results = _search_wikimedia_jpg_urls(query=query, width=width, limit=10, offset=0)
    if not results:
        raise RuntimeError(f"No Wikimedia image found for query: {query!r}")
    return results[0]


def _download_as_jpg(url: str, dest: Path, *, retries: int = 5) -> None:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if response.status_code == 429:
                wait = min(30, 2 ** attempt)
                print(f"  rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            dest.parent.mkdir(parents=True, exist_ok=True)
            image.save(dest, format="JPEG", quality=92)
            time.sleep(1.5)
            return
        except requests.RequestException as exc:
            last_error = exc
            wait = min(30, 2 ** attempt)
            print(f"  download failed ({exc}), retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to download {url}") from last_error


def _search_query_for_prompt(prompt: str) -> str:
    stem = _sanitize_stem(prompt)
    if stem in SEARCH_TERMS:
        return SEARCH_TERMS[stem]
    if prompt in SEARCH_TERMS:
        return SEARCH_TERMS[prompt]
    return prompt


def fetch_named_images(out_dir: Path, names: list[str], width: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out_dir}")

    for name in names:
        query = SEARCH_TERMS.get(name, name)
        dest = out_dir / f"{name}.jpg"
        print(f"[{name}] searching Wikimedia for: {query!r}")
        url, title = _search_wikimedia_jpg_url(query=query, width=width)
        print(f"  source={title}")
        print(f"  url={url}")
        _download_as_jpg(url, dest)
        print(f"  saved={dest}")

    print(f"Done. Downloaded {len(names)} image(s).")
    return 0


def _existing_image_state(out_dir: Path) -> tuple[int, dict[str, int]]:
    per_prompt_counts: dict[str, int] = {}
    numbered = re.compile(r"^(.+)_(\d+)\.jpg$")
    plain = re.compile(r"^(.+)\.jpg$")
    total = 0
    for path in out_dir.glob("*.jpg"):
        total += 1
        match = numbered.match(path.name)
        if match:
            stem, num = match.group(1), int(match.group(2))
            per_prompt_counts[stem] = max(per_prompt_counts.get(stem, 0), num)
            continue
        match = plain.match(path.name)
        if match:
            stem = match.group(1)
            per_prompt_counts[stem] = max(per_prompt_counts.get(stem, 0), 1)
    return total, per_prompt_counts


def _next_image_for_prompt(
    prompt: str,
    *,
    width: int,
    seen_urls: set[str],
    search_offsets: dict[str, int],
    cached_results: dict[str, list[tuple[str, str]]],
    result_cursor: dict[str, int],
) -> tuple[str, str] | None:
    query = _search_query_for_prompt(prompt)
    stem = _sanitize_stem(prompt)

    for _ in range(6):
        cursor = result_cursor.get(stem, 0)
        cache = cached_results.get(stem, [])
        if cursor >= len(cache):
            offset = search_offsets.get(stem, 0)
            batch = _search_wikimedia_jpg_urls(query=query, width=width, limit=20, offset=offset)
            search_offsets[stem] = offset + 20
            if not batch:
                return None
            cache = cached_results.setdefault(stem, []) + batch
            cached_results[stem] = cache

        while cursor < len(cache):
            url, title = cache[cursor]
            result_cursor[stem] = cursor + 1
            if url not in seen_urls:
                return url, title
            cursor += 1

    return None


def fetch_count_images(out_dir: Path, count: int, width: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded, per_prompt_counts = _existing_image_state(out_dir)
    if downloaded >= count:
        print(f"Already have {downloaded} image(s) in {out_dir}")
        return 0

    prompts = list(PROMPTS)
    max_per_category = (count + len(prompts) - 1) // len(prompts)
    print(
        f"Saving {count} image(s) to {out_dir} "
        f"({downloaded} already present, max {max_per_category} per category)"
    )

    seen_urls: set[str] = set()
    search_offsets: dict[str, int] = {}
    cached_results: dict[str, list[tuple[str, str]]] = {}
    result_cursor: dict[str, int] = {}

    while downloaded < count:
        made_progress = False
        for prompt in prompts:
            if downloaded >= count:
                break

            stem = _sanitize_stem(prompt)
            if per_prompt_counts.get(stem, 0) >= max_per_category:
                continue

            found = _next_image_for_prompt(
                prompt,
                width=width,
                seen_urls=seen_urls,
                search_offsets=search_offsets,
                cached_results=cached_results,
                result_cursor=result_cursor,
            )
            if found is None:
                continue

            url, title = found
            seen_urls.add(url)
            per_prompt_counts[stem] = per_prompt_counts.get(stem, 0) + 1
            dest = out_dir / f"{stem}_{per_prompt_counts[stem]:02d}.jpg"
            if dest.exists():
                continue

            print(f"[{downloaded + 1}/{count}] {prompt!r} -> {dest.name}")
            print(f"  source={title}")
            _download_as_jpg(url, dest)
            print(f"  saved={dest}")
            downloaded += 1
            made_progress = True

        if not made_progress:
            break

    if downloaded < count:
        raise RuntimeError(f"Only downloaded {downloaded}/{count} unique images")

    print(f"Done. Downloaded {downloaded} image(s).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download real topic-matched JPG test images from Wikimedia Commons"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: assets/test_images)",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Download one image per base filename (legacy mode)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Download this many diverse images using validator prompts (e.g. --count 100)",
    )
    parser.add_argument("--width", type=int, default=480)
    args = parser.parse_args()

    try:
        out_dir = args.out_dir.resolve()
        if args.count > 0:
            return fetch_count_images(out_dir=out_dir, count=args.count, width=args.width)
        names = list(args.names) if args.names is not None else sorted(SEARCH_TERMS.keys())
        return fetch_named_images(out_dir=out_dir, names=names, width=args.width)
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
