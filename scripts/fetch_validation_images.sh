#!/usr/bin/env bash
# Download ImageNet-100 validation JPGs from Hugging Face into assets/validation_images/.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv. Run: bash ./scripts/setup_common.sh miner"
  exit 1
fi

source .venv/bin/activate
pip install -q datasets huggingface_hub
python scripts/fetch_validation_images.py "$@"

echo "Run: python scripts/test_miner_forward.py --list-images"
