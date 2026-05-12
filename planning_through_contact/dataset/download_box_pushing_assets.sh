#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRIVE_URL="${DRIVE_URL:-https://drive.google.com/drive/folders/1j9C6yO3iEgct5gvyPnisgnCZ7Ezb8z-4?usp=sharing}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/planning_through_contact/dataset/data/box_pushing}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/checkpoints/gcs_gnn}"

if ! python3 -c "import gdown" >/dev/null 2>&1; then
  python3 -m pip install gdown
fi

tmp_dir="$(mktemp -d)"
download_dir="$tmp_dir/box_pushing"
trap 'rm -rf "$tmp_dir"' EXIT

python3 -m gdown --folder "$DRIVE_URL" -O "$download_dir"

for file in global_features.csv node_features.csv gcs_solutions.h5 box_pushing.ckpt; do
  if [[ ! -f "$download_dir/$file" ]]; then
    echo "Error: missing downloaded file: $file" >&2
    exit 1
  fi
done

mkdir -p "$DATA_DIR" "$CKPT_DIR"
cp "$download_dir"/global_features.csv "$DATA_DIR/global_features.csv"
cp "$download_dir"/node_features.csv "$DATA_DIR/node_features.csv"
cp "$download_dir"/gcs_solutions.h5 "$DATA_DIR/gcs_solutions.h5"
cp "$download_dir"/box_pushing.ckpt "$CKPT_DIR/box_pushing.ckpt"

echo "Downloaded box-pushing dataset to $DATA_DIR"
echo "Downloaded checkpoint to $CKPT_DIR/box_pushing.ckpt"
