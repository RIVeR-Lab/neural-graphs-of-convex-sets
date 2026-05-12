#!/usr/bin/env bash
# Download pretrained quadrotor artifacts from Google Drive.
#
# Usage:
#   bash scripts/quadrotor_download.sh

set -euo pipefail

FOLDER_URL="https://drive.google.com/drive/folders/1G_4j4bxx_djooq2KYhV3jObGUzDky6SB"
TMP_DIR="/tmp/quadrotor_download"

echo "=== Quadrotor: downloading pretrained artifacts ==="

if ! python -c "import gdown" 2>/dev/null; then
    echo "Installing gdown..."
    pip install -q gdown
fi

rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

echo "Downloading from Google Drive..."
gdown --folder "$FOLDER_URL" -O "$TMP_DIR" --quiet

mkdir -p quadrotor/dataset
mkdir -p quadrotor/checkpoints/quadrotor_gnn
mkdir -p quadrotor/checkpoints/quadrotor_ranknet

mv "$TMP_DIR/quadrotor_gcs_dataset.h5"  quadrotor/dataset/
mv "$TMP_DIR/quadrotor_flow_gnn.ckpt"   quadrotor/checkpoints/quadrotor_gnn/
mv "$TMP_DIR/quadrotor_ranknet.ckpt"    quadrotor/checkpoints/quadrotor_ranknet/

rm -rf "$TMP_DIR"

echo ""
echo "Done. Files saved to:"
echo "  quadrotor/dataset/quadrotor_gcs_dataset.h5"
echo "  quadrotor/checkpoints/quadrotor_gnn/quadrotor_flow_gnn.ckpt"
echo "  quadrotor/checkpoints/quadrotor_ranknet/quadrotor_ranknet.ckpt"
