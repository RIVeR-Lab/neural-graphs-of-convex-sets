#!/usr/bin/env bash
# Download dataset and checkpoints from Google Drive.
# Requires: pip install gdown
#
# Google Drive folder: https://drive.google.com/drive/folders/1WGXLxlpctM_KHSMEiCVzDsGjrM8w0DvB
#
# Expected layout after download:
#   planning_through_contact/dataset/data/sugar_box/{gcs_solutions.h5,global_features.csv,node_features.csv}
#   planning_through_contact/dataset/data/tee/{gcs_solutions.h5,global_features.csv,node_features.csv}
#   checkpoints/gcs_gnn/sugar_box_flow.ckpt
#   checkpoints/gcs_gnn/tee_flow.ckpt
#   checkpoints/ranknet/sugar_box_ranker.ckpt
#   checkpoints/ranknet/tee_ranker.ckpt

set -euo pipefail

# Resolve repo root (script lives in scripts/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python - <<'EOF'
import subprocess, sys
try:
    import gdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown
EOF

GDRIVE_FOLDER_ID="1WGXLxlpctM_KHSMEiCVzDsGjrM8w0DvB"

echo "Downloading data folder from Google Drive..."
python -c "
import gdown, pathlib
pathlib.Path('_gdrive_tmp').mkdir(exist_ok=True)
gdown.download_folder(
    id='${GDRIVE_FOLDER_ID}',
    output='_gdrive_tmp',
    quiet=False,
    use_cookies=False,
)
"

# --- sugar_box data --------------------------------------------------------
SUGAR_DST="planning_through_contact/dataset/data/sugar_box"
mkdir -p "$SUGAR_DST"
cp -v _gdrive_tmp/box_pushing/sugar_box/gcs_solutions.h5   "$SUGAR_DST/"
cp -v _gdrive_tmp/box_pushing/sugar_box/global_features.csv "$SUGAR_DST/"
cp -v _gdrive_tmp/box_pushing/sugar_box/node_features.csv   "$SUGAR_DST/"

# --- tee data --------------------------------------------------------------
TEE_DST="planning_through_contact/dataset/data/tee"
mkdir -p "$TEE_DST"
cp -v _gdrive_tmp/tee/gcs_solutions.h5    "$TEE_DST/" 2>/dev/null || echo "[skip] tee/gcs_solutions.h5 not found"
cp -v _gdrive_tmp/tee/global_features.csv "$TEE_DST/" 2>/dev/null || echo "[skip] tee/global_features.csv not found"
cp -v _gdrive_tmp/tee/node_features.csv   "$TEE_DST/" 2>/dev/null || echo "[skip] tee/node_features.csv not found"

# --- checkpoints -----------------------------------------------------------
mkdir -p checkpoints/gcs_gnn checkpoints/ranknet

cp -v _gdrive_tmp/box_pushing/sugar_box/gnn.ckpt     checkpoints/gcs_gnn/sugar_box_flow.ckpt
cp -v _gdrive_tmp/box_pushing/sugar_box/ranknet.ckpt checkpoints/ranknet/sugar_box_ranker.ckpt

cp -v _gdrive_tmp/tee/gnn.ckpt     checkpoints/gcs_gnn/tee_flow.ckpt     2>/dev/null || echo "[skip] tee gnn.ckpt not found"
cp -v _gdrive_tmp/tee/ranknet.ckpt checkpoints/ranknet/tee_ranker.ckpt   2>/dev/null || echo "[skip] tee ranknet.ckpt not found"

# --- cleanup ---------------------------------------------------------------
rm -rf _gdrive_tmp
echo ""
echo "Done. Data and checkpoints are in place."
