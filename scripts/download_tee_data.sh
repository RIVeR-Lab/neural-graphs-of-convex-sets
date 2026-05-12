#!/usr/bin/env bash
# Download the tee planar-pushing dataset (gcs_solutions.h5, global_features.csv,
# node_features.csv) from Google Drive. Use this when you want to train the tee
# model from scratch — it does NOT pull pretrained checkpoints.
#
# Requires: pip install gdown
#
# Google Drive folder: https://drive.google.com/drive/folders/1b0I7oo_K53031Ldv_wbnhCEygDQYeAX5
#
# After download:
#   planning_through_contact/dataset/data/tee/{gcs_solutions.h5,global_features.csv,node_features.csv}

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python - <<'EOF'
import subprocess, sys
try:
    import gdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
EOF

GDRIVE_FOLDER_ID="1b0I7oo_K53031Ldv_wbnhCEygDQYeAX5"
TMP_DIR="_gdrive_tee_tmp"
DST="planning_through_contact/dataset/data/tee"

mkdir -p "$DST"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

echo "Downloading tee dataset folder from Google Drive..."
python - <<EOF
import gdown
gdown.download_folder(
    id='${GDRIVE_FOLDER_ID}',
    output='${TMP_DIR}',
    quiet=False,
    use_cookies=False,
)
EOF

# gdown creates a subdirectory named after the source folder (likely "tee"),
# but layout can vary — scan recursively for the three expected files.
echo ""
echo "Copying files into $DST/ ..."
for fname in gcs_solutions.h5 global_features.csv node_features.csv; do
    src="$(find "$TMP_DIR" -type f -name "$fname" | head -1)"
    if [[ -z "$src" ]]; then
        echo "[error] $fname not found in download." >&2
        exit 1
    fi
    cp -v "$src" "$DST/$fname"
done

rm -rf "$TMP_DIR"
echo ""
echo "Done. Tee dataset is at: $DST/"
ls -la "$DST"