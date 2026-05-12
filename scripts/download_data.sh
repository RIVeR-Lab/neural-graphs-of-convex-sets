#!/usr/bin/env bash
# Download training datasets from a Google Drive Training Dataset folder.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FOLDER_ID="${DATA_DRIVE_FOLDER_ID:-1L1zp8i1pfKtUGlzk26C3Z8RFUzr7A8Bn}"
TMP_DIR="${TMP_DIR:-/tmp/neural_gcs_data_download}"
DOMAIN_FILTER=""
BODY_FILTER=""

usage() {
    cat <<'EOF'
Usage: bash scripts/download_data.sh [OPTIONS]

Download training datasets into the repo tree.

Options:
  --folder-id ID       Google Drive folder id (default: Training Dataset folder)
  --folder-url URL     Google Drive folder share URL
  --domain NAME        quadrotor | manipulation | planning_through_contact
  --body NAME          sugar_box | tee (planning_through_contact only)
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --folder-id) FOLDER_ID="$2"; shift 2 ;;
        --folder-url)
            FOLDER_ID="$(python3 - "$2" <<'PY'
import re
import sys

match = re.search(r"/folders/([^/?]+)", sys.argv[1])
if match is None:
    raise SystemExit(f"Could not parse Google Drive folder id from: {sys.argv[1]}")
print(match.group(1))
PY
)"
            shift 2
            ;;
        --domain) DOMAIN_FILTER="$2"; shift 2 ;;
        --body) BODY_FILTER="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

python3 - <<PY
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import gdown  # type: ignore
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown  # type: ignore

repo_root = Path("${REPO_ROOT}")
folder_id = "${FOLDER_ID}"
tmp_dir = Path("${TMP_DIR}")
domain_filter = "${DOMAIN_FILTER}" or None
body_filter = "${BODY_FILTER}" or None

mappings = [
  ("quadrotor/convex/quadrotor_gcs_convex.h5", "quadrotor/dataset/quadrotor_gcs_convex.h5", "quadrotor", None),
  ("quadrotor/nonconvex/quadrotor_gcs_nonconvex.h5", "quadrotor/dataset/quadrotor_gcs_nonconvex.h5", "quadrotor", None),
  ("manipulation/convex/manipulation_gcs_convex.h5", "manipulation/dataset/manipulation_gcs_convex.h5", "manipulation", None),
  ("manipulation/nonconvex/manipulation_gcs_nonconvex.h5", "manipulation/dataset/manipulation_gcs_nonconvex.h5", "manipulation", None),
  ("manipulation/iris/IRIS.reg", "manipulation/data/IRIS.reg", "manipulation", None),
  ("planning_through_contact/sugar_box/gcs_solutions.h5", "planning_through_contact/dataset/data/sugar_box/gcs_solutions.h5", "planning_through_contact", "sugar_box"),
  ("planning_through_contact/sugar_box/global_features.csv", "planning_through_contact/dataset/data/sugar_box/global_features.csv", "planning_through_contact", "sugar_box"),
  ("planning_through_contact/sugar_box/node_features.csv", "planning_through_contact/dataset/data/sugar_box/node_features.csv", "planning_through_contact", "sugar_box"),
  ("planning_through_contact/tee/gcs_solutions.h5", "planning_through_contact/dataset/data/tee/gcs_solutions.h5", "planning_through_contact", "tee"),
  ("planning_through_contact/tee/global_features.csv", "planning_through_contact/dataset/data/tee/global_features.csv", "planning_through_contact", "tee"),
  ("planning_through_contact/tee/node_features.csv", "planning_through_contact/dataset/data/tee/node_features.csv", "planning_through_contact", "tee"),
]

if domain_filter is not None:
    mappings = [m for m in mappings if m[2] == domain_filter]
if body_filter is not None:
    mappings = [m for m in mappings if m[3] == body_filter]
if not mappings:
    raise SystemExit("No files selected. Check --domain / --body filters.")

if tmp_dir.exists():
    shutil.rmtree(tmp_dir)
tmp_dir.mkdir(parents=True)

print("=== Downloading training datasets ===")
print(f"Google Drive folder: {folder_id}")
gdown.download_folder(id=folder_id, output=str(tmp_dir), quiet=False, use_cookies=False)

def find_downloaded(relative_path: str) -> Path:
    suffix = Path(relative_path).parts
    matches = [
        path
        for path in tmp_dir.rglob(Path(relative_path).name)
        if path.is_file() and path.parts[-len(suffix):] == suffix
    ]
    if not matches:
        raise FileNotFoundError(f"Missing file in downloaded folder: {relative_path}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files matched {relative_path}: {matches}")
    return matches[0]

for src_rel, dst_rel, _, _ in mappings:
    src = find_downloaded(src_rel)
    dst = repo_root / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"{src_rel} -> {dst_rel}")

shutil.rmtree(tmp_dir)
print(f"Done. Installed {len(mappings)} training dataset file(s).")
PY
