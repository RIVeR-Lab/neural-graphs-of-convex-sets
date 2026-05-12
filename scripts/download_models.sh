#!/usr/bin/env bash
# Download inference checkpoints from a Google Drive Models folder.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FOLDER_ID="${MODELS_DRIVE_FOLDER_ID:-1kNkK0T6jgN9gCZASmhLA3uVY9hpIHat3}"
TMP_DIR="${TMP_DIR:-/tmp/neural_gcs_models_download}"
DOMAIN_FILTER=""

usage() {
    cat <<'EOF'
Usage: bash scripts/download_models.sh [OPTIONS]

Download inference checkpoints into the repo tree.

Options:
  --folder-id ID       Google Drive folder id (default: Trained Models folder)
  --folder-url URL     Google Drive folder share URL
  --domain NAME        quadrotor | manipulation | planning_through_contact
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --folder-id) FOLDER_ID="$2"; shift 2 ;;
        --domain) DOMAIN_FILTER="$2"; shift 2 ;;
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

mappings = [
    ("quadrotor/convex/quadrotor_convex_flow_gnn.ckpt", "quadrotor/checkpoints/quadrotor_convex/quadrotor_convex_flow_gnn.ckpt", "quadrotor"),
    ("quadrotor/convex/quadrotor_convex_ranknet.ckpt", "quadrotor/checkpoints/quadrotor_convex/quadrotor_convex_ranknet.ckpt", "quadrotor"),
    ("quadrotor/nonconvex/quadrotor_nonconvex_flow_gnn.ckpt", "quadrotor/checkpoints/quadrotor_nonconvex/quadrotor_nonconvex_flow_gnn.ckpt", "quadrotor"),
    ("quadrotor/nonconvex/quadrotor_nonconvex_ranknet.ckpt", "quadrotor/checkpoints/quadrotor_nonconvex/quadrotor_nonconvex_ranknet.ckpt", "quadrotor"),
    ("manipulation/convex/manipulation_convex_flow_gnn.ckpt", "manipulation/checkpoints/manipulation_convex/manipulation_convex_flow_gnn.ckpt", "manipulation"),
    ("manipulation/convex/manipulation_convex_ranknet.ckpt", "manipulation/checkpoints/manipulation_convex/manipulation_convex_ranknet.ckpt", "manipulation"),
    ("manipulation/nonconvex/manipulation_nonconvex_flow_gnn.ckpt", "manipulation/checkpoints/manipulation_nonconvex/manipulation_nonconvex_flow_gnn.ckpt", "manipulation"),
    ("manipulation/nonconvex/manipulation_nonconvex_ranknet.ckpt", "manipulation/checkpoints/manipulation_nonconvex/manipulation_nonconvex_ranknet.ckpt", "manipulation"),
    ("planning_through_contact/sugar_box/sugar_box_flow.ckpt", "planning_through_contact/checkpoints/sugar_box/sugar_box_flow.ckpt", "planning_through_contact"),
    ("planning_through_contact/sugar_box/sugar_box_ranker.ckpt", "planning_through_contact/checkpoints/sugar_box/sugar_box_ranker.ckpt", "planning_through_contact"),
    ("planning_through_contact/tee/tee_flow.ckpt", "planning_through_contact/checkpoints/tee/tee_flow.ckpt", "planning_through_contact"),
    ("planning_through_contact/tee/tee_ranker.ckpt", "planning_through_contact/checkpoints/tee/tee_ranker.ckpt", "planning_through_contact"),
]

if domain_filter is not None:
    mappings = [m for m in mappings if m[2] == domain_filter]
if not mappings:
    raise SystemExit("No checkpoints selected. Check --domain filter.")

if tmp_dir.exists():
    shutil.rmtree(tmp_dir)
tmp_dir.mkdir(parents=True)

print("=== Downloading inference checkpoints ===")
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
        raise FileNotFoundError(f"Missing checkpoint in downloaded folder: {relative_path}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files matched {relative_path}: {matches}")
    return matches[0]

for src_rel, dst_rel, _ in mappings:
    src = find_downloaded(src_rel)
    dst = repo_root / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"{src_rel} -> {dst_rel}")

shutil.rmtree(tmp_dir)
print(f"Done. Installed {len(mappings)} inference checkpoint(s).")
PY
