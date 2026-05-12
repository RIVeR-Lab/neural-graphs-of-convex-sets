#!/usr/bin/env bash
# Download planar pushing training data and pretrained checkpoints.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

bash scripts/download_data.sh --domain planning_through_contact "$@"
bash scripts/download_models.sh --domain planning_through_contact "$@"
