#!/usr/bin/env bash
# Download manipulation training data and pretrained checkpoints.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

bash scripts/download_data.sh --domain manipulation "$@"
bash scripts/download_models.sh --domain manipulation "$@"
