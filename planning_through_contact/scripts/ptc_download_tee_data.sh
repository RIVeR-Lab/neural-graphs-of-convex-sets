#!/usr/bin/env bash
# Download the tee planar pushing dataset only.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

exec bash scripts/download_data.sh --domain planning_through_contact --body tee "$@"
