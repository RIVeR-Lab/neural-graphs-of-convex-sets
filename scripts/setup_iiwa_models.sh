#!/usr/bin/env bash
# Download IIWA models required for the shelf GCS demo.
#
# Drake's pip wheel does not ship manipulation/models; this script vendors
# iiwa_description from RobotLocomotion/models into manipulation/vendor/drake/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/manipulation/vendor/drake/manipulation/models"
TMP="${REPO_ROOT}/.tmp_robotlocomotion_models"

echo "=== IIWA model setup ==="
echo "  Destination: ${DEST}/iiwa_description"

if [[ -f "${DEST}/iiwa_description/urdf/iiwa14_spheres_collision.urdf" \
   && -f "${DEST}/wsg_50_description/meshes/wsg_body.gltf" ]]; then
    echo "  Already installed."
    exit 0
fi

rm -rf "${TMP}"
git clone --depth 1 https://github.com/RobotLocomotion/models.git "${TMP}"

mkdir -p "${DEST}"
for pkg in iiwa_description wsg_50_description; do
    rm -rf "${DEST}/${pkg}"
    cp -r "${TMP}/${pkg}" "${DEST}/"
done
rm -rf "${TMP}"

echo "  Done."
echo "  URDF: ${DEST}/iiwa_description/urdf/iiwa14_spheres_collision.urdf"
