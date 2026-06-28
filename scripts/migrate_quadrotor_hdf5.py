#!/usr/bin/env python3
"""
Migrate existing quadrotor GCS HDF5 files to store raw region halfspaces (A, b).

The original collectors stored only precomputed 9-d AABB ``node_features``.
PointNet-over-facets training needs the raw halfspaces instead. Quadrotor
regions are deterministic from ``building_seed`` + ``grid_size``, so we
regenerate the convex decomposition (no GCS solves) and write, per instance:

  samples/<id>/regions/<i:04d>/A   float32 (m, 3)
  samples/<id>/regions/<i:04d>/b   float32 (m,)

Region index i matches node "v{i}" (convex) and "Subgraph0: Region{i}" (nonlinear).

Usage:
  python scripts/migrate_quadrotor_hdf5.py \
      quadrotor/dataset/quadrotor_gcs_convex.h5 \
      quadrotor/dataset/quadrotor_gcs_nonlinear.h5
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from tqdm import tqdm

logging.getLogger("drake").setLevel(logging.WARNING)


def _load_generalization_module():
    spec = importlib.util.spec_from_file_location(
        "collect_quadrotor_generalization",
        REPO_ROOT / "scripts" / "collect_quadrotor_generalization.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _regions_for(gen, grid_size: int, building_seed: int, sdf_path: Path):
    """Regenerate the convex region decomposition only (skip the O(N^2) adjacency
    LPs in generate_scene, which we don't need)."""
    from quadrotor.building_generation import (
        compile_sdf,
        diagonal_start_goal,
        generate_grid_world,
    )

    shape = (grid_size, grid_size)
    grid_start, grid_goal = diagonal_start_goal(shape)
    grid, indoor_edges, outdoor_edges = generate_grid_world(
        shape=shape,
        start=grid_start,
        goal=grid_goal,
        seed=building_seed,
        grow_probability=gen.MOSTLY_INDOOR_GROW_PROBABILITY,
        start_indoor=gen.MOSTLY_INDOOR_START_INDOOR,
    )
    return compile_sdf(
        str(sdf_path),
        grid,
        grid_start,
        grid_goal,
        indoor_edges,
        outdoor_edges,
        seed=building_seed,
        indoor_options=gen.DOORS_WINDOWS_INDOOR_OPTIONS,
        wall_options=gen.DOORS_WINDOWS_WALL_OPTIONS,
        tree_probability=gen.MOSTLY_INDOOR_TREE_PROBABILITY,
    )


def migrate_file(h5_path: Path, *, overwrite: bool) -> None:
    import h5py

    gen = _load_generalization_module()
    region_cache: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    tmp_dir = Path(tempfile.mkdtemp(prefix="quad_migrate_"))

    with h5py.File(h5_path, "a") as h5:
        samples = h5["samples"]
        keys = sorted(samples.keys(), key=lambda k: int(k))
        print(f"\n{h5_path.name}: {len(keys)} instances")

        for k in tqdm(keys, desc=h5_path.name):
            grp = samples[k]
            if "grid_size" not in grp.attrs:
                raise RuntimeError(
                    f"Instance {k} has no grid_size attr; only generalization "
                    "datasets are supported by this migration."
                )
            if "regions" in grp and not overwrite:
                continue
            if "regions" in grp and overwrite:
                del grp["regions"]

            grid_size = int(grp.attrs["grid_size"])
            building_seed = int(grp.attrs["building_seed"])
            cache_key = (grid_size, building_seed)

            if cache_key not in region_cache:
                sdf_path = tmp_dir / f"b_{grid_size}_{building_seed}.sdf"
                regions = _regions_for(gen, grid_size, building_seed, sdf_path)
                region_cache[cache_key] = [
                    (np.asarray(r.A(), dtype=np.float32), np.asarray(r.b(), dtype=np.float32))
                    for r in regions
                ]

            ab_list = region_cache[cache_key]
            reg_grp = grp.create_group("regions")
            for i, (A, b) in enumerate(ab_list):
                rg = reg_grp.create_group(f"{i:04d}")
                rg.create_dataset("A", data=A)
                rg.create_dataset("b", data=b)

        h5.require_group("meta").attrs["has_region_halfspaces"] = True

    print(f"  done ({len(region_cache)} unique scenes regenerated).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add (A,b) region halfspaces to quadrotor HDF5.")
    parser.add_argument("h5_paths", nargs="+", type=Path, help="HDF5 files to migrate in-place.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute regions even if already present.",
    )
    args = parser.parse_args()

    for h5_path in args.h5_paths:
        if not h5_path.is_file():
            print(f"Skipping missing file: {h5_path}", file=sys.stderr)
            continue
        migrate_file(h5_path, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
