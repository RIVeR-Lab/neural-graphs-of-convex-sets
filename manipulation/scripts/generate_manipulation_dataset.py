#!/usr/bin/env python3
"""Generate IIWA shelf GCS datasets for convex and nonconvex planners."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from pydrake.solvers import MosekSolver
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.getLogger("drake").setLevel(logging.WARNING)

from manipulation.dataset_collect import (
    SPLITS,
    build_adjacency,
    collect_linear_instance,
    collect_nonlinear_instance,
    count_split,
    next_instance_id,
    sample_start_goal,
    write_region_meta,
)
from manipulation.iiwa_helpers import build_shelf_plant
from manipulation.paths import DEFAULT_DATASET_DIR, DEFAULT_REGIONS_PATH
from manipulation.paths import manipulation_models_hint, manipulation_models_ready
from manipulation.shelf_gcs import load_regions

CONVEX_FILENAME = "manipulation_gcs_convex.h5"
NONCONVEX_FILENAME = "manipulation_gcs_nonconvex.h5"
PLANNERS = ("convex", "nonconvex")
MAX_QUERY_ATTEMPTS_PER_SAMPLE = 500
SPLIT_SEED_OFFSETS = {"train": 0, "val": 1_000_000, "test": 2_000_000}


def parse_planners(value: str) -> tuple[str, ...]:
    if value == "both":
        return PLANNERS
    planners = tuple(p.strip() for p in value.split(",") if p.strip())
    unknown = sorted(set(planners) - set(PLANNERS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown planner(s): {', '.join(unknown)}; expected convex, nonconvex, or both"
        )
    if not planners:
        raise argparse.ArgumentTypeError("at least one planner is required")
    return planners


def parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(s.strip() for s in value.split(",") if s.strip())
    unknown = sorted(set(splits) - set(SPLITS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown split(s): {', '.join(unknown)}")
    if not splits:
        raise argparse.ArgumentTypeError("at least one split is required")
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manipulation datasets for convex and nonconvex GCS.",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS_PATH)
    parser.add_argument(
        "--planners",
        type=parse_planners,
        default=PLANNERS,
        help="Dataset(s) to generate: convex, nonconvex, convex,nonconvex, or both.",
    )
    parser.add_argument("--splits", type=parse_splits, default=tuple(SPLITS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=SPLITS["train"])
    parser.add_argument("--val-count", type=int, default=SPLITS["val"])
    parser.add_argument("--test-count", type=int, default=SPLITS["test"])
    parser.add_argument("--max-query-attempts", type=int, default=MAX_QUERY_ATTEMPTS_PER_SAMPLE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_targets(args: argparse.Namespace) -> dict[str, int]:
    return {
        "train": int(args.train_count),
        "val": int(args.val_count),
        "test": int(args.test_count),
    }


def query_seed(base_seed: int, split: str, attempt: int) -> int:
    return int(base_seed + SPLIT_SEED_OFFSETS[split] + attempt)


def open_outputs(args: argparse.Namespace, h5py):
    paths = {
        "convex": args.output_dir / CONVEX_FILENAME,
        "nonconvex": args.output_dir / NONCONVEX_FILENAME,
    }
    if args.overwrite:
        for planner in args.planners:
            paths[planner].unlink(missing_ok=True)
    return {
        planner: h5py.File(paths[planner], "a")
        for planner in args.planners
    }, paths


def prepare_file(h5_file, regions, *, str_dtype, seed: int, planner: str):
    write_region_meta(h5_file, regions, str_dtype=str_dtype)
    meta = h5_file.require_group("meta")
    meta.attrs["base_seed"] = seed
    meta.attrs["planner"] = planner
    meta.attrs["model"] = planner
    return h5_file.require_group("samples")


def main() -> None:
    args = parse_args()
    if not manipulation_models_ready():
        print(manipulation_models_hint(), file=sys.stderr)
        sys.exit(1)
    if not args.regions.is_file():
        print(f"IRIS regions not found: {args.regions}", file=sys.stderr)
        print("Run: python scripts/iiwa_shelf_scenes.py --generate-regions --regions-only", file=sys.stderr)
        sys.exit(1)

    import h5py

    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = split_targets(args)
    solver = MosekSolver()

    print("Building IIWA shelf scene...")
    plant, _, _, _ = build_shelf_plant()
    print(f"Loading IRIS regions from {args.regions}")
    regions = load_regions(args.regions)
    adj = build_adjacency(regions)
    print(f"  {len(regions)} regions")

    files, paths = open_outputs(args, h5py)
    try:
        str_dtype = h5py.string_dtype(encoding="utf-8")
        samples = {
            planner: prepare_file(
                files[planner], regions, str_dtype=str_dtype, seed=args.seed, planner=planner,
            )
            for planner in args.planners
        }
        require_all = len(args.planners) > 1

        for split in args.splits:
            target = targets[split]
            if target <= 0:
                continue

            saved = {planner: count_split(samples[planner], split) for planner in args.planners}
            instance_id = max(next_instance_id(samples[planner]) for planner in args.planners)
            progress_start = min(saved.values()) if require_all else next(iter(saved.values()))

            print(f"\n=== {split.upper()} | target {target} per selected planner ===")
            for planner, count in saved.items():
                if count:
                    print(f"  resuming {planner}: {count}/{target}")

            stats = {"query_fail": 0, "convex_fail": 0, "nonconvex_fail": 0}
            pbar = tqdm(initial=progress_start, total=target, desc=split)
            attempt = 0
            max_attempts = max(1, args.max_query_attempts) * target

            while any(saved[planner] < target for planner in args.planners):
                if attempt >= max_attempts:
                    print(f"  stopping {split}: exceeded attempt budget ({max_attempts})")
                    break

                seed = query_seed(args.seed, split, attempt)
                rng = np.random.default_rng(seed)
                sampled = sample_start_goal(regions, adj, rng)
                attempt += 1
                if sampled is None:
                    stats["query_fail"] += 1
                    continue

                q_start, q_goal, _, _ = sampled
                accepted: dict[str, bool] = {}

                if "convex" in args.planners and saved["convex"] < target:
                    accepted["convex"] = collect_linear_instance(
                        regions,
                        q_start,
                        q_goal,
                        solver=solver,
                        str_dtype=str_dtype,
                        h5_samples=samples["convex"],
                        instance_id=instance_id,
                        split=split,
                        query_seed=seed,
                    )
                    if not accepted["convex"]:
                        stats["convex_fail"] += 1

                if "nonconvex" in args.planners and saved["nonconvex"] < target:
                    accepted["nonconvex"] = collect_nonlinear_instance(
                        regions,
                        q_start,
                        q_goal,
                        plant=plant,
                        str_dtype=str_dtype,
                        h5_samples=samples["nonconvex"],
                        instance_id=instance_id,
                        split=split,
                        query_seed=seed,
                    )
                    if not accepted["nonconvex"]:
                        stats["nonconvex_fail"] += 1

                if require_all and all(accepted.get(planner, saved[planner] >= target) for planner in args.planners):
                    for planner in args.planners:
                        if planner in accepted:
                            saved[planner] += 1
                    instance_id += 1
                    pbar.update(1)
                elif require_all:
                    for planner, ok in accepted.items():
                        if ok and str(instance_id) in samples[planner]:
                            del samples[planner][str(instance_id)]
                else:
                    planner = args.planners[0]
                    if accepted.get(planner):
                        saved[planner] += 1
                        instance_id += 1
                        pbar.update(1)

                pbar.set_postfix(**saved)

            pbar.close()
            print(
                "  saved "
                + ", ".join(f"{planner}={saved[planner]}/{target}" for planner in args.planners)
                + f" | failures query={stats['query_fail']} convex={stats['convex_fail']} nonconvex={stats['nonconvex_fail']}"
            )
    finally:
        for h5_file in files.values():
            h5_file.close()

    for planner in args.planners:
        print(f"{planner.capitalize()} HDF5: {paths[planner]}")


if __name__ == "__main__":
    main()
