#!/usr/bin/env python3
"""
Standalone IIWA shelf GCS demo (extracted from gcs/reproduction/prm_comparison).

Builds the Kuka IIWA + shelf + bins scene, loads or generates IRIS configuration-space
regions, plans joint-space paths with linear and/or nonlinear GCS, and exports Meshcat HTML.

Examples:
  bash scripts/setup_iiwa_models.sh
  python scripts/iiwa_shelf_scenes.py --demo circle
  python scripts/iiwa_shelf_scenes.py --planner nonlinear --demo circle
  python scripts/iiwa_shelf_scenes.py --generate-regions --demo circle   # force recompute IRIS
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.getLogger("drake").setLevel(logging.WARNING)

from pydrake.geometry import StartMeshcat

from manipulation.iiwa_helpers import build_shelf_plant, trajectory_length, visualize_trajectory
from manipulation.paths import DEFAULT_REGIONS_PATH, DEFAULT_OUTPUT_DIR
from manipulation.shelf_gcs import (
    build_demo_sequences,
    build_seed_points,
    planning_configurations,
    generate_regions,
    load_regions,
    plan_and_make_trajectory,
    save_regions,
)
from manipulation.trajopt import plan_nonlinear_and_make_trajectory

DEMO_CHOICES = ("a", "b", "c", "d", "e", "f", "circle")
PLANNER_CHOICES = ("linear", "nonlinear", "both")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IIWA shelf GCS planning + Meshcat viz.")
    parser.add_argument(
        "--demo", type=str, default="circle", choices=DEMO_CHOICES,
        help="Waypoint sequence to execute (default: circle).",
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="both",
        choices=PLANNER_CHOICES,
        help="GCS planner: linear (config-space), nonlinear (GcsTrajectoryOptimization), or both.",
    )
    parser.add_argument(
        "--regions", type=Path, default=DEFAULT_REGIONS_PATH,
        help="Pickle file with IRIS regions (default: manipulation/data/IRIS.reg).",
    )
    parser.add_argument(
        "--generate-regions", action="store_true",
        help="Force (re)generation of IRIS regions before planning.",
    )
    parser.add_argument(
        "--regions-only", action="store_true",
        help="Only generate/save IRIS regions; skip GCS planning and viz.",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel workers for IRIS (default: all CPU cores, same as gcs notebook).",
    )
    parser.add_argument("--seed", type=int, default=17, help="GCS rounding RNG seed (linear).")
    parser.add_argument("--speed", type=float, default=2.0, help="Playback speed for linear trajectory.")
    parser.add_argument(
        "--plan-wait", type=float, default=3.0,
        help="Pause between sequential plans in Meshcat when --planner both.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for Meshcat HTML output.",
    )
    parser.add_argument(
        "--show-line", action="store_true",
        help="Draw end-effector path polylines in Meshcat.",
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Plan only; skip Meshcat HTML export.",
    )
    return parser.parse_args()


def ensure_regions(args: argparse.Namespace, plant, diagram) -> dict:
    if args.regions.exists() and not args.generate_regions:
        print(f"Loading IRIS regions from {args.regions}")
        return load_regions(args.regions)

    if args.regions.exists():
        print(f"Regenerating IRIS regions (--generate-regions), overwriting {args.regions}")
    else:
        print(f"No cached regions at {args.regions}; generating IRIS regions (several minutes)...")

    seed_points = build_seed_points()
    regions = generate_regions(plant, diagram, seed_points, workers=args.workers)
    save_regions(regions, args.regions)
    print(f"Saved {len(regions)} regions to {args.regions}")
    return regions


def main() -> None:
    args = parse_args()

    from manipulation.paths import manipulation_models_ready, manipulation_models_hint
    if not manipulation_models_ready():
        print(manipulation_models_hint(), file=sys.stderr)
        sys.exit(1)

    print("Building IIWA shelf scene...")
    plant, _, diagram, _ = build_shelf_plant()
    regions = ensure_regions(args, plant, diagram)

    if args.regions_only:
        print("Done (--regions-only).")
        return

    print("Computing demonstration configurations...")
    demo_configs = planning_configurations(regions)
    seed_points = build_seed_points()
    demos = build_demo_sequences(demo_configs, seed_points)
    sequence = demos[args.demo]
    print(f"Planning demo '{args.demo}' ({len(sequence)} waypoints)...")

    linear_traj = None
    nonlinear_traj = None
    linear_solve_time = 0.0
    nonlinear_solve_time = 0.0

    if args.planner in ("linear", "both"):
        print("  Linear GCS...")
        path, linear_traj, linear_solve_time = plan_and_make_trajectory(
            regions, sequence, seed=args.seed, speed=args.speed, verbose=True,
        )
        if path is None:
            print("Linear GCS planning failed.", file=sys.stderr)
            sys.exit(1)
        length = trajectory_length(linear_traj)
        print(f"    Path length: {length:.3f}")
        print(f"    Solve time:  {linear_solve_time:.3f} s")
        print(f"    Duration:    {linear_traj.start_time():.2f}s → {linear_traj.end_time():.2f}s")

    if args.planner in ("nonlinear", "both"):
        print("  Nonlinear GCS...")
        nonlinear_traj, nonlinear_solve_time = plan_nonlinear_and_make_trajectory(
            regions, sequence, plant=plant, verbose=True,
        )
        if nonlinear_traj is None:
            print("Nonlinear GCS planning failed.", file=sys.stderr)
            sys.exit(1)
        length = trajectory_length(nonlinear_traj)
        print(f"    Path length: {length:.3f}")
        print(f"    Solve time:  {nonlinear_solve_time:.3f} s")
        print(f"    Duration:    {nonlinear_traj.start_time():.2f}s → {nonlinear_traj.end_time():.2f}s")

    if args.no_viz:
        return

    viz_trajs = []
    if linear_traj is not None:
        viz_trajs.append(linear_traj)
    if nonlinear_traj is not None:
        viz_trajs.append(nonlinear_traj)

    print("Rendering Meshcat HTML...")
    meshcat = StartMeshcat()
    visualize_trajectory(
        meshcat,
        viz_trajs,
        show_line=args.show_line,
        ghost_configs=sequence,
        alpha=0.3,
        plan_wait=args.plan_wait if len(viz_trajs) > 1 else 2.0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.output_dir / f"shelf_demo_{args.demo}_{args.planner}_seed{args.seed}.html"
    html_path.write_text(meshcat.StaticHtml())
    print(f"Saved {html_path.resolve()}")


if __name__ == "__main__":
    main()
