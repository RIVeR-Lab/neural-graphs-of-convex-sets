"""Shelf-based IIWA Linear GCS planning (extracted from gcs/reproduction/prm_comparison)."""

from __future__ import annotations

import logging
import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np

from pydrake.geometry.optimization import IrisNp, IrisOptions
from pydrake.solvers import MosekSolver

from manipulation.iiwa_helpers import inverse_kinematics, make_traj, trajectory_length
from manipulation.paths import DEFAULT_REGIONS_PATH
from quadrotor.gcs.linear import LinearGCS
from quadrotor.gcs.rounding import randomForwardPathSearch

logger = logging.getLogger(__name__)

DEFAULT_Q0 = [0, 0.3, 0, -1.8, 0, 1, 1.57]

# IK seeds for IRIS region generation (notebook milestones + approach configs).
MILESTONES = {
    "Above Shelve": [[0.75, 0, 0.9], [0, -np.pi, -np.pi / 2]],
    "Top Rack": [[0.75, 0, 0.67], [0, -np.pi, -np.pi / 2]],
    "Middle Rack": [[0.75, 0, 0.41], [0, -np.pi, -np.pi / 2]],
    "Left Bin": [[0.0, 0.6, 0.22], [np.pi / 2, np.pi, 0]],
    "Right Bin": [[0.0, -0.6, 0.22], [np.pi / 2, np.pi, np.pi]],
}

ADDITIONAL_SEED_POINTS = {
    "Front to Shelve": np.array([0, 0.2, 0, -2.09, 0, -0.3, np.pi / 2]),
    "Left to Shelve": np.array([0.8, 0.7, 0, -1.6, 0, 0, np.pi / 2]),
    "Right to Shelve": np.array([-0.8, 0.7, 0, -1.6, 0, 0, np.pi / 2]),
}

# Slightly offset gripper poses used for the motion demos.
DEMONSTRATION = {
    "Above Shelve": [[0.75, -0.12, 0.9], [0, -np.pi, -np.pi / 2]],
    "Top Rack": [[0.75, 0.12, 0.67], [0, -np.pi, -np.pi / 2]],
    "Middle Rack": [[0.75, 0.12, 0.41], [0, -np.pi, -np.pi / 2]],
    "Left Bin": [[0.08, 0.6, 0.22], [np.pi / 2, np.pi, 0]],
    "Right Bin": [[-0.08, -0.6, 0.22], [np.pi / 2, np.pi, np.pi]],
}


def compute_configurations(poses: dict, q0=None) -> dict[str, np.ndarray]:
    q0 = DEFAULT_Q0 if q0 is None else q0
    configs = {}
    for name, (translation, rpy) in poses.items():
        q = inverse_kinematics(q0, translation, rpy)
        if q is None:
            raise RuntimeError(f"IK failed for pose '{name}'")
        configs[name] = q
    return configs


def _configs_in_regions(q: np.ndarray, regions: dict) -> list[str]:
    return [name for name, region in regions.items() if region.PointInSet(q)]


def planning_configurations(regions: dict, q0=None) -> dict[str, np.ndarray]:
    """
    Joint configs for GCS planning.

    Prefer demonstration IK (notebook motion targets). If a demo config falls outside
    every IRIS region — common with freshly generated IrisNp regions — fall back to
    the milestone seed used to grow that named region.
    """
    demo = compute_configurations(DEMONSTRATION, q0=q0)
    seeds = build_seed_points(q0=q0)
    resolved: dict[str, np.ndarray] = {}

    for name, q_demo in demo.items():
        if _configs_in_regions(q_demo, regions):
            resolved[name] = q_demo
            continue
        q_seed = seeds.get(name)
        if q_seed is not None and _configs_in_regions(q_seed, regions):
            print(
                f"  Warning: demo '{name}' is outside all IRIS regions; "
                f"using milestone seed config instead.",
                flush=True,
            )
            resolved[name] = q_seed
            continue
        raise RuntimeError(
            f"Neither demonstration nor milestone config for '{name}' "
            f"lies in any IRIS region. Regenerate regions or adjust poses."
        )
    return resolved


def build_seed_points(q0=None) -> dict[str, np.ndarray]:
    milestone_configs = compute_configurations(MILESTONES, q0=q0)
    return {**milestone_configs, **ADDITIONAL_SEED_POINTS}


def default_iris_options() -> IrisOptions:
    opts = IrisOptions()
    opts.require_sample_point_is_contained = True
    opts.iteration_limit = 10
    opts.termination_threshold = -1
    opts.relative_termination_threshold = 0.01
    return opts


def _calc_region_worker(seed: np.ndarray):
    global _WORKER_PLANT, _WORKER_DIAGRAM, _WORKER_IRIS_OPTIONS
    context = _WORKER_DIAGRAM.CreateDefaultContext()
    plant_context = _WORKER_PLANT.GetMyMutableContextFromRoot(context)
    _WORKER_PLANT.SetPositions(plant_context, seed)
    t0 = time.time()
    hpoly = IrisNp(_WORKER_PLANT, plant_context, _WORKER_IRIS_OPTIONS)
    print(f"  IRIS seed done in {time.time() - t0:.1f}s", flush=True)
    return hpoly


_WORKER_PLANT = None
_WORKER_DIAGRAM = None
_WORKER_IRIS_OPTIONS = None


def generate_regions(plant, diagram, seed_points: dict, *, workers: int | None = None) -> dict:
    """Grow IRIS regions in configuration space from named seed configs."""
    global _WORKER_PLANT, _WORKER_DIAGRAM, _WORKER_IRIS_OPTIONS
    if workers is None:
        workers = mp.cpu_count()
    names = list(seed_points.keys())
    seeds = list(seed_points.values())
    iris_options = default_iris_options()

    print(f"Generating {len(seeds)} IRIS regions with {workers} worker(s)...")
    t0 = time.time()
    _WORKER_PLANT, _WORKER_DIAGRAM, _WORKER_IRIS_OPTIONS = plant, diagram, iris_options
    if workers <= 1:
        regions = [_calc_region_worker(seed) for seed in seeds]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers) as pool:
            regions = pool.map(_calc_region_worker, seeds)
    print(f"Generated {len(regions)} IRIS regions in {time.time() - t0:.1f}s")
    return dict(zip(names, regions))


def load_regions(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_regions(regions: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(regions, f)


def build_demo_sequences(
    demonstration_configs: dict[str, np.ndarray],
    seed_points: dict[str, np.ndarray],
) -> dict[str, list[np.ndarray]]:
    d = demonstration_configs
    return {
        "a": [d["Above Shelve"], d["Top Rack"]],
        "b": [d["Top Rack"], d["Middle Rack"]],
        "c": [d["Middle Rack"], d["Left Bin"]],
        "d": [d["Left Bin"], d["Right Bin"]],
        "e": [d["Right Bin"], d["Above Shelve"]],
        "f": [d["Above Shelve"], seed_points["Left to Shelve"]],
        "circle": [
            d["Above Shelve"],
            d["Top Rack"],
            d["Middle Rack"],
            d["Left Bin"],
            d["Right Bin"],
            d["Above Shelve"],
        ],
    }


def plan_gcs_path(
    regions: dict,
    sequence: list[np.ndarray],
    *,
    seed: int = 17,
    verbose: bool = False,
) -> tuple[np.ndarray | None, float, dict]:
    """
    Plan through a sequence of joint-space waypoints with Linear GCS.

    Returns (path, total_solve_time, segment_results).
    """
    path = [sequence[0]]
    run_time = 0.0
    segment_results = []

    for start_pt, goal_pt in zip(sequence[:-1], sequence[1:]):
        gcs = LinearGCS(regions.copy())
        gcs.addSourceTarget(start_pt, goal_pt)
        gcs.setRoundingStrategy(
            randomForwardPathSearch, max_paths=10, max_trials=100, seed=seed,
        )
        gcs.setSolver(MosekSolver())
        waypoints, results = gcs.SolvePath(rounding=True, verbose=False, preprocessing=True)
        if waypoints is None:
            if verbose:
                logger.warning("GCS failed between %s and %s", start_pt, goal_pt)
            return None, run_time, segment_results

        run_time += results["preprocessing_stats"]["linear_programs"]
        run_time += results["relaxation_solver_time"]
        run_time += results["total_rounded_solver_time"]
        segment_results.append(results)

        if verbose:
            gap = (results["rounded_cost"] - results["relaxation_cost"]) / results["relaxation_cost"]
            logger.info(
                "segment: relax=%.3f rounded=%.3f gap=%.3f",
                results["relaxation_cost"], results["rounded_cost"], gap,
            )

        path += waypoints.T[1:].tolist()

    return np.stack(path).T, run_time, segment_results


def plan_and_make_trajectory(
    regions: dict,
    sequence: list[np.ndarray],
    *,
    seed: int = 17,
    speed: float = 2.0,
    verbose: bool = False,
):
    path, solve_time, _ = plan_gcs_path(regions, sequence, seed=seed, verbose=verbose)
    if path is None:
        return None, None, solve_time
    traj = make_traj(path, speed=speed)
    return path, traj, solve_time
