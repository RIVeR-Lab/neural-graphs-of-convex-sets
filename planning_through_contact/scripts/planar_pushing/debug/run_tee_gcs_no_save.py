"""
Run vanilla GCS tee-pushing instances without saving artifacts.

This is a lightweight diagnostic for checking whether tee GCS planning itself
fits in memory, separate from dataset/HDF5 candidate-path collection.
"""

from __future__ import annotations

import argparse
import gc
import time

from planning_through_contact.experiments.utils import (
    get_default_experiment_plans,
    get_default_plan_config,
    get_default_solver_params,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num", type=int, default=3)
    parser.add_argument("--traj", type=int, default=None)
    parser.add_argument("--body", type=str, default="tee")
    parser.add_argument("--rounding_steps", type=int, default=100)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = get_default_plan_config(
        slider_type=args.body,
        pusher_radius=0.015,
        use_case="normal",
    )
    solver_params = get_default_solver_params(args.debug, clarabel=False)
    solver_params.rounding_steps = int(args.rounding_steps)

    num_to_sample = args.num if args.traj is None else max(args.num, args.traj + 1)
    plans = get_default_experiment_plans(args.seed, num_to_sample, config)
    if args.traj is not None:
        plans = [plans[int(args.traj)]]

    for local_idx, plan in enumerate(plans):
        plan_idx = int(args.traj) if args.traj is not None else local_idx
        print(f"\n[{plan_idx}] planning body={args.body} rounding_steps={solver_params.rounding_steps}")
        config.start_and_goal = plan
        planner = PlanarPushingPlanner(config)

        t0 = time.perf_counter()
        planner.formulate_problem()
        path = planner.plan_path(solver_params)
        elapsed = time.perf_counter() - t0

        ok = path is not None and path.rounded_result is not None and path.rounded_result.is_success()
        cost = path.rounded_result.get_optimal_cost() if ok else None
        print(f"[{plan_idx}] ok={ok} elapsed_s={elapsed:.3f} cost={cost}")

        del planner, path
        gc.collect()


if __name__ == "__main__":
    main()
