"""
Print the randomized start/goal samples used by `scripts/planar_pushing/create_plans.py`.

This script does NOT solve any planning problems. It only samples the random
initial conditions (given a seed and count) and prints the resulting poses.

Example:
  python -m planning_through_contact.tests.test_randomization --seed 1
"""

from __future__ import annotations

import argparse

from planning_through_contact.experiments.utils import (
    get_default_experiment_plans,
    get_default_plan_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        help="Random seed for sampling start conditions.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num",
        help="Number of random samples to generate.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--body",
        help='Which slider body to use (e.g. "sugar_box", "box", "tee").',
        type=str,
        default="sugar_box",
    )
    parser.add_argument(
        "--workspace_size",
        help="Workspace size used by the sampler (matches get_default_experiment_plans).",
        type=float,
        default=0.6,
    )
    args = parser.parse_args()

    config = get_default_plan_config(slider_type=args.body)
    plans = get_default_experiment_plans(
        seed=args.seed,
        num_trajs=args.num,
        config=config,
        workspace_size=args.workspace_size,
    )

    print(f"body={args.body} seed={args.seed} num={args.num} workspace_size={args.workspace_size}")
    print("")

    for i, plan in enumerate(plans):
        print(f"sample {i}")
        print(f"  slider_initial_pose: {plan.slider_initial_pose}")
        print(f"  slider_target_pose:  {plan.slider_target_pose}")
        print(f"  pusher_initial_pose: {plan.pusher_initial_pose}")
        print(f"  pusher_target_pose:  {plan.pusher_target_pose}")
        print("")


if __name__ == "__main__":
    main()

