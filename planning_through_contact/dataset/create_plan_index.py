"""
Create a deterministic list of planar pushing planning queries (start/goal poses).

This mirrors the sampling logic used by:
  `planning_through_contact/scripts/planar_pushing/create_plans.py`

It does NOT solve any GCS problems. It only generates and saves the sampled plans so
multiple machines can consume disjoint index ranges reproducibly.

Outputs:
- CSV (Excel-friendly): always written
- XLSX: written if `pandas` and `openpyxl` are installed

Example:
  python dataset/create_plan_index.py --body sugar_box --seed 0
  python dataset/create_plan_index.py --body tee --seed 0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from planning_through_contact.experiments.utils import (
    get_default_experiment_plans,
    get_default_plan_config,
)
from planning_through_contact.geometry.planar.non_collision import NonCollisionMode
from tqdm import tqdm


def _pose_to_dict(prefix: str, pose) -> dict[str, float]:
    if pose is None:
        return {
            f"{prefix}_x": float("nan"),
            f"{prefix}_y": float("nan"),
            f"{prefix}_theta": float("nan"),
        }
    return {
        f"{prefix}_x": float(pose.x),
        f"{prefix}_y": float(pose.y),
        f"{prefix}_theta": float(pose.theta),
    }


def _entry_onehot(entry_idx0: int, num_entries: int) -> list[int]:
    v = [0 for _ in range(num_entries)]
    if 0 <= entry_idx0 < num_entries:
        v[entry_idx0] = 1
    return v


def _wrap_angle(theta: float) -> float:
    # Wrap to (-pi, pi]
    return float(np.arctan2(np.sin(theta), np.cos(theta)))


def _rot2(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_train", type=int, default=500)
    parser.add_argument("--num_val", type=int, default=100)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--body", type=str, default="sugar_box")
    parser.add_argument("--workspace_size", type=float, default=0.6)
    parser.add_argument("--pusher_radius", type=float, default=0.015)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to planning_through_contact/dataset/data/<body>.",
    )
    parser.add_argument("--output_stem", type=str, default=None)
    parser.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help="Random seed used ONLY for shuffling train/val/test split assignment.",
    )
    args = parser.parse_args()

    if args.num_train < 0 or args.num_val < 0 or args.num_test < 0:
        raise ValueError("--num_train, --num_val, and --num_test must be non-negative")

    num_train = int(args.num_train)
    num_val = int(args.num_val)
    num_test = int(args.num_test)
    num_plans = num_train + num_val + num_test

    config = get_default_plan_config(
        slider_type=args.body,
        pusher_radius=args.pusher_radius,
        use_case="normal",
    )

    plans = get_default_experiment_plans(
        seed=args.seed,
        num_trajs=num_plans,
        config=config,
        workspace_size=args.workspace_size,
    )

    out_dir = Path(args.output_dir) if args.output_dir is not None else Path("planning_through_contact/dataset/data") / args.body
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = args.output_stem or "global_features"
    csv_path = out_dir / f"{stem}.csv"
    xlsx_path = out_dir / f"{stem}.xlsx"

    num_entries = int(config.slider_geometry.num_collision_free_regions)

    # Classical ML split: shuffle indices deterministically, then assign first Ntrain to train.
    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(num_plans)
    train_ids = set(int(i) for i in perm[:num_train])
    val_ids = set(int(i) for i in perm[num_train : num_train + num_val])
    test_ids = set(int(i) for i in perm[num_train + num_val : num_train + num_val + num_test])

    rows: list[dict[str, Any]] = []
    for plan_id, plan in enumerate(tqdm(plans, desc="Building plan index")):
        if plan_id in train_ids:
            split = "train"
        elif plan_id in val_ids:
            split = "val"
        else:
            split = "test"

        # Determine which non-contact region the source connects to (before any GCS solve).
        # This is used to construct g_entry in the PDF.
        assert plan.pusher_initial_pose is not None
        src_mode = NonCollisionMode.create_source_or_target_mode(
            config=config,
            slider_pose_world=plan.slider_initial_pose,
            pusher_pose_world=plan.pusher_initial_pose,
            initial_or_final="initial",
            set_slider_pose=True,
            terminal_cost=False,
        )
        entry_idx0 = int(src_mode.contact_location.idx)
        entry_oh = _entry_onehot(entry_idx0, num_entries)

        # Updated PDF (ICRA2027_Ananya.pdf): express global conditioning in the
        # slider's initial body frame S.
        # Goal pose is fixed at world origin with theta=0.
        ps_W = np.array(
            [float(plan.slider_initial_pose.x), float(plan.slider_initial_pose.y)],
            dtype=np.float64,
        )
        theta_s = float(plan.slider_initial_pose.theta)
        R_WS = _rot2(theta_s)  # maps S->W; so R_WS^T maps W->S

        # p_goal,S = R_WS^T (0 - p_s,W) = -R_WS^T p_s,W
        p_goal_S = -(R_WS.T @ ps_W.reshape((2, 1))).reshape((2,))
        theta_goal = _wrap_angle(-theta_s)

        # p_pusher,S = R_WS^T (p_pusher,W - p_s,W)
        assert plan.pusher_initial_pose is not None
        pp_W = np.array(
            [float(plan.pusher_initial_pose.x), float(plan.pusher_initial_pose.y)],
            dtype=np.float64,
        )
        p_pusher_S = (R_WS.T @ (pp_W - ps_W).reshape((2, 1))).reshape((2,))

        g_pose = [
            float(p_goal_S[0]),
            float(p_goal_S[1]),
            float(np.sin(theta_goal)),
            float(np.cos(theta_goal)),
            float(p_pusher_S[0]),
            float(p_pusher_S[1]),
        ]

        row: dict[str, Any] = {
            "plan_id": plan_id,
            "split": split,
            "split_seed": int(args.split_seed),
            "body": args.body,
            "seed": int(args.seed),
            "workspace_size": float(args.workspace_size),
            "pusher_radius": float(args.pusher_radius),
            "entry_idx0": entry_idx0,
            **{f"g_pose_{i}": g_pose[i] for i in range(6)},
            **{f"g_entry_{i}": int(entry_oh[i]) for i in range(num_entries)},
            **_pose_to_dict("slider_init", plan.slider_initial_pose),
            **_pose_to_dict("slider_goal", plan.slider_target_pose),
            **_pose_to_dict("pusher_init", plan.pusher_initial_pose),
            **_pose_to_dict("pusher_goal", plan.pusher_target_pose),
        }
        rows.append(row)

    print("Plan index built. Writing CSV/XLSX to disk...")

    # Stable header order: basic fields, then g, then poses.
    fieldnames: list[str] = []
    for key in rows[0].keys():
        fieldnames.append(key)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Optional XLSX for convenient viewing.
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows)
        df.to_excel(xlsx_path, index=False)  # requires openpyxl
    except Exception:
        # CSV is always written; XLSX is best-effort.
        pass

    print(f"Wrote {len(rows)} plans to {csv_path}")
    if xlsx_path.exists():
        print(f"Wrote Excel file to {xlsx_path}")


if __name__ == "__main__":
    main()

