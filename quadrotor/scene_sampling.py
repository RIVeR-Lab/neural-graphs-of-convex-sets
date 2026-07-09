"""Start/goal sampling for quadrotor GCS in compile_sdf regions."""

from __future__ import annotations

import numpy as np

DELTA = 0.3


def box_bounds(region):
    A, b = region.A(), region.b()
    lb = np.full(3, -np.inf)
    ub = np.full(3, np.inf)
    for i in range(A.shape[0]):
        row = A[i]
        nz = np.nonzero(row)[0]
        if len(nz) == 1:
            dim = nz[0]
            if row[dim] > 0:
                ub[dim] = min(ub[dim], b[i] / row[dim])
            else:
                lb[dim] = max(lb[dim], b[i] / row[dim])
    return lb, ub


def sample_inside(region, rng):
    lb, ub = box_bounds(region)
    lo, hi = lb + DELTA, ub - DELTA
    if np.any(lo >= hi):
        return None
    return rng.uniform(lo, hi)


def pick_corner_regions(regions, rng, diagonal="bl_to_tr"):
    """Pick start/goal from opposite corners (skip first 4 exterior corridor regions)."""
    finite = []
    for i, r in enumerate(regions[4:], start=4):
        lb, ub = box_bounds(r)
        if not np.any(np.isinf(lb)) and not np.any(np.isinf(ub)):
            finite.append((i, lb, ub))

    if not finite:
        return None

    centers = np.array([(lb + ub) / 2 for _, lb, ub in finite])

    if diagonal == "bl_to_tr":
        scores = centers[:, 0] + centers[:, 1]
        start_reg_idx = finite[int(np.argmin(scores))][0]
        goal_reg_idx = finite[int(np.argmax(scores))][0]
    elif diagonal == "br_to_tl":
        scores = centers[:, 0] - centers[:, 1]
        start_reg_idx = finite[int(np.argmin(scores))][0]
        goal_reg_idx = finite[int(np.argmax(scores))][0]
    else:
        scores = centers[:, 0] + centers[:, 1]
        start_reg_idx = finite[int(np.argmax(scores))][0]
        goal_reg_idx = finite[int(np.argmin(scores))][0]

    start_lb, start_ub = box_bounds(regions[start_reg_idx])
    goal_lb, goal_ub = box_bounds(regions[goal_reg_idx])

    start_pt = sample_inside(regions[start_reg_idx], rng)
    goal_pt = sample_inside(regions[goal_reg_idx], rng)
    if start_pt is None or goal_pt is None:
        return None

    if diagonal == "high_to_low":
        start_pt[2] = np.clip(start_ub[2] - DELTA - 0.1, start_lb[2] + DELTA, start_ub[2] - DELTA)
        goal_pt[2] = np.clip(goal_lb[2] + DELTA + 0.1, goal_lb[2] + DELTA, goal_ub[2] - DELTA)
    else:
        start_pt = np.clip(
            start_pt + np.array([-0.8, -0.6, -0.2]),
            start_lb + DELTA, start_ub - DELTA,
        )
        direction = goal_pt - start_pt
        direction /= np.linalg.norm(direction) + 1e-6
        goal_pt = np.clip(
            goal_pt + direction * 0.5 + np.array([0.0, 0.0, 0.7]),
            goal_lb + DELTA, goal_ub - DELTA,
        )

    return start_pt, goal_pt
