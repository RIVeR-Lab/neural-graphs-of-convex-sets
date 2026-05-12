"""Nonconvex GCS planning for IIWA shelf manipulation (GcsTrajectoryOptimization)."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from pydrake.geometry.optimization import GraphOfConvexSetsOptions, HPolyhedron, Point
from pydrake.planning import GcsTrajectoryOptimization
from underactuated.uav_environment import NONLINEAR_GCS_OPTION

from manipulation.iiwa_helpers import combine_trajectory

NUM_POSITIONS = 7
REGION_ORDER = 5
H_MIN = 0.0
H_MAX = 15.0


def region_list(regions: dict) -> list[HPolyhedron]:
    return list(regions.values())


def build_region_edges(regions: Sequence[HPolyhedron]) -> list[tuple[int, int]]:
    """Directed edges between intersecting IRIS regions."""
    edges: list[tuple[int, int]] = []
    for ii in range(len(regions)):
        for jj in range(ii + 1, len(regions)):
            if regions[ii].IntersectsWith(regions[jj]):
                edges.append((ii, jj))
                edges.append((jj, ii))
    return edges


def iiwa_kinematic_limits(plant) -> tuple[np.ndarray, np.ndarray]:
    """Default IIWA joint velocity and acceleration magnitudes from the plant model."""
    vel = np.asarray(plant.GetVelocityUpperLimits(), dtype=float)
    accel = np.asarray(plant.GetAccelerationUpperLimits(), dtype=float)
    return vel, accel


def _nonlinear_gcs_options(*, restriction: bool = False) -> GraphOfConvexSetsOptions:
    opts = GraphOfConvexSetsOptions()
    opts.convex_relaxation = True
    opts.max_rounded_paths = 0
    opts.preprocessing = NONLINEAR_GCS_OPTION.preprocessing
    opts.solver = NONLINEAR_GCS_OPTION.solver
    opts.solver_options = NONLINEAR_GCS_OPTION.solver_options
    if restriction:
        opts.restriction_solver = NONLINEAR_GCS_OPTION.restriction_solver
        opts.restriction_solver_options = NONLINEAR_GCS_OPTION.restriction_solver_options
    return opts


def build_nonlinear_gcs_problem(
    regions: Sequence[HPolyhedron],
    edges_between_regions: Sequence[tuple[int, int]],
    start_q: np.ndarray,
    goal_q: np.ndarray,
    *,
    vel_limits: np.ndarray,
    accel_limits: np.ndarray,
):
    """
    Build a nonconvex GCS problem (without solving).

    Returns (gcs, graph, source_vertex, target_vertex).
    """
    n = len(start_q)
    assert n == NUM_POSITIONS

    gcs = GcsTrajectoryOptimization(n)
    main = gcs.AddRegions(
        list(regions),
        list(edges_between_regions),
        order=REGION_ORDER,
        h_min=H_MIN,
        h_max=H_MAX,
    )
    source = gcs.AddRegions([Point(start_q)], order=0, h_min=0, h_max=0, name="source")
    target = gcs.AddRegions([Point(goal_q)], order=0, h_min=0, h_max=0, name="target")
    source_to_main = gcs.AddEdges(source, main)
    main_to_target = gcs.AddEdges(main, target)

    source_to_main.AddZeroDerivativeConstraints(1)
    main_to_target.AddZeroDerivativeConstraints(1)
    source_to_main.AddZeroDerivativeConstraints(2)
    main_to_target.AddNonlinearDerivativeBounds(n * [0.0], n * [0.0], 2)

    gcs.AddContinuityConstraints(1)
    gcs.AddContinuityConstraints(2)
    gcs.AddContinuityConstraints(3)

    gcs.AddVelocityBounds(-vel_limits, vel_limits)
    gcs.AddNonlinearDerivativeBounds(-accel_limits, accel_limits, 2)

    gcs.AddTimeCost()
    gcs.AddPathLengthCost()

    graph = gcs.graph_of_convex_sets()
    return gcs, graph, source.Vertices()[0], target.Vertices()[0]


def plan_nonlinear_gcs_segment(
    regions: dict,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    *,
    plant,
    gcs_options=NONLINEAR_GCS_OPTION,
):
    """
    Plan one joint-space segment with nonconvex GCS.

    Returns (trajectory, result, solve_time). trajectory is None if the solve fails.
    """
    region_polys = region_list(regions)
    edges = build_region_edges(region_polys)
    vel_limits, accel_limits = iiwa_kinematic_limits(plant)

    gcs = GcsTrajectoryOptimization(NUM_POSITIONS)
    main = gcs.AddRegions(
        region_polys, edges, order=REGION_ORDER, h_min=H_MIN, h_max=H_MAX,
    )
    source = gcs.AddRegions([Point(start_q)], order=0, h_min=0, h_max=0, name="source")
    target = gcs.AddRegions([Point(goal_q)], order=0, h_min=0, h_max=0, name="target")
    source_to_main = gcs.AddEdges(source, main)
    main_to_target = gcs.AddEdges(main, target)

    source_to_main.AddZeroDerivativeConstraints(1)
    main_to_target.AddZeroDerivativeConstraints(1)
    source_to_main.AddZeroDerivativeConstraints(2)
    main_to_target.AddNonlinearDerivativeBounds(NUM_POSITIONS * [0.0], NUM_POSITIONS * [0.0], 2)

    gcs.AddContinuityConstraints(1)
    gcs.AddContinuityConstraints(2)
    gcs.AddContinuityConstraints(3)

    gcs.AddVelocityBounds(-vel_limits, vel_limits)
    gcs.AddNonlinearDerivativeBounds(-accel_limits, accel_limits, 2)

    gcs.AddTimeCost()
    gcs.AddPathLengthCost()

    t0 = time.time()
    traj, result = gcs.SolvePath(source, target, gcs_options)
    solve_time = time.time() - t0
    if not result.is_success():
        return None, result, solve_time
    return traj, result, solve_time


def plan_nonlinear_gcs_path(
    regions: dict,
    sequence: list[np.ndarray],
    *,
    plant,
    verbose: bool = False,
) -> tuple[object | None, float, list]:
    """
    Plan through a sequence of joint-space waypoints with nonconvex GCS.

    Returns (combined_trajectory, total_solve_time, segment_results).
    """
    traj_segments = []
    run_time = 0.0
    segment_results = []

    for start_pt, goal_pt in zip(sequence[:-1], sequence[1:]):
        traj, result, seg_time = plan_nonlinear_gcs_segment(
            regions, start_pt, goal_pt, plant=plant,
        )
        run_time += seg_time
        segment_results.append(result)
        if traj is None:
            if verbose:
                print(f"  Nonconvex GCS failed between {start_pt} and {goal_pt}", flush=True)
            return None, run_time, segment_results
        traj_segments.append(traj)

    combined = combine_trajectory(traj_segments, wait=0.0)
    return combined, run_time, segment_results


def solve_nonlinear_relaxation(graph, source_vertex, target_vertex):
    """Convex relaxation of the nonconvex GCS (labels for flow GNN)."""
    return graph.SolveShortestPath(
        source_vertex, target_vertex, _nonlinear_gcs_options(restriction=False),
    )


def solve_nonlinear_restriction(graph, source_vertex, target_vertex, path_edges):
    """Fix phi to a rounded path and solve the nonlinear restriction (SNOPT)."""
    for edge in graph.Edges():
        edge.AddPhiConstraint(edge in path_edges)
    result = graph.SolveShortestPath(
        source_vertex, target_vertex, _nonlinear_gcs_options(restriction=True),
    )
    for edge in graph.Edges():
        edge.ClearPhiConstraints()
    return result


def plan_nonlinear_and_make_trajectory(
    regions: dict,
    sequence: list[np.ndarray],
    *,
    plant,
    verbose: bool = False,
):
    """Plan a nonconvex GCS path and return (trajectory, solve_time)."""
    traj, solve_time, _ = plan_nonlinear_gcs_path(
        regions, sequence, plant=plant, verbose=verbose,
    )
    return traj, solve_time
