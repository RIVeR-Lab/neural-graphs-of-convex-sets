"""Nonlinear GCS planning via Drake GcsTrajectoryOptimization (Wrangel et al.)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pydrake.geometry.optimization import GraphOfConvexSetsOptions, HPolyhedron, Point
from pydrake.planning import GcsTrajectoryOptimization
from underactuated.uav_environment import NONLINEAR_GCS_OPTION

# Match gcs_quadrotor.ipynb (Skydio 2–style limits).
QDT_MAX = 16.0
QDDT_MAX = 10.0
REGION_ORDER = 6
H_MIN = 0.0
H_MAX = 20.0


def adjacency_to_edges(adj: dict[int, list[int]]) -> list[tuple[int, int]]:
    """Directed edge list for GcsTrajectoryOptimization.AddRegions."""
    edges: set[tuple[int, int]] = set()
    for i, neighbors in adj.items():
        for j in neighbors:
            edges.add((i, j))
    return sorted(edges)


def build_nonlinear_gcs_problem(
    regions: Sequence[HPolyhedron],
    edges_between_regions: Sequence[tuple[int, int]],
    start_pose: np.ndarray,
    goal_pose: np.ndarray,
):
    """
    Build a nonlinear GCS problem (without solving).

    Returns (gcs, graph, source_vertex, target_vertex).
    """
    gcs = GcsTrajectoryOptimization(3)
    main = gcs.AddRegions(
        list(regions), list(edges_between_regions), order=REGION_ORDER, h_min=H_MIN, h_max=H_MAX,
    )
    source = gcs.AddRegions(
        [Point(start_pose)], order=0, h_min=0, h_max=0, name="source",
    )
    target = gcs.AddRegions(
        [Point(goal_pose)], order=0, h_min=0, h_max=0, name="target",
    )
    source_to_main = gcs.AddEdges(source, main)
    main_to_target = gcs.AddEdges(main, target)

    source_to_main.AddZeroDerivativeConstraints(1)
    main_to_target.AddZeroDerivativeConstraints(1)
    source_to_main.AddZeroDerivativeConstraints(2)
    main_to_target.AddNonlinearDerivativeBounds(3 * [0.0], 3 * [0.0], 2)

    gcs.AddContinuityConstraints(1)
    gcs.AddContinuityConstraints(2)
    gcs.AddContinuityConstraints(3)
    gcs.AddContinuityConstraints(4)

    gcs.AddVelocityBounds(3 * [-QDT_MAX], 3 * [QDT_MAX])
    gcs.AddNonlinearDerivativeBounds(3 * [-QDDT_MAX], 3 * [QDDT_MAX], 2)

    gcs.AddTimeCost()
    gcs.AddPathLengthCost()

    graph = gcs.graph_of_convex_sets()
    return gcs, graph, source.Vertices()[0], target.Vertices()[0]


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


def solve_nonlinear_relaxation(graph, source_vertex, target_vertex):
    """Convex relaxation of the nonlinear GCS (same phase as BezierGCS rounding=True step 1)."""
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


def plan_nonlinear_gcs(
    regions: Sequence[HPolyhedron],
    edges_between_regions: Sequence[tuple[int, int]],
    start_pose: np.ndarray,
    goal_pose: np.ndarray,
    *,
    gcs_options=NONLINEAR_GCS_OPTION,
):
    """
    Shortest-path nonlinear GCS (accel limits + continuity through snap).

    Returns (trajectory, result). trajectory is None if the solve fails.
    """
    gcs = GcsTrajectoryOptimization(3)
    main = gcs.AddRegions(
        list(regions), list(edges_between_regions), order=REGION_ORDER, h_min=H_MIN, h_max=H_MAX,
    )
    source = gcs.AddRegions(
        [Point(start_pose)], order=0, h_min=0, h_max=0, name="source",
    )
    target = gcs.AddRegions(
        [Point(goal_pose)], order=0, h_min=0, h_max=0, name="target",
    )
    source_to_main = gcs.AddEdges(source, main)
    main_to_target = gcs.AddEdges(main, target)

    source_to_main.AddZeroDerivativeConstraints(1)
    main_to_target.AddZeroDerivativeConstraints(1)
    source_to_main.AddZeroDerivativeConstraints(2)
    main_to_target.AddNonlinearDerivativeBounds(3 * [0.0], 3 * [0.0], 2)

    gcs.AddContinuityConstraints(1)
    gcs.AddContinuityConstraints(2)
    gcs.AddContinuityConstraints(3)
    gcs.AddContinuityConstraints(4)

    gcs.AddVelocityBounds(3 * [-QDT_MAX], 3 * [QDT_MAX])
    gcs.AddNonlinearDerivativeBounds(3 * [-QDDT_MAX], 3 * [QDDT_MAX], 2)

    gcs.AddTimeCost()
    gcs.AddPathLengthCost()

    traj, result = gcs.SolvePath(source, target, gcs_options)
    if not result.is_success():
        return None, result
    return traj, result
