import os
import pickle
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydrake.solvers import MathematicalProgramResult

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.non_collision import (
    check_finger_pose_in_contact_location,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_path import (
    PlanarPushingPath,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    PlanarPushingWorkspace,
    PlanarSolverParams,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)
from planning_through_contact.visualize.colors import COLORS
from planning_through_contact.visualize.planar_pushing import (
    compare_trajs,
    make_traj_figure,
    plot_forces,
    visualize_planar_pushing_start_and_goal,
    visualize_planar_pushing_trajectory,
)


@dataclass
class SingleRunResult:
    relaxed_gcs_cost: float
    relaxed_gcs_success: bool
    relaxed_gcs_time: float
    binary_flows_cost: Optional[float]
    binary_flows_success: bool
    binary_flows_time: Optional[float]
    feasible_cost: Optional[float]
    feasible_success: Optional[bool]
    feasible_time: Optional[float]
    relaxed_mean_determinant: Optional[float]
    rounded_mean_determinant: Optional[float]
    start_and_goal: PlanarPushingStartAndGoal
    config: PlanarPlanConfig
    num_binary_rounded_paths: Optional[int] = None
    num_feasible_rounded_paths: Optional[int] = None
    solver_params: Optional[PlanarSolverParams] = None
    name: Optional[str] = None
    cost_term_vals: Optional[Dict[str, Dict]] = None

    @property
    def total_rounding_time(self) -> Optional[float]:
        if self.binary_flows_time is None or self.feasible_time is None:
            return None
        else:
            return self.binary_flows_time + self.feasible_time

    @property
    def optimality_gap(self) -> Optional[float]:
        if self.feasible_cost is None or self.numerical_difficulties:
            return None
        else:
            return (
                (self.feasible_cost - self.relaxed_gcs_cost) / self.relaxed_gcs_cost
            ) * 100

    @property
    def binary_flows_optimality_gap(self) -> Optional[float]:
        if self.binary_flows_cost is None:
            return None
        else:
            return (
                (self.binary_flows_cost - self.relaxed_gcs_cost) / self.relaxed_gcs_cost
            ) * 100

    @property
    def optimality_percentage(self) -> Optional[float]:
        if self.optimality_gap is None:
            return None
        else:
            return 100 - self.optimality_gap

    @property
    def binary_flows_optimality_percentage(self) -> Optional[float]:
        if self.binary_flows_optimality_gap is None:
            return None
        else:
            return 100 - self.binary_flows_optimality_gap

    @property
    def distance(self) -> float:
        start = self.start_and_goal.slider_initial_pose.pos()
        end = self.start_and_goal.slider_target_pose.pos()
        dist = float(np.linalg.norm(start - end))
        return dist

    @property
    def numerical_difficulties(self) -> Optional[bool]:
        if self.feasible_cost is None or self.binary_flows_cost is None:
            return None
        else:
            TOL = 1e-2
            return self.feasible_cost < self.binary_flows_cost - TOL

    def save(self, filename: str) -> None:
        with open(Path(filename), "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str) -> "SingleRunResult":
        with open(Path(filename), "rb") as file:
            run_result = pickle.load(file)
            run_result.name = filename
            return run_result

    def __str__(self):
        # Manually add property strings
        property_strings = [
            f"optimality_gap: {self.optimality_gap}",
            f"sdp_optimality_gap: {self.binary_flows_optimality_gap}",
        ]

        field_attributes = {
            field.name: getattr(self, field.name) for field in fields(self)
        }
        field_strings = []

        # Avoid printing scientific and with too many decimals
        np.set_printoptions(precision=3, suppress=True)

        for key, val in field_attributes.items():
            if type(val) is dict:
                field_strings.append(f"{key}: ----")
                for d_key, d_val in val.items():
                    if type(d_val) is dict:
                        field_strings.append(f"   {d_key}: ----")
                        for dd_key, dd_val in d_val.items():
                            field_strings.append(f"      {dd_key}: {dd_val}")
                    else:
                        field_strings.append(f"   {d_key}: {d_val}")
            else:
                field_strings.append(f"{key}: {val}")

        # Combine field and property strings
        all_strings = field_strings + property_strings
        return "\n".join(all_strings)

    def save_as_text(self, filename: str) -> None:
        with open(filename, "w") as file:
            file.write(str(self))


def do_one_run_get_path(
    plan_config: PlanarPlanConfig,
    solver_params: PlanarSolverParams,
    start_and_goal: PlanarPushingStartAndGoal,
    save_cost_vals: bool = False,
    graph_filename: Optional[str] = None,
) -> Tuple[SingleRunResult, Optional[PlanarPushingPath]]:
    plan_config.start_and_goal = start_and_goal

    planner = PlanarPushingPlanner(plan_config)
    planner.formulate_problem()

    if graph_filename is not None:
        planner.create_graph_diagram(graph_filename)

    paths = planner._plan_paths(solver_params)
    if paths is None:
        num_binary_rounded_paths = 0
        num_feasible_rounded_paths = None
        path = None

        binary_flows_cost = None
        binary_flows_success = False
        binary_flows_time = None

        feasible_success = False
        feasible_cost = None
        feasible_time = None
    else:
        num_binary_rounded_paths = len(paths)

        feasible_paths = planner._get_rounded_paths(solver_params, paths)
        if feasible_paths is None:
            num_feasible_rounded_paths = 0

            # Still record binary path
            binary_flows_best_idx = np.argmin([p.relaxed_cost for p in paths])
            path = paths[binary_flows_best_idx]
            binary_flows_success = True
            binary_flows_cost = path.relaxed_cost
            binary_flows_time = path.solve_time

            feasible_success = False
            feasible_cost = None
            feasible_time = None
        else:
            num_feasible_rounded_paths = len(feasible_paths)
            path = planner._pick_best_path(feasible_paths)

            binary_flows_success = True
            binary_flows_cost = path.relaxed_cost
            binary_flows_time = path.solve_time

            feasible_success = True
            feasible_cost = path.rounded_cost
            feasible_time = path.rounding_time

    assert planner.source is not None  # avoid typing errors
    assert planner.target is not None  # avoid typing errors

    relaxed_mean_determinant = (
        float(np.mean(path.get_determinants())) if path is not None else None
    )

    rounded_mean_determinant = (
        float(np.mean(path.get_determinants(rounded=True)))
        if path is not None
        else None
    )

    assert planner.relaxed_gcs_result is not None

    return (
        SingleRunResult(
            relaxed_gcs_cost=planner.relaxed_gcs_result.get_optimal_cost(),
            relaxed_gcs_success=planner.relaxed_gcs_result.is_success(),
            relaxed_gcs_time=planner.relaxed_gcs_result.get_solver_details().optimizer_time,  # type: ignore
            binary_flows_cost=binary_flows_cost,
            binary_flows_success=binary_flows_success,
            binary_flows_time=binary_flows_time,
            feasible_cost=feasible_cost,
            feasible_success=feasible_success,
            feasible_time=feasible_time,
            relaxed_mean_determinant=relaxed_mean_determinant,
            rounded_mean_determinant=rounded_mean_determinant,
            start_and_goal=start_and_goal,
            config=plan_config,
            cost_term_vals=(
                path.get_cost_terms() if path is not None and save_cost_vals else None
            ),
            solver_params=solver_params,
            num_binary_rounded_paths=num_binary_rounded_paths,
            num_feasible_rounded_paths=num_feasible_rounded_paths,
        ),
        path,
    )


def do_one_run(
    plan_config: PlanarPlanConfig,
    solver_params: PlanarSolverParams,
    start_and_goal: PlanarPushingStartAndGoal,
):
    run, _ = do_one_run_get_path(plan_config, solver_params, start_and_goal)
    return run


def sample_random_plan(
    x_and_y_limits: Tuple[float, float, float, float] = (-0.5, 0.5, -0.5, 0.5),
    slider_target_pose: Optional[PlanarPose] = None,
):
    x_min, x_max, y_min, y_max = x_and_y_limits

    # Default target is origin
    if slider_target_pose is None:
        slider_target_pose = PlanarPose(0, 0, 0)

    # Draw random initial pose for slider
    x_initial = np.random.uniform(x_min, x_max)
    y_initial = np.random.uniform(y_min, y_max)
    th_initial = np.random.uniform(-np.pi + 0.1, np.pi - 0.1)

    slider_initial_pose = PlanarPose(x_initial, y_initial, th_initial)

    # Fix pusher pose to upper right corner, outside of where the
    # slider will be
    BUFFER = 0.5  # This is just a hardcoded distance number
    pusher_pose = PlanarPose(x_max + BUFFER, y_max + BUFFER, 0)

    plan = PlanarPushingStartAndGoal(
        slider_initial_pose, slider_target_pose, pusher_pose, pusher_pose
    )
    return plan


def _slider_within_workspace(
    workspace: PlanarPushingWorkspace, pose: PlanarPose, slider: CollisionGeometry
) -> bool:
    """
    Checks whether the entire slider is within the workspace
    """
    R_WB = pose.two_d_rot_matrix()
    p_WB = pose.pos()

    p_Wv_s = [
        slider.get_p_Wv_i(vertex_idx, R_WB, p_WB).flatten()
        for vertex_idx in range(len(slider.vertices))
    ]

    lb, ub = workspace.slider.bounds
    vertices_within_workspace: bool = np.all([v <= ub for v in p_Wv_s]) and np.all(  # type: ignore
        [v >= lb for v in p_Wv_s]
    )
    return vertices_within_workspace


def _check_collision(
    pusher_pose_world: PlanarPose,
    slider_pose_world: PlanarPose,
    config: PlanarPlanConfig,
) -> bool:
    p_WP = pusher_pose_world.pos()
    R_WB = slider_pose_world.two_d_rot_matrix()
    p_WB = slider_pose_world.pos()

    # We need to compute the pusher pos in the frame of the slider
    p_BP = R_WB.T @ (p_WP - p_WB)
    pusher_pose_body = PlanarPose(p_BP[0, 0], p_BP[1, 0], 0)

    # we always add all non-collision modes, even when we don't add all contact modes
    # (think of maneuvering around the object etc)
    locations = [
        PolytopeContactLocation(ContactLocation.FACE, idx)
        for idx in range(config.slider_geometry.num_collision_free_regions)
    ]
    matching_locs = [
        loc
        for loc in locations
        if check_finger_pose_in_contact_location(pusher_pose_body, loc, config)
    ]
    if len(matching_locs) == 0:
        return True
    else:
        return False


def _get_slider_pose_within_workspace(
    workspace: PlanarPushingWorkspace,
    slider: CollisionGeometry,
    pusher_pose: PlanarPose,
    config: PlanarPlanConfig,
    limit_rotations: bool = False,
    enforce_entire_slider_within_workspace: bool = True,
) -> PlanarPose:
    valid_pose = False

    slider_pose = None
    while not valid_pose:
        x_initial = np.random.uniform(workspace.slider.x_min, workspace.slider.x_max)
        y_initial = np.random.uniform(workspace.slider.y_min, workspace.slider.y_max)
        EPS = 0.01
        if limit_rotations:
            th_initial = np.random.uniform(-np.pi / 2 + EPS, np.pi / 2 - EPS)
        else:
            th_initial = np.random.uniform(-np.pi + EPS, np.pi - EPS)

        slider_pose = PlanarPose(x_initial, y_initial, th_initial)

        collides_with_pusher = _check_collision(pusher_pose, slider_pose, config)
        within_workspace = _slider_within_workspace(workspace, slider_pose, slider)

        if enforce_entire_slider_within_workspace:
            valid_pose = within_workspace and not collides_with_pusher
        else:
            valid_pose = not collides_with_pusher

    assert slider_pose is not None  # fix LSP errors

    return slider_pose


def get_plan_start_and_goals_to_point(
    seed: int,
    num_plans: int,
    workspace: PlanarPushingWorkspace,
    config: PlanarPlanConfig,
    point: Tuple[float, float] = (0, 0),  # Default is origin
    limit_rotations: bool = True,  # Use this to start with
) -> List[PlanarPushingStartAndGoal]:
    # We want the plans to always be the same
    np.random.seed(seed)

    slider = config.slider_geometry

    # Hardcoded pusher start pose to be at top edge
    # of workspace
    ws = workspace.slider.new_workspace_with_buffer(new_buffer=0)
    pusher_pose = PlanarPose(ws.x_min, 0, 0)

    plans = []
    from tqdm import tqdm

    print(f"Sampling {num_plans} random initial conditions with random seed {seed}")

    for _ in tqdm(range(num_plans)):
        slider_initial_pose = _get_slider_pose_within_workspace(
            workspace, slider, pusher_pose, config, limit_rotations
        )

        slider_target_pose = PlanarPose(point[0], point[1], 0)

        plans.append(
            PlanarPushingStartAndGoal(
                slider_initial_pose, slider_target_pose, pusher_pose, pusher_pose
            )
        )

    return plans


def create_plan(
    start_and_target: PlanarPushingStartAndGoal,
    config: PlanarPlanConfig,
    solver_params: PlanarSolverParams,
    output_folder: str = "",
    output_name: str = "Untitled_traj",
    save_video: bool = True,
    do_rounding: bool = True,
    interpolate_video: bool = False,
    animation_lims: Optional[Tuple[float, float, float, float]] = None,
    save_traj: bool = True,
    debug: bool = False,
    save_relaxed: bool = False,
    save_graph_edge_labels: bool = False,
) -> SingleRunResult | None:
    """
    Creates a planar pushing plan.

    @param start_and_target: Starting and target configuration for the system.
    @param config: Config for the system and planner.
    @param solver_params: Parameters for the underlying optimization solver.
    """
    # Set up folders
    folder_name = f"{output_folder}/{output_name}"
    video_folder = folder_name
    os.makedirs(folder_name, exist_ok=True)
    trajectory_folder = f"{folder_name}/trajectory"
    os.makedirs(trajectory_folder, exist_ok=True)
    # NOTE: analysis/ablation utilities were removed during pruning; keep outputs minimal.

    if debug:
        visualize_planar_pushing_start_and_goal(
            config.dynamics_config.slider.geometry,
            config.dynamics_config.pusher_radius,
            start_and_target,
            # show=True,
            save=True,
            filename=f"{folder_name}/start_and_goal",
        )

    planner: PlanarPushingPlanner | None = None
    planner = PlanarPushingPlanner(config)
    planner.config.start_and_goal = start_and_target
    planner.formulate_problem()

    # --- Timing breakdown (matched to paper-style reporting) ---
    # 1) "SDP solve time" = solver-reported time for the convex relaxation solve.
    relaxed_result = planner._solve(solver_params)
    planner.relaxed_gcs_result = relaxed_result
    sdp_solve_time_s = float(relaxed_result.get_solver_details().optimizer_time)  # type: ignore

    # 2) "Rounding time" = SNOPT-only wall-clock solve time for the chosen feasible rounded path.
    # Note: the overall rounding *pipeline* (sampling paths + convex restrictions + SNOPT across candidates)
    # can be much larger; we still record it separately for debugging.
    rounding_pipeline_wall_time_s: float | None = None
    nonlinear_rounding_time_s: float | None = None

    path = None
    if do_rounding:
        t_round_start = time.perf_counter()
        candidate_paths = planner.get_solution_paths(relaxed_result, solver_params)
        if candidate_paths is not None:
            feasible_paths = planner._get_rounded_paths(
                solver_params, list(candidate_paths)
            )
            rounding_pipeline_wall_time_s = time.perf_counter() - t_round_start
            if feasible_paths is not None:
                path = planner._pick_best_path(feasible_paths)
                rt = getattr(path, "rounding_time", None)
                nonlinear_rounding_time_s = None if rt is None else float(rt)
        else:
            rounding_pipeline_wall_time_s = time.perf_counter() - t_round_start
    else:
        candidate_paths = planner.get_solution_paths(relaxed_result, solver_params)
        if candidate_paths is not None:
            # Pick a binary path (no nonlinear rounding) if requested.
            path = candidate_paths[0] if hasattr(candidate_paths, "__getitem__") else None

    print(f"[{output_name}] SDP solve time: {sdp_solve_time_s:.3f} s")
    if nonlinear_rounding_time_s is None:
        print(f"[{output_name}] Rounding time (SNOPT only): N/A")
    else:
        print(f"[{output_name}] Rounding time (SNOPT only): {nonlinear_rounding_time_s:.3f} s")
    if rounding_pipeline_wall_time_s is not None and debug:
        print(
            f"[{output_name}] Rounding pipeline time (sample+restrictions+SNOPT): {rounding_pipeline_wall_time_s:.3f} s"
        )

    # We may get infeasible
    if path is not None:
        traj_relaxed = path.to_traj()

        if do_rounding:
            traj_rounded = path.to_traj(rounded=True)
        else:
            traj_rounded = None

        if save_traj:
            if save_relaxed:
                traj_relaxed.save(f"{trajectory_folder}/traj_relaxed.pkl")  # type: ignore

            if traj_rounded is not None:
                traj_rounded.save(f"{trajectory_folder}/traj_rounded.pkl")  # type: ignore

        slider_color = COLORS["aquamarine4"].diffuse()

        if save_relaxed:
            make_traj_figure(
                traj_relaxed,
                filename=f"{trajectory_folder}/relaxed_traj",
                slider_color=slider_color,
                split_on_mode_type=True,
                show_workspace=False,
            )

        if traj_rounded is not None:
            make_traj_figure(
                traj_rounded,
                filename=f"{trajectory_folder}/rounded_traj",
                slider_color=slider_color,
                split_on_mode_type=True,
                show_workspace=False,
            )

        if save_video:
            if save_relaxed:
                ani = visualize_planar_pushing_trajectory(
                    traj_relaxed,  # type: ignore
                    save=True,
                    filename=f"{video_folder}/relaxed_traj",
                    visualize_knot_points=not interpolate_video,
                    lims=animation_lims,
                )

            if traj_rounded is not None:
                ani = visualize_planar_pushing_trajectory(
                    traj_rounded,  # type: ignore
                    save=True,
                    filename=f"{video_folder}/rounded_traj",
                    visualize_knot_points=not interpolate_video,
                    lims=animation_lims,
                )

    if save_graph_edge_labels:
        # Export the full directed edge list with a 0/1 label per edge,
        # where 1 indicates the edge is on the chosen discrete path.
        import json

        try:
            import pydot  # type: ignore
        except ModuleNotFoundError:
            pydot = None  # type: ignore

        def _edge_key(e) -> tuple[str, str]:
            return (e.u().name(), e.v().name())

        # Use the already-built planner when available; otherwise rebuild the graph (no solves).
        planner_for_labels = planner
        if planner_for_labels is None:
            planner_for_labels = PlanarPushingPlanner(config)
            planner_for_labels.config.start_and_goal = start_and_target
            planner_for_labels.formulate_problem()

        all_edges = list(planner_for_labels.gcs.Edges())
        all_edge_keys = [_edge_key(e) for e in all_edges]

        # Best-effort: grab relaxed (convex-relaxation) flow values for each edge.
        # These live on the relaxed GCS solve result, not on the rounded path.
        relaxed_result: MathematicalProgramResult | None = getattr(
            planner_for_labels, "relaxed_gcs_result", None
        )
        phi_vals: list[float | None] = []
        if relaxed_result is not None:
            for e in all_edges:
                try:
                    phi_vals.append(float(relaxed_result.GetSolution(e.phi())))
                except Exception:
                    phi_vals.append(None)
        else:
            phi_vals = [None for _ in all_edges]

        path_edge_keys: set[tuple[str, str]] = set()
        path_edge_order: dict[tuple[str, str], list[int]] = {}
        path_vertex_names: list[str] = []
        chosen_path_kind = "none"
        if path is not None:
            path_edge_keys = {_edge_key(e) for e in path.edges}
            for idx, e in enumerate(path.edges, start=1):
                path_edge_order.setdefault(_edge_key(e), []).append(idx)
            path_vertex_names = path.get_path_names()
            if path.rounded_result is not None and path.rounded_result.is_success():
                chosen_path_kind = "feasible"
            else:
                chosen_path_kind = "binary"

        y = [1 if key in path_edge_keys else 0 for key in all_edge_keys]

        out_dir = Path(folder_name) / "graph_edge_labels"
        out_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "chosen_path_kind": chosen_path_kind,  # "feasible" | "binary" | "none"
            "relaxed_cost": (
                None
                if relaxed_result is None
                else float(relaxed_result.get_optimal_cost())
            ),
            "relaxed_solve_time": (
                None
                if relaxed_result is None
                else float(relaxed_result.get_solver_details().optimizer_time)  # type: ignore
            ),
            # Paper-style rounding time: SNOPT-only time for the chosen feasible rounded path.
            "rounding_time": nonlinear_rounding_time_s,
            # For debugging only: includes path sampling + convex restrictions + SNOPT across candidates.
            "rounding_pipeline_time": rounding_pipeline_wall_time_s,
            "path_cost": (
                None
                if path is None
                else (
                    float(path.rounded_result.get_optimal_cost())
                    if (path.rounded_result is not None and path.rounded_result.is_success())
                    else float(path.relaxed_cost)
                )
            ),
            "num_vertices": len(list(planner_for_labels.gcs.Vertices())),
            "num_edges": len(all_edges),
            "num_positive_edges": int(sum(y)),
            "path_vertex_names": path_vertex_names,
            "edges": [
                {
                    "u": u,
                    "v": v,
                    "y": int(label),
                    "phi": (None if phi is None else float(phi)),
                    **(
                        {"order": path_edge_order.get((u, v), [])}
                        if int(label) == 1
                        else {}
                    ),
                }
                for ((u, v), label, phi) in zip(all_edge_keys, y, phi_vals)
            ],
        }
        (out_dir / "edges.json").write_text(json.dumps(payload, indent=2))

        # Best-effort visualization (requires pydot; SVG also requires graphviz).
        if pydot is not None:
            dot = pydot.Dot(graph_type="digraph", rankdir="LR")

            for v in planner_for_labels.gcs.Vertices():
                name = v.name()
                attrs: dict[str, str] = {}
                if name == "source":
                    attrs.update({"shape": "doublecircle", "color": "darkgreen"})
                elif name == "target":
                    attrs.update({"shape": "doublecircle", "color": "darkblue"})
                dot.add_node(pydot.Node(name, **attrs))

            def _fmt_phi(phi: float | None) -> str:
                return "?" if phi is None else f"{phi:.3f}"

            for (u, v), label, phi in zip(all_edge_keys, y, phi_vals):
                if label == 1:
                    order = path_edge_order.get((u, v), [])
                    order_str = ",".join(str(i) for i in order) if len(order) > 0 else "?"
                    edge_attrs = {
                        # Use an HTML-like label so we can color only the traversal index.
                        "label": f'<<FONT COLOR="black">1</FONT><FONT COLOR="blue"> {order_str}</FONT><FONT COLOR="gray40">, {_fmt_phi(phi)}</FONT>>',
                        "color": "red",
                        "penwidth": "3",
                    }
                else:
                    edge_attrs = {
                        "label": f'<<FONT COLOR="black">0</FONT><FONT COLOR="gray40">, {_fmt_phi(phi)}</FONT>>',
                        "color": "gray70",
                        "penwidth": "1",
                    }
                dot.add_edge(pydot.Edge(u, v, **edge_attrs))

            dot_path = out_dir / "graph_edge_labels.dot"
            dot.write_raw(str(dot_path))
            try:
                dot.write_svg(str(out_dir / "graph_edge_labels.svg"))
            except Exception:
                pass

    return None
