from itertools import combinations
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pydot
import pydrake.geometry.optimization as opt
from pydrake.solvers import (
    ClarabelSolver,
    CommonSolverOption,
    MathematicalProgramResult,
    MosekSolver,
    SolverOptions,
)

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.face_contact import FaceContactMode
from planning_through_contact.geometry.planar.non_collision import NonCollisionMode
from planning_through_contact.geometry.planar.non_collision_subgraph import (
    NonCollisionSubGraph,
    VertexModePair,
    gcs_add_edge_with_continuity,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_path import (
    PlanarPushingPath,
)
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    PlanarSolverParams,
)

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge
BidirGcsEdge = Tuple[GcsEdge, GcsEdge]


class PlanarPushingPlanner:
    """
    A planner that generates motion plans for pushing an object (the "slider") with a point finger (the "pusher").
    The motion planner formulates the problem as a Graph-of-Convex-Sets problem, where each vertex in the graph
    corresponds to a contact mode.
    """

    def __init__(
        self,
        config: PlanarPlanConfig,
        contact_locations: Optional[List[PolytopeContactLocation]] = None,
    ):
        self.slider = config.dynamics_config.slider
        self.config = config

        self.source = None
        self.target = None
        self.relaxed_gcs_result = None

        if (
            self.config.non_collision_cost.avoid_object
            and config.num_knot_points_non_collision <= 2
        ):
            raise ValueError(
                "It is not possible to avoid object with only 2 knot points."
            )

        # TODO(bernhardpg): should just extract faces, rather than relying on the
        # object to only pass faces as contact locations
        self.contact_locations = contact_locations
        if self.contact_locations is None:
            self.contact_locations = self.slider.geometry.contact_locations

    def formulate_problem(self) -> None:
        assert self.config.start_and_goal is not None
        self.slider_pose_initial = self.config.start_and_goal.slider_initial_pose
        self.slider_pose_target = self.config.start_and_goal.slider_target_pose
        self.pusher_pose_initial = self.config.start_and_goal.pusher_initial_pose
        self.pusher_pose_target = self.config.start_and_goal.pusher_target_pose

        self.gcs = opt.GraphOfConvexSets()
        self._formulate_contact_modes()
        self._build_graph()

        # costs for non-collisions are added by each of the separate subgraphs
        for m, v in zip(self.contact_modes, self.contact_vertices):
            m.add_cost_to_vertex(v)

    @property
    def num_contact_modes(self) -> int:
        return len(self.contact_modes)

    def _formulate_contact_modes(self):
        assert self.contact_locations is not None

        if not all([loc.pos == ContactLocation.FACE for loc in self.contact_locations]):
            raise RuntimeError("Only face contacts are supported for planar pushing.")

        self.contact_modes = [
            FaceContactMode.create_from_plan_spec(
                loc,
                self.config,
            )
            for loc in self.contact_locations
        ]

        for mode in self.contact_modes:
            mode.add_so2_cut(
                self.slider_pose_initial.theta, self.slider_pose_target.theta
            )

    def _build_graph(self):
        self.contact_vertices = [
            self.gcs.AddVertex(mode.get_convex_set(), mode.name)
            for mode in self.contact_modes
        ]

        self.edges = {}
        if self.config.allow_teleportation:
            for i, j in combinations(range(self.num_contact_modes), 2):
                self.edges[(self.contact_modes[i].name, self.contact_modes[j].name)] = (
                    gcs_add_edge_with_continuity(
                        self.gcs,
                        VertexModePair(self.contact_vertices[i], self.contact_modes[i]),
                        VertexModePair(self.contact_vertices[j], self.contact_modes[j]),
                        only_continuity_on_slider=True,
                    )
                )
                self.edges[(self.contact_modes[j].name, self.contact_modes[i].name)] = (
                    gcs_add_edge_with_continuity(
                        self.gcs,
                        VertexModePair(self.contact_vertices[j], self.contact_modes[j]),
                        VertexModePair(self.contact_vertices[i], self.contact_modes[i]),
                        only_continuity_on_slider=True,
                    )
                )
        else:
            # connect contact modes through NonCollisionSubGraphs
            connections = list(combinations(range(self.num_contact_modes), 2))

            self.subgraphs = [
                self._build_subgraph_between_contact_modes(
                    mode_i, mode_j, self.config.no_cycles
                )
                for mode_i, mode_j in connections
            ]

            if self.config.use_entry_and_exit_subgraphs:
                self.source_subgraph = self._create_entry_or_exit_subgraph("entry")
                self.target_subgraph = self._create_entry_or_exit_subgraph("exit")

        self._set_initial_poses(self.pusher_pose_initial, self.slider_pose_initial)
        self._set_target_poses(self.pusher_pose_target, self.slider_pose_target)

    def _build_subgraph_between_contact_modes(
        self,
        first_contact_mode_idx: int,
        second_contact_mode_idx: int,
        no_cycles: bool = False,
    ) -> NonCollisionSubGraph:
        subgraph = NonCollisionSubGraph.create_with_gcs(
            self.gcs,
            self.config,
            f"FACE_{first_contact_mode_idx}_to_FACE_{second_contact_mode_idx}",
        )
        if no_cycles:  # only connect lower idx faces to higher idx faces
            if first_contact_mode_idx <= second_contact_mode_idx:
                incoming_idx = first_contact_mode_idx
                outgoing_idx = second_contact_mode_idx
            else:  # second_contact_mode_idx <= first_contact_mode_idx
                outgoing_idx = first_contact_mode_idx
                incoming_idx = second_contact_mode_idx

            subgraph.connect_with_continuity_constraints(
                self.slider.geometry.get_collision_free_region_for_loc_idx(
                    incoming_idx
                ),
                VertexModePair(
                    self.contact_vertices[incoming_idx],
                    self.contact_modes[incoming_idx],
                ),
                incoming=True,
                outgoing=False,
            )
            subgraph.connect_with_continuity_constraints(
                self.slider.geometry.get_collision_free_region_for_loc_idx(
                    outgoing_idx
                ),
                VertexModePair(
                    self.contact_vertices[outgoing_idx],
                    self.contact_modes[outgoing_idx],
                ),
                incoming=False,
                outgoing=True,
            )
        else:
            for idx in (first_contact_mode_idx, second_contact_mode_idx):
                subgraph.connect_with_continuity_constraints(
                    self.slider.geometry.get_collision_free_region_for_loc_idx(idx),
                    VertexModePair(
                        self.contact_vertices[idx],
                        self.contact_modes[idx],
                    ),
                )
        return subgraph

    def _get_all_vertex_mode_pairs(self) -> Dict[str, VertexModePair]:
        all_pairs = {
            v.name(): VertexModePair(vertex=v, mode=m)
            for v, m in zip(self.contact_vertices, self.contact_modes)
        }
        # Add all vertices from subgraphs
        if not self.config.allow_teleportation:
            for subgraph in self.subgraphs:
                all_pairs.update(subgraph.get_all_vertex_mode_pairs())

        # Add source and target vertices (and possibly the ones associated
        # with the entry and exit subgraphs)
        if (
            self.config.allow_teleportation
            or not self.config.use_entry_and_exit_subgraphs
        ):
            assert self.source is not None
            assert self.target is not None

            all_pairs[self.source.mode.name] = self.source
            all_pairs[self.target.mode.name] = self.target
        else:
            for subgraph in (self.source_subgraph, self.target_subgraph):
                all_pairs.update(subgraph.get_all_vertex_mode_pairs())

        return all_pairs

    def _create_entry_or_exit_subgraph(
        self, entry_or_exit: Literal["entry", "exit"]
    ) -> NonCollisionSubGraph:
        if entry_or_exit == "entry":
            name = "ENTRY"
            kwargs = {"outgoing": True, "incoming": False}
        else:
            name = "EXIT"
            kwargs = {"outgoing": False, "incoming": True}

        subgraph = NonCollisionSubGraph.create_with_gcs(self.gcs, self.config, name)

        for idx, (vertex, mode) in enumerate(
            zip(self.contact_vertices, self.contact_modes)
        ):
            subgraph.connect_with_continuity_constraints(
                self.slider.geometry.get_collision_free_region_for_loc_idx(idx),
                VertexModePair(vertex, mode),
                **kwargs,
            )
        return subgraph

    def _set_initial_poses(
        self,
        pusher_pose: PlanarPose,
        slider_pose: PlanarPose,
    ) -> None:
        if (
            self.config.allow_teleportation
            or not self.config.use_entry_and_exit_subgraphs
        ):
            self.source = self._add_single_source_or_target(
                pusher_pose, slider_pose, "initial"
            )
        else:
            self.source_subgraph.set_initial_poses(pusher_pose, slider_pose)
            self.source = self.source_subgraph.source

    def _set_target_poses(
        self,
        pusher_pose: PlanarPose,
        slider_pose: PlanarPose,
    ) -> None:
        if (
            self.config.allow_teleportation
            or not self.config.use_entry_and_exit_subgraphs
        ):
            self.target = self._add_single_source_or_target(
                pusher_pose, slider_pose, "final"
            )
        else:
            self.target_subgraph.set_final_poses(pusher_pose, slider_pose)
            self.target = self.target_subgraph.target

    def _add_single_source_or_target(
        self,
        pusher_pose: PlanarPose,
        slider_pose: PlanarPose,
        initial_or_final: Literal["initial", "final"],
    ) -> VertexModePair:
        set_slider_pose = True
        terminal_cost = False

        mode = NonCollisionMode.create_source_or_target_mode(
            self.config,
            slider_pose,
            pusher_pose,
            initial_or_final,
            set_slider_pose=set_slider_pose,
            terminal_cost=terminal_cost,
        )
        vertex = self.gcs.AddVertex(mode.get_convex_set(), mode.name)
        pair = VertexModePair(vertex, mode)

        if terminal_cost:  # add cost on target vertex
            mode.add_cost_to_vertex(vertex)

        # connect source or target to all contact modes
        if initial_or_final == "initial":
            # source to contact modes
            for contact_vertex, contact_mode in zip(
                self.contact_vertices, self.contact_modes
            ):
                self.edges[("source", contact_mode.name)] = (
                    gcs_add_edge_with_continuity(
                        self.gcs,
                        pair,
                        VertexModePair(contact_vertex, contact_mode),
                        only_continuity_on_slider=True,
                    )
                )
        else:  # contact modes to target
            for contact_vertex, contact_mode in zip(
                self.contact_vertices, self.contact_modes
            ):
                self.edges[(contact_mode.name, "target")] = (
                    gcs_add_edge_with_continuity(
                        self.gcs,
                        VertexModePair(contact_vertex, contact_mode),
                        pair,
                        only_continuity_on_slider=True,
                    )
                )

        return pair

    def _get_mosek_params(
        self,
        solver_params: PlanarSolverParams,
        tolerance: float = 1e-5,
        presolve: bool = True,
    ) -> SolverOptions:
        solver_options = SolverOptions()
        if solver_params.print_solver_output:
            # solver_options.SetOption(CommonSolverOption.kPrintFileName, "optimization_log.txt")  # type: ignore
            solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

        if solver_params.save_solver_output:
            solver_options.SetOption(CommonSolverOption.kPrintFileName, "solver_log.txt")  # type: ignore

        mosek = MosekSolver()
        solver_options.SetOption(
            mosek.solver_id(), "MSK_DPAR_INTPNT_CO_TOL_PFEAS", tolerance
        )
        solver_options.SetOption(
            mosek.solver_id(), "MSK_DPAR_INTPNT_CO_TOL_DFEAS", tolerance
        )
        solver_options.SetOption(
            mosek.solver_id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", tolerance
        )

        solver_options.SetOption(
            mosek.solver_id(),
            "MSK_DPAR_OPTIMIZER_MAX_TIME",
            solver_params.max_mosek_solve_time,
        )

        if not presolve:
            solver_options.SetOption(mosek.solver_id(), "MSK_IPAR_PRESOLVE_USE", 0)

        return solver_options

    def _solve(self, solver_params: PlanarSolverParams) -> MathematicalProgramResult:
        """
        Returns the relaxed GCS result, potentially with non-binary flow values.
        """
        options = opt.GraphOfConvexSetsOptions()

        options.convex_relaxation = solver_params.gcs_convex_relaxation
        if solver_params.gcs_convex_relaxation:
            options.preprocessing = True
            # We want to solve only the convex relaxation first
            options.max_rounded_paths = 0

        if solver_params.solver == "mosek":
            mosek = MosekSolver()
            options.solver = mosek
            options.solver_options = self._get_mosek_params(
                solver_params, 1e-4, presolve=False
            )
        else:  # clarabel
            clarabel = ClarabelSolver()
            options.solver = clarabel
            options.solver_options.SetOption(clarabel.solver_id(), "tol_feas", 1e-4)
            options.solver_options.SetOption(clarabel.solver_id(), "tol_gap_rel", 1e-4)
            options.solver_options.SetOption(clarabel.solver_id(), "tol_gap_abs", 1e-4)

        assert self.source is not None
        assert self.target is not None

        # TODO: The following commented out code allows you to pick which path to choose
        # active_vertices = ["source", "FACE_2", "FACE_0", "target"]
        # active_edges = [
        #     self.edges[(active_vertices[i], active_vertices[i + 1])]
        #     for i in range(len(active_vertices) - 1)
        # ]
        # result = self.gcs.SolveConvexRestriction(active_edges, options)

        result = self.gcs.SolveShortestPath(
            self.source.vertex, self.target.vertex, options
        )

        if solver_params.print_flows:
            self._print_edge_flows(result)

        return result

    def get_solution_paths(
        self,
        result: MathematicalProgramResult,
        solver_params: PlanarSolverParams,
        profile: Optional[dict[str, Any]] = None,
    ) -> Optional[List[PlanarPushingPath]]:
        """
        Returns N solution paths, sorted in increasing order based on optimal cost,
        where N = solver_params.rounding_steps.
        """
        assert self.source is not None
        assert self.target is not None

        options = opt.GraphOfConvexSetsOptions()
        options.max_rounded_paths = solver_params.rounding_steps
        options.max_rounding_trials = solver_params.max_rounding_trials

        options.convex_relaxation = True
        options.preprocessing = True

        options.solver_options = self._get_mosek_params(solver_params, 1e-5)

        t_sample = time.perf_counter()
        paths = self.gcs.SamplePaths(
            self.source.vertex, self.target.vertex, result, options
        )
        if profile is not None:
            profile["path_sampling_s"] = time.perf_counter() - t_sample
            profile["num_unique_paths"] = len({self._edge_path_key(path) for path in paths})

        entries: list[dict[str, Any]] = []
        results = []
        for i, path in enumerate(paths):
            t_solve = time.perf_counter()
            res = self.gcs.SolveConvexRestriction(path, options)
            solve_time_s = time.perf_counter() - t_solve
            results.append(res)
            entry = {
                "path_index": i,
                "convex_restriction_s": solve_time_s,
                "convex_restriction_success": res.is_success(),
                "relaxed_cost": float(res.get_optimal_cost()) if res.is_success() else None,
                "rounding_success": False,
                "rounding_s": None,
                "rounded_cost": None,
            }
            try:
                entry["convex_restriction_solver_s"] = float(res.get_solver_details().optimizer_time)  # type: ignore
            except Exception:
                entry["convex_restriction_solver_s"] = None
            entries.append(entry)

        flows = [result.GetSolution(e.phi()) for e in self.gcs.Edges()]

        paths_and_results = zip(paths, results, entries)
        only_successful_res = [
            pair for pair in paths_and_results if pair[1].is_success()
        ]

        if len(only_successful_res) == 0:
            if profile is not None:
                profile["paths"] = entries
            return None

        # if len(only_successful_res) == 0:
        #     raise RuntimeError("No trajectories rounded succesfully")

        sorted_res = sorted(
            only_successful_res, key=lambda pair: pair[1].get_optimal_cost()
        )
        paths, results, sorted_entries = zip(*sorted_res)
        if profile is not None:
            failed_entries = [entry for entry in entries if not entry["convex_restriction_success"]]
            profile["paths"] = list(sorted_entries) + failed_entries

        paths = [
            PlanarPushingPath.from_path(
                self.gcs,
                result,
                path,
                self._get_all_vertex_mode_pairs(),
                assert_nan_values=solver_params.assert_nan_values,
            )
            for path, result in zip(paths, results)
        ]
        if profile is not None:
            for path, entry in zip(paths, profile.get("paths", [])):
                entry["path_object_id"] = id(path)

        return paths

    def get_solution_paths_from_flows(
        self,
        edge_flows: np.ndarray,
        solver_params: PlanarSolverParams,
        *,
        max_paths: Optional[int] = None,
        max_steps: int = 512,
        seed: int = 0,
        profile: Optional[dict[str, Any]] = None,
    ) -> Optional[List[PlanarPushingPath]]:
        """
        Mimics the create_plans.py rounding pipeline, but *replaces* Drake's
        `GraphOfConvexSets.SamplePaths(..., result, ...)` with sampling directly from provided
        per-edge flows.

        Pipeline:
          1) Sample discrete edge-paths using probabilities proportional to `edge_flows`.
             (This is the flow-guided "path proposal" role played by SamplePaths(result, ...) in create_plans.)
          2) Solve convex restriction for each sampled path: SolveConvexRestriction(path, ...).
          3) Wrap as PlanarPushingPath.
             SNOPT nonlinear rounding is done later via `_get_rounded_paths` -> `PlanarPushingPath.do_rounding`.

        `edge_flows` must be aligned with `list(self.gcs.Edges())` (same order).
        """
        assert self.source is not None
        assert self.target is not None

        all_edges = list(self.gcs.Edges())
        edge_flows = np.asarray(edge_flows, dtype=np.float64).reshape((-1,))
        if edge_flows.shape[0] != len(all_edges):
            raise ValueError(
                f"edge_flows has length {edge_flows.shape[0]} but graph has {len(all_edges)} edges."
            )

        options = opt.GraphOfConvexSetsOptions()
        options.convex_relaxation = True
        options.preprocessing = True
        options.solver_options = self._get_mosek_params(solver_params, 1e-5)

        n_paths = int(solver_params.rounding_steps if max_paths is None else max_paths)
        rng = np.random.default_rng(int(seed))

        # Build outgoing adjacency for sampling.
        outgoing: Dict[str, List[int]] = {}
        for idx, e in enumerate(all_edges):
            outgoing.setdefault(e.u().name(), []).append(idx)

        def _sample_one() -> Optional[List[GcsEdge]]:
            cur = self.source.vertex.name()
            target = self.target.vertex.name()
            path_edge_indices: List[int] = []
            for _ in range(int(max_steps)):
                if cur == target:
                    break
                cand = outgoing.get(cur, [])
                if len(cand) == 0:
                    return None
                w = np.maximum(edge_flows[cand], 0.0)
                s = float(np.sum(w))
                if s <= 0:
                    return None
                # Flow-guided transition probabilities:
                #   p(e | v) = ϕ_e / Σ_{e' ∈ out(v)} ϕ_{e'}
                # implemented as: w = max(ϕ, 0), then p = w / sum(w)
                p = w / s
                choice = int(rng.choice(len(cand), p=p))
                eidx = int(cand[choice])
                path_edge_indices.append(eidx)
                cur = all_edges[eidx].v().name()
            if cur != target:
                return None
            return [all_edges[i] for i in path_edge_indices]

        sampled_paths: List[List[GcsEdge]] = []
        seen: set[tuple[str, ...]] = set()
        trials = 0
        max_trials = int(solver_params.max_rounding_trials)
        t_sample = time.perf_counter()
        while len(sampled_paths) < n_paths and trials < max_trials:
            trials += 1
            ep = _sample_one()
            if ep is None or len(ep) == 0:
                continue
            key = tuple(f"{e.u().name()}->{e.v().name()}" for e in ep)
            if key in seen:
                continue
            seen.add(key)
            sampled_paths.append(ep)
        if profile is not None:
            profile["path_sampling_s"] = time.perf_counter() - t_sample
            profile["num_unique_paths"] = len(sampled_paths)

        if len(sampled_paths) == 0:
            if profile is not None:
                profile["paths"] = []
            return None

        entries: list[dict[str, Any]] = []
        results = []
        for i, path in enumerate(sampled_paths):
            t_solve = time.perf_counter()
            res = self.gcs.SolveConvexRestriction(path, options)
            solve_time_s = time.perf_counter() - t_solve
            results.append(res)
            entry = {
                "path_index": i,
                "convex_restriction_s": solve_time_s,
                "convex_restriction_success": res.is_success(),
                "relaxed_cost": float(res.get_optimal_cost()) if res.is_success() else None,
                "rounding_success": False,
                "rounding_s": None,
                "rounded_cost": None,
            }
            try:
                entry["convex_restriction_solver_s"] = float(res.get_solver_details().optimizer_time)  # type: ignore
            except Exception:
                entry["convex_restriction_solver_s"] = None
            entries.append(entry)
        only_successful = [
            (path, res, entry)
            for path, res, entry in zip(sampled_paths, results, entries)
            if res.is_success()
        ]
        if len(only_successful) == 0:
            if profile is not None:
                profile["paths"] = entries
            return None

        sorted_res = sorted(only_successful, key=lambda pair: pair[1].get_optimal_cost())
        sampled_paths, results, sorted_entries = zip(*sorted_res)
        if profile is not None:
            failed_entries = [entry for entry in entries if not entry["convex_restriction_success"]]
            profile["paths"] = list(sorted_entries) + failed_entries

        all_pairs = self._get_all_vertex_mode_pairs()
        paths = [
            PlanarPushingPath.from_path(
                self.gcs,
                result,
                path,
                all_pairs,
                assert_nan_values=solver_params.assert_nan_values,
            )
            for path, result in zip(sampled_paths, results)
        ]
        if profile is not None:
            for path, entry in zip(paths, profile.get("paths", [])):
                entry["path_object_id"] = id(path)
        return list(paths)

    def _plan_paths(
        self, solver_params: PlanarSolverParams
    ) -> Optional[List[PlanarPushingPath]]:
        """
        Plans a path.
        """
        assert self.source is not None
        assert self.target is not None

        gcs_result = self._solve(solver_params)
        self.relaxed_gcs_result = gcs_result

        if solver_params.assert_result:
            assert gcs_result.is_success()
        else:
            if not gcs_result.is_success():
                print("WARNING: Solver did not find a solution!")

        if not gcs_result.is_success():
            return None

        if solver_params.measure_solve_time:
            print(
                f"Total elapsed optimization time: {gcs_result.get_solver_details().optimizer_time}"
            )

        if solver_params.print_cost:
            cost = gcs_result.get_optimal_cost()
            print(f"Cost: {cost}")

        # Get N paths from GCS rounding, pick the best one
        paths = self.get_solution_paths(
            gcs_result,
            solver_params,
        )

        if paths is None:
            print("No gcs paths found")
            return None
        else:
            if solver_params.print_rounding_details:
                print(f"num rounded paths: {len(paths)}")

            return paths

    def _get_rounded_paths(
        self,
        solver_params: PlanarSolverParams,
        paths: List[PlanarPushingPath],
        profile: Optional[dict[str, Any]] = None,
    ) -> Optional[List[PlanarPushingPath]]:
        if solver_params.rounding_steps > 0:
            profile_entries = profile.get("paths", []) if profile is not None else []
            for i, path in enumerate(paths):
                t_round = time.perf_counter()
                path.do_rounding(solver_params)
                rounding_s = getattr(path, "rounding_time", time.perf_counter() - t_round)
                if i < len(profile_entries):
                    entry = profile_entries[i]
                    entry["rounding_s"] = rounding_s
                    entry["rounding_success"] = path.rounded_result is not None and path.rounded_result.is_success()
                    entry["rounded_cost"] = (
                        float(path.rounded_result.get_optimal_cost())
                        if entry["rounding_success"]
                        else None
                    )

            feasible_paths = [
                p
                for p in paths
                if p.rounded_result is not None and p.rounded_result.is_success()
            ]

            if solver_params.print_rounding_details:
                print(f"num rounded feasible paths: {len(feasible_paths)}")

            if len(feasible_paths) == 0:
                return None
            else:
                return feasible_paths
        else:
            raise NotImplementedError("Must enable rounding steps")

    def _pick_best_path(self, paths: List[PlanarPushingPath]) -> PlanarPushingPath:
        rounded_costs = [
            p.rounded_result.get_optimal_cost()
            for p in paths
            if p.rounded_result is not None  # type
        ]

        best_idx = np.argmin(rounded_costs)
        path = paths[best_idx]

        return path

    @staticmethod
    def _edge_path_key(path: List[GcsEdge]) -> tuple[str, ...]:
        return tuple(f"{e.u().name()}->{e.v().name()}" for e in path)

    @staticmethod
    def _fmt_profile_value(value: Any, suffix: str = "") -> str:
        if value is None:
            return "N/A"
        return f"{float(value):.3f}{suffix}"

    def print_rounding_profile(
        self,
        output_name: str,
        profile: dict[str, Any],
        chosen_path: Optional[PlanarPushingPath],
    ) -> None:
        print(
            f"[{output_name}] Path sampling time: "
            f"{self._fmt_profile_value(profile.get('path_sampling_s'), ' s')}"
        )
        print(f"[{output_name}] Unique sampled paths: {profile.get('num_unique_paths', 0)}")
        entries = profile.get("paths", [])
        chosen_idx = None
        chosen_cost = None
        if chosen_path is not None:
            for i, entry in enumerate(entries):
                if entry.get("path_object_id") == id(chosen_path):
                    chosen_idx = i
                    chosen_cost = entry.get("rounded_cost")
                    break
        for i, entry in enumerate(entries):
            status = "success" if entry.get("rounding_success") else "failed"
            print(
                f"[{output_name}] path_{i}: "
                f"restriction_s={self._fmt_profile_value(entry.get('convex_restriction_s'))}, "
                f"restriction_cost={self._fmt_profile_value(entry.get('relaxed_cost'))}, "
                f"rounding_s={self._fmt_profile_value(entry.get('rounding_s'))}, "
                f"rounded_cost={self._fmt_profile_value(entry.get('rounded_cost'))}, "
                f"status={status}"
            )
        if chosen_idx is None:
            print(f"[{output_name}] Chosen path: N/A")
        else:
            print(
                f"[{output_name}] Chosen path: path_{chosen_idx} "
                f"(cost={self._fmt_profile_value(chosen_cost)})"
            )

    def plan_path(
        self, solver_params: PlanarSolverParams
    ) -> Optional[PlanarPushingPath]:
        paths = self._plan_paths(solver_params)
        if paths is None:
            return None

        feasible_paths = self._get_rounded_paths(solver_params, paths)
        if feasible_paths is None:
            return None

        self.path = self._pick_best_path(feasible_paths)

        if solver_params.print_path:
            print(f"path: {self.path.get_path_names()}")

        return self.path

    def _print_edge_flows(self, result: MathematicalProgramResult) -> None:
        """
        Used for debugging.
        """
        edge_phis = {
            (e.u().name(), e.v().name()): result.GetSolution(e.phi())
            for e in self.gcs.Edges()
        }
        sorted_flows = sorted(edge_phis.items(), key=lambda item: item[0])
        for name, flow in sorted_flows:
            print(f"{name}: {flow}")

    def create_graph_diagram(
        self,
        filename: Optional[str] = None,
        result: Optional[MathematicalProgramResult] = None,
    ) -> pydot.Dot:
        """
        Optionally saves the graph to file if a string is given for the 'filepath' argument.
        """
        if result:
            graphviz = self.gcs.GetGraphvizString(
                result=result, show_slacks=False, precision=2, active_path=[]
            )
        else:
            graphviz = self.gcs.GetGraphvizString(
                show_slacks=False, precision=2, active_path=[]
            )

        data = pydot.graph_from_dot_data(graphviz)[0]  # type: ignore
        if filename is not None:
            data.write_png(filename + ".png")

        return data
