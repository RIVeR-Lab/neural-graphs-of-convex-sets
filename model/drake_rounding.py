from __future__ import annotations

import csv
import time
from typing import Any, Optional

import numpy as np
import torch

from model.inference import project_flows_qp
from model.model import GCSFlowPredictor
from planning_through_contact.geometry.planar.planar_pushing_path import PlanarPushingPath
from planning_through_contact.planning.planar.planar_plan_config import PlanarSolverParams
from planning_through_contact.planning.planar.planar_pushing_planner import PlanarPushingPlanner


def round_from_predicted_flows(
    *,
    planner: PlanarPushingPlanner,
    edge_flows: np.ndarray,
    solver_params: PlanarSolverParams,
    max_paths: Optional[int] = None,
    max_steps: int = 512,
    seed: int = 0,
    profile: Optional[dict[str, Any]] = None,
) -> Optional[PlanarPushingPath]:
    """
    Run the planar pushing rounding stack with predicted edge flows:

      flows -> (sample discrete paths) -> SolveConvexRestriction -> SNOPT nonlinear rounding -> pick best
    """
    candidate_paths = planner.get_solution_paths_from_flows(
        edge_flows=edge_flows,
        solver_params=solver_params,
        max_paths=max_paths,
        max_steps=max_steps,
        seed=seed,
        profile=profile,
    )
    if candidate_paths is None:
        return None
    feasible_paths = planner._get_rounded_paths(solver_params, list(candidate_paths), profile=profile)
    if feasible_paths is None:
        return None
    return planner._pick_best_path(feasible_paths)


def build_gnn_batch_for_planner(
    *,
    planner: PlanarPushingPlanner,
    g: np.ndarray,
    node_features_csv: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build (x, edge_index, g) tensors for the GNN, in the same order as predict_edge_flows_for_planner.
    Returns CPU tensors; caller can .to(device) once for profiling.
    """
    if not hasattr(planner, "gcs"):
        raise ValueError("Planner has no graph yet. Did you call planner.formulate_problem()?")

    all_edges = list(planner.gcs.Edges())
    edge_u = [e.u().name() for e in all_edges]
    edge_v = [e.v().name() for e in all_edges]
    node_names = sorted(set(edge_u) | set(edge_v))

    with open(node_features_csv, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"Empty CSV: {node_features_csv}")
        x_cols = [c for c in reader.fieldnames if c.startswith("x_")]
        rows = {r["node_name"]: np.array([float(r[c]) for c in x_cols], dtype=np.float32) for r in reader}

    x_dim = int(next(iter(rows.values())).shape[0]) if len(rows) > 0 else 0
    x_mat = []
    for n in node_names:
        if n in rows:
            x_mat.append(rows[n])
        else:
            x_mat.append(np.zeros((x_dim,), dtype=np.float32))
    x = torch.tensor(np.stack(x_mat, axis=0), dtype=torch.float32)

    name_to_idx = {n: i for i, n in enumerate(node_names)}
    src = torch.tensor([name_to_idx[u] for u in edge_u], dtype=torch.long)
    dst = torch.tensor([name_to_idx[v] for v in edge_v], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    # Verify edge ordering matches planner (same check as predict_edge_flows_for_planner).
    for i, e in enumerate(all_edges):
        if edge_u[i] != e.u().name() or edge_v[i] != e.v().name():
            raise RuntimeError(
                f"Edge ordering mismatch at index {i}: "
                f"expected ({e.u().name()}, {e.v().name()}), got ({edge_u[i]}, {edge_v[i]})."
            )

    g_t = torch.tensor(g, dtype=torch.float32)
    return x, edge_index, g_t


def predict_edge_flows_for_planner(
    *,
    planner: PlanarPushingPlanner,
    model: GCSFlowPredictor,
    g: np.ndarray | torch.Tensor,
    node_features_csv: str = "planning_through_contact/dataset/data/sugar_box/node_features.csv",
    device: Optional[torch.device] = None,
    enforce_flow_conservation: bool = True,
    timings: Optional[dict] = None,
) -> np.ndarray:
    """
    Builds the graph for `planner` in the exact edge order `list(planner.gcs.Edges())`,
    runs the GNN, and returns per-edge flows aligned with that order.

    If enforce_flow_conservation=True (QP): project sigmoid output onto the flow polytope (Eq. 39).
    If enforce_flow_conservation=False (direct): return raw sigmoid output; path sampling will
    normalize per node (Direct Randomised Rounding, Sec. H).

    If timings is a dict (e.g. {}), fills timings["gnn_s"] (GNN forward+sigmoid) and, when
    enforce_flow_conservation, timings["qp_s"] (QP projection time in seconds).
    """
    if not hasattr(planner, "gcs"):
        raise ValueError("Planner has no graph yet. Did you call planner.formulate_problem()?")

    all_edges = list(planner.gcs.Edges())
    edge_u = [e.u().name() for e in all_edges]
    edge_v = [e.v().name() for e in all_edges]
    node_names = sorted(set(edge_u) | set(edge_v))

    # Load static node features.
    with open(node_features_csv, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"Empty CSV: {node_features_csv}")
        x_cols = [c for c in reader.fieldnames if c.startswith("x_")]
        rows = {r["node_name"]: np.array([float(r[c]) for c in x_cols], dtype=np.float32) for r in reader}

    x_dim = int(next(iter(rows.values())).shape[0]) if len(rows) > 0 else 0
    x_mat = []
    for n in node_names:
        if n in rows:
            x_mat.append(rows[n])
        else:
            # source/target (and any other unexpected nodes) get zero features
            x_mat.append(np.zeros((x_dim,), dtype=np.float32))
    x = torch.tensor(np.stack(x_mat, axis=0), dtype=torch.float32)

    name_to_idx = {n: i for i, n in enumerate(node_names)}
    src = torch.tensor([name_to_idx[u] for u in edge_u], dtype=torch.long)
    dst = torch.tensor([name_to_idx[v] for v in edge_v], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    # Defensive check: verify edge ordering matches planner's expectation.
    # Critical: model outputs must align with list(planner.gcs.Edges()) order.
    if len(edge_u) != len(all_edges):
        raise RuntimeError(
            f"Edge count mismatch: built {len(edge_u)} edges but planner has {len(all_edges)} edges."
        )
    for i, e in enumerate(all_edges):
        if edge_u[i] != e.u().name() or edge_v[i] != e.v().name():
            raise RuntimeError(
                f"Edge ordering mismatch at index {i}: "
                f"expected ({e.u().name()}, {e.v().name()}), "
                f"got ({edge_u[i]}, {edge_v[i]}). "
                f"This would cause flows to be misaligned silently."
            )

    if isinstance(g, np.ndarray):
        g_t = torch.tensor(g, dtype=torch.float32)
    else:
        g_t = g.float()

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0_gnn = time.perf_counter()
    with torch.no_grad():
        out = model(
            x=x.to(device),
            edge_index=edge_index.to(device),
            g=g_t.to(device),
            batch=None,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        phi_hat = torch.sigmoid(out.edge_logits).detach().cpu()
    if timings is not None:
        timings["gnn_s"] = time.perf_counter() - t0_gnn

    if not enforce_flow_conservation:
        return phi_hat.numpy()

    # Project to flow-conserving polytope (Eq. 39–41).
    source_idx = name_to_idx.get("source", None)
    target_idx = name_to_idx.get("target", None)
    if source_idx is None or target_idx is None:
        raise RuntimeError("Could not find 'source'/'target' nodes to enforce flow conservation.")

    t0_qp = time.perf_counter()
    phi_proj = project_flows_qp(
        edge_index=edge_index,
        phi_hat=phi_hat,
        num_nodes=len(node_names),
        source_idx=source_idx,
        target_idx=target_idx,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if timings is not None:
        timings["qp_s"] = time.perf_counter() - t0_qp
    return phi_proj.numpy()


def plan_with_gnn_flows(
    *,
    planner: PlanarPushingPlanner,
    model: GCSFlowPredictor,
    g: np.ndarray | torch.Tensor,
    solver_params: PlanarSolverParams,
    node_features_csv: str = "planning_through_contact/dataset/data/sugar_box/node_features.csv",
    max_paths: Optional[int] = None,
    max_steps: int = 512,
    seed: int = 0,
    device: Optional[torch.device] = None,
    enforce_flow_conservation: bool = True,
) -> Optional[PlanarPushingPath]:
    """
    Full glue:
      GNN logits -> flows (aligned with planner edges) -> sample discrete paths -> convex restrictions -> SNOPT rounding
    """
    edge_flows = predict_edge_flows_for_planner(
        planner=planner,
        model=model,
        g=g,
        node_features_csv=node_features_csv,
        device=device,
        enforce_flow_conservation=enforce_flow_conservation,
    )
    return round_from_predicted_flows(
        planner=planner,
        edge_flows=edge_flows,
        solver_params=solver_params,
        max_paths=max_paths,
        max_steps=max_steps,
        seed=seed,
    )

