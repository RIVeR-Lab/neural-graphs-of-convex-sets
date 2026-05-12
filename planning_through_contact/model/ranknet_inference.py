from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from planning_through_contact.model.drake_rounding import build_gnn_batch_for_planner
from planning_through_contact.model.inference import project_flows_qp
from planning_through_contact.model.model import GCSFlowPredictor
from planning_through_contact.model.ranknet import PathRankNet, RankNetConfig
from planning_through_contact.planning.planar.planar_plan_config import PlanarSolverParams
from planning_through_contact.planning.planar.planar_pushing_planner import PlanarPushingPlanner


def load_ranknet_from_checkpoint(
    ckpt_path: str | Path,
    *,
    cfg: RankNetConfig = RankNetConfig(),
    map_location: str = "cpu",
) -> PathRankNet:
    ckpt = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    ranker_state = {
        key.removeprefix("ranker."): value
        for key, value in state_dict.items()
        if key.startswith("ranker.")
    }
    if not ranker_state:
        ranker_state = state_dict
    ranker = PathRankNet(cfg)
    missing, unexpected = ranker.load_state_dict(ranker_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Could not load RankNet checkpoint: missing={missing[:5]} unexpected={unexpected[:5]}")
    return ranker


def _path_tensors_from_edge_paths(planner: PlanarPushingPlanner, edge_paths) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_edges = list(planner.gcs.Edges())
    edge_u = [e.u().name() for e in all_edges]
    edge_v = [e.v().name() for e in all_edges]
    node_names = sorted(set(edge_u) | set(edge_v))
    node_to_idx = {name: i for i, name in enumerate(node_names)}
    edge_to_idx = {(u, v): i for i, (u, v) in enumerate(zip(edge_u, edge_v))}

    path_nodes: list[list[int]] = []
    path_edges: list[list[int]] = []
    for path in edge_paths:
        if len(path) == 0:
            continue
        cand_u = [e.u().name() for e in path]
        cand_v = [e.v().name() for e in path]
        path_nodes.append([node_to_idx[cand_u[0]]] + [node_to_idx[v] for v in cand_v])
        path_edges.append([-1] + [edge_to_idx[(u, v)] for u, v in zip(cand_u, cand_v)])

    max_len = max(len(p) for p in path_nodes)
    node_arr = torch.full((len(path_nodes), max_len), -1, dtype=torch.long)
    edge_arr = torch.full((len(path_edges), max_len), -1, dtype=torch.long)
    mask = torch.zeros((len(path_nodes), max_len), dtype=torch.bool)
    for i, (nodes, edges) in enumerate(zip(path_nodes, path_edges)):
        L = len(nodes)
        node_arr[i, :L] = torch.tensor(nodes, dtype=torch.long)
        edge_arr[i, :L] = torch.tensor(edges, dtype=torch.long)
        mask[i, :L] = True
    return node_arr, edge_arr, mask


def ranknet_round_from_flow_model(
    *,
    planner: PlanarPushingPlanner,
    flow_model: GCSFlowPredictor,
    ranker: PathRankNet,
    g: np.ndarray | torch.Tensor,
    node_features_csv: str,
    solver_params: PlanarSolverParams,
    max_paths: Optional[int] = None,
    max_steps: int = 512,
    seed: int = 0,
    device: Optional[torch.device] = None,
    profile: Optional[dict[str, Any]] = None,
) -> tuple[Optional[Any], np.ndarray]:
    if device is None:
        device = next(flow_model.parameters()).device

    x, edge_index, g_t = build_gnn_batch_for_planner(
        planner=planner,
        g=np.asarray(g, dtype=np.float32) if isinstance(g, np.ndarray) else g.detach().cpu().numpy(),
        node_features_csv=node_features_csv,
    )
    x = x.to(device)
    edge_index = edge_index.to(device)
    g_t = g_t.to(device)

    flow_model.eval()
    ranker.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        flow_out = flow_model(x=x, edge_index=edge_index, g=g_t, batch=None)
        phi_hat = torch.sigmoid(flow_out.edge_logits).detach().cpu()
    if profile is not None:
        profile["gnn_s"] = time.perf_counter() - t0

    all_edges = list(planner.gcs.Edges())
    edge_u = [e.u().name() for e in all_edges]
    edge_v = [e.v().name() for e in all_edges]
    node_names = sorted(set(edge_u) | set(edge_v))
    source_idx = node_names.index("source")
    target_idx = node_names.index("target")
    t_qp = time.perf_counter()
    edge_flows = project_flows_qp(
        edge_index=edge_index.detach().cpu(),
        phi_hat=phi_hat,
        num_nodes=len(node_names),
        source_idx=source_idx,
        target_idx=target_idx,
    )
    if profile is not None:
        profile["qp_s"] = time.perf_counter() - t_qp

    edge_paths = planner.sample_edge_paths_from_flows(
        edge_flows=edge_flows.numpy(),
        solver_params=solver_params,
        max_paths=max_paths,
        max_steps=max_steps,
        seed=seed,
        profile=profile,
    )
    if len(edge_paths) == 0:
        return None, edge_flows.numpy()

    path_node_idx, path_edge_idx, path_mask = _path_tensors_from_edge_paths(planner, edge_paths)
    t_rank = time.perf_counter()
    with torch.no_grad():
        scores = ranker(
            node_embeddings=flow_out.node_embeddings,
            edge_flows=edge_flows.to(device),
            path_node_indices=path_node_idx,
            path_edge_indices=path_edge_idx,
            path_mask=path_mask,
        )
    ranked_indices = torch.argsort(scores, descending=True).detach().cpu().tolist()
    if profile is not None:
        profile["ranknet_s"] = time.perf_counter() - t_rank
        profile["ranknet_scores"] = [float(s) for s in scores.detach().cpu().tolist()]

    path = planner.solve_ranked_edge_paths_until_feasible(
        edge_paths,
        ranked_indices,
        solver_params=solver_params,
        profile=profile,
    )
    return path, edge_flows.numpy()
