from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from planning_through_contact.model.cuda_required import require_cuda
from planning_through_contact.experiments.utils import (
    get_default_experiment_plans,
    get_default_plan_config,
    get_default_solver_params,
    get_time_as_str,
)
from planning_through_contact.geometry.planar.non_collision import NonCollisionMode
from planning_through_contact.model.drake_rounding import predict_edge_flows_for_planner, round_from_predicted_flows
from planning_through_contact.model.hparams import DecoderHParams, EncoderHParams
from planning_through_contact.model.model import GCSFlowPredictor
from planning_through_contact.planning.planar.planar_pushing_planner import PlanarPushingPlanner


def _wrap_angle(theta: float) -> float:
    return float(np.arctan2(np.sin(theta), np.cos(theta)))


def _rot2(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def compute_g(config, plan) -> np.ndarray:
    """
    Compute g = [g_pose (6) || g_entry (num_entries)] in the slider initial body frame,
    matching `dataset/create_plan_index.py`.
    """
    num_entries = int(config.slider_geometry.num_collision_free_regions)

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
    g_entry = np.zeros((num_entries,), dtype=np.float32)
    if 0 <= entry_idx0 < num_entries:
        g_entry[entry_idx0] = 1.0

    ps_W = np.array([float(plan.slider_initial_pose.x), float(plan.slider_initial_pose.y)], dtype=np.float64)
    theta_s = float(plan.slider_initial_pose.theta)
    R_WS = _rot2(theta_s)

    p_goal_S = -(R_WS.T @ ps_W.reshape((2, 1))).reshape((2,))
    theta_goal = _wrap_angle(-theta_s)

    pp_W = np.array([float(plan.pusher_initial_pose.x), float(plan.pusher_initial_pose.y)], dtype=np.float64)
    p_pusher_S = (R_WS.T @ (pp_W - ps_W).reshape((2, 1))).reshape((2,))

    g_pose = np.array(
        [
            float(p_goal_S[0]),
            float(p_goal_S[1]),
            float(np.sin(theta_goal)),
            float(np.cos(theta_goal)),
            float(p_pusher_S[0]),
            float(p_pusher_S[1]),
        ],
        dtype=np.float32,
    )
    return np.concatenate([g_pose, g_entry], axis=0)


def load_gcs_flow_predictor_from_lightning_ckpt(
    ckpt_path: str | Path,
    *,
    x_dim: int,
    g_dim: int,
    encoder_hp: EncoderHParams,
    decoder_hp: DecoderHParams,
    map_location: str = "cpu",
) -> GCSFlowPredictor:
    """
    Loads weights from a Lightning checkpoint produced by `train_gcs_gnn.py`.
    """
    ckpt = torch.load(
        str(ckpt_path), map_location=map_location, weights_only=False
    )  # Lightning checkpoints contain state_dict + other objects
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # LightningModule stores predictor under `model.*`
    model_state: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            model_state[k[len("model.") :]] = v

    model = GCSFlowPredictor(x_dim=x_dim, g_dim=g_dim, encoder_hp=encoder_hp, decoder_hp=decoder_hp)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if len(unexpected) > 0:
        raise RuntimeError(f"Unexpected keys when loading checkpoint: {unexpected[:10]}")
    if len(missing) > 0:
        raise RuntimeError(f"Missing keys when loading checkpoint: {missing[:10]}")
    return model


def export_edge_labels(
    *,
    out_dir: Path,
    planner: PlanarPushingPlanner,
    path,
    edge_flows: Optional[np.ndarray] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_edges = list(planner.gcs.Edges())
    edge_u = [e.u().name() for e in all_edges]
    edge_v = [e.v().name() for e in all_edges]
    edge_keys = list(zip(edge_u, edge_v))

    path_edge_keys: set[tuple[str, str]] = set()
    path_edge_order: dict[tuple[str, str], list[int]] = {}
    path_vertex_names: list[str] = []
    if path is not None:
        path_edge_keys = {(e.u().name(), e.v().name()) for e in path.edges}
        for idx, e in enumerate(path.edges, start=1):
            path_edge_order.setdefault((e.u().name(), e.v().name()), []).append(idx)
        path_vertex_names = path.get_path_names()

    y = [1 if k in path_edge_keys else 0 for k in edge_keys]
    phi = None if edge_flows is None else [float(x) for x in np.asarray(edge_flows).reshape((-1,)).tolist()]

    payload: dict[str, Any] = {
        "num_vertices": len(list(planner.gcs.Vertices())),
        "num_edges": len(all_edges),
        "num_positive_edges": int(sum(y)),
        "path_vertex_names": path_vertex_names,
        "edges": [
            {
                "u": u,
                "v": v,
                "y": int(label),
                **({"phi": float(phi_i)} if phi is not None else {}),
                **({"order": path_edge_order.get((u, v), [])} if int(label) == 1 else {}),
            }
            for ((u, v), label, phi_i) in zip(edge_keys, y, (phi if phi is not None else [0.0] * len(y)))
        ]
        if phi is not None
        else [{"u": u, "v": v, "y": int(label), **({"order": path_edge_order.get((u, v), [])} if int(label) == 1 else {})}
              for ((u, v), label) in zip(edge_keys, y)],
    }

    (out_dir / "edges.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    require_cuda()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traj", type=int, default=None)
    parser.add_argument("--body", type=str, default="sugar_box")
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="trajectories")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--node_features_csv", type=str, default="planning_through_contact/dataset/data/node_features.csv")

    parser.add_argument("--ckpt_path", type=str, required=True, help="Lightning .ckpt from train_gcs_gnn.py")
    parser.add_argument("--device", type=str, default="cuda", help="Device for model (default: cuda; requires GPU).")
    parser.add_argument("--no_flow_projection", action="store_true", help="Disable QP flow projection.")
    parser.add_argument("--max_paths", type=int, default=None, help="Override number of candidate paths to sample.")
    parser.add_argument("--max_steps", type=int, default=512)

    # Model hparams (must match training)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ffn_hidden_mult", type=int, default=2)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--decoder_hidden", type=str, default="256,256")
    parser.add_argument("--decoder_dropout_p", type=float, default=0.1)

    parser.add_argument("--save_graph_edge_labels", action="store_true", default=True)
    args = parser.parse_args()

    if not args.debug:
        import logging

        logging.getLogger("drake").setLevel(logging.WARNING)

    config = get_default_plan_config(slider_type=args.body, pusher_radius=0.015, use_case="normal")
    solver_params = get_default_solver_params(args.debug, clarabel=False)

    # Build plans like create_plans.py
    plans = get_default_experiment_plans(args.seed, args.num, config)
    if args.traj is not None:
        plans = [plans[int(args.traj)]]

    # GNN-specific output folder so it's clear these are GNN inference results
    folder_name = f"{args.output_dir}/gnn_run_{get_time_as_str()}_{args.body}"
    if args.traj is not None:
        folder_name += f"_traj_{args.traj}"
    Path(folder_name).mkdir(parents=True, exist_ok=True)

    # Infer x_dim from CSV
    import csv as _csv

    with open(args.node_features_csv, newline="") as f:
        r = _csv.DictReader(f)
        if r.fieldnames is None:
            raise RuntimeError(f"Empty CSV: {args.node_features_csv}")
        x_dim = len([c for c in r.fieldnames if c.startswith("x_")])

    g_dim = 6 + int(config.slider_geometry.num_collision_free_regions)

    encoder_hp = EncoderHParams(
        d_model=int(args.d_model),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        ffn_hidden_mult=int(args.ffn_hidden_mult),
        dropout_p=float(args.dropout_p),
    )
    hidden = tuple(int(s) for s in str(args.decoder_hidden).split(",") if s.strip() != "")
    decoder_hp = DecoderHParams(hidden_dims=hidden, dropout_p=float(args.decoder_dropout_p))

    model = load_gcs_flow_predictor_from_lightning_ckpt(
        args.ckpt_path, x_dim=x_dim, g_dim=g_dim, encoder_hp=encoder_hp, decoder_hp=decoder_hp, map_location="cpu"
    )
    device = torch.device(args.device)
    model.to(device)

    for idx, plan in enumerate(plans):
        out = Path(folder_name) / f"traj_{idx}"
        out.mkdir(parents=True, exist_ok=True)

        config.start_and_goal = plan
        planner = PlanarPushingPlanner(config)
        planner.formulate_problem()

        g = compute_g(config, plan)

        t0 = time.perf_counter()
        edge_flows = predict_edge_flows_for_planner(
            planner=planner,
            model=model,
            g=g,
            node_features_csv=args.node_features_csv,
            device=device,
            enforce_flow_conservation=not bool(args.no_flow_projection),
        )
        t_pred = time.perf_counter() - t0

        t1 = time.perf_counter()
        path = round_from_predicted_flows(
            planner=planner,
            edge_flows=edge_flows,
            solver_params=solver_params,
            max_paths=args.max_paths,
            max_steps=int(args.max_steps),
            seed=int(args.seed + idx),
        )
        t_round = time.perf_counter() - t1

        summary = {
            "ok": path is not None and path.rounded_result is not None and path.rounded_result.is_success(),
            "pred_time_s": float(t_pred),
            "rounding_pipeline_time_s": float(t_round),
            "path_cost": None
            if path is None
            else float(
                path.rounded_result.get_optimal_cost()
                if (path.rounded_result is not None and path.rounded_result.is_success())
                else path.relaxed_cost
            ),
        }
        (out / "gnn_plan_summary.json").write_text(json.dumps(summary, indent=2))

        if args.save_graph_edge_labels:
            export_edge_labels(out_dir=out / "graph_edge_labels", planner=planner, path=path, edge_flows=edge_flows)

        # Save trajectories (pickles) if we have a path
        if path is not None:
            traj_dir = out / "trajectory"
            traj_dir.mkdir(parents=True, exist_ok=True)
            traj_relaxed = path.to_traj()
            traj_relaxed.save(str(traj_dir / "traj_relaxed.pkl"))  # type: ignore
            if path.rounded_result is not None and path.rounded_result.is_success():
                traj_rounded = path.to_traj(rounded=True)
                traj_rounded.save(str(traj_dir / "traj_rounded.pkl"))  # type: ignore


if __name__ == "__main__":
    main()

