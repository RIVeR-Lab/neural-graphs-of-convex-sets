from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from planning_through_contact.experiments.utils import (
    get_default_experiment_plans,
    get_default_plan_config,
    get_default_solver_params,
)
from planning_through_contact.geometry.planar.non_collision import NonCollisionMode
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.model.drake_rounding import predict_edge_flows_for_planner, round_from_predicted_flows
from planning_through_contact.model.hparams import DecoderHParams, EncoderHParams
from planning_through_contact.model.model import GCSFlowPredictor
from planning_through_contact.model.ranknet import RankNetConfig
from planning_through_contact.model.ranknet_inference import (
    load_ranknet_from_checkpoint,
    ranknet_round_from_flow_model,
)
from planning_through_contact.planning.planar.planar_pushing_planner import PlanarPushingPlanner
from planning_through_contact.planning.planar.planar_plan_config import PlanarPushingStartAndGoal
from planning_through_contact.visualize.planar_pushing import visualize_planar_pushing_trajectory


def resolve_torch_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available. Use --device cpu.")
    return device

def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _pose_from_row(row: dict[str, str], prefix: str) -> PlanarPose:
    return PlanarPose(
        _float(row, f"{prefix}_x"),
        _float(row, f"{prefix}_y"),
        _float(row, f"{prefix}_theta"),
    )


def load_test_plans_from_csv(
    csv_path: Path,
) -> tuple[list[tuple[int, PlanarPushingStartAndGoal]], str]:
    """Load plans with split=='test' from plan index CSV. Returns ((plan_id, plan), ...) and body from first row."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("split") == "test"]
    if not rows:
        return [], "sugar_box"
    body = str(rows[0].get("body", "sugar_box"))
    out: list[tuple[int, PlanarPushingStartAndGoal]] = []
    for r in rows:
        plan_id = int(float(r["plan_id"]))
        plan = PlanarPushingStartAndGoal(
            slider_initial_pose=_pose_from_row(r, "slider_init"),
            slider_target_pose=_pose_from_row(r, "slider_goal"),
            pusher_initial_pose=_pose_from_row(r, "pusher_init"),
            pusher_target_pose=_pose_from_row(r, "pusher_goal"),
        )
        out.append((plan_id, plan))
    return out, body


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



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traj", type=int, default=None)
    parser.add_argument("--body", type=str, default="sugar_box")
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="trajectories")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--node_features_csv",
        type=str,
        default=None,
    )

    parser.add_argument("--ckpt_path", type=str, required=True, help="Lightning .ckpt from train_gcs_gnn.py")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="cuda",
        help="Device for model inference. Use 'cuda' for GPU or 'auto' to use CUDA when available.",
    )
    parser.add_argument(
        "--rounding_flow",
        type=str,
        choices=["qp", "direct"],
        default="direct",
        help="Flow for path sampling: 'direct' = sigmoid, normalize per node (default); 'qp' = project to flow polytope.",
    )
    parser.add_argument("--no_flow_projection", action="store_true", help="Use direct rounding (same as --rounding_flow direct).")
    parser.add_argument(
        "--use_test_plans",
        action="store_true",
        help="Load test split from plan_index CSV and run inference on those plans only.",
    )
    parser.add_argument(
        "--plan_index_csv",
        type=str,
        default=None,
        help="CSV with plan_id, split, poses. Used when --use_test_plans.",
    )
    parser.add_argument(
        "--max_test_plans",
        type=int,
        default=None,
        help="Optional cap on number of test plans processed (in CSV order).",
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        default=None,
        help="Solutions HDF5. When --use_test_plans: test loss and ground-truth SVG.",
    )
    parser.add_argument(
        "--max_paths",
        type=int,
        default=None,
        help="Number of candidate paths to sample (default 100, same as nominal GCS).",
    )
    parser.add_argument("--max_steps", type=int, default=512)

    # Model hparams (must match training)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ffn_hidden_mult", type=int, default=2)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--decoder_hidden", type=str, default="256,256")
    parser.add_argument("--decoder_dropout_p", type=float, default=0.1)
    parser.add_argument(
        "--ranker_ckpt_path",
        type=str,
        default=None,
        help="Optional RankNet checkpoint. If set, sampled paths are ranked before restriction/rounding solves.",
    )
    parser.add_argument("--ranker_layers", type=int, default=3)
    parser.add_argument("--ranker_heads", type=int, default=4)
    parser.add_argument("--ranker_ffn_hidden", type=int, default=256)
    parser.add_argument("--ranker_score_hidden", type=int, default=64)
    parser.add_argument("--ranker_dropout_p", type=float, default=0.1)

    parser.add_argument(
        "--save_video",
        action="store_true",
        default=True,
        help="Save motion videos: prediction.mp4 (GNN+rounding) and, with --use_test_plans and h5_path, ground_truth.mp4 (SDP+rounding).",
    )
    parser.add_argument("--no_save_video", action="store_true", help="Disable saving motion videos (overrides --save_video).")
    parser.add_argument(
        "--show_contact_legend",
        action="store_true",
        default=True,
        help="Show top-left legend for pusher contact state in videos.",
    )
    parser.add_argument(
        "--no_contact_legend",
        action="store_true",
        help="Hide pusher contact state legend in videos.",
    )
    parser.add_argument("--video_width_px", type=int, default=1920, help="Output video width in pixels.")
    parser.add_argument("--video_height_px", type=int, default=1080, help="Output video height in pixels.")
    parser.add_argument(
        "--video_dpi",
        type=int,
        default=100,
        help="Video export DPI used with figure size to match pixel dimensions.",
    )
    args = parser.parse_args()
    if args.no_save_video:
        args.save_video = False
    if args.no_contact_legend:
        args.show_contact_legend = False
    data_dir = Path("planning_through_contact/dataset/data") / args.body
    if args.node_features_csv is None:
        args.node_features_csv = str(data_dir / "node_features.csv")
    if args.plan_index_csv is None:
        args.plan_index_csv = str(data_dir / "global_features.csv")

    if not args.debug:
        import logging

        logging.getLogger("drake").setLevel(logging.WARNING)

    # Plans: either test split from CSV or random from seed/num
    if args.use_test_plans:
        plan_index_csv = Path(args.plan_index_csv)
        if not plan_index_csv.exists():
            raise FileNotFoundError(f"Plan index CSV not found: {plan_index_csv}")
        test_plans_with_ids, body = load_test_plans_from_csv(plan_index_csv)
        plans_with_ids = [(pid, p) for pid, p in test_plans_with_ids]
        if args.max_test_plans is not None:
            plans_with_ids = plans_with_ids[: max(0, int(args.max_test_plans))]
        if args.traj is not None:
            k = int(args.traj)
            if not plans_with_ids:
                raise SystemExit("No test plans in CSV (split=='test').")
            if k < 0 or k >= len(plans_with_ids):
                raise SystemExit(
                    f"--traj {k} is out of range for {len(plans_with_ids)} test plan(s) (use 0..{len(plans_with_ids) - 1})."
                )
            plans_with_ids = [plans_with_ids[k]]
        body_for_run = body
        if not plans_with_ids:
            raise SystemExit("No test plans found in CSV (split=='test'). Check plan_index_csv and split column.")
    else:
        config_temp = get_default_plan_config(slider_type=args.body, pusher_radius=0.015, use_case="normal")
        plans = get_default_experiment_plans(args.seed, args.num, config_temp)
        if args.traj is not None:
            plans = [plans[int(args.traj)]]
        plans_with_ids = [(None, p) for p in plans]
        body_for_run = args.body

    config = get_default_plan_config(slider_type=body_for_run, pusher_radius=0.015, use_case="normal")
    solver_params = get_default_solver_params(args.debug, clarabel=False)

    base = Path(args.output_dir) / body_for_run
    base.mkdir(parents=True, exist_ok=True)
    existing = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("video_")]
    next_n = max((int(d.name.split("_")[-1]) for d in existing if d.name.split("_")[-1].isdigit()), default=0) + 1
    folder_name = str(base / f"video_{next_n}")
    if args.traj is not None and not args.use_test_plans:
        folder_name += f"_traj_{args.traj}"
    Path(folder_name).mkdir(parents=True, exist_ok=True)

    h5_path = Path(args.h5_path) if args.h5_path else None

    # Infer x_dim from CSV
    with open(args.node_features_csv, newline="") as f:
        r = csv.DictReader(f)
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
    device = resolve_torch_device(args.device)
    print(f"GNN inference device: {device}")
    model.to(device)
    ranker = None
    if args.ranker_ckpt_path is not None:
        ranker_cfg = RankNetConfig(
            d_model=int(args.d_model),
            num_layers=int(args.ranker_layers),
            num_heads=int(args.ranker_heads),
            ffn_hidden_dim=int(args.ranker_ffn_hidden),
            score_hidden_dim=int(args.ranker_score_hidden),
            dropout_p=float(args.ranker_dropout_p),
        )
        ranker = load_ranknet_from_checkpoint(args.ranker_ckpt_path, cfg=ranker_cfg, map_location="cpu")
        ranker.to(device)
        print(f"RankNet inference enabled: {args.ranker_ckpt_path}")

    max_paths_val = args.max_paths if args.max_paths is not None else 100
    enforce_flow = (args.rounding_flow == "qp") and not args.no_flow_projection

    for idx, (plan_id, plan) in enumerate(plans_with_ids):
        out = Path(folder_name) / (f"plan_{plan_id}" if plan_id is not None else f"traj_{idx}")
        out.mkdir(parents=True, exist_ok=True)
        if plan_id is not None:
            (out / "plan_id.txt").write_text(str(plan_id))

        config.start_and_goal = plan
        planner = PlanarPushingPlanner(config)
        planner.formulate_problem()

        g = compute_g(config, plan)

        rounding_profile: dict[str, Any] = {}
        timings: dict[str, float] = {}
        if ranker is None:
            t0 = time.perf_counter()
            edge_flows = predict_edge_flows_for_planner(
                planner=planner,
                model=model,
                g=g,
                node_features_csv=args.node_features_csv,
                device=device,
                enforce_flow_conservation=enforce_flow,
                timings=timings,
            )
            t_pred = time.perf_counter() - t0

            t1 = time.perf_counter()
            path = round_from_predicted_flows(
                planner=planner,
                edge_flows=edge_flows,
                solver_params=solver_params,
                max_paths=max_paths_val,
                max_steps=int(args.max_steps),
                seed=int(args.seed + idx),
                profile=rounding_profile,
            )
            t_round = time.perf_counter() - t1
            inference_mode = "flow_rounding"
        else:
            t0 = time.perf_counter()
            path, edge_flows = ranknet_round_from_flow_model(
                planner=planner,
                flow_model=model,
                ranker=ranker,
                g=g,
                node_features_csv=args.node_features_csv,
                solver_params=solver_params,
                max_paths=max_paths_val,
                max_steps=int(args.max_steps),
                seed=int(args.seed + idx),
                device=device,
                profile=rounding_profile,
            )
            t_pred = float(rounding_profile.get("gnn_s", time.perf_counter() - t0))
            t_round = time.perf_counter() - t0
            timings.update({k: v for k, v in rounding_profile.items() if isinstance(v, float)})
            inference_mode = "ranknet"

        output_name = f"plan_{plan_id}" if plan_id is not None else f"traj_{idx}"
        ok = path is not None and path.rounded_result is not None and path.rounded_result.is_success()
        print(f"[{output_name}] ok={ok}  gnn={timings.get('gnn_s', t_pred):.3f}s  total={t_round:.3f}s")

        if path is not None and path.rounded_result is not None and path.rounded_result.is_success():
            traj_rounded = path.to_traj(rounded=True)
            traj_dir = out / "trajectory"
            traj_dir.mkdir(parents=True, exist_ok=True)

            if args.save_video:
                animation_lims = traj_rounded.get_pos_limits(buffer=0.12)
                visualize_planar_pushing_trajectory(
                    traj_rounded,
                    save=True,
                    filename=str(traj_dir / "prediction"),
                    visualize_knot_points=True,
                    lims=animation_lims,
                    show_contact_legend=args.show_contact_legend,
                    video_width_px=args.video_width_px,
                    video_height_px=args.video_height_px,
                    video_dpi=args.video_dpi,
                )
                # Ground-truth video: run nominal GCS (SDP + rounding) for this plan so you can compare motion
                if args.use_test_plans and h5_path is not None and plan_id is not None and h5_path.exists():
                    planner_gt = PlanarPushingPlanner(config)
                    planner_gt.config.start_and_goal = plan
                    planner_gt.formulate_problem()
                    path_gt = planner_gt.plan_path(solver_params)
                    if (
                        path_gt is not None
                        and path_gt.rounded_result is not None
                        and path_gt.rounded_result.is_success()
                    ):
                        traj_gt = path_gt.to_traj(rounded=True)
                        visualize_planar_pushing_trajectory(
                            traj_gt,
                            save=True,
                            filename=str(traj_dir / "ground_truth"),
                            visualize_knot_points=True,
                            lims=animation_lims,
                            show_contact_legend=args.show_contact_legend,
                            video_width_px=args.video_width_px,
                            video_height_px=args.video_height_px,
                            video_dpi=args.video_dpi,
                        )


if __name__ == "__main__":
    main()

