from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

logging.getLogger("drake").setLevel(logging.ERROR)

import numpy as np
import torch

from planning_through_contact.experiments.baseline_comparison.direct_trajectory_optimization import (
    SmoothingSchedule,
    direct_trajopt_through_contact,
)
from planning_through_contact.experiments.utils import (
    get_default_plan_config,
    get_default_solver_params,
    get_time_as_str,
)
from planning_through_contact.model.checkpoint_utils import (
    dataset_paths_for_body,
    flow_checkpoint_name,
    ranker_checkpoint_name,
    validate_body,
)
from planning_through_contact.model.drake_rounding import (
    predict_edge_flows_for_planner,
    round_from_predicted_flows,
)
from planning_through_contact.model.hparams import DecoderHParams, EncoderHParams
from planning_through_contact.model.ranknet import RankNetConfig
from planning_through_contact.model.ranknet_inference import (
    load_ranknet_from_checkpoint,
    ranknet_round_from_flow_model,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)
from planning_through_contact.scripts.planar_pushing.plan_with_gnn import (
    compute_g,
    load_gcs_flow_predictor_from_lightning_ckpt,
    load_test_plans_from_csv,
    resolve_torch_device,
)

# --- Method labels for the results table ----------------------------------

METHOD_PREFIXES = [
    "vanilla",
    "vanilla_short",
    "neural_ranknet",
    "neural_ranknet_short",
    "neural_no_ranknet",
    "neural_no_ranknet_short",
    "contact_implicit",
]


def method_label(prefix: str, max_paths: int, max_paths_short: int) -> str:
    labels = {
        "vanilla": f"GCS ({max_paths} steps)",
        "vanilla_short": f"GCS ({max_paths_short} steps)",
        "neural_ranknet": f"Neural GCS w/ RankNet ({max_paths} paths)",
        "neural_ranknet_short": f"Neural GCS w/ RankNet ({max_paths_short} paths)",
        "neural_no_ranknet": f"Neural GCS w/o RankNet ({max_paths} paths)",
        "neural_no_ranknet_short": f"Neural GCS w/o RankNet ({max_paths_short} paths)",
        "contact_implicit": "Contact-Implicit",
    }
    return labels.get(prefix, prefix)


# --- Helpers ---------------------------------------------------------------

def _safe_cost(path) -> Optional[float]:
    if path is None or path.rounded_result is None or not path.rounded_result.is_success():
        return None
    return float(path.rounded_result.get_optimal_cost())


def _solve_vanilla(planner: PlanarPushingPlanner, solver_params) -> tuple[Optional[Any], dict[str, Any]]:
    profile: dict[str, Any] = {}
    t_total = time.perf_counter()

    t_relax = time.perf_counter()
    relaxed = planner._solve(solver_params)
    planner.relaxed_gcs_result = relaxed
    profile["relaxation_wall_s"] = time.perf_counter() - t_relax
    profile["relaxation_success"] = relaxed.is_success()
    if not relaxed.is_success():
        profile["total_s"] = time.perf_counter() - t_total
        return None, profile

    profile["relaxation_cost"] = float(relaxed.get_optimal_cost())
    try:
        profile["relaxation_solver_s"] = float(relaxed.get_solver_details().optimizer_time)  # type: ignore
    except Exception:
        profile["relaxation_solver_s"] = None

    paths = planner.get_solution_paths(relaxed, solver_params, profile=profile)
    if paths is None:
        profile["total_s"] = time.perf_counter() - t_total
        return None, profile

    feasible_paths = planner._get_rounded_paths(solver_params, paths, profile=profile)
    if feasible_paths is None:
        profile["total_s"] = time.perf_counter() - t_total
        return None, profile

    path = planner._pick_best_path(feasible_paths)
    profile["total_s"] = time.perf_counter() - t_total
    profile["rounded_cost"] = _safe_cost(path)
    if profile["rounded_cost"] is not None and profile.get("relaxation_cost", 0.0) > 0:
        profile["relative_gap_upper_bound"] = (
            (float(profile["rounded_cost"]) - float(profile["relaxation_cost"]))
            / float(profile["relaxation_cost"])
        )
    else:
        profile["relative_gap_upper_bound"] = None
    return path, profile


def _make_trajopt_config(body: str):
    """Build a PlanarPlanConfig suitable for direct trajopt (requires dt_contact == dt_non_collision).
    Values copied verbatim from check_please/planning_through_contact/experiments/utils.py:get_baseline_comparison_configs.
    """
    import math
    config = get_default_plan_config(slider_type=body, pusher_radius=0.015, use_case="normal")
    dt = 0.25
    config.num_knot_points_contact = 3
    config.time_in_contact = config.num_knot_points_contact * dt
    config.num_knot_points_non_collision = 4
    config.time_non_collision = config.num_knot_points_non_collision * dt
    config.contact_config.cost.force_regularization = 1000
    config.contact_config.cost.keypoint_velocity_regularization = 10
    config.non_collision_cost.pusher_arc_length = 1
    config.non_collision_cost.pusher_velocity_regularization = 5
    config.contact_config.slider_velocity_constraint = 0.3
    config.non_collision_cost.pusher_velocity_constraint = 0.3
    config.contact_config.slider_rot_velocity_constraint = (2 * math.pi) / 4
    config.dynamics_config.force_scale = 1
    config.contact_config.lam_min = 0
    config.contact_config.lam_max = 1
    assert config.dt_contact == config.dt_non_collision
    return config


def _solve_direct_trajopt(body: str, start_and_goal, solver_params, smoothing: Optional[SmoothingSchedule] = None) -> dict[str, Any]:
    import logging, tempfile
    logging.getLogger("drake").setLevel(logging.ERROR)
    config = _make_trajopt_config(body)
    config.start_and_goal = start_and_goal
    profile: dict[str, Any] = {}
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        result = direct_trajopt_through_contact(
            start_and_goal=start_and_goal,
            config=config,
            solver_params=solver_params,
            output_name="bench",
            output_folder=tmp,
            visualize=False,
            smoothing=smoothing,
            save_traj=False,
            save_cost=False,
        )
    profile["total_s"] = time.perf_counter() - t0
    profile["success"] = bool(result.is_success())
    return profile


def _cumulative_rounding_solver_s(profile: dict[str, Any]) -> Optional[float]:
    """Sum (convex_restriction_solver_s + rounding_s) across all paths."""
    paths = profile.get("paths")
    if not isinstance(paths, list):
        return None
    total = 0.0
    found = False
    for p in paths:
        if p.get("convex_restriction_solver_s") is not None:
            total += float(p["convex_restriction_solver_s"])
            found = True
        if p.get("rounding_s") is not None:
            total += float(p["rounding_s"])
            found = True
    return total if found else None


def _flatten_result(prefix: str, profile: dict[str, Any], path) -> dict[str, Any]:
    rows_tried = profile.get("num_ranked_paths_tried")
    if rows_tried is None and isinstance(profile.get("paths"), list):
        rows_tried = len(profile["paths"])
    return {
        f"{prefix}_ok": path is not None and _safe_cost(path) is not None,
        f"{prefix}_cost": _safe_cost(path),
        f"{prefix}_total_s": profile.get("total_s"),
        f"{prefix}_relaxation_solver_s": profile.get("relaxation_solver_s"),
        f"{prefix}_relaxation_wall_s": profile.get("relaxation_wall_s"),
        f"{prefix}_gnn_s": profile.get("gnn_s"),
        f"{prefix}_qp_s": profile.get("qp_s"),
        f"{prefix}_path_sampling_s": profile.get("path_sampling_s"),
        f"{prefix}_num_unique_paths": profile.get("num_unique_paths"),
        f"{prefix}_num_paths_tried": rows_tried,
        f"{prefix}_ranknet_s": profile.get("ranknet_s"),
        f"{prefix}_rounding_solver_cumulative_s": _cumulative_rounding_solver_s(profile),
        f"{prefix}_relaxation_cost": profile.get("relaxation_cost"),
        f"{prefix}_relative_gap_upper_bound": profile.get("relative_gap_upper_bound"),
    }


def _aggregate(rows: list[dict[str, Any]], max_paths: int, max_paths_short: int) -> dict[str, Any]:
    def _mean_std(key: str) -> tuple[Optional[float], Optional[float]]:
        vals = [r[key] for r in rows if r.get(key) is not None and not isinstance(r.get(key), bool)]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    out: dict[str, Any] = {
        "num_instances": len(rows),
        "max_paths": max_paths,
        "max_paths_short": max_paths_short,
    }

    gcs_prefixes = [
        "vanilla", "vanilla_short",
        "neural_ranknet", "neural_ranknet_short",
        "neural_no_ranknet", "neural_no_ranknet_short",
    ]
    for prefix in gcs_prefixes:
        ok_key = f"{prefix}_ok"
        ok = [bool(r.get(ok_key, False)) for r in rows if ok_key in r]
        out[f"{prefix}_success_rate"] = float(np.mean(ok)) if ok else None
        for suffix in (
            "total_s",
            "relaxation_wall_s",
            "relaxation_solver_s",
            "gnn_s",
            "qp_s",
            "path_sampling_s",
            "num_paths_tried",
            "ranknet_s",
            "rounding_solver_cumulative_s",
            "relative_gap_upper_bound",
            "cost",
            "gap_vs_gcs_relax",
        ):
            m, s = _mean_std(f"{prefix}_{suffix}")
            out[f"{prefix}_{suffix}_mean"] = m
            out[f"{prefix}_{suffix}_std"] = s

    ci_ok = [bool(r.get("contact_implicit_ok", False)) for r in rows if "contact_implicit_ok" in r]
    out["contact_implicit_success_rate"] = float(np.mean(ci_ok)) if ci_ok else None
    m, s = _mean_std("contact_implicit_total_s")
    out["contact_implicit_total_s_mean"] = m
    out["contact_implicit_total_s_std"] = s

    return out


# --- Neural GCS helpers ----------------------------------------------------

def _run_neural_no_ranknet(
    config, flow_model, g, node_features_csv, solver_params, max_paths, max_steps, seed, device
) -> tuple[Any, dict[str, Any]]:
    planner = PlanarPushingPlanner(config)
    planner.formulate_problem()
    profile: dict[str, Any] = {}
    t0 = time.perf_counter()
    edge_flows = predict_edge_flows_for_planner(
        planner=planner,
        model=flow_model,
        g=g,
        node_features_csv=str(node_features_csv),
        device=device,
        enforce_flow_conservation=True,
        timings=profile,
    )
    path = round_from_predicted_flows(
        planner=planner,
        edge_flows=edge_flows,
        solver_params=solver_params,
        max_paths=int(max_paths),
        max_steps=int(max_steps),
        seed=seed,
        profile=profile,
    )
    profile["total_s"] = time.perf_counter() - t0
    return path, profile


def _run_neural_ranknet(
    config, flow_model, ranker, g, node_features_csv, solver_params, max_paths, max_steps, seed, device
) -> tuple[Any, dict[str, Any]]:
    planner = PlanarPushingPlanner(config)
    planner.formulate_problem()
    profile: dict[str, Any] = {}
    t0 = time.perf_counter()
    path, _ = ranknet_round_from_flow_model(
        planner=planner,
        flow_model=flow_model,
        ranker=ranker,
        g=g,
        node_features_csv=str(node_features_csv),
        solver_params=solver_params,
        max_paths=int(max_paths),
        max_steps=int(max_steps),
        seed=seed,
        device=device,
        profile=profile,
    )
    profile["total_s"] = time.perf_counter() - t0
    return path, profile


# --- Main ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", type=str, default="sugar_box", choices=["sugar_box", "tee"])
    parser.add_argument("--data_root", type=str, default="planning_through_contact/dataset/data")
    parser.add_argument("--flow_ckpt_path", type=str, default=None)
    parser.add_argument("--ranker_ckpt_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="benchmark_results")
    parser.add_argument("--max_test_plans", type=int, default=100)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--max_paths", type=int, default=100)
    parser.add_argument("--max_paths_short", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--skip_vanilla", action="store_true")
    parser.add_argument("--skip_neural", action="store_true")
    parser.add_argument("--skip_trajopt", action="store_true")
    parser.add_argument("--trajopt_smoothing", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ffn_hidden_mult", type=int, default=2)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--decoder_hidden", type=str, default="256,256")
    parser.add_argument("--decoder_dropout_p", type=float, default=0.1)

    parser.add_argument("--ranker_layers", type=int, default=3)
    parser.add_argument("--ranker_heads", type=int, default=4)
    parser.add_argument("--ranker_ffn_hidden", type=int, default=256)
    parser.add_argument("--ranker_score_hidden", type=int, default=64)
    parser.add_argument("--ranker_dropout_p", type=float, default=0.1)
    args = parser.parse_args()

    body = validate_body(args.body)
    paths = dataset_paths_for_body(body, data_root=args.data_root)
    flow_ckpt_path = (
        Path(args.flow_ckpt_path)
        if args.flow_ckpt_path is not None
        else Path("checkpoints/gcs_gnn") / f"{flow_checkpoint_name(body)}.ckpt"
    )
    ranker_ckpt_path = (
        Path(args.ranker_ckpt_path)
        if args.ranker_ckpt_path is not None
        else Path("checkpoints/ranknet") / f"{ranker_checkpoint_name(body)}.ckpt"
    )

    if not args.skip_neural and not flow_ckpt_path.exists():
        raise FileNotFoundError(flow_ckpt_path)
    use_ranker = ranker_ckpt_path.exists()
    if not args.skip_neural and not use_ranker:
        print(f"[warn] Ranker checkpoint not found at {ranker_ckpt_path} — neural_ranknet variants will be skipped.")

    test_plans, csv_body = load_test_plans_from_csv(paths.global_features_csv)
    if csv_body != body:
        raise RuntimeError(f"CSV body {csv_body!r} does not match requested body {body!r}")
    test_plans = test_plans[: max(0, int(args.max_test_plans))]
    if not test_plans:
        raise RuntimeError(f"No test plans found in {paths.global_features_csv}")

    config = get_default_plan_config(slider_type=body, pusher_radius=0.015, use_case="normal")
    solver_params = get_default_solver_params(args.debug, clarabel=False)
    solver_params.rounding_steps = int(args.max_paths)

    solver_params_short = copy.copy(solver_params)
    solver_params_short.rounding_steps = int(args.max_paths_short)

    flow_model = None
    ranker = None
    if not args.skip_neural:
        with open(paths.node_features_csv, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise RuntimeError(f"Empty CSV: {paths.node_features_csv}")
            x_dim = len([c for c in reader.fieldnames if c.startswith("x_")])
        g_dim = 6 + int(config.slider_geometry.num_collision_free_regions)

        encoder_hp = EncoderHParams(
            d_model=int(args.d_model),
            num_layers=int(args.num_layers),
            num_heads=int(args.num_heads),
            ffn_hidden_mult=int(args.ffn_hidden_mult),
            dropout_p=float(args.dropout_p),
        )
        hidden = tuple(int(s) for s in str(args.decoder_hidden).split(",") if s.strip())
        decoder_hp = DecoderHParams(hidden_dims=hidden, dropout_p=float(args.decoder_dropout_p))
        flow_model = load_gcs_flow_predictor_from_lightning_ckpt(
            flow_ckpt_path,
            x_dim=x_dim,
            g_dim=g_dim,
            encoder_hp=encoder_hp,
            decoder_hp=decoder_hp,
            map_location="cpu",
        )
        device = resolve_torch_device(args.device)
        flow_model.to(device)

        if use_ranker:
            ranker_cfg = RankNetConfig(
                d_model=int(args.d_model),
                num_layers=int(args.ranker_layers),
                num_heads=int(args.ranker_heads),
                ffn_hidden_dim=int(args.ranker_ffn_hidden),
                score_hidden_dim=int(args.ranker_score_hidden),
                dropout_p=float(args.ranker_dropout_p),
            )
            ranker = load_ranknet_from_checkpoint(ranker_ckpt_path, cfg=ranker_cfg, map_location="cpu")
            ranker.to(device)
    else:
        device = resolve_torch_device(args.device)

    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    existing = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(f"{body}_")]
    next_n = max((int(d.name.split("_")[-1]) for d in existing if d.name.split("_")[-1].isdigit()), default=0) + 1
    out_dir = base / f"{body}_{next_n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    warmup_done = False
    for idx, (plan_id, plan) in enumerate(test_plans):
        config.start_and_goal = plan
        g = compute_g(config, plan) if not args.skip_neural else None

        if not warmup_done:
            print(f"[warmup] plan_id={plan_id} — warming up solvers/JIT, timings discarded.")
            if not args.skip_neural:
                _wp = PlanarPushingPlanner(config)
                _wp.formulate_problem()
                _wprof: dict[str, Any] = {}
                _wflows = predict_edge_flows_for_planner(planner=_wp, model=flow_model, g=g, node_features_csv=str(paths.node_features_csv), device=device, enforce_flow_conservation=True, timings=_wprof)
                round_from_predicted_flows(planner=_wp, edge_flows=_wflows, solver_params=solver_params, max_paths=1, max_steps=1, seed=0, profile=_wprof)
                if use_ranker:
                    _wp2 = PlanarPushingPlanner(config)
                    _wp2.formulate_problem()
                    ranknet_round_from_flow_model(planner=_wp2, flow_model=flow_model, ranker=ranker, g=g, node_features_csv=str(paths.node_features_csv), solver_params=solver_params, max_paths=1, max_steps=1, seed=0, device=device, profile={})
            if not args.skip_vanilla:
                _wv = PlanarPushingPlanner(config)
                _wv.formulate_problem()
                _solve_vanilla(_wv, solver_params)
            if not args.skip_trajopt:
                _wsmoothing = SmoothingSchedule(0.01, 5, "exp") if args.trajopt_smoothing else None
                _solve_direct_trajopt(body, plan, solver_params, smoothing=_wsmoothing)
            print("[warmup] Done — starting real benchmark.\n")
            warmup_done = True

        n_total = len(test_plans)
        print(f"\n[{idx + 1}/{n_total}] plan_id={plan_id}")
        row: dict[str, Any] = {"plan_id": int(plan_id), "body": body}

        # --- Neural GCS w/ RankNet -----------------------------------------
        if not args.skip_neural and use_ranker:
            print(f"  → Neural GCS w/ RankNet ({args.max_paths} paths) ...", flush=True)
            path, profile = _run_neural_ranknet(config, flow_model, ranker, g, paths.node_features_csv, solver_params, args.max_paths, args.max_steps, idx, device)
            row.update(_flatten_result("neural_ranknet", profile, path))
            print(f"     ok={row.get('neural_ranknet_ok')}  t={profile.get('total_s', 0):.2f}s")

            print(f"  → Neural GCS w/ RankNet ({args.max_paths_short} paths) ...", flush=True)
            path, profile = _run_neural_ranknet(config, flow_model, ranker, g, paths.node_features_csv, solver_params_short, args.max_paths_short, args.max_steps, idx, device)
            row.update(_flatten_result("neural_ranknet_short", profile, path))
            print(f"     ok={row.get('neural_ranknet_short_ok')}  t={profile.get('total_s', 0):.2f}s")

        # --- Neural GCS w/o RankNet ----------------------------------------
        if not args.skip_neural:
            print(f"  → Neural GCS w/o RankNet ({args.max_paths} paths) ...", flush=True)
            path, profile = _run_neural_no_ranknet(config, flow_model, g, paths.node_features_csv, solver_params, args.max_paths, args.max_steps, idx, device)
            row.update(_flatten_result("neural_no_ranknet", profile, path))
            print(f"     ok={row.get('neural_no_ranknet_ok')}  t={profile.get('total_s', 0):.2f}s")

            print(f"  → Neural GCS w/o RankNet ({args.max_paths_short} paths) ...", flush=True)
            path, profile = _run_neural_no_ranknet(config, flow_model, g, paths.node_features_csv, solver_params_short, args.max_paths_short, args.max_steps, idx, device)
            row.update(_flatten_result("neural_no_ranknet_short", profile, path))
            print(f"     ok={row.get('neural_no_ranknet_short_ok')}  t={profile.get('total_s', 0):.2f}s")

        # --- Vanilla GCS ---------------------------------------------------
        if not args.skip_vanilla:
            print(f"  → GCS ({args.max_paths} steps) ...", flush=True)
            planner = PlanarPushingPlanner(config)
            planner.formulate_problem()
            path, profile = _solve_vanilla(planner, solver_params)
            row.update(_flatten_result("vanilla", profile, path))
            print(f"     ok={row.get('vanilla_ok')}  t={profile.get('total_s', 0):.2f}s")

            print(f"  → GCS ({args.max_paths_short} steps) ...", flush=True)
            planner = PlanarPushingPlanner(config)
            planner.formulate_problem()
            path, profile = _solve_vanilla(planner, solver_params_short)
            row.update(_flatten_result("vanilla_short", profile, path))
            print(f"     ok={row.get('vanilla_short_ok')}  t={profile.get('total_s', 0):.2f}s")

        # --- Contact-Implicit ----------------------------------------------
        if not args.skip_trajopt:
            print(f"  → Contact-Implicit ...", flush=True)
            smoothing = SmoothingSchedule(0.01, 5, "exp") if args.trajopt_smoothing else None
            profile = _solve_direct_trajopt(body, plan, solver_params, smoothing=smoothing)
            row["contact_implicit_ok"] = profile["success"]
            row["contact_implicit_total_s"] = profile["total_s"]
            print(f"     ok={row.get('contact_implicit_ok')}  t={profile.get('total_s', 0):.2f}s")

        # Compute neural gap vs vanilla GCS relaxation lower bound
        vanilla_relax = row.get("vanilla_relaxation_cost")
        if vanilla_relax is not None and float(vanilla_relax) > 0:
            for neural_prefix in ("neural_ranknet", "neural_ranknet_short", "neural_no_ranknet", "neural_no_ranknet_short"):
                c_round = row.get(f"{neural_prefix}_cost")
                if c_round is not None:
                    row[f"{neural_prefix}_gap_vs_gcs_relax"] = (float(c_round) - float(vanilla_relax)) / float(vanilla_relax)

        rows.append(row)

    summary = _aggregate(rows, max_paths=int(args.max_paths), max_paths_short=int(args.max_paths_short))
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    out_json = out_dir / "summary.json"
    out_json.write_text(json.dumps({"rows": rows, "aggregate": summary}, indent=2))
    print(f"\nWrote benchmark results to {out_dir}")


if __name__ == "__main__":
    main()
