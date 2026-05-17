from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

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


def _flatten_result(prefix: str, profile: dict[str, Any], path) -> dict[str, Any]:
    rows_tried = profile.get("num_ranked_paths_tried")
    if rows_tried is None and isinstance(profile.get("paths"), list):
        rows_tried = len(profile["paths"])
    return {
        f"{prefix}_ok": path is not None and _safe_cost(path) is not None,
        f"{prefix}_cost": _safe_cost(path),
        f"{prefix}_total_s": profile.get("total_s"),
        f"{prefix}_gnn_s": profile.get("gnn_s"),
        f"{prefix}_qp_s": profile.get("qp_s"),
        f"{prefix}_ranknet_s": profile.get("ranknet_s"),
        f"{prefix}_path_sampling_s": profile.get("path_sampling_s"),
        f"{prefix}_num_unique_paths": profile.get("num_unique_paths"),
        f"{prefix}_num_paths_tried": rows_tried,
        f"{prefix}_relaxation_cost": profile.get("relaxation_cost"),
        f"{prefix}_relaxation_solver_s": profile.get("relaxation_solver_s"),
        f"{prefix}_relative_gap_upper_bound": profile.get("relative_gap_upper_bound"),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_numeric(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r.get(key) is not None and not isinstance(r.get(key), bool)]
        return float(np.mean(vals)) if vals else None

    out: dict[str, Any] = {"num_instances": len(rows)}
    for prefix in ("learned", "vanilla"):
        ok_key = f"{prefix}_ok"
        ok = [bool(r.get(ok_key, False)) for r in rows]
        out[f"{prefix}_success_rate"] = float(np.mean(ok)) if ok else 0.0
        for suffix in (
            "cost",
            "total_s",
            "gnn_s",
            "qp_s",
            "ranknet_s",
            "path_sampling_s",
            "num_unique_paths",
            "num_paths_tried",
            "relaxation_cost",
            "relaxation_solver_s",
            "relative_gap_upper_bound",
        ):
            out[f"{prefix}_{suffix}_mean"] = mean_numeric(f"{prefix}_{suffix}")

    speedups = [
        r["vanilla_total_s"] / r["learned_total_s"]
        for r in rows
        if r.get("vanilla_total_s") is not None
        and r.get("learned_total_s") is not None
        and float(r["learned_total_s"]) > 0
    ]
    out["speedup_mean"] = float(np.mean(speedups)) if speedups else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", type=str, default="sugar_box", choices=["sugar_box", "tee"])
    parser.add_argument("--data_root", type=str, default="planning_through_contact/dataset/data")
    parser.add_argument("--flow_ckpt_path", type=str, default=None)
    parser.add_argument("--ranker_ckpt_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="benchmark_results")
    parser.add_argument("--max_test_plans", type=int, default=10)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="cpu")
    parser.add_argument("--max_paths", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--skip_vanilla", action="store_true")
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

    if not flow_ckpt_path.exists():
        raise FileNotFoundError(flow_ckpt_path)
    use_ranker = ranker_ckpt_path.exists()

    test_plans, csv_body = load_test_plans_from_csv(paths.global_features_csv)
    if csv_body != body:
        raise RuntimeError(f"CSV body {csv_body!r} does not match requested body {body!r}")
    test_plans = test_plans[: max(0, int(args.max_test_plans))]
    if not test_plans:
        raise RuntimeError(f"No test plans found in {paths.global_features_csv}")

    config = get_default_plan_config(slider_type=body, pusher_radius=0.015, use_case="normal")
    solver_params = get_default_solver_params(args.debug, clarabel=False)
    solver_params.rounding_steps = int(args.max_paths)

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

    ranker = None
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

    out_dir = Path(args.output_dir) / f"benchmark_{get_time_as_str()}_{body}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for idx, (plan_id, plan) in enumerate(test_plans):
        config.start_and_goal = plan
        g = compute_g(config, plan)
        row: dict[str, Any] = {"plan_id": int(plan_id), "body": body}

        learned_planner = PlanarPushingPlanner(config)
        learned_planner.formulate_problem()
        learned_profile: dict[str, Any] = {}
        t_learned = time.perf_counter()
        if ranker is None:
            edge_flows = predict_edge_flows_for_planner(
                planner=learned_planner,
                model=flow_model,
                g=g,
                node_features_csv=str(paths.node_features_csv),
                device=device,
                enforce_flow_conservation=True,
                timings=learned_profile,
            )
            learned_path = round_from_predicted_flows(
                planner=learned_planner,
                edge_flows=edge_flows,
                solver_params=solver_params,
                max_paths=int(args.max_paths),
                max_steps=int(args.max_steps),
                seed=idx,
                profile=learned_profile,
            )
        else:
            learned_path, _ = ranknet_round_from_flow_model(
                planner=learned_planner,
                flow_model=flow_model,
                ranker=ranker,
                g=g,
                node_features_csv=str(paths.node_features_csv),
                solver_params=solver_params,
                max_paths=int(args.max_paths),
                max_steps=int(args.max_steps),
                seed=idx,
                device=device,
                profile=learned_profile,
            )
        learned_profile["total_s"] = time.perf_counter() - t_learned
        row.update(_flatten_result("learned", learned_profile, learned_path))

        if not args.skip_vanilla:
            vanilla_planner = PlanarPushingPlanner(config)
            vanilla_planner.formulate_problem()
            vanilla_path, vanilla_profile = _solve_vanilla(vanilla_planner, solver_params)
            row.update(_flatten_result("vanilla", vanilla_profile, vanilla_path))

        rows.append(row)
        print(json.dumps(row, indent=2))

    summary = _aggregate(rows)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows, "aggregate": summary}, indent=2))
    print(f"Wrote benchmark results to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
