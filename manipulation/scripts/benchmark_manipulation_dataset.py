#!/usr/bin/env python3
"""Benchmark manipulation GCS methods on held-out HDF5 test instances."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from pydrake.geometry.optimization import HPolyhedron
from pydrake.solvers import MosekSolver
from tqdm import tqdm

from manipulation.dataset_collect import _linear_gcs_options, _solve_linear_restriction
from manipulation.iiwa_helpers import build_shelf_plant
from manipulation.trajopt import (
    build_nonlinear_gcs_problem,
    build_region_edges,
    iiwa_kinematic_limits,
    region_list,
    solve_nonlinear_relaxation,
    solve_nonlinear_restriction,
)
from model.hparams import DecoderHParams, EncoderHParams
from model.inference import project_flows_qp
from model.ranknet import RankNetConfig
from quadrotor.gcs.linear import LinearGCS
from quadrotor.gcs.preprocessing import removeRedundancies
from quadrotor.gcs.rounding import randomForwardPathSearch
from scripts.eval_manipulation_circle_demo import (
    MAX_PATHS,
    MAX_ROUNDING_TRIALS,
    build_graph_tensors,
    load_flow_model,
    load_ranknet,
    rank_candidate_paths,
    sample_candidate_paths_from_flows,
    sync_device,
)


DATASETS = {
    "convex": "manipulation/dataset/manipulation_gcs_convex.h5",
    "nonconvex": "manipulation/dataset/manipulation_gcs_nonconvex.h5",
}


def _elapsed(t0: float, device: torch.device) -> float:
    sync_device(device)
    return time.perf_counter() - t0


def load_regions(h5) -> dict[str, HPolyhedron]:
    names = [
        name.decode("utf-8") if isinstance(name, bytes) else str(name)
        for name in h5["meta"]["region_names"][()]
    ]
    return {
        name: HPolyhedron(h5["regions"][name]["A"][()], h5["regions"][name]["b"][()])
        for name in names
    }


def build_problem(planner: str, regions: dict, q_start, q_goal, *, solver, plant):
    if planner == "convex":
        gcs = LinearGCS(regions.copy())
        gcs.setSolver(solver)
        gcs.addSourceTarget(q_start, q_goal)
        removeRedundancies(gcs.gcs, gcs.source, gcs.target, verbose=False)
        return (
            gcs.gcs,
            gcs.source,
            gcs.target,
            lambda: gcs.gcs.SolveShortestPath(
                gcs.source, gcs.target, _linear_gcs_options(gcs),
            ),
            lambda path_edges: _solve_linear_restriction(gcs, path_edges),
        )

    polys = region_list(regions)
    vel_limits, accel_limits = iiwa_kinematic_limits(plant)
    _, graph, source, target = build_nonlinear_gcs_problem(
        polys,
        build_region_edges(polys),
        q_start,
        q_goal,
        vel_limits=vel_limits,
        accel_limits=accel_limits,
    )
    return (
        graph,
        source,
        target,
        lambda: solve_nonlinear_relaxation(graph, source, target),
        lambda path_edges: solve_nonlinear_restriction(graph, source, target, path_edges),
    )


def predict_projected_flows(graph_tensors, flow_model, q_start, q_goal, device):
    g_t = torch.from_numpy(np.concatenate([q_start, q_goal]).astype(np.float32)).to(device)

    sync_device(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        flow_out = flow_model(
            facets=graph_tensors.facets,
            facet_mask=graph_tensors.facet_mask,
            node_flags=graph_tensors.node_flags,
            edge_index=graph_tensors.edge_index,
            g=g_t,
            batch=None,
        )
        phi_hat = torch.sigmoid(flow_out.edge_logits).detach().cpu()
    t_gnn = _elapsed(t0, device)

    sync_device(device)
    t0 = time.perf_counter()
    phi_proj = project_flows_qp(
        edge_index=graph_tensors.edge_index.cpu(),
        phi_hat=phi_hat,
        num_nodes=graph_tensors.n_nodes,
        source_idx=graph_tensors.source_idx,
        target_idx=graph_tensors.target_idx,
    )
    t_qp = _elapsed(t0, device)
    return flow_out, phi_proj, t_gnn, t_qp


def _try_paths(candidates, solve_restriction, *, stop_on_first: bool):
    tried = 0
    best_cost = float("nan")
    success = False
    t_restriction = 0.0
    for path_edges in candidates or []:
        if path_edges is None:
            continue
        tried += 1
        t0 = time.perf_counter()
        res = solve_restriction(path_edges)
        t_restriction += time.perf_counter() - t0
        if not res.is_success():
            continue
        cost = float(res.get_optimal_cost())
        if stop_on_first:
            return True, cost, tried, t_restriction
        if not success or cost < best_cost:
            success = True
            best_cost = cost
    return success, best_cost, tried, t_restriction


def benchmark_instance(grp, planner, regions, flow_model, ranker, solver, plant, device, seed):
    g = np.asarray(grp["g"][()], dtype=np.float64)
    q_start, q_goal = g[:7], g[7:]

    graph, source, target, solve_relaxation, solve_restriction = build_problem(
        planner, regions, q_start, q_goal, solver=solver, plant=plant,
    )
    graph_tensors = build_graph_tensors(graph, regions, device=device)

    out: dict[str, dict] = {}

    t0 = time.perf_counter()
    relaxed = solve_relaxation()
    t_relax = time.perf_counter() - t0
    if not relaxed.is_success():
        return None

    t0 = time.perf_counter()
    vanilla_candidates = randomForwardPathSearch(
        graph, relaxed, source, target,
        max_paths=MAX_PATHS, max_trials=MAX_ROUNDING_TRIALS, seed=seed,
    )
    t_sampling_vanilla = time.perf_counter() - t0
    v_success, v_cost, v_tried, v_restriction = _try_paths(
        vanilla_candidates, solve_restriction, stop_on_first=False,
    )
    out["vanilla"] = {
        "success": v_success,
        "cost": v_cost,
        "paths_tried": v_tried,
        "relaxation_s": t_relax,
        "sampling_s": t_sampling_vanilla,
        "restriction_s": v_restriction,
        "total_s": t_relax + t_sampling_vanilla + v_restriction,
    }

    flow_out, phi_proj, t_gnn, t_qp = predict_projected_flows(
        graph_tensors, flow_model, q_start, q_goal, device,
    )
    t0 = time.perf_counter()
    gnn_candidates = sample_candidate_paths_from_flows(
        graph_tensors=graph_tensors,
        phi=phi_proj,
        seed=seed + 1,
    )
    t_sampling_gnn = time.perf_counter() - t0
    g_success, g_cost, g_tried, g_restriction = _try_paths(
        gnn_candidates, solve_restriction, stop_on_first=False,
    )
    out["gnn_only"] = {
        "success": g_success,
        "cost": g_cost,
        "paths_tried": g_tried,
        "gnn_s": t_gnn,
        "qp_s": t_qp,
        "sampling_s": t_sampling_gnn,
        "restriction_s": g_restriction,
        "total_s": t_gnn + t_qp + t_sampling_gnn + g_restriction,
    }

    rank_timings: dict[str, float] = {}
    ranked = rank_candidate_paths(
        graph_tensors=graph_tensors,
        candidates=gnn_candidates,
        flow_out=flow_out,
        phi_proj=phi_proj,
        ranker=ranker,
        regions=regions,
        device=device,
        timings=rank_timings,
    )
    ranked_candidates = [gnn_candidates[i] for i in ranked]
    r_success, r_cost, r_tried, r_restriction = _try_paths(
        ranked_candidates, solve_restriction, stop_on_first=True,
    )
    t_ranknet = rank_timings.get("ranknet_s", 0.0)
    out["gnn_ranknet"] = {
        "success": r_success,
        "cost": r_cost,
        "paths_tried": r_tried,
        "gnn_s": t_gnn,
        "qp_s": t_qp,
        "sampling_s": t_sampling_gnn,
        "ranknet_s": t_ranknet,
        "restriction_s": r_restriction,
        "total_s": t_gnn + t_qp + t_sampling_gnn + t_ranknet + r_restriction,
    }
    return out


def _stats(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return None, None, None
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.std()), float(np.median(arr))


def aggregate(results):
    agg: dict[str, float | None] = {"num_instances": len(results)}
    for method in ("vanilla", "gnn_only", "gnn_ranknet"):
        rows = [r[method] for r in results if method in r]
        if not rows:
            continue
        agg[f"{method}_success_rate"] = float(np.mean([r["success"] for r in rows]))
        for key in (
            "paths_tried",
            "relaxation_s",
            "gnn_s",
            "qp_s",
            "sampling_s",
            "ranknet_s",
            "restriction_s",
            "total_s",
            "cost",
        ):
            m, s, med = _stats([r.get(key) for r in rows])
            agg[f"{method}_{key}_mean"] = m
            agg[f"{method}_{key}_std"] = s
            agg[f"{method}_{key}_median"] = med
    return agg


def _fmt(mean, std, median=None, *, seconds=True):
    if mean is None:
        return "-"
    suffix = " s" if seconds else ""
    if median is None:
        return f"{mean:.3f} +/- {std:.3f}{suffix}"
    return f"{mean:.3f} +/- {std:.3f}\nmed {median:.3f}{suffix}"


def print_results(agg):
    headers = [
        "Method", "Success", "Paths", "CR/GNN", "QP", "Sampling",
        "RankNet", "Restriction", "Total", "Cost",
    ]
    rows = _table_rows(agg)
    widths = [max(len(str(x).replace("\n", " ")) for x in col) for col in zip(headers, *rows)]
    print()
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("=" * w for w in widths))
    for row in rows:
        flat = [str(x).replace("\n", " ") for x in row]
        print("  ".join(x.ljust(w) for x, w in zip(flat, widths)))
    print(f"\nInstances evaluated: {int(agg['num_instances'])}")


def _table_rows(agg):
    def first_present(*keys):
        for key in keys:
            value = agg.get(key)
            if value is not None:
                return value
        return None

    rows = []
    labels = [
        ("vanilla", "Vanilla GCS"),
        ("gnn_only", "GNN only"),
        ("gnn_ranknet", "GNN + RankNet"),
    ]
    for key, label in labels:
        if f"{key}_success_rate" not in agg:
            continue
        rows.append([
            label,
            f"{100 * agg[f'{key}_success_rate']:.1f}%",
            _fmt(
                agg.get(f"{key}_paths_tried_mean"),
                agg.get(f"{key}_paths_tried_std"),
                agg.get(f"{key}_paths_tried_median"),
                seconds=False,
            ),
            _fmt(
                first_present(f"{key}_relaxation_s_mean", f"{key}_gnn_s_mean"),
                first_present(f"{key}_relaxation_s_std", f"{key}_gnn_s_std"),
                first_present(f"{key}_relaxation_s_median", f"{key}_gnn_s_median"),
            ),
            _fmt(agg.get(f"{key}_qp_s_mean"), agg.get(f"{key}_qp_s_std"), agg.get(f"{key}_qp_s_median")),
            _fmt(
                agg.get(f"{key}_sampling_s_mean"),
                agg.get(f"{key}_sampling_s_std"),
                agg.get(f"{key}_sampling_s_median"),
            ),
            _fmt(
                agg.get(f"{key}_ranknet_s_mean"),
                agg.get(f"{key}_ranknet_s_std"),
                agg.get(f"{key}_ranknet_s_median"),
            ),
            _fmt(
                agg.get(f"{key}_restriction_s_mean"),
                agg.get(f"{key}_restriction_s_std"),
                agg.get(f"{key}_restriction_s_median"),
            ),
            _fmt(agg.get(f"{key}_total_s_mean"), agg.get(f"{key}_total_s_std"), agg.get(f"{key}_total_s_median")),
            _fmt(
                agg.get(f"{key}_cost_mean"),
                agg.get(f"{key}_cost_std"),
                agg.get(f"{key}_cost_median"),
                seconds=False,
            ),
        ])
    return rows


def render_pdf(agg, out_path: Path, *, planner: str):
    headers = [
        "Method", "Success", "Paths", "CR / GNN", "QP", "Sampling",
        "RankNet", "Restriction", "Total", "Cost",
    ]
    rows = _table_rows(agg)
    col_widths = [1.6, 0.75, 1.25, 1.45, 1.15, 1.35, 1.35, 1.45, 1.45, 1.35]
    total_w = sum(col_widths)
    row_h = 0.68
    header_h = 0.6
    fig_w = total_w + 0.4
    fig_h = header_h + len(rows) * row_h + 0.45
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    fig.text(0.5, 0.98, f"Manipulation {planner} GCS Test Benchmark", ha="center", va="top",
             fontsize=10, fontweight="bold")

    x_starts = [sum(col_widths[:i]) / total_w for i in range(len(headers))]
    x_centers = [x_starts[i] + col_widths[i] / 2 / total_w for i in range(len(headers))]
    top = 0.86
    header_bg = "#2c3e50"
    alt_bg = "#ecf0f1"
    border = "#7f8c8d"

    for c, (hdr, xc) in enumerate(zip(headers, x_centers)):
        x0 = x_starts[c]
        w = col_widths[c] / total_w
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, top - header_h / fig_h), w, header_h / fig_h,
            boxstyle="square,pad=0", linewidth=0.5, edgecolor=border,
            facecolor=header_bg, transform=ax.transAxes, clip_on=False,
        ))
        ax.text(xc, top - (header_h / fig_h) / 2, hdr,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="white")

    for r, row in enumerate(rows):
        bg = alt_bg if r % 2 == 0 else "#ffffff"
        y_top = top - header_h / fig_h - r * (row_h / fig_h)
        for c, (cell, xc) in enumerate(zip(row, x_centers)):
            x0 = x_starts[c]
            w = col_widths[c] / total_w
            ax.add_patch(mpatches.FancyBboxPatch(
                (x0, y_top - row_h / fig_h), w, row_h / fig_h,
                boxstyle="square,pad=0", linewidth=0.5, edgecolor=border,
                facecolor=bg, transform=ax.transAxes, clip_on=False,
            ))
            x_pos = x_starts[c] + 0.01 if c == 0 else xc
            ax.text(x_pos, y_top - (row_h / fig_h) / 2, cell,
                    transform=ax.transAxes,
                    ha="left" if c == 0 else "center",
                    va="center",
                    fontsize=6.6,
                    fontweight="bold" if c == 0 else "normal")
    fig.savefig(str(out_path), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Results table -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Manipulation held-out test-set benchmark.")
    p.add_argument("--planner", choices=("convex", "nonconvex"), default="convex")
    p.add_argument("--h5_path", default=None)
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--flow_ckpt", default=None)
    p.add_argument("--ranknet_ckpt", default=None)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--max_instances", type=int, default=None)
    p.add_argument("--output_dir", default="manipulation/results/shelf_viz/dataset_benchmark")
    p.add_argument("--device", default="cpu")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--ffn_hidden_mult", type=int, default=2)
    p.add_argument("--dropout_p", type=float, default=0.1)
    p.add_argument("--decoder_hidden", default="256,256")
    p.add_argument("--decoder_dropout_p", type=float, default=0.1)
    p.add_argument("--pointnet_hidden", type=int, default=64)
    p.add_argument("--ranker_layers", type=int, default=3)
    p.add_argument("--ranker_heads", type=int, default=4)
    p.add_argument("--ranker_ffn_hidden", type=int, default=256)
    p.add_argument("--ranker_score_hidden", type=int, default=64)
    p.add_argument("--ranker_dropout_p", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()
    h5_path = Path(args.h5_path or DATASETS[args.planner])
    ckpt_dir = Path(args.ckpt_dir or f"manipulation/checkpoints/manipulation_{args.planner}")
    flow_ckpt = Path(args.flow_ckpt or ckpt_dir / f"manipulation_{args.planner}_flow_gnn.ckpt")
    ranknet_ckpt = Path(args.ranknet_ckpt or ckpt_dir / f"manipulation_{args.planner}_ranknet.ckpt")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {h5_path}")

    plant = None
    if args.planner == "nonconvex":
        plant, _, _, _ = build_shelf_plant()
    solver = MosekSolver()

    with h5py.File(h5_path, "r") as h5:
        regions = load_regions(h5)
        keys = [k for k in h5["samples"].keys() if h5["samples"][k].attrs.get("split") == args.split]
        keys.sort(key=int)
        if args.max_instances is not None:
            keys = keys[: args.max_instances + 1]
        if len(keys) < 2:
            raise SystemExit("Need at least one warmup and one benchmark instance.")

        sample0 = h5["samples"][keys[0]]
        g0 = np.asarray(sample0["g"][()], dtype=np.float64)
        graph0, _, _, _, _ = build_problem(
            args.planner, regions, g0[:7], g0[7:], solver=solver, plant=plant,
        )
        facet_dim = build_graph_tensors(graph0, regions, device=device).facet_dim

        encoder_hp = EncoderHParams(
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ffn_hidden_mult=args.ffn_hidden_mult,
            dropout_p=args.dropout_p,
        )
        decoder_hp = DecoderHParams(
            hidden_dims=tuple(int(s) for s in args.decoder_hidden.split(",") if s.strip()),
            dropout_p=args.decoder_dropout_p,
        )
        flow_model = load_flow_model(
            flow_ckpt,
            facet_dim=facet_dim,
            g_dim=14,
            encoder_hp=encoder_hp,
            decoder_hp=decoder_hp,
            pointnet_hidden=args.pointnet_hidden,
            device=device,
        )
        ranker = load_ranknet(
            ranknet_ckpt,
            cfg=RankNetConfig(
                d_model=args.d_model,
                num_layers=args.ranker_layers,
                num_heads=args.ranker_heads,
                ffn_hidden_dim=args.ranker_ffn_hidden,
                score_hidden_dim=args.ranker_score_hidden,
                dropout_p=args.ranker_dropout_p,
            ),
            device=device,
        )
        print(f"Loaded flow: {flow_ckpt}")
        print(f"Loaded ranknet: {ranknet_ckpt}")
        print(f"Evaluating {len(keys) - 1} instances from split='{args.split}' (1 warmup excluded)")

        print("Running warmup instance...")
        benchmark_instance(
            sample0, args.planner, regions, flow_model, ranker, solver, plant, device, seed=0,
        )
        print("Warmup done.\n")

        results = []
        for i, key in enumerate(tqdm(keys[1:], desc="benchmark")):
            res = benchmark_instance(
                h5["samples"][key],
                args.planner,
                regions,
                flow_model,
                ranker,
                solver,
                plant,
                device,
                seed=i + 1,
            )
            if res is None:
                print(f"  [WARN] instance {key}: relaxation infeasible")
                continue
            results.append(res)

    agg = aggregate(results)
    print(f"\nInstances evaluated: {int(agg['num_instances'])}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_pdf(agg, out_dir / f"results_table_{args.planner}.pdf", planner=args.planner)


if __name__ == "__main__":
    main()
