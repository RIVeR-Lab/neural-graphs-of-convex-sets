"""
Benchmark Vanilla GCS, Neural GCS, and FastPathPlanning on the quadrotor test set.

Methods
-------
Vanilla GCS      : solve convex relaxation → randomized rounding (up to MAX_PATHS paths)
GNN only         : GNN forward + QP projection → rounding (no re-ranking)
GNN + RankNet    : GNN forward + RankNet scoring → round in ranked order, stop at first feasible
FastPathPlanning : polygonal shortest path → smooth trajectory (non-GCS baseline)

Timing
------
- The first instance is used as a warmup and excluded from all reported statistics.
- GCS methods report: relaxation/GNN time, rounding time, total time.
- FastPathPlanning reports: cumulative total time (preprocess + plan).

Outputs
-------
- Console summary table
- quadrotor/results/results_table_<planner>.pdf
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from quadrotor.scripts._bootstrap import REPO_ROOT  # noqa: F401

from model.hparams import DecoderHParams, EncoderHParams
from model.inference import project_flows_qp
from model.model import GCSFlowPredictor
from model.ranknet import PathRankNet, RankNetConfig
from pydrake.geometry.optimization import GraphOfConvexSetsOptions, HPolyhedron
from pydrake.solvers import MosekSolver

logging.getLogger("drake").setLevel(logging.WARNING)

from quadrotor.fastpathplanning import SafeSet as FPPSafeSet, plan as fpp_plan
from quadrotor.gcs.data_generation import build_adjacency
from quadrotor.gcs.rounding import randomForwardPathSearch
from quadrotor.gcs.trajopt import (
    adjacency_to_edges,
    build_nonlinear_gcs_problem,
    solve_nonlinear_relaxation,
    solve_nonlinear_restriction,
)
from quadrotor.helpers import build_bezier_gcs
from model.dataset import _decode_str

MAX_PATHS = 10
MAX_ROUNDING_TRIALS = 100
_FPP_ALPHA = [0, 0, 0, 1]
_FPP_ZERO_DERIVS = {1: np.zeros(3), 2: np.zeros(3), 3: np.zeros(3)}


# ---------- fastpathplanning helpers ----------

def _region_box_bounds(region):
    A, b = region.A(), region.b()
    lb = np.full(3, -np.inf)
    ub = np.full(3, np.inf)
    for i in range(A.shape[0]):
        row = A[i]
        nz = np.nonzero(row)[0]
        if len(nz) == 1:
            dim = nz[0]
            if row[dim] > 0:
                ub[dim] = min(ub[dim], b[i] / row[dim])
            else:
                lb[dim] = max(lb[dim], b[i] / row[dim])
    return lb, ub


def _build_safe_set(regions):
    L, U = [], []
    for r in regions:
        lb, ub = _region_box_bounds(r)
        L.append(lb.tolist())
        U.append(ub.tolist())
    return FPPSafeSet(L, U, verbose=False)


# ---------- model loading ----------

def load_flow_model(ckpt_path, *, x_dim, g_dim, encoder_hp, decoder_hp, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model_state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model = GCSFlowPredictor(x_dim=x_dim, g_dim=g_dim, encoder_hp=encoder_hp, decoder_hp=decoder_hp)
    model.load_state_dict(model_state, strict=False)
    return model.eval().to(device)


def load_ranknet(ckpt_path, *, cfg, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    ranker_state = {k.removeprefix("ranker."): v for k, v in state.items() if k.startswith("ranker.")}
    if not ranker_state:
        ranker_state = state
    ranker = PathRankNet(cfg)
    ranker.load_state_dict(ranker_state, strict=False)
    return ranker.eval().to(device)


# ---------- GCS helpers ----------

def _gcs_options(convex_relaxation, gcs_obj):
    opts = GraphOfConvexSetsOptions()
    opts.convex_relaxation = convex_relaxation
    opts.max_rounded_paths = 0
    opts.preprocessing = False
    if gcs_obj.solver is not None:
        opts.solver = gcs_obj.solver
    if gcs_obj.options is not None:
        opts.solver_options = gcs_obj.options
    return opts


def solve_relaxation(gcs_obj):
    return gcs_obj.gcs.SolveShortestPath(gcs_obj.source, gcs_obj.target, _gcs_options(True, gcs_obj))


def solve_restriction(gcs_obj, path_edges):
    edge_set = set(path_edges)
    for edge in gcs_obj.gcs.Edges():
        edge.AddPhiConstraint(edge in edge_set)
    result = gcs_obj.gcs.SolveShortestPath(gcs_obj.source, gcs_obj.target, _gcs_options(True, gcs_obj))
    for edge in gcs_obj.gcs.Edges():
        edge.ClearPhiConstraints()
    return result


# ---------- RankNet path tensor helpers ----------

def _build_path_tensors(candidate_edge_lists, edge_lookup, name_to_idx):
    path_nodes_list, path_edges_list = [], []
    for path_edges in candidate_edge_lists:
        if not path_edges:
            continue
        node_seq = [name_to_idx[path_edges[0].u().name()]] + [
            name_to_idx[e.v().name()] for e in path_edges
        ]
        edge_seq = [-1] + [edge_lookup[(e.u().name(), e.v().name())] for e in path_edges]
        path_nodes_list.append(node_seq)
        path_edges_list.append(edge_seq)

    if not path_nodes_list:
        return None, None, None

    max_len = max(len(p) for p in path_nodes_list)
    P = len(path_nodes_list)
    node_arr = torch.full((P, max_len), -1, dtype=torch.long)
    edge_arr = torch.full((P, max_len), -1, dtype=torch.long)
    mask = torch.zeros((P, max_len), dtype=torch.bool)
    for i, (nodes, edges) in enumerate(zip(path_nodes_list, path_edges_list)):
        L = len(nodes)
        node_arr[i, :L] = torch.tensor(nodes, dtype=torch.long)
        edge_arr[i, :L] = torch.tensor(edges, dtype=torch.long)
        mask[i, :L] = True
    return node_arr, edge_arr, mask


# ---------- per-instance benchmark ----------

def _load_regions(grp):
    return [
        HPolyhedron(
            grp["regions"][key]["A"][()],
            grp["regions"][key]["b"][()],
        )
        for key in sorted(grp["regions"].keys(), key=lambda x: int(x))
    ]


def _build_problem(planner, regions, start_pose, goal_pose, solver):
    if planner == "convex":
        gcs_obj = build_bezier_gcs(regions, solver)
        gcs_obj.addSourceTarget(start_pose, goal_pose, zero_deriv_boundary=3)
        return (
            gcs_obj.gcs,
            gcs_obj.source,
            gcs_obj.target,
            lambda: solve_relaxation(gcs_obj),
            lambda path_edges: solve_restriction(gcs_obj, path_edges),
        )

    adj = build_adjacency(regions)
    _, graph, source, target = build_nonlinear_gcs_problem(
        regions, adjacency_to_edges(adj), start_pose, goal_pose,
    )
    return (
        graph,
        source,
        target,
        lambda: solve_nonlinear_relaxation(graph, source, target),
        lambda path_edges: solve_nonlinear_restriction(graph, source, target, path_edges),
    )


def benchmark_instance(grp, planner, flow_model, ranker, solver, device):

    g_np = np.array(grp["g"][()], dtype=np.float32)
    start_pose, goal_pose = g_np[:3], g_np[3:]
    x_np = np.array(grp["node_features"][()], dtype=np.float32)
    edge_u_stored = [_decode_str(s) for s in grp["edge_u"][()]]
    edge_v_stored = [_decode_str(s) for s in grp["edge_v"][()]]

    source_idx = int(np.argmax(x_np[:, 7]))
    target_idx = int(np.argmax(x_np[:, 8]))

    regions = _load_regions(grp)
    try:
        graph, source, target, solve_relaxed, solve_path_restriction = _build_problem(
            planner, regions, start_pose, goal_pose, solver,
        )
    except ValueError:
        return None

    all_edges = list(graph.Edges())
    edge_lookup = {(e.u().name(), e.v().name()): i for i, e in enumerate(all_edges)}
    name_to_idx = {v.name(): i for i, v in enumerate(graph.Vertices())}

    result: dict = {}

    # ── Vanilla GCS ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    relaxed = solve_relaxed()
    t_cr = time.perf_counter() - t0

    if not relaxed.is_success():
        return None

    relaxed_cost = float(relaxed.get_optimal_cost())

    vanilla_candidates = randomForwardPathSearch(
        graph, relaxed, source, target,
        max_paths=MAX_PATHS, max_trials=MAX_ROUNDING_TRIALS, seed=0,
    )

    t_round_vanilla = 0.0
    vanilla_best_cost = float("nan")
    vanilla_success = False
    vanilla_paths_tried = 0
    for path_edges in (vanilla_candidates or []):
        if path_edges is None:
            continue
        vanilla_paths_tried += 1
        t1 = time.perf_counter()
        res = solve_path_restriction(path_edges)
        t_round_vanilla += time.perf_counter() - t1
        if res.is_success():
            c = float(res.get_optimal_cost())
            if not vanilla_success or c < vanilla_best_cost:
                vanilla_best_cost = c
                vanilla_success = True

    result["vanilla"] = {
        "t_cr_s": t_cr,
        "t_rounding_s": t_round_vanilla,
        "t_total_s": t_cr + t_round_vanilla,
        "relaxed_cost": relaxed_cost,
        "cost": vanilla_best_cost,
        "success": vanilla_success,
        "n_candidates": len(vanilla_candidates) if vanilla_candidates else 0,
        "num_paths_tried": vanilla_paths_tried,
    }

    # ── GNN shared forward pass ───────────────────────────────────────────────
    x_t = torch.tensor(x_np).to(device)
    g_t = torch.tensor(g_np).to(device)
    src_idx_t = torch.tensor([name_to_idx.get(u, 0) for u in edge_u_stored], dtype=torch.long)
    dst_idx_t = torch.tensor([name_to_idx.get(v, 0) for v in edge_v_stored], dtype=torch.long)
    edge_index_t = torch.stack([src_idx_t, dst_idx_t], dim=0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        flow_out = flow_model(x=x_t, edge_index=edge_index_t, g=g_t, batch=None)
        phi_hat = torch.sigmoid(flow_out.edge_logits).detach().cpu()
    t_gnn_forward = time.perf_counter() - t0

    t1 = time.perf_counter()
    phi_proj = project_flows_qp(
        edge_index=edge_index_t.cpu(),
        phi_hat=phi_hat,
        num_nodes=int(x_np.shape[0]),
        source_idx=source_idx,
        target_idx=target_idx,
    )
    t_qp = time.perf_counter() - t1
    t_gnn = t_gnn_forward + t_qp

    gnn_candidates = randomForwardPathSearch(
        graph, relaxed, source, target,
        max_paths=MAX_PATHS, max_trials=MAX_ROUNDING_TRIALS, seed=1,
    )

    # ── GNN only ─────────────────────────────────────────────────────────────
    t_round_gnn = 0.0
    gnn_best_cost = float("nan")
    gnn_success = False
    gnn_paths_tried = 0
    for path_edges in (gnn_candidates or []):
        if path_edges is None:
            continue
        gnn_paths_tried += 1
        t1 = time.perf_counter()
        res = solve_path_restriction(path_edges)
        t_round_gnn += time.perf_counter() - t1
        if res.is_success():
            c = float(res.get_optimal_cost())
            if not gnn_success or c < gnn_best_cost:
                gnn_best_cost = c
                gnn_success = True

    result["gnn_only"] = {
        "t_gnn_s": t_gnn,
        "t_gnn_forward_s": t_gnn_forward,
        "t_qp_s": t_qp,
        "t_rounding_s": t_round_gnn,
        "t_total_s": t_gnn + t_round_gnn,
        "relaxed_cost": relaxed_cost,
        "cost": gnn_best_cost,
        "success": gnn_success,
        "n_candidates": len(gnn_candidates) if gnn_candidates else 0,
        "num_paths_tried": gnn_paths_tried,
    }

    # ── GNN + RankNet ────────────────────────────────────────────────────────
    if ranker is not None and gnn_candidates:
        node_arr, edge_arr, path_mask = _build_path_tensors(gnn_candidates, edge_lookup, name_to_idx)
        t_rank = 0.0
        scores = None
        if node_arr is not None:
            t1 = time.perf_counter()
            with torch.no_grad():
                scores = ranker(
                    node_embeddings=flow_out.node_embeddings.to(device),
                    edge_flows=phi_proj.to(device),
                    path_node_indices=node_arr,
                    path_edge_indices=edge_arr,
                    path_mask=path_mask,
                )
            t_rank = time.perf_counter() - t1

        ranked_indices = torch.argsort(scores, descending=True).cpu().tolist() if scores is not None else []
        t_round_ranknet = 0.0
        ranknet_cost = float("nan")
        ranknet_success = False
        ranknet_paths_tried = 0
        for ri in ranked_indices:
            path_edges = gnn_candidates[ri]
            if path_edges is None:
                continue
            ranknet_paths_tried += 1
            t1 = time.perf_counter()
            res = solve_path_restriction(path_edges)
            t_round_ranknet += time.perf_counter() - t1
            if res.is_success():
                ranknet_cost = float(res.get_optimal_cost())
                ranknet_success = True
                break

        result["gnn_ranknet"] = {
            "t_gnn_s": t_gnn,
            "t_ranknet_s": t_rank,
            "t_rounding_s": t_round_ranknet,
            "t_total_s": t_gnn + t_rank + t_round_ranknet,
            "relaxed_cost": relaxed_cost,
            "cost": ranknet_cost,
            "success": ranknet_success,
            "num_paths_tried": ranknet_paths_tried,
        }

    # ── FastPathPlanning ──────────────────────────────────────────────────────
    # The FPP paper (Marcucci et al., 2024) excludes CVXPY problem-construction
    # overhead from reported runtimes, on the basis that talking to the solver
    # directly would make it negligible. We mirror that convention here so our
    # FPP numbers are apples-to-apples with the paper.
    try:
        t0 = time.perf_counter()
        safe_set = _build_safe_set(regions)
        t_preprocess = time.perf_counter() - t0 - getattr(safe_set, "cvxpy_time", 0.0)

        dist = float(np.linalg.norm(goal_pose - start_pose))
        T = max(dist, 1.0)

        t1 = time.perf_counter()
        path, plan_cvxpy_time = fpp_plan(
            safe_set, start_pose, goal_pose, T, _FPP_ALPHA,
            der_init=_FPP_ZERO_DERIVS, der_term=_FPP_ZERO_DERIVS,
            verbose=False,
        )
        t_plan = time.perf_counter() - t1 - plan_cvxpy_time
        fpp_success = path is not None
    except Exception:
        t_preprocess = float("nan")
        t_plan = float("nan")
        fpp_success = False

    result["fastpathplanning"] = {
        "t_preprocess_s": t_preprocess,
        "t_plan_s": t_plan,
        "t_total_s": t_preprocess + t_plan if fpp_success else float("nan"),
        "success": fpp_success,
    }

    return result


# ---------- aggregation ----------

def _stats(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return None, None, None
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.std()), float(np.median(arr))


def aggregate(all_results):
    agg = {"num_instances": len(all_results)}

    for method in ("vanilla", "gnn_only", "gnn_ranknet"):
        rows = [r[method] for r in all_results if method in r]
        if not rows:
            continue
        agg[f"{method}_success_rate"] = float(np.mean([r["success"] for r in rows]))
        for key in ("t_total_s", "t_rounding_s", "cost", "num_paths_tried"):
            m, s, med = _stats([r.get(key) for r in rows])
            agg[f"{method}_{key}_mean"] = m
            agg[f"{method}_{key}_std"] = s
            agg[f"{method}_{key}_median"] = med
        if method == "vanilla":
            m, s, med = _stats([r["t_cr_s"] for r in rows])
            agg["vanilla_t_cr_s_mean"] = m
            agg["vanilla_t_cr_s_std"] = s
            agg["vanilla_t_cr_s_median"] = med
        else:
            m, s, med = _stats([r["t_gnn_s"] for r in rows])
            agg[f"{method}_t_gnn_s_mean"] = m
            agg[f"{method}_t_gnn_s_std"] = s
            agg[f"{method}_t_gnn_s_median"] = med
        if method == "gnn_only":
            m, s, med = _stats([r["t_gnn_forward_s"] for r in rows])
            agg["gnn_only_t_gnn_forward_s_mean"] = m
            agg["gnn_only_t_gnn_forward_s_std"] = s
            agg["gnn_only_t_gnn_forward_s_median"] = med
            m, s, med = _stats([r["t_qp_s"] for r in rows])
            agg["gnn_only_t_qp_s_mean"] = m
            agg["gnn_only_t_qp_s_std"] = s
            agg["gnn_only_t_qp_s_median"] = med
        if method == "gnn_ranknet":
            m, s, med = _stats([r["t_ranknet_s"] for r in rows])
            agg["gnn_ranknet_t_ranknet_s_mean"] = m
            agg["gnn_ranknet_t_ranknet_s_std"] = s
            agg["gnn_ranknet_t_ranknet_s_median"] = med

    fpp_rows = [r["fastpathplanning"] for r in all_results if "fastpathplanning" in r]
    if fpp_rows:
        agg["fpp_success_rate"] = float(np.mean([r["success"] for r in fpp_rows]))
        m, s, med = _stats([r["t_preprocess_s"] for r in fpp_rows])
        agg["fpp_t_preprocess_s_mean"] = m
        agg["fpp_t_preprocess_s_std"] = s
        agg["fpp_t_preprocess_s_median"] = med
        m, s, med = _stats([r["t_plan_s"] for r in fpp_rows if r["success"]])
        agg["fpp_t_plan_s_mean"] = m
        agg["fpp_t_plan_s_std"] = s
        agg["fpp_t_plan_s_median"] = med
        m, s, med = _stats([r["t_total_s"] for r in fpp_rows if r["success"]])
        agg["fpp_t_total_s_mean"] = m
        agg["fpp_t_total_s_std"] = s
        agg["fpp_t_total_s_median"] = med

    return agg


# ---------- console table ----------

def print_results(all_results, agg):
    def _fmt(m, s, med=None):
        if m is None:
            return "  —  "
        if med is not None:
            return f"{m:.3f}±{s:.3f} ({med:.3f})"
        return f"{m:.3f}±{s:.3f}"

    def _fmt_paths(m, s, med=None):
        if m is None:
            return "  —  "
        if med is not None:
            return f"{m:.1f}±{s:.1f} ({med:.1f})"
        return f"{m:.1f}±{s:.1f}"

    header = (
        f"{'Method':<20} {'Success %':>9}  {'Paths Used':>20}  "
        f"{'CR/GNN (s)':>22}  {'Round (s)':>22}  "
        f"{'Total (s)':>22}  {'C_round':>13}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    n = len(all_results)

    def _row(label, success_rate, paths, t_first, t_round, t_total, c_round):
        return (
            f"{label:<20} {success_rate:>9.1%}  "
            f"{_fmt_paths(*paths):>20}  "
            f"{_fmt(*t_first):>22}  {_fmt(*t_round):>22}  "
            f"{_fmt(*t_total):>22}  {_fmt(*c_round):>13}"
        )

    for method, label, cr_key in [
        ("vanilla",     "Vanilla GCS",        "vanilla_t_cr_s"),
        ("gnn_only",    "GNN only",            "gnn_only_t_gnn_s"),
        ("gnn_ranknet", "GNN + RankNet",       "gnn_ranknet_t_gnn_s"),
    ]:
        if f"{method}_success_rate" not in agg:
            continue
        print(_row(
            label,
            agg[f"{method}_success_rate"],
            (
                agg.get(f"{method}_num_paths_tried_mean"),
                agg.get(f"{method}_num_paths_tried_std"),
                agg.get(f"{method}_num_paths_tried_median"),
            ),
            (agg.get(f"{cr_key}_mean"), agg.get(f"{cr_key}_std"), agg.get(f"{cr_key}_median")),
            (
                agg.get(f"{method}_t_rounding_s_mean"),
                agg.get(f"{method}_t_rounding_s_std"),
                agg.get(f"{method}_t_rounding_s_median"),
            ),
            (
                agg.get(f"{method}_t_total_s_mean"),
                agg.get(f"{method}_t_total_s_std"),
                agg.get(f"{method}_t_total_s_median"),
            ),
            (
                agg.get(f"{method}_cost_mean"),
                agg.get(f"{method}_cost_std"),
                agg.get(f"{method}_cost_median"),
            ),
        ))

    if "fpp_success_rate" in agg:
        print(f"{'─'*len(header)}")
        total_str = _fmt(
            agg.get("fpp_t_total_s_mean"),
            agg.get("fpp_t_total_s_std"),
            agg.get("fpp_t_total_s_median"),
        )
        print(
            f"{'FastPathPlanning':<20} {agg['fpp_success_rate']:>9.1%}  "
            f"{'—':>20}  "
            f"{'prep+plan':>22}  {'—':>22}  "
            f"{total_str:>22}  {'—':>13}"
        )

    print(sep)
    if "gnn_only_t_gnn_forward_s_mean" in agg:
        print(
            f"  GNN breakdown — forward: {agg['gnn_only_t_gnn_forward_s_mean']:.3f}±"
            f"{agg['gnn_only_t_gnn_forward_s_std']:.3f}s "
            f"(median {agg['gnn_only_t_gnn_forward_s_median']:.3f}s)  |  "
            f"QP: {agg['gnn_only_t_qp_s_mean']:.3f}±{agg['gnn_only_t_qp_s_std']:.3f}s "
            f"(median {agg['gnn_only_t_qp_s_median']:.3f}s)"
        )
    print(f"  Instances evaluated: {n}")


# ---------- PDF table ----------

def _fmt_cell(mean, std, median=None, pct=False):
    if mean is None:
        return "—"
    if pct:
        return f"{mean:.1f} ± {std:.1f}%"
    if median is not None:
        return f"{mean:.3f} ± {std:.3f}\nmed {median:.3f} s"
    return f"{mean:.3f} ± {std:.3f} s"


def render_pdf(agg, out_path):
    headers = ["Method", "Success %", "Paths Used", "CR / GNN (s)",
               "Rounding (s)", "Total (s)", "C_round ↓"]

    def success_str(key):
        r = agg.get(key)
        return f"{r * 100:.1f}%" if r is not None else "—"

    def _fmt_paths(m, s, med=None):
        if m is None:
            return "—"
        if med is not None:
            return f"{m:.1f} ± {s:.1f}\nmed {med:.1f}"
        return f"{m:.1f} ± {s:.1f}"

    def _fmt_cost(m, s, med=None):
        if m is None:
            return "—"
        if med is not None:
            return f"{m:.3f} ± {s:.3f}\nmed {med:.3f}"
        return f"{m:.3f} ± {s:.3f}"

    rows = []

    for method, label, cr_key in [
        ("vanilla",     "Vanilla GCS",   "vanilla_t_cr_s"),
        ("gnn_only",    "Neural GCS",    "gnn_only_t_gnn_s"),
        ("gnn_ranknet", "Neural GCS\n+ RankNet", "gnn_ranknet_t_gnn_s"),
    ]:
        if f"{method}_success_rate" not in agg:
            continue
        rows.append([
            label,
            success_str(f"{method}_success_rate"),
            _fmt_paths(agg.get(f"{method}_num_paths_tried_mean"),
                       agg.get(f"{method}_num_paths_tried_std"),
                       agg.get(f"{method}_num_paths_tried_median")),
            _fmt_cell(agg.get(f"{cr_key}_mean"), agg.get(f"{cr_key}_std"), agg.get(f"{cr_key}_median")),
            _fmt_cell(
                agg.get(f"{method}_t_rounding_s_mean"),
                agg.get(f"{method}_t_rounding_s_std"),
                agg.get(f"{method}_t_rounding_s_median"),
            ),
            _fmt_cell(
                agg.get(f"{method}_t_total_s_mean"),
                agg.get(f"{method}_t_total_s_std"),
                agg.get(f"{method}_t_total_s_median"),
            ),
            _fmt_cost(
                agg.get(f"{method}_cost_mean"),
                agg.get(f"{method}_cost_std"),
                agg.get(f"{method}_cost_median"),
            ),
        ])

    if "fpp_success_rate" in agg:
        rows.append([
            "FastPathPlanning",
            success_str("fpp_success_rate"),
            "—",
            "—",
            "—",
            _fmt_cell(
                agg.get("fpp_t_total_s_mean"),
                agg.get("fpp_t_total_s_std"),
                agg.get("fpp_t_total_s_median"),
            ),
            "—",
        ])

    n_rows = len(rows)
    n_cols = len(headers)
    col_widths = [2.2, 0.85, 1.35, 1.65, 1.55, 1.55, 1.6]
    total_w = sum(col_widths)
    fig_w = total_w + 0.3
    row_h = 0.52
    header_h = 0.58
    fig_h = header_h + n_rows * row_h + 0.3

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    fig.text(0.5, 0.98, "Quadrotor Motion Planning Benchmark", ha="center", va="top",
             fontsize=10, fontweight="bold")

    x_starts = [sum(col_widths[:i]) / total_w for i in range(n_cols)]
    x_centers = [x_starts[i] + col_widths[i] / 2 / total_w for i in range(n_cols)]
    top = 0.88

    HEADER_BG = "#2c3e50"
    ALT_BG = "#ecf0f1"
    WHITE = "#ffffff"
    BORDER = "#7f8c8d"

    for c, (hdr, xc) in enumerate(zip(headers, x_centers)):
        x0 = x_starts[c]
        w = col_widths[c] / total_w
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, top - header_h / fig_h), w, header_h / fig_h,
            boxstyle="square,pad=0", linewidth=0.5, edgecolor=BORDER,
            facecolor=HEADER_BG, transform=ax.transAxes, clip_on=False,
        ))
        ax.text(xc, top - (header_h / fig_h) / 2, hdr,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")

    for r, row_data in enumerate(rows):
        bg = ALT_BG if r % 2 == 0 else WHITE
        y_top = top - header_h / fig_h - r * (row_h / fig_h)
        for c, (cell, xc) in enumerate(zip(row_data, x_centers)):
            x0 = x_starts[c]
            w = col_widths[c] / total_w
            ax.add_patch(mpatches.FancyBboxPatch(
                (x0, y_top - row_h / fig_h), w, row_h / fig_h,
                boxstyle="square,pad=0", linewidth=0.5, edgecolor=BORDER,
                facecolor=bg, transform=ax.transAxes, clip_on=False,
            ))
            x_pos = x_starts[c] + 0.01 if c == 0 else xc
            ha = "left" if c == 0 else "center"
            ax.text(x_pos, y_top - (row_h / fig_h) / 2, cell,
                    transform=ax.transAxes, ha=ha, va="center",
                    fontsize=7.5, fontweight="bold" if c == 0 else "normal")

    fig.savefig(str(out_path), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Results table → {out_path}")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", default="quadrotor/dataset/quadrotor_gcs_convex.h5")
    from quadrotor.checkpoint_paths import (
        add_planner_checkpoint_args,
        resolve_flow_ckpt,
        resolve_ranknet_ckpt,
    )
    add_planner_checkpoint_args(parser, default_planner="convex")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--output_dir", default="quadrotor/results")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ffn_hidden_mult", type=int, default=2)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--decoder_hidden", default="256,256")
    parser.add_argument("--decoder_dropout_p", type=float, default=0.1)
    parser.add_argument("--ranker_layers", type=int, default=3)
    parser.add_argument("--ranker_heads", type=int, default=4)
    parser.add_argument("--ranker_ffn_hidden", type=int, default=256)
    parser.add_argument("--ranker_score_hidden", type=int, default=64)
    parser.add_argument("--ranker_dropout_p", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.flow_ckpt = resolve_flow_ckpt(args)

    import h5py
    from tqdm import tqdm

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoder_hp = EncoderHParams(
        d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads,
        ffn_hidden_mult=args.ffn_hidden_mult, dropout_p=args.dropout_p,
    )
    hidden = tuple(int(s) for s in args.decoder_hidden.split(",") if s.strip())
    decoder_hp = DecoderHParams(hidden_dims=hidden, dropout_p=args.decoder_dropout_p)

    flow_model = load_flow_model(
        args.flow_ckpt, x_dim=9, g_dim=6,
        encoder_hp=encoder_hp, decoder_hp=decoder_hp, device=device,
    )

    if args.ranknet_ckpt is not None:
        ranknet_path = args.ranknet_ckpt
    else:
        ranknet_path = resolve_ranknet_ckpt(args)
        if not Path(ranknet_path).is_file():
            ranknet_path = None

    ranker = None
    if ranknet_path is not None:
        ranker_cfg = RankNetConfig(
            d_model=args.d_model, num_layers=args.ranker_layers, num_heads=args.ranker_heads,
            ffn_hidden_dim=args.ranker_ffn_hidden, score_hidden_dim=args.ranker_score_hidden,
            dropout_p=args.ranker_dropout_p,
        )
        ranker = load_ranknet(ranknet_path, cfg=ranker_cfg, device=device)
        print(f"RankNet loaded from {ranknet_path}")
    else:
        print("No RankNet checkpoint — skipping GNN+RankNet column.")

    solver = MosekSolver()

    with h5py.File(args.h5_path, "r") as h5:
        keys = [k for k in h5["samples"].keys()
                if _decode_str(h5["samples"][k].attrs.get("split", "")) == args.split]
        keys.sort(key=int)
        if args.max_instances is not None:
            keys = keys[: args.max_instances + 1]  # +1 for warmup
        print(f"\nEvaluating {len(keys) - 1} instances from split='{args.split}' (1 warmup excluded)")

        # Warmup: run first instance, discard result
        print("Running warmup instance...")
        warmup_grp = h5["samples"][keys[0]]
        benchmark_instance(warmup_grp, args.planner, flow_model, ranker, solver, device)
        print("Warmup done.\n")

        all_results = []
        for k in tqdm(keys[1:], desc="benchmark"):
            grp = h5["samples"][k]
            try:
                res = benchmark_instance(grp, args.planner, flow_model, ranker, solver, device)
                if res is not None:
                    all_results.append(res)
                else:
                    print(f"  [WARN] instance {k}: CR infeasible or invalid query")
            except Exception as e:
                import traceback
                print(f"  [WARN] instance {k} failed: {e}")
                traceback.print_exc()

    agg = aggregate(all_results)
    print_results(all_results, agg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_pdf(agg, out_dir / f"results_table_{args.planner}.pdf")


if __name__ == "__main__":
    main()
