#!/usr/bin/env python3
"""
Chance-constrained circle demo: inflate shelf/bin/table obstacles, regenerate IRIS
regions, and plan convex or nonconvex vanilla/neural GCS for nominal vs inflated scenes.

Usage:
    bash scripts/setup_iiwa_models.sh
    python scripts/plan_manipulation_chance_constraint.py
    python scripts/plan_manipulation_chance_constraint.py --planner nonconvex
    python scripts/plan_manipulation_chance_constraint.py --planner nonconvex --method neural_ranknet
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

logging.getLogger("drake").setLevel(logging.WARNING)

from pydrake.geometry import Mesh, Rgba, StartMeshcat
from pydrake.math import RigidTransform

from eval_manipulation_circle_demo import (
    build_graph_tensors,
    load_flow_model,
    load_ranknet,
    plan_circle,
)
from manipulation.iiwa_helpers import build_shelf_plant, visualize_trajectory
from manipulation.paths import (
    DEFAULT_DIRECTIVES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REGIONS_PATH,
    manipulation_models_ready,
    manipulation_models_hint,
)
from manipulation.scene_obstacle_inflation import inflated_directives_yaml
from manipulation.shelf_gcs import (
    build_demo_sequences,
    build_seed_points,
    generate_regions,
    load_regions,
    plan_and_make_trajectory,
    planning_configurations,
    save_regions,
)
from manipulation.trajopt import (
    build_nonlinear_gcs_problem,
    build_region_edges,
    iiwa_kinematic_limits,
    plan_nonlinear_gcs_path,
    region_list,
)
from model.hparams import DecoderHParams, EncoderHParams
from model.ranknet import RankNetConfig
from quadrotor.chance_constraints import inflate_regions, write_legend_gltf
from quadrotor.gcs.linear import LinearGCS
from quadrotor.obstacle_inflation import inflation_margin

INFLATED_REGIONS_PATH = DEFAULT_REGIONS_PATH.parent / "IRIS_inflated.reg"

SEED = 17
SIGMA = 0.02
DELTA = 0.1
SPEED = 2.0
METHODS = ("vanilla", "neural_ranknet")

TRAJ_NOMINAL = Rgba(0.0, 0.0, 1.0, 1.0)
TRAJ_INFLATED = Rgba(1.0, 0.75, 0.0, 1.0)
LEGEND_POS = [0.45, 0.0, 1.35]


def inflate_regions_dict(regions_nom: dict, margin: float) -> dict:
    names = list(regions_nom.keys())
    polys = inflate_regions(list(regions_nom.values()), margin)
    return dict(zip(names, polys))


def merge_inflated_regions(regions_nom: dict, iris_inflated: dict, margin: float) -> dict:
    regions = inflate_regions_dict(regions_nom, margin)
    for name, poly in iris_inflated.items():
        regions[name] = poly
    print(
        f"  Inflated region set: {len(iris_inflated)} workspace IRIS + "
        f"{len(regions) - len(iris_inflated)} tightened nominal"
    )
    return regions


def ensure_regions(
    plant,
    diagram,
    seed_points: dict,
    path: Path,
    *,
    regenerate: bool,
    workers: int | None,
    skip_failed: bool = False,
) -> dict:
    if path.exists() and not regenerate:
        print(f"Loading IRIS regions from {path}")
        return load_regions(path)

    print(f"Generating IRIS regions -> {path} (this may take several minutes)...")
    regions = generate_regions(
        plant, diagram, seed_points, workers=workers, skip_failed=skip_failed,
    )
    save_regions(regions, path)
    print(f"Saved {len(regions)} regions to {path}")
    return regions


def load_neural_models(planner: str, regions: dict, sequence, *, plant, device):
    ckpt_dir = f"manipulation/checkpoints/manipulation_{planner}"
    flow_ckpt = f"{ckpt_dir}/manipulation_{planner}_flow_gnn.ckpt"
    ranknet_ckpt = f"{ckpt_dir}/manipulation_{planner}_ranknet.ckpt"

    encoder_hp = EncoderHParams(d_model=128, num_layers=4, num_heads=4, ffn_hidden_mult=2, dropout_p=0.1)
    decoder_hp = DecoderHParams(hidden_dims=(256, 256), dropout_p=0.1)
    if planner == "convex":
        tmp = LinearGCS(regions.copy())
        tmp.addSourceTarget(sequence[0], sequence[1])
        tmp_graph = tmp.gcs
    else:
        polys = region_list(regions)
        vel_limits, accel_limits = iiwa_kinematic_limits(plant)
        _, tmp_graph, _, _ = build_nonlinear_gcs_problem(
            polys,
            build_region_edges(polys),
            sequence[0],
            sequence[1],
            vel_limits=vel_limits,
            accel_limits=accel_limits,
        )
    facet_dim = build_graph_tensors(tmp_graph, regions, device=device).facet_dim
    flow_model = load_flow_model(
        flow_ckpt,
        facet_dim=facet_dim,
        g_dim=14,
        encoder_hp=encoder_hp,
        decoder_hp=decoder_hp,
        pointnet_hidden=64,
        device=device,
    )
    ranker = load_ranknet(
        ranknet_ckpt,
        cfg=RankNetConfig(d_model=128, num_layers=3, num_heads=4),
        device=device,
    )
    print(f"  Loaded flow: {flow_ckpt}")
    print(f"  Loaded ranknet: {ranknet_ckpt}")
    return flow_model, ranker


def plan_circle_cc(
    regions,
    sequence,
    *,
    planner: str,
    method: str,
    plant,
    seed: int,
    speed: float,
    flow_model=None,
    ranker=None,
    device=None,
    verbose: bool,
):
    if method != "vanilla":
        mode = "neural"
        result = plan_circle(
            planner=planner,
            mode=mode,
            regions=regions,
            sequence=sequence,
            plant=plant,
            seed=seed,
            speed=speed,
            flow_model=flow_model,
            ranker=ranker,
            device=device,
        )
        return result.trajectory, result.elapsed_s, result.cost

    if planner == "convex":
        path, traj, solve_time, gcs_cost = plan_and_make_trajectory(
            regions, sequence, seed=seed, speed=speed, verbose=verbose,
        )
        return traj, solve_time, gcs_cost

    traj, solve_time, segment_results = plan_nonlinear_gcs_path(
        regions, sequence, plant=plant, verbose=verbose,
    )
    if traj is None:
        return None, solve_time, float("nan")
    gcs_cost = sum(
        float(r.get_optimal_cost())
        for r in segment_results
        if r.is_success()
    )
    return traj, solve_time, gcs_cost


def attach_legend(meshcat) -> None:
    assets_dir = Path(tempfile.mkdtemp(prefix="cc_manip_legend_"))
    try:
        legend_entries = [
            ("Nominal GCS", TRAJ_NOMINAL),
            ("Chance constraint GCS", TRAJ_INFLATED),
        ]
        gltf_path = write_legend_gltf(legend_entries, assets_dir, quad_w=6.0, quad_h=2.5)
        meshcat.SetObject("/legend", Mesh(str(gltf_path), 1.0))
        meshcat.SetTransform("/legend", RigidTransform(LEGEND_POS))
    finally:
        shutil.rmtree(assets_dir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=("convex", "nonconvex"), default="convex")
    ap.add_argument("--method", choices=METHODS, default="vanilla")
    args = ap.parse_args()

    assert 0.0 < DELTA < 0.5

    if not manipulation_models_ready():
        print(manipulation_models_hint(), file=sys.stderr)
        sys.exit(1)

    margin_m = inflation_margin(SIGMA, DELTA)
    print(f"sigma={SIGMA}  delta={DELTA}  workspace margin={margin_m:.4f} m")

    seed_points = build_seed_points()
    print(f"IRIS seeds: {list(seed_points.keys())}")

    print("\n[Nominal] Building scene...")
    plant_nom, _, diagram_nom, _ = build_shelf_plant()
    regions_nom = ensure_regions(
        plant_nom, diagram_nom, seed_points, DEFAULT_REGIONS_PATH,
        regenerate=False,
        workers=None,
    )

    infl_yaml = inflated_directives_yaml(DEFAULT_DIRECTIVES, margin_m)
    print(f"\n[Inflated] Building scene (margin={margin_m:.4f} m)...")
    print(f"  directives: {infl_yaml}")
    plant_infl, _, diagram_infl, _ = build_shelf_plant(directives_path=infl_yaml)
    iris_infl = ensure_regions(
        plant_infl, diagram_infl, seed_points, INFLATED_REGIONS_PATH,
        regenerate=not INFLATED_REGIONS_PATH.exists(),
        workers=None,
        skip_failed=True,
    )
    regions_infl = merge_inflated_regions(regions_nom, iris_infl, margin_m)

    demo_configs_nom = planning_configurations(regions_nom)
    sequence_nom = build_demo_sequences(demo_configs_nom, seed_points)["circle"]
    demo_configs_infl = planning_configurations(regions_infl)
    sequence_infl = build_demo_sequences(demo_configs_infl, seed_points)["circle"]
    print(f"\nCircle sequence: {len(sequence_nom)} waypoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow_model = ranker = None
    if args.method != "vanilla":
        print(f"\nLoading neural models ({args.method}, {args.planner})...")
        flow_model, ranker = load_neural_models(
            args.planner, regions_nom, sequence_nom, plant=plant_nom, device=device,
        )

    plans = {}
    costs = {}
    for label, regions, sequence, plant, tag in (
        ("nominal", regions_nom, sequence_nom, plant_nom, "nominal"),
        ("chance-constrained", regions_infl, sequence_infl, plant_infl, "inflated"),
    ):
        print(f"\n[{label}] {args.method} {args.planner} GCS circle plan...")
        traj, solve_time, gcs_cost = plan_circle_cc(
            regions, sequence,
            planner=args.planner,
            method=args.method,
            plant=plant,
            seed=SEED,
            speed=SPEED,
            flow_model=flow_model,
            ranker=ranker,
            device=device,
            verbose=True,
        )
        if traj is None:
            print(f"  [{label}] planning failed.")
            continue
        costs[tag] = gcs_cost
        print(f"  GCS path cost={gcs_cost:.3f}  solve={solve_time:.2f}s  "
              f"duration={traj.start_time():.2f}->{traj.end_time():.2f}s")
        plans[tag] = traj

    if costs:
        print("\n--- Path costs ---")
        if "nominal" in costs:
            print(f"  nominal:              {costs['nominal']:.3f}")
        if "inflated" in costs:
            print(f"  chance-constrained:   {costs['inflated']:.3f}")

    if len(plans) < 2:
        print("\nDone, but combined viz skipped because a plan failed.")
        sys.exit(1)

    print("\nRendering Meshcat HTML...")
    meshcat = StartMeshcat()
    visualize_trajectory(
        meshcat,
        [plans["nominal"], plans["inflated"]],
        show_line=True,
        ghost_configs=sequence_nom,
        alpha=0.25,
        plan_wait=3.0,
    )
    attach_legend(meshcat)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"sigma{SIGMA:.2f}".replace(".", "p")
    html_path = DEFAULT_OUTPUT_DIR / f"cc_circle_{args.planner}_{args.method}_{tag}_seed{SEED}.html"
    html_path.write_text(meshcat.StaticHtml())
    print(f"\nSaved -> {html_path}")
    print("EE traces: blue=Nominal GCS, yellow=Chance constraint GCS")


if __name__ == "__main__":
    main()
