#!/usr/bin/env python3
"""Physics demo: blue block right bin -> top shelf; red block left bin -> top shelf.

Neural GCS (convex or nonlinear) plans coarse arm waypoints through IRIS regions.
Drake contact simulation runs the IIWA, actuated WSG, and two free-floating blocks.
Vertical pick/place phases are spliced onto the GCS trajectory; gripper
close/open is timed by a simple state machine.

GCS waypoints:
  1. home_ready
  2. right_bin_pregrasp
  3. right_bin_grasp
  4. top_shelf_pre_place
  5. left_bin_pregrasp
  6. left_bin_grasp
  7. via_front
  8. red_top_shelf_pre_place
  9. home_second_shelf (Top Rack)

Execution splices (physics):
  blue: pick right bin -> reach top shelf pre-place, hold 0.5s, open, hold 1.3s, move on
  red:  pick left bin -> same drop at red shelf pre-place -> home

Examples:
  python scripts/demonstrate_pick_and_place.py
  python scripts/demonstrate_pick_and_place.py --planner convex
  python scripts/demonstrate_pick_and_place.py --mode vanilla
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
    InverseDynamicsController,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    RigidTransform,
    Role,
    SchunkWsgPositionController,
    SpatialInertia,
    SpatialVelocity,
    StartMeshcat,
    StateInterpolatorWithDiscreteDerivative,
    UnitInertia,
)
from pydrake.geometry import Box as DrakeBox
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource, TrajectorySource
from pydrake.trajectories import PiecewisePolynomial

from manipulation.iiwa_helpers import build_shelf_plant, inverse_kinematics
from manipulation.paths import DEFAULT_OUTPUT_DIR, DEFAULT_REGIONS_PATH, iiwa_urdf_path, register_package_maps
from manipulation.shelf_gcs import build_seed_points, load_regions, planning_configurations

# Scene / task constants
BLOCK_DIMS = np.array([0.06, 0.04, 0.14])
BLOCK_HALF_HEIGHT = float(BLOCK_DIMS[2] / 2.0)
BIN_FLOOR_Z = 0.015
SHELF_ORIGIN_Z = 0.4
SHELF_THICKNESS = 0.016
SHELF_COLLISION_THICKNESS = 0.04
EXTERIOR_TOP_LOCAL_Z = 0.3995

WSG_OPEN = 0.09
WSG_CLOSED = 0.02
WSG_FULLY_CLOSED = 0.0
WSG_RAMP_S = 0.8
WSG_FORCE_N = 120.0

RIGHT_BIN_X = -0.08
RIGHT_BIN_Y = -0.60
LEFT_BIN_X = 0.08
LEFT_BIN_Y = 0.60
BIN_PREGRASP_Z = 0.38
BIN_GRASP_PLAN_Z = 0.22
BIN_LIFT_Z = 0.38
FINGER_CENTER_Z_OFFSET = 0.052

BLUE_SHELF_X = 0.75
BLUE_SHELF_Y = -0.12  # align with "Above Shelve" IRIS region (not shelf geometric center)
RED_SHELF_X = 0.75
RED_SHELF_Y = 0.12  # symmetric partner on exterior top shelf
BLUE_SHELF_ABOVE_Z = 0.95
DROP_HOLD_S = 0.5
POST_OPEN_HOLD_S = 0.5  # hold after gripper fully open, before moving on

RIGHT_RPY = [np.pi / 2.0, np.pi, np.pi]
LEFT_RPY = [np.pi / 2.0, np.pi, 0.0]
TOP_RPY = [0.0, -np.pi, -np.pi / 2.0]

DESCENT_S = 1.5
LIFT_S = 2.5
GRASP_SETTLE_S = 1.5
PLACE_SETTLE_S = 2.0
CARRY_HOLD_S = 1.0
REST_LIN_VEL_TOL = 0.03
REST_ANG_VEL_TOL = 0.15
REST_Z_TOL = 0.012
REST_SETTLE_S = 0.35
DROP_LAND_TIMEOUT_S = 2.5
CLOSE_DELAY_S = 0.3
SHELF_ORIGIN_X = 0.85

ACTUATED_DIRECTIVES = REPO_ROOT / "manipulation" / "models" / "iiwa14_spheres_collision_actuated_gripper.yaml"

_eval_spec = importlib.util.spec_from_file_location(
    "eval_manipulation_circle_demo",
    REPO_ROOT / "scripts" / "eval_manipulation_circle_demo.py",
)
eval_demo = importlib.util.module_from_spec(_eval_spec)
assert _eval_spec.loader is not None
sys.modules["eval_manipulation_circle_demo"] = eval_demo
_eval_spec.loader.exec_module(eval_demo)

_viz_spec = importlib.util.spec_from_file_location(
    "visualize_manipulation_books_neural",
    REPO_ROOT / "scripts" / "visualize_manipulation_books_neural.py",
)
book_viz = importlib.util.module_from_spec(_viz_spec)
sys.modules["visualize_manipulation_books_neural"] = book_viz
_viz_spec.loader.exec_module(book_viz)


def block_center_z() -> float:
    return BIN_FLOOR_Z + BLOCK_HALF_HEIGHT


def bin_grasp_actual_z() -> float:
    return block_center_z() + FINGER_CENTER_Z_OFFSET


def shelf_top_z() -> float:
    return SHELF_ORIGIN_Z + EXTERIOR_TOP_LOCAL_Z + SHELF_COLLISION_THICKNESS / 2.0


def wsg_attach_s(close_s: float) -> float:
    return close_s + WSG_RAMP_S


def wsg_release_s(open_s: float) -> float:
    return open_s + WSG_RAMP_S


def resolve_torch_device(device_arg: str) -> torch.device:
    """Pick a torch device, falling back to CPU when CUDA init fails."""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg in ("auto", "cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError:
            return torch.device("cpu")
    device = torch.device(device_arg)
    if device.type == "cuda":
        try:
            torch.zeros(1, device=device)
        except RuntimeError:
            return torch.device("cpu")
    return device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physics: blue right bin->top shelf, red left bin->top shelf."
    )
    parser.add_argument("--planner", choices=("convex", "nonlinear"), default="nonlinear")
    parser.add_argument("--mode", choices=("neural", "vanilla"), default="neural")
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "blue_shelf_physics",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--speed", type=float, default=1.5)
    parser.add_argument("--action-wait", type=float, default=1.5)
    parser.add_argument("--time-step", type=float, default=0.002)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flow-ckpt", default=None)
    parser.add_argument("--ranknet-ckpt", default=None)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-hidden-mult", type=int, default=2)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--decoder-hidden", default="256,256")
    parser.add_argument("--decoder-dropout-p", type=float, default=0.1)
    parser.add_argument("--pointnet-hidden", type=int, default=64)
    parser.add_argument("--ranker-layers", type=int, default=3)
    parser.add_argument("--ranker-heads", type=int, default=4)
    parser.add_argument(
        "--blue-lock-after-open-s",
        type=float,
        default=1.2,
        help="Seconds after blue gripper fully opens to lock the blue block pose.",
    )
    parser.add_argument(
        "--red-lock-after-open-s",
        type=float,
        default=1.0,
        help="Seconds after red gripper fully opens to lock the red block pose.",
    )
    return parser.parse_args(argv)


def _require_ik(name: str, xyz, rpy, q0) -> np.ndarray:
    q = inverse_kinematics(q0, xyz, rpy)
    if q is None:
        raise RuntimeError(f"IK failed for {name}: xyz={xyz}, rpy={rpy}")
    return q


def _resolve_in_regions(
    name: str,
    q_ik: np.ndarray,
    regions: dict,
    *,
    fallbacks: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Prefer task IK; fall back to named seed configs inside IRIS (same as shelf_gcs)."""
    candidates = [q_ik, *(fallbacks or [])]
    for q in candidates:
        if any(region.PointInSet(q) for region in regions.values()):
            return q
    raise RuntimeError(f"Waypoint '{name}' is outside all IRIS regions.")


_SECTION_RULE = "----"


def _planning_banner(planner: str, mode: str) -> str:
    planner_label = "non-convex" if planner == "nonlinear" else "convex"
    mode_label = "Neural" if mode == "neural" else "vanilla"
    return (
        f"Planning blue and red boxes to the top of the shelf "
        f"using {planner_label} {mode_label} GCS...."
    )


def _print_section_start(message: str) -> None:
    print(_SECTION_RULE)
    print(message)


def _print_section_end(message: str) -> None:
    print(message)
    print(_SECTION_RULE)


def build_task(regions: dict) -> tuple[list[str], list[np.ndarray], dict]:
    configs = planning_configurations(regions)

    q_home = configs["Above Shelve"]
    q_pre = _require_ik(
        "right_bin_pregrasp",
        [RIGHT_BIN_X, RIGHT_BIN_Y, BIN_PREGRASP_Z],
        RIGHT_RPY,
        q_home,
    )
    q_grasp_plan = _require_ik(
        "right_bin_grasp",
        [RIGHT_BIN_X, RIGHT_BIN_Y, BIN_GRASP_PLAN_Z],
        RIGHT_RPY,
        q_pre,
    )
    q_lift = _require_ik(
        "right_bin_lift",
        [RIGHT_BIN_X, RIGHT_BIN_Y, BIN_LIFT_Z],
        RIGHT_RPY,
        q_grasp_plan,
    )
    q_shelf_above = _require_ik(
        "top_shelf_pre_place",
        [BLUE_SHELF_X, BLUE_SHELF_Y, BLUE_SHELF_ABOVE_Z],
        TOP_RPY,
        configs["Above Shelve"],
    )

    block_on_shelf_z = shelf_top_z() + BLOCK_HALF_HEIGHT
    q_grasp_actual = _require_ik(
        "right_bin_grasp_actual",
        [RIGHT_BIN_X, RIGHT_BIN_Y, bin_grasp_actual_z()],
        RIGHT_RPY,
        q_grasp_plan,
    )

    seeds = build_seed_points()
    q_left_pre = _require_ik(
        "left_bin_pregrasp",
        [LEFT_BIN_X, LEFT_BIN_Y, BIN_PREGRASP_Z],
        LEFT_RPY,
        configs["Top Rack"],
    )
    q_left_grasp = _require_ik(
        "left_bin_grasp",
        [LEFT_BIN_X, LEFT_BIN_Y, BIN_GRASP_PLAN_Z],
        LEFT_RPY,
        q_left_pre,
    )
    q_left_lift = _require_ik(
        "left_bin_lift",
        [LEFT_BIN_X, LEFT_BIN_Y, BIN_LIFT_Z],
        LEFT_RPY,
        q_left_grasp,
    )
    q_via_front = seeds["Front to Shelve"]
    q_red_shelf_above = _require_ik(
        "red_top_shelf_pre_place",
        [RED_SHELF_X, RED_SHELF_Y, BLUE_SHELF_ABOVE_Z],
        TOP_RPY,
        q_via_front,
    )
    q_home_final = configs["Top Rack"]

    names = [
        "home_ready",
        "right_bin_pregrasp",
        "right_bin_grasp",
        "top_shelf_pre_place",
        "left_bin_pregrasp",
        "left_bin_grasp",
        "via_front",
        "red_top_shelf_pre_place",
        "home_second_shelf",
    ]
    shelf_fallbacks = [configs["Above Shelve"], configs["Top Rack"]]
    sequence = [
        _resolve_in_regions("home_ready", q_home, regions),
        _resolve_in_regions("right_bin_pregrasp", q_pre, regions, fallbacks=[configs["Right Bin"]]),
        _resolve_in_regions("right_bin_grasp", q_grasp_plan, regions, fallbacks=[configs["Right Bin"]]),
        _resolve_in_regions(
            "top_shelf_pre_place", q_shelf_above, regions, fallbacks=shelf_fallbacks,
        ),
        _resolve_in_regions(
            "left_bin_pregrasp", q_left_pre, regions, fallbacks=[configs["Left Bin"], configs["Top Rack"]],
        ),
        _resolve_in_regions("left_bin_grasp", q_left_grasp, regions, fallbacks=[configs["Left Bin"]]),
        _resolve_in_regions("via_front", q_via_front, regions),
        _resolve_in_regions(
            "red_top_shelf_pre_place", q_red_shelf_above, regions, fallbacks=shelf_fallbacks,
        ),
        _resolve_in_regions(
            "home_second_shelf", q_home_final, regions, fallbacks=[configs["Top Rack"]],
        ),
    ]

    q_red_grasp_actual = _require_ik(
        "left_bin_grasp_actual",
        [LEFT_BIN_X, LEFT_BIN_Y, bin_grasp_actual_z()],
        LEFT_RPY,
        q_left_grasp,
    )

    extras = {
        "block_initial": RigidTransform([RIGHT_BIN_X, RIGHT_BIN_Y, block_center_z()]),
        "block_target": RigidTransform([BLUE_SHELF_X, BLUE_SHELF_Y, block_on_shelf_z]),
        "red_initial": RigidTransform([LEFT_BIN_X, LEFT_BIN_Y, block_center_z()]),
        "red_target": RigidTransform([RED_SHELF_X, RED_SHELF_Y, block_on_shelf_z]),
        "q_home": sequence[0],
        "q_grasp_actual": q_grasp_actual,
        "q_lift": q_lift,
        "q_shelf_above": q_shelf_above,
        "q_red_grasp_actual": q_red_grasp_actual,
        "q_red_lift": q_left_lift,
        "q_red_shelf_above": q_red_shelf_above,
        "q_home_final": q_home_final,
    }
    return names, sequence, extras


def splice_vertical_move(
    arm_traj: PiecewisePolynomial,
    start_s: float,
    q_end: np.ndarray,
    duration_s: float,
) -> tuple[PiecewisePolynomial, float]:
    times = [float(t) for t in arm_traj.get_segment_times()]
    knots = arm_traj.vector_values(times)
    insert_idx = max(i for i, t in enumerate(times) if t <= start_s + 1e-9)
    t0 = float(times[insert_idx])
    t1 = t0 + duration_s

    pre_times = times[: insert_idx + 1]
    pre_knots = knots[:, : insert_idx + 1]
    post_times = [t + duration_s for t in times[insert_idx + 1 :]]
    post_knots = knots[:, insert_idx + 1 :]

    merged_times = pre_times + [t1]
    merged_knots = np.column_stack([pre_knots, np.asarray(q_end).reshape(-1, 1)])
    if post_knots.size:
        merged_times.extend(post_times)
        merged_knots = np.column_stack([merged_knots, post_knots])
    return PiecewisePolynomial.FirstOrderHold(merged_times, merged_knots), t1


def insert_hold(
    arm_traj: PiecewisePolynomial,
    hold_start_s: float,
    duration_s: float,
) -> tuple[PiecewisePolynomial, float]:
    times = [float(t) for t in arm_traj.get_segment_times()]
    knots = arm_traj.vector_values(times)
    insert_idx = max(i for i, t in enumerate(times) if t <= hold_start_s + 1e-9)
    q_hold = knots[:, insert_idx]
    t1 = times[insert_idx] + duration_s

    pre_times = times[: insert_idx + 1]
    pre_knots = knots[:, : insert_idx + 1]
    post_times = [t + duration_s for t in times[insert_idx + 1 :]]
    post_knots = knots[:, insert_idx + 1 :]

    merged_times = pre_times + [t1]
    merged_knots = np.column_stack([pre_knots, q_hold.reshape(-1, 1)])
    if post_knots.size:
        merged_times.extend(post_times)
        merged_knots = np.column_stack([merged_knots, post_knots])
    return PiecewisePolynomial.FirstOrderHold(merged_times, merged_knots), t1


def run_pick_splice(
    arm_traj: PiecewisePolynomial,
    *,
    arrival_wait_end_s: float,
    q_grasp_actual: np.ndarray,
    q_lift: np.ndarray,
) -> tuple[PiecewisePolynomial, float, float]:
    arm_traj, grasp_end_s = splice_vertical_move(
        arm_traj, arrival_wait_end_s, q_grasp_actual, DESCENT_S,
    )
    close_s = grasp_end_s + CLOSE_DELAY_S
    arm_traj, lift_start_s = insert_hold(arm_traj, grasp_end_s, GRASP_SETTLE_S)
    arm_traj, lift_end_s = splice_vertical_move(arm_traj, lift_start_s, q_lift, LIFT_S)
    arm_traj, _ = insert_hold(arm_traj, lift_end_s, CARRY_HOLD_S)
    added = DESCENT_S + GRASP_SETTLE_S + LIFT_S + CARRY_HOLD_S
    return arm_traj, close_s, added


def drop_open_hold(
    arm_traj: PiecewisePolynomial,
    arrival_wait_end_s: float,
) -> tuple[PiecewisePolynomial, float, float]:
    """Hold at arrival, open gripper, wait for full open + extra settle, then move on."""
    arm_traj, open_s = insert_hold(arm_traj, arrival_wait_end_s, DROP_HOLD_S)
    arm_traj, _ = insert_hold(arm_traj, open_s, WSG_RAMP_S + POST_OPEN_HOLD_S)
    added = DROP_HOLD_S + WSG_RAMP_S + POST_OPEN_HOLD_S
    return arm_traj, open_s, added


def append_joint_move(
    arm_traj: PiecewisePolynomial,
    q_target: np.ndarray,
    duration_s: float,
) -> tuple[PiecewisePolynomial, float]:
    times = [float(t) for t in arm_traj.get_segment_times()]
    knots = arm_traj.vector_values(times)
    t_end = float(times[-1] + duration_s)
    return PiecewisePolynomial.FirstOrderHold(
        times + [t_end],
        np.column_stack([knots, np.asarray(q_target).reshape(-1, 1)]),
    ), t_end


def make_wsg_traj(
    t0: float,
    events: list[tuple[float, float]],
    tf: float,
) -> PiecewisePolynomial:
    times = [float(t0)]
    values = [WSG_OPEN]
    current = WSG_OPEN
    for t_cmd, target in events:
        t_cmd = float(t_cmd)
        target = float(target)
        if t_cmd > times[-1] + 1e-9:
            times.append(t_cmd)
            values.append(current)
        t_end = t_cmd + WSG_RAMP_S
        times.append(t_end)
        values.append(target)
        current = target
    if tf > times[-1] + 1e-9:
        times.append(float(tf))
        values.append(current)
    return PiecewisePolynomial.FirstOrderHold(times, np.array([values]))


def add_block(plant, *, name: str, color: np.ndarray):
    mass = 0.12
    body = plant.AddRigidBody(
        name,
        plant.AddModelInstance(f"{name}_model"),
        SpatialInertia(mass, [0, 0, 0], UnitInertia.SolidBox(*BLOCK_DIMS)),
    )
    X_BG = RigidTransform()
    plant.RegisterVisualGeometry(body, X_BG, DrakeBox(*BLOCK_DIMS), f"{name}_visual", color)
    plant.RegisterCollisionGeometry(
        body, X_BG, DrakeBox(*BLOCK_DIMS), f"{name}_collision", CoulombFriction(1.2, 0.9),
    )
    return body


def weld_exterior_top_shelf(plant) -> None:
    """Welded collision plank on the exterior top shelf (thick enough to catch drops)."""
    plank_z = SHELF_ORIGIN_Z + EXTERIOR_TOP_LOCAL_Z
    model = plant.AddModelInstance("exterior_top_plank")
    body = plant.AddRigidBody(
        "exterior_top_plank",
        model,
        SpatialInertia(
            5.0,
            [0, 0, 0],
            UnitInertia.SolidBox(0.32, 0.62, SHELF_COLLISION_THICKNESS),
        ),
    )
    plant.RegisterCollisionGeometry(
        body,
        RigidTransform(),
        DrakeBox(0.32, 0.62, SHELF_COLLISION_THICKNESS),
        "exterior_top_plank_collision",
        CoulombFriction(1.2, 0.9),
    )
    X_WB = RigidTransform([SHELF_ORIGIN_X, 0.0, plank_z])
    plant.WeldFrames(plant.world_frame(), body.body_frame(), X_WB)


def make_controller_plant():
    from pydrake.multibody.plant import MultibodyPlant

    p = MultibodyPlant(0.0)
    parser = Parser(p)
    register_package_maps(parser)
    parser.AddModels(str(iiwa_urdf_path()))
    p.WeldFrames(p.world_frame(), p.GetFrameByName("base"))
    p.Finalize()
    return p


def load_neural_models(args, regions, sequence, plant, device):
    ckpt_dir = f"checkpoints/manipulation_{args.planner}"
    flow_ckpt = args.flow_ckpt or f"{ckpt_dir}/manipulation_{args.planner}_flow_gnn.ckpt"
    ranknet_ckpt = args.ranknet_ckpt or f"{ckpt_dir}/manipulation_{args.planner}_ranknet.ckpt"
    encoder_hp = book_viz.EncoderHParams(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_hidden_mult=args.ffn_hidden_mult,
        dropout_p=args.dropout_p,
    )
    decoder_hp = book_viz.DecoderHParams(
        hidden_dims=tuple(int(s) for s in args.decoder_hidden.split(",") if s.strip()),
        dropout_p=args.decoder_dropout_p,
    )
    facet_dim = book_viz.facet_dim_for_first_segment(args, regions, sequence, plant, device)
    flow_model = eval_demo.load_flow_model(
        flow_ckpt,
        facet_dim=facet_dim,
        g_dim=14,
        encoder_hp=encoder_hp,
        decoder_hp=decoder_hp,
        pointnet_hidden=args.pointnet_hidden,
        device=device,
    )
    ranker = eval_demo.load_ranknet(
        ranknet_ckpt,
        cfg=book_viz.RankNetConfig(
            d_model=args.d_model,
            num_layers=args.ranker_layers,
            num_heads=args.ranker_heads,
        ),
        device=device,
    )
    return flow_model, ranker


def _lock_block(plant, plant_context, block, pose: RigidTransform) -> None:
    plant.SetFreeBodyPose(plant_context, block, pose)
    plant.SetFreeBodySpatialVelocity(
        plant_context,
        block,
        SpatialVelocity(np.zeros(3), np.zeros(3)),
    )


def _attach_block_to_gripper(
    plant,
    plant_context,
    block,
    gripper_body,
    X_gripper_block: RigidTransform,
) -> None:
    X_WG = plant.EvalBodyPoseInWorld(plant_context, gripper_body)
    X_WB = X_WG @ X_gripper_block
    plant.SetFreeBodyPose(plant_context, block, X_WB)
    v_g = plant.EvalBodySpatialVelocityInWorld(plant_context, gripper_body)
    plant.SetFreeBodySpatialVelocity(plant_context, block, v_g)


def advance_physics_with_blocks(
    simulator: Simulator,
    plant,
    wsg,
    block_phases: list[dict],
    *,
    tf: float,
    time_step: float,
) -> None:
    """Simulate with gripper-attached carries and post-drop shelf freezes."""
    gripper_body = plant.GetBodyByName("body", wsg)
    context = simulator.get_mutable_context()
    simulator.Initialize()
    t = 0.0
    while t < tf - 1e-12:
        plant_context = plant.GetMyMutableContextFromRoot(context)
        for phase in block_phases:
            block = phase["block"]
            if phase["frozen"]:
                _lock_block(plant, plant_context, block, phase["X_frozen"])
                continue
            if phase["attach_s"] <= t < phase["release_s"]:
                if phase["X_gripper_block"] is None:
                    X_WG = plant.EvalBodyPoseInWorld(plant_context, gripper_body)
                    X_WB = plant.EvalBodyPoseInWorld(plant_context, block)
                    phase["X_gripper_block"] = X_WG.InvertAndCompose(X_WB)
                _attach_block_to_gripper(
                    plant, plant_context, block, gripper_body, phase["X_gripper_block"],
                )

        t_next = min(t + time_step, tf)
        simulator.AdvanceTo(t_next)
        plant_context = plant.GetMyMutableContextFromRoot(context)

        for phase in block_phases:
            block = phase["block"]
            if phase["frozen"]:
                _lock_block(plant, plant_context, block, phase["X_frozen"])
                continue
            if t_next < phase["release_s"]:
                continue

            lock_after_open_s = phase.get("lock_after_open_s")
            if lock_after_open_s is not None:
                if t_next >= phase["release_s"] + lock_after_open_s:
                    phase["X_frozen"] = plant.EvalBodyPoseInWorld(plant_context, block)
                    phase["frozen"] = True
                if phase["frozen"]:
                    _lock_block(plant, plant_context, block, phase["X_frozen"])
                continue

            p = plant.EvalBodyPoseInWorld(plant_context, block).translation()
            v = plant.EvalBodySpatialVelocityInWorld(plant_context, block)
            block_bottom = p[2] - BLOCK_HALF_HEIGHT
            shelf_z = phase["shelf_surface_z"]
            on_shelf = abs(block_bottom - shelf_z) <= REST_Z_TOL
            lin_speed = float(np.linalg.norm(v.translational()))
            ang_speed = float(np.linalg.norm(v.rotational()))
            at_rest = lin_speed < REST_LIN_VEL_TOL and ang_speed < REST_ANG_VEL_TOL
            landed = on_shelf and at_rest
            timed_out = t_next >= phase["release_s"] + DROP_LAND_TIMEOUT_S

            if landed:
                if phase["resting_since"] is None:
                    phase["resting_since"] = t_next
                elif t_next - phase["resting_since"] >= REST_SETTLE_S:
                    phase["X_frozen"] = plant.EvalBodyPoseInWorld(plant_context, block)
                    phase["frozen"] = True
            elif timed_out and block_bottom >= shelf_z - REST_Z_TOL:
                phase["X_frozen"] = plant.EvalBodyPoseInWorld(plant_context, block)
                phase["frozen"] = True
            else:
                phase["resting_since"] = None

            if phase["frozen"]:
                _lock_block(plant, plant_context, block, phase["X_frozen"])

        t = t_next


def build_physics_trajectories(args):
    """Plan GCS path and splice pick/place phases. Returns trajectories + phase times."""
    from manipulation.paths import manipulation_models_hint, manipulation_models_ready

    if not manipulation_models_ready():
        raise RuntimeError(manipulation_models_hint())
    if not args.regions.exists():
        raise FileNotFoundError(f"Missing IRIS regions: {args.regions}")

    device = resolve_torch_device(args.device)
    if args.device != "cpu" and device.type == "cpu":
        print(f"Warning: CUDA unavailable ({args.device!r} requested); using CPU.")
    warnings.filterwarnings(
        "ignore",
        message="enable_nested_tensor is True",
        category=UserWarning,
        module="torch.nn.modules.transformer",
    )

    planning_plant, _, _, _ = build_shelf_plant()
    with contextlib.redirect_stdout(io.StringIO()):
        regions = load_regions(args.regions)
        waypoint_names, sequence, extras = build_task(regions)
        flow_model = ranker = None
        if args.mode == "neural":
            flow_model, ranker = load_neural_models(
                args, regions, sequence, planning_plant, device,
            )
        plan = eval_demo.plan_circle(
            planner=args.planner,
            mode=args.mode,
            regions=regions,
            sequence=sequence,
            plant=planning_plant,
            seed=args.seed,
            speed=args.speed,
            flow_model=flow_model,
            ranker=ranker,
            device=device,
        )
    if not plan.success:
        raise RuntimeError(f"{args.mode.capitalize()} GCS planning failed.")

    arm_traj, timings = book_viz.combine_segments_with_waits(plan.segments, wait=args.action_wait)

    def timing(name: str):
        return timings[waypoint_names.index(name) - 1]

    grasp_arrival_s = timing("right_bin_grasp").wait_end_s
    arm_traj, close_s, pick_shift = run_pick_splice(
        arm_traj,
        arrival_wait_end_s=grasp_arrival_s,
        q_grasp_actual=extras["q_grasp_actual"],
        q_lift=extras["q_lift"],
    )
    blue_lift_start_s = grasp_arrival_s + DESCENT_S + GRASP_SETTLE_S
    blue_lift_end_s = blue_lift_start_s + LIFT_S
    blue_shelf_arrival_s = timing("top_shelf_pre_place").wait_end_s + pick_shift

    arm_traj, open_s, drop_shift = drop_open_hold(
        arm_traj,
        blue_shelf_arrival_s,
    )
    time_shift = pick_shift + drop_shift

    arm_traj, red_close_s, red_pick_shift = run_pick_splice(
        arm_traj,
        arrival_wait_end_s=timing("left_bin_grasp").wait_end_s + time_shift,
        q_grasp_actual=extras["q_red_grasp_actual"],
        q_lift=extras["q_red_lift"],
    )
    time_shift += red_pick_shift

    arm_traj, red_open_s, red_drop_shift = drop_open_hold(
        arm_traj,
        timing("red_top_shelf_pre_place").wait_end_s + time_shift,
    )
    time_shift += red_drop_shift

    times = list(arm_traj.get_segment_times())
    knots = arm_traj.vector_values(times)
    t_hold_end = float(times[-1] + args.action_wait)
    arm_traj = PiecewisePolynomial.FirstOrderHold(
        times + [t_hold_end],
        np.column_stack([knots, knots[:, -1]]),
    )

    wsg_traj = make_wsg_traj(
        arm_traj.start_time(),
        [
            (close_s, WSG_CLOSED),
            (open_s, WSG_OPEN),
            (red_close_s, WSG_CLOSED),
            (red_open_s, WSG_OPEN),
        ],
        arm_traj.end_time(),
    )

    phase_times = {
        "blue_lift_start_s": blue_lift_start_s,
        "blue_lift_end_s": blue_lift_end_s,
        "blue_shelf_arrival_s": blue_shelf_arrival_s,
        "blue_close_s": close_s,
        "blue_open_s": open_s,
        "red_close_s": red_close_s,
        "red_open_s": red_open_s,
    }
    return arm_traj, wsg_traj, extras, phase_times


def main() -> None:
    args = parse_args()

    _print_section_start(_planning_banner(args.planner, args.mode))
    arm_traj, wsg_traj, extras, phase_times = build_physics_trajectories(args)
    close_s = phase_times["blue_close_s"]
    open_s = phase_times["blue_open_s"]
    red_close_s = phase_times["red_close_s"]
    red_open_s = phase_times["red_open_s"]
    blue_release_s = wsg_release_s(open_s)
    red_release_s = wsg_release_s(red_open_s)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=args.time_step)
    parser = Parser(plant, scene_graph)
    register_package_maps(parser)
    book_viz.ProcessModelDirectives(
        book_viz.LoadModelDirectives(str(ACTUATED_DIRECTIVES)), plant, parser,
    )
    weld_exterior_top_shelf(plant)
    blue_block = add_block(plant, name="blue_block", color=np.array([0.05, 0.20, 1.00, 1.0]))
    red_block = add_block(plant, name="red_block", color=np.array([0.95, 0.15, 0.10, 1.0]))
    plant.Finalize()

    iiwa = plant.GetModelInstanceByName("iiwa")
    wsg = plant.GetModelInstanceByName("wsg")

    iiwa_controller = builder.AddSystem(
        InverseDynamicsController(
            make_controller_plant(),
            kp=[350] * 7,
            ki=[0] * 7,
            kd=[60] * 7,
            has_reference_acceleration=False,
        )
    )
    builder.Connect(plant.get_state_output_port(iiwa), iiwa_controller.get_input_port_estimated_state())
    desired_state = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(7, args.time_step, suppress_initial_transient=True)
    )
    builder.Connect(
        builder.AddSystem(TrajectorySource(arm_traj)).get_output_port(),
        desired_state.get_input_port(),
    )
    builder.Connect(desired_state.get_output_port(), iiwa_controller.get_input_port_desired_state())
    builder.Connect(iiwa_controller.get_output_port_control(), plant.get_actuation_input_port(iiwa))

    wsg_controller = builder.AddSystem(SchunkWsgPositionController())
    builder.Connect(wsg_controller.get_generalized_force_output_port(), plant.get_actuation_input_port(wsg))
    builder.Connect(plant.get_state_output_port(wsg), wsg_controller.get_state_input_port())
    builder.Connect(
        builder.AddSystem(TrajectorySource(wsg_traj)).get_output_port(),
        wsg_controller.get_desired_position_input_port(),
    )
    builder.Connect(
        builder.AddSystem(ConstantVectorSource([WSG_FORCE_N])).get_output_port(),
        wsg_controller.get_force_limit_input_port(),
    )

    meshcat = StartMeshcat()
    meshcat.SetProperty("/Grid", "visible", False)
    meshcat.SetProperty("/Axes", "visible", False)
    params = MeshcatVisualizerParams()
    params.delete_on_initialization_event = False
    params.role = Role.kIllustration
    meshcat_viz = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    plant.SetPositions(plant_context, iiwa, np.squeeze(arm_traj.value(arm_traj.start_time())))
    plant.SetPositions(plant_context, wsg, np.array([-WSG_OPEN / 2.0, WSG_OPEN / 2.0]))
    plant.SetFreeBodyPose(plant_context, blue_block, extras["block_initial"])
    plant.SetFreeBodyPose(plant_context, red_block, extras["red_initial"])

    meshcat_viz.StartRecording()
    simulator.set_target_realtime_rate(0.0)
    shelf_z = shelf_top_z()
    advance_physics_with_blocks(
        simulator,
        plant,
        wsg,
        [
            {
                "block": blue_block,
                "attach_s": wsg_attach_s(close_s),
                "release_s": blue_release_s,
                "lock_after_open_s": args.blue_lock_after_open_s,
                "shelf_surface_z": shelf_z,
                "X_gripper_block": None,
                "frozen": False,
                "X_frozen": None,
                "resting_since": None,
            },
            {
                "block": red_block,
                "attach_s": wsg_attach_s(red_close_s),
                "release_s": red_release_s,
                "lock_after_open_s": args.red_lock_after_open_s,
                "shelf_surface_z": shelf_z,
                "X_gripper_block": None,
                "frozen": False,
                "X_frozen": None,
                "resting_since": None,
            },
        ],
        tf=arm_traj.end_time(),
        time_step=args.time_step,
    )
    recording = meshcat.get_mutable_recording()
    recording.set_autoplay(True)
    recording.set_repetitions(1)
    recording.set_clamp_when_finished(True)
    meshcat_viz.PublishRecording()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.output_dir / "pick_and_place.html"
    html_path.write_text(eval_demo.patch_meshcat_html_play_once(meshcat.StaticHtml()))
    _print_section_end(f"Open {html_path} in your browser to visualize.")


if __name__ == "__main__":
    main()
