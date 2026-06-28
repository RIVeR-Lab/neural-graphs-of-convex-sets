#!/usr/bin/env python3
"""Physics bin task: blue from right bin to top shelf, red from left bin to right bin.

Each pick/place uses explicit execution phases on top of neural GCS:
  pre-grasp -> [descend, hold, close, hold, lift, carry hold] -> carry ->
  pre-place -> [descend, hold, open, settle, lift]

Neural GCS plans coarse arm waypoints through IRIS regions; Drake contact sim runs
the WSG and free-floating block bodies.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
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

_viz_spec = importlib.util.spec_from_file_location(
    "visualize_manipulation_books_neural",
    REPO_ROOT / "scripts" / "visualize_manipulation_books_neural.py",
)
book_viz = importlib.util.module_from_spec(_viz_spec)
assert _viz_spec.loader is not None
sys.modules["visualize_manipulation_books_neural"] = book_viz
_viz_spec.loader.exec_module(book_viz)

# Tall narrow block: thin in y so side grasps at the bins can pinch it.
BLOCK_DIMS = np.array([0.06, 0.04, 0.14])
BLOCK_HALF_HEIGHT = float(BLOCK_DIMS[2] / 2.0)
WSG_OPEN = 0.09
WSG_CLOSED = 0.02
BIN_FLOOR_Z = book_viz.BIN_FLOOR_Z
# GCS bin regions only allow high grasps; descend to the block before closing.
BIN_GRASP_PLAN_Z = 0.22
BIN_PREGRASP_Z = 0.38
BIN_LIFT_Z = 0.38
FINGER_CENTER_Z_OFFSET = 0.052
DESCENT_MOVE_S = 1.5
LIFT_MOVE_S = 2.5
CARRY_HOLD_S = 1.0
PLACE_SETTLE_S = 2.5

RIGHT_BIN_PICK_X = -0.08
RIGHT_BIN_PICK_Y = -0.60
LEFT_BIN_PICK_X = 0.08
LEFT_BIN_PICK_Y = 0.60

EXTERIOR_TOP_LOCAL_Z = 0.3995
BLUE_SHELF_X = 0.75
BLUE_SHELF_Y = -0.12
BLUE_SHELF_ABOVE_Z = 0.95
BLUE_SHELF_DESCENT_S = DESCENT_MOVE_S
RED_BIN_DESCENT_S = DESCENT_MOVE_S
TOP_RPY = [0.0, -np.pi, -np.pi / 2.0]
RIGHT_RPY = [np.pi / 2.0, np.pi, np.pi]
LEFT_RPY = [np.pi / 2.0, np.pi, 0.0]


def block_center_z() -> float:
    return BIN_FLOOR_Z + BLOCK_HALF_HEIGHT


def bin_grasp_actual_z() -> float:
    return block_center_z() + FINGER_CENTER_Z_OFFSET


def shelf_collision_top_z(local_center_z: float) -> float:
    return book_viz.SHELF_ORIGIN_Z + local_center_z + book_viz.SHELF_THICKNESS / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physics demo: blue right-bin to top shelf, red left-bin to right bin."
    )
    parser.add_argument("--planner", choices=("convex", "nonlinear"), default="convex")
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "book_stacking_physics")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--speed", type=float, default=2.0)
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
    return parser.parse_args()


def make_wsg_command_traj(
    t0: float,
    timed_commands: list[tuple[float, float]],
    tf: float,
    *,
    ramp_s: float = 0.8,
) -> PiecewisePolynomial:
    """Hold gripper width constant, then ramp only during close/open events."""
    times = [float(t0)]
    values = [WSG_OPEN]
    current = WSG_OPEN

    for t_command, target in timed_commands:
        t_command = float(t_command)
        target = float(target)
        if t_command > times[-1] + 1e-9:
            times.append(t_command)
            values.append(current)
        t_ramp_end = t_command + ramp_s
        times.append(t_ramp_end)
        values.append(target)
        current = target

    if tf > times[-1] + 1e-9:
        times.append(float(tf))
        values.append(current)
    return PiecewisePolynomial.FirstOrderHold(times, np.array([values]))


def add_block_body(plant, *, name: str, color: np.ndarray):
    mass = 0.12
    inertia = UnitInertia.SolidBox(*BLOCK_DIMS)
    model = plant.AddModelInstance(f"{name}_model")
    body = plant.AddRigidBody(name, model, SpatialInertia(mass, [0, 0, 0], inertia))
    X_BG = RigidTransform()
    plant.RegisterVisualGeometry(
        body, X_BG, DrakeBox(*BLOCK_DIMS), f"{name}_visual", color,
    )
    plant.RegisterCollisionGeometry(
        body, X_BG, DrakeBox(*BLOCK_DIMS), f"{name}_collision", CoulombFriction(1.2, 0.9),
    )
    return body


def make_controller_plant() -> object:
    controller_plant = book_viz.MultibodyPlant(time_step=0.0)
    parser = Parser(controller_plant)
    register_package_maps(parser)
    parser.AddModels(str(iiwa_urdf_path()))
    controller_plant.WeldFrames(
        controller_plant.world_frame(),
        controller_plant.GetFrameByName("base"),
    )
    controller_plant.Finalize()
    return controller_plant


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
    flow_model = book_viz.eval_demo.load_flow_model(
        flow_ckpt,
        facet_dim=facet_dim,
        g_dim=14,
        encoder_hp=encoder_hp,
        decoder_hp=decoder_hp,
        pointnet_hidden=args.pointnet_hidden,
        device=device,
    )
    ranker = book_viz.eval_demo.load_ranknet(
        ranknet_ckpt,
        cfg=book_viz.RankNetConfig(
            d_model=args.d_model,
            num_layers=args.ranker_layers,
            num_heads=args.ranker_heads,
        ),
        device=device,
    )
    return flow_model, ranker


def _require_ik(name: str, xyz, rpy, q0) -> np.ndarray:
    q = inverse_kinematics(q0, xyz, rpy)
    if q is None:
        raise RuntimeError(f"IK failed for {name}: xyz={xyz}, rpy={rpy}")
    return q


def _require_in_regions(name: str, q: np.ndarray, regions: dict) -> np.ndarray:
    containing = [region_name for region_name, region in regions.items() if region.PointInSet(q)]
    if not containing:
        raise RuntimeError(f"Waypoint '{name}' is outside all IRIS regions.")
    print(f"  {name}: {containing}")
    return q


def build_bin_task_sequence(
    regions: dict,
) -> tuple[list[str], list[np.ndarray], dict[str, RigidTransform | np.ndarray]]:
    configs = planning_configurations(regions)
    seeds = build_seed_points()

    q_ready = configs["Top Rack"]
    q_right_pre = _require_ik(
        "right_bin_pregrasp",
        [RIGHT_BIN_PICK_X, RIGHT_BIN_PICK_Y, BIN_PREGRASP_Z],
        RIGHT_RPY,
        q_ready,
    )
    q_right_grasp = _require_ik(
        "right_bin_grasp",
        [RIGHT_BIN_PICK_X, RIGHT_BIN_PICK_Y, BIN_GRASP_PLAN_Z],
        RIGHT_RPY,
        q_right_pre,
    )
    q_right_lift = _require_ik(
        "right_bin_lift",
        [RIGHT_BIN_PICK_X, RIGHT_BIN_PICK_Y, BIN_LIFT_Z],
        RIGHT_RPY,
        q_right_grasp,
    )
    q_blue_above = _require_ik(
        "blue_top_shelf_above",
        [BLUE_SHELF_X, BLUE_SHELF_Y, BLUE_SHELF_ABOVE_Z],
        TOP_RPY,
        q_right_lift,
    )
    q_left_pre = _require_ik(
        "left_bin_pregrasp",
        [LEFT_BIN_PICK_X, LEFT_BIN_PICK_Y, BIN_PREGRASP_Z],
        LEFT_RPY,
        configs["Top Rack"],
    )
    q_left_grasp = _require_ik(
        "left_bin_grasp",
        [LEFT_BIN_PICK_X, LEFT_BIN_PICK_Y, BIN_GRASP_PLAN_Z],
        LEFT_RPY,
        q_left_pre,
    )
    q_left_lift = _require_ik(
        "left_bin_lift",
        [LEFT_BIN_PICK_X, LEFT_BIN_PICK_Y, BIN_LIFT_Z],
        LEFT_RPY,
        q_left_grasp,
    )
    q_via_front = seeds["Front to Shelve"]
    q_via_upper = configs["Top Rack"]
    q_red_bin_pre = _require_ik(
        "red_right_bin_pregrasp",
        [RIGHT_BIN_PICK_X, RIGHT_BIN_PICK_Y, BIN_PREGRASP_Z],
        RIGHT_RPY,
        q_via_upper,
    )
    q_far = seeds["Front to Shelve"]

    names = [
        "top_shelf_ready",
        "right_bin_pregrasp",
        "right_bin_grasp",
        "blue_top_shelf_above",
        "left_bin_pregrasp",
        "left_bin_grasp",
        "via_front",
        "via_upper_rack",
        "red_right_bin_pregrasp",
        "far_retreat",
    ]
    sequence = [
        _require_in_regions("top_shelf_ready", q_ready, regions),
        _require_in_regions("right_bin_pregrasp", q_right_pre, regions),
        _require_in_regions("right_bin_grasp", q_right_grasp, regions),
        _require_in_regions("blue_top_shelf_above", q_blue_above, regions),
        _require_in_regions("left_bin_pregrasp", q_left_pre, regions),
        _require_in_regions("left_bin_grasp", q_left_grasp, regions),
        _require_in_regions("via_front", q_via_front, regions),
        _require_in_regions("via_upper_rack", q_via_upper, regions),
        _require_in_regions("red_right_bin_pregrasp", q_red_bin_pre, regions),
        _require_in_regions("far_retreat", q_far, regions),
    ]

    blue_top = shelf_collision_top_z(EXTERIOR_TOP_LOCAL_Z)
    blue_xyz = [BLUE_SHELF_X, BLUE_SHELF_Y, blue_top + BLOCK_HALF_HEIGHT]
    red_xyz = [RIGHT_BIN_PICK_X, RIGHT_BIN_PICK_Y, block_center_z()]

    task = {
        "blue_initial": RigidTransform([RIGHT_BIN_PICK_X, RIGHT_BIN_PICK_Y, block_center_z()]),
        "red_initial": RigidTransform([LEFT_BIN_PICK_X, LEFT_BIN_PICK_Y, block_center_z()]),
        "blue_final": RigidTransform(blue_xyz),
        "red_final": RigidTransform(red_xyz),
        "q_right_grasp_actual": _require_ik(
            "right_bin_grasp_actual",
            [RIGHT_BIN_PICK_X, RIGHT_BIN_PICK_Y, bin_grasp_actual_z()],
            RIGHT_RPY,
            q_right_grasp,
        ),
        "q_right_lift": q_right_lift,
        "q_left_grasp_actual": _require_ik(
            "left_bin_grasp_actual",
            [LEFT_BIN_PICK_X, LEFT_BIN_PICK_Y, bin_grasp_actual_z()],
            LEFT_RPY,
            q_left_grasp,
        ),
        "q_left_lift": q_left_lift,
        "q_blue_drop_actual": _require_ik(
            "blue_top_shelf_drop_actual",
            [BLUE_SHELF_X, BLUE_SHELF_Y, blue_xyz[2] + FINGER_CENTER_Z_OFFSET - 0.015],
            TOP_RPY,
            q_blue_above,
        ),
        "q_blue_above": q_blue_above,
        "q_red_drop_actual": _require_ik(
            "red_right_bin_drop_actual",
            [RIGHT_BIN_PICK_X, RIGHT_BIN_PICK_Y, bin_grasp_actual_z()],
            RIGHT_RPY,
            q_red_bin_pre,
        ),
        "q_red_bin_pre": q_red_bin_pre,
    }
    return names, sequence, task


def append_final_hold(traj: PiecewisePolynomial, hold_s: float) -> PiecewisePolynomial:
    times = list(traj.get_segment_times())
    knots = traj.vector_values(times)
    times.append(float(times[-1] + hold_s))
    knots = np.column_stack([knots, knots[:, -1]])
    return PiecewisePolynomial.FirstOrderHold(times, knots)


def splice_descent(
    arm_traj: PiecewisePolynomial,
    descent_start_s: float,
    q_end: np.ndarray,
    duration_s: float,
) -> tuple[PiecewisePolynomial, float]:
    times = [float(t) for t in arm_traj.get_segment_times()]
    knots = arm_traj.vector_values(times)
    insert_idx = max(i for i, t in enumerate(times) if t <= descent_start_s + 1e-9)
    q_start = knots[:, insert_idx]
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


def splice_vertical_move(
    arm_traj: PiecewisePolynomial,
    move_start_s: float,
    q_end: np.ndarray,
    duration_s: float,
) -> tuple[PiecewisePolynomial, float]:
    return splice_descent(arm_traj, move_start_s, q_end, duration_s)


def run_pick_splice(
    arm_traj: PiecewisePolynomial,
    *,
    grasp_timing,
    time_shift: float,
    q_grasp_actual: np.ndarray,
    q_lift: np.ndarray,
    action_wait: float,
) -> tuple[PiecewisePolynomial, float, float, float]:
    """Descend to grasp depth, close immediately, settle, lift to carry height."""
    arm_traj, grasp_end_s = splice_vertical_move(
        arm_traj,
        grasp_timing.wait_end_s + time_shift,
        q_grasp_actual,
        DESCENT_MOVE_S,
    )
    close_s = grasp_end_s + 0.3
    arm_traj, lift_start_s = insert_hold(arm_traj, grasp_end_s, action_wait)
    arm_traj, lift_end_s = splice_vertical_move(
        arm_traj,
        lift_start_s,
        q_lift,
        LIFT_MOVE_S,
    )
    arm_traj, _carry_end_s = insert_hold(arm_traj, lift_end_s, CARRY_HOLD_S)
    added = DESCENT_MOVE_S + action_wait + LIFT_MOVE_S + CARRY_HOLD_S
    return arm_traj, close_s, lift_start_s, added


def run_place_splice(
    arm_traj: PiecewisePolynomial,
    *,
    place_timing,
    time_shift: float,
    q_place_actual: np.ndarray,
    q_retreat: np.ndarray,
    action_wait: float,
) -> tuple[PiecewisePolynomial, float, float, float]:
    """Descend to place depth, settle, open, settle on shelf, lift back up."""
    arm_traj, place_end_s = splice_vertical_move(
        arm_traj,
        place_timing.wait_end_s + time_shift,
        q_place_actual,
        DESCENT_MOVE_S,
    )
    arm_traj, release_s = insert_hold(arm_traj, place_end_s, action_wait)
    open_s = release_s
    arm_traj, retreat_start_s = insert_hold(arm_traj, release_s, PLACE_SETTLE_S)
    arm_traj, _retreat_s = splice_vertical_move(
        arm_traj,
        retreat_start_s,
        q_retreat,
        LIFT_MOVE_S,
    )
    added = DESCENT_MOVE_S + action_wait + PLACE_SETTLE_S + LIFT_MOVE_S
    return arm_traj, open_s, retreat_start_s, added


def main() -> None:
    args = parse_args()

    from manipulation.paths import manipulation_models_hint, manipulation_models_ready

    if not manipulation_models_ready():
        print(manipulation_models_hint(), file=sys.stderr)
        sys.exit(1)
    if not args.regions.exists():
        print(f"Missing IRIS regions: {args.regions}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Planning bin task with neural GCS...")
    planning_plant, _, _, _ = build_shelf_plant()
    regions = load_regions(args.regions)
    waypoint_names, sequence, task = build_bin_task_sequence(regions)
    print("Bin task (blue->top shelf, red->right bin) with explicit pick/place splices:")
    for i, name in enumerate(waypoint_names, start=1):
        print(f"  {i}. {name}")

    flow_model, ranker = load_neural_models(args, regions, sequence, planning_plant, device)
    plan = book_viz.eval_demo.plan_circle(
        planner=args.planner,
        mode="neural",
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
        print("Neural GCS plan failed.", file=sys.stderr)
        sys.exit(1)

    arm_traj, timings = book_viz.combine_segments_with_waits(plan.segments, wait=args.action_wait)

    def arrival_timing(waypoint_name: str):
        return timings[waypoint_names.index(waypoint_name) - 1]

    time_shift = 0.0

    arm_traj, right_close_s, _, pick_shift = run_pick_splice(
        arm_traj,
        grasp_timing=arrival_timing("right_bin_grasp"),
        time_shift=time_shift,
        q_grasp_actual=task["q_right_grasp_actual"],
        q_lift=task["q_right_lift"],
        action_wait=args.action_wait,
    )
    time_shift += pick_shift

    arm_traj, blue_open_s, _, place_shift = run_place_splice(
        arm_traj,
        place_timing=arrival_timing("blue_top_shelf_above"),
        time_shift=time_shift,
        q_place_actual=task["q_blue_drop_actual"],
        q_retreat=task["q_blue_above"],
        action_wait=args.action_wait,
    )
    time_shift += place_shift

    arm_traj, left_close_s, _, pick_shift = run_pick_splice(
        arm_traj,
        grasp_timing=arrival_timing("left_bin_grasp"),
        time_shift=time_shift,
        q_grasp_actual=task["q_left_grasp_actual"],
        q_lift=task["q_left_lift"],
        action_wait=args.action_wait,
    )
    time_shift += pick_shift

    arm_traj, red_open_s, _, place_shift = run_place_splice(
        arm_traj,
        place_timing=arrival_timing("red_right_bin_pregrasp"),
        time_shift=time_shift,
        q_place_actual=task["q_red_drop_actual"],
        q_retreat=task["q_red_bin_pre"],
        action_wait=args.action_wait,
    )
    time_shift += place_shift
    arm_traj = append_final_hold(arm_traj, args.action_wait)

    wsg_traj = make_wsg_command_traj(
        arm_traj.start_time(),
        [
            (right_close_s, WSG_CLOSED),
            (blue_open_s, WSG_OPEN),
            (left_close_s, WSG_CLOSED),
            (red_open_s, WSG_OPEN),
        ],
        arm_traj.end_time(),
    )

    print("Building Drake contact simulation...")
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=args.time_step)
    parser = Parser(plant, scene_graph)
    register_package_maps(parser)
    ProcessModelDirectives = book_viz.ProcessModelDirectives
    ProcessModelDirectives(book_viz.LoadModelDirectives(str(book_viz.ACTUATED_DIRECTIVES)), plant, parser)
    blue_body = add_block_body(
        plant,
        name="blue_block",
        color=np.array([0.05, 0.20, 1.00, 1.0]),
    )
    red_body = add_block_body(
        plant,
        name="red_block",
        color=np.array([0.95, 0.15, 0.10, 1.0]),
    )
    plant.Finalize()

    iiwa = plant.GetModelInstanceByName("iiwa")
    wsg = plant.GetModelInstanceByName("wsg")

    controller_plant = make_controller_plant()
    iiwa_controller = builder.AddSystem(
        InverseDynamicsController(
            controller_plant,
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
    arm_source = builder.AddSystem(TrajectorySource(arm_traj))
    builder.Connect(arm_source.get_output_port(), desired_state.get_input_port())
    builder.Connect(desired_state.get_output_port(), iiwa_controller.get_input_port_desired_state())
    builder.Connect(iiwa_controller.get_output_port_control(), plant.get_actuation_input_port(iiwa))

    wsg_controller = builder.AddSystem(SchunkWsgPositionController())
    builder.Connect(wsg_controller.get_generalized_force_output_port(), plant.get_actuation_input_port(wsg))
    builder.Connect(plant.get_state_output_port(wsg), wsg_controller.get_state_input_port())
    wsg_source = builder.AddSystem(TrajectorySource(wsg_traj))
    builder.Connect(wsg_source.get_output_port(), wsg_controller.get_desired_position_input_port())
    force_limit = builder.AddSystem(ConstantVectorSource([120.0]))
    builder.Connect(force_limit.get_output_port(), wsg_controller.get_force_limit_input_port())

    meshcat = StartMeshcat()
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
    plant.SetFreeBodyPose(plant_context, blue_body, task["blue_initial"])
    plant.SetFreeBodyPose(plant_context, red_body, task["red_initial"])

    meshcat_viz.StartRecording()
    simulator.set_target_realtime_rate(0.0)
    simulator.Initialize()
    simulator.AdvanceTo(arm_traj.end_time())
    recording = meshcat.get_mutable_recording()
    recording.set_autoplay(True)
    recording.set_repetitions(1)
    recording.set_clamp_when_finished(True)
    meshcat_viz.PublishRecording()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.output_dir / f"book_stacking_physics_{args.planner}_seed{args.seed}.html"
    html_path.write_text(book_viz.eval_demo.patch_meshcat_html_play_once(meshcat.StaticHtml()))
    X_blue = plant.EvalBodyPoseInWorld(plant_context, blue_body)
    X_red = plant.EvalBodyPoseInWorld(plant_context, red_body)
    print(f"Final blue block pose: {np.round(X_blue.translation(), 3)}")
    print(f"Final red block pose:  {np.round(X_red.translation(), 3)}")
    print(f"Saved HTML -> {html_path}")


if __name__ == "__main__":
    main()
