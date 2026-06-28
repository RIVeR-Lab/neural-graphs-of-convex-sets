#!/usr/bin/env python3
"""Visualize a Neural GCS two-block bin task (kinematic attach/release).

Neural GCS plans 7-DOF arm motions through shelf IRIS regions. Blocks are
kinematically attached to the gripper at bin grasp waits and released at place
waits — no contact simulation. Suitable for stable paper/demo animations.

Task: blue block right bin → exterior top shelf; red block left bin → right bin.

Examples:
  python3 scripts/visualize_manipulation_books_neural.py
  python3 scripts/visualize_manipulation_books_neural.py --planner convex --show-line
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from pydrake.geometry import (
    Box as DrakeBox,
    MeshcatAnimation,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Rgba,
    Role,
    SceneGraph,
    StartMeshcat,
)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.parsing import LoadModelDirectives, Parser, ProcessModelDirectives
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import RevoluteJoint
from pydrake.perception import PointCloud
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource, Multiplexer, TrajectorySource
from pydrake.trajectories import PiecewisePolynomial

from manipulation.iiwa_helpers import (
    _add_gcs_package,
    _lower_alpha,
    build_shelf_plant,
    forward_kinematics,
    inverse_kinematics,
)
from manipulation.paths import DEFAULT_OUTPUT_DIR, DEFAULT_REGIONS_PATH
from manipulation.shelf_gcs import (
    build_seed_points,
    load_regions,
    planning_configurations,
)
from manipulation.trajopt import (
    build_nonlinear_gcs_problem,
    build_region_edges,
    iiwa_kinematic_limits,
    region_list,
)
from planning_through_contact.model.hparams import DecoderHParams, EncoderHParams
from planning_through_contact.model.ranknet import RankNetConfig
from quadrotor.gcs.linear import LinearGCS

_eval_spec = importlib.util.spec_from_file_location(
    "eval_manipulation_circle_demo",
    REPO_ROOT / "scripts" / "eval_manipulation_circle_demo.py",
)
eval_demo = importlib.util.module_from_spec(_eval_spec)
assert _eval_spec.loader is not None
sys.modules["eval_manipulation_circle_demo"] = eval_demo
_eval_spec.loader.exec_module(eval_demo)

BOOK_DIMS = np.array([0.06, 0.04, 0.14])
BOOK_HALF_HEIGHT = float(BOOK_DIMS[2] / 2.0)
BIN_FLOOR_Z = 0.015
SHELF_ORIGIN_Z = 0.4
SHELF_THICKNESS = 0.016
LOWER_SHELF_CENTER_Z = SHELF_ORIGIN_Z - 0.13115
UPPER_SHELF_CENTER_Z = SHELF_ORIGIN_Z + 0.13115
WSG_OPEN = 0.05
WSG_CLOSED = 0.02
ACTUATED_DIRECTIVES = (
    REPO_ROOT / "manipulation" / "models" / "iiwa14_spheres_collision_actuated_gripper.yaml"
)


@dataclass(frozen=True)
class BlockTask:
    waypoint_names: list[str]
    sequence: list[np.ndarray]
    blue_initial: RigidTransform
    blue_final: RigidTransform
    red_initial: RigidTransform
    red_final: RigidTransform
    blue_grasp_idx: int
    red_grasp_idx: int
    blue_close_timing_idx: int
    blue_place_timing_idx: int
    red_close_timing_idx: int
    red_place_timing_idx: int

    @property
    def right_initial(self) -> RigidTransform:
        return self.blue_initial

    @property
    def left_initial(self) -> RigidTransform:
        return self.red_initial


@dataclass(frozen=True)
class SegmentTiming:
    name: str
    start_s: float
    end_s: float
    wait_end_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neural GCS Meshcat visualization for a two-book shelf-stacking task."
    )
    parser.add_argument("--planner", choices=("convex", "nonlinear"), default="nonlinear")
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGIONS_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "book_stacking_neural",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--speed", type=float, default=1.0, help="Arm joint-space speed (lower = slower).")
    parser.add_argument("--action-wait", type=float, default=1.5, help="Pause at each waypoint (seconds).")
    parser.add_argument("--max-paths", type=int, default=eval_demo.MAX_PATHS)
    parser.add_argument("--max-trials", type=int, default=eval_demo.MAX_ROUNDING_TRIALS)
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
        "--show-line",
        action="store_true",
        help="Draw the end-effector path in the exported Meshcat scene.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=32,
        help="Recording frame rate (lower = faster HTML export).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Open Meshcat in the browser and skip slow StaticHtml export.",
    )
    return parser.parse_args()


def facet_dim_for_first_segment(args, regions, sequence, plant, device: torch.device) -> int:
    if args.planner == "convex":
        gcs = LinearGCS(regions.copy())
        gcs.addSourceTarget(sequence[0], sequence[1])
        graph = gcs.gcs
    else:
        polys = region_list(regions)
        vel_limits, accel_limits = iiwa_kinematic_limits(plant)
        _, graph, _, _ = build_nonlinear_gcs_problem(
            polys,
            build_region_edges(polys),
            sequence[0],
            sequence[1],
            vel_limits=vel_limits,
            accel_limits=accel_limits,
        )
    return eval_demo.build_graph_tensors(graph, regions, device=device).facet_dim


def make_pose(xyz, rpy=(0.0, 0.0, 0.0)) -> RigidTransform:
    return RigidTransform(RollPitchYaw(rpy), np.asarray(xyz, dtype=float))


RIGHT_BIN_X = -0.08
RIGHT_BIN_Y = -0.60
LEFT_BIN_X = 0.08
LEFT_BIN_Y = 0.60
BLUE_SHELF_X = 0.75
BLUE_SHELF_Y = -0.12
BLUE_SHELF_ABOVE_Z = 0.95
BIN_GRASP_PLAN_Z = 0.22
BIN_PREGRASP_Z = 0.38
EXTERIOR_TOP_LOCAL_Z = 0.3995
RIGHT_RPY = [np.pi / 2.0, np.pi, np.pi]
LEFT_RPY = [np.pi / 2.0, np.pi, 0.0]
TOP_RPY = [0.0, -np.pi, -np.pi / 2.0]


def _require_ik(name: str, xyz, rpy, q0) -> np.ndarray:
    q = inverse_kinematics(q0, xyz, rpy)
    if q is None:
        raise RuntimeError(f"IK failed for {name}: xyz={xyz}, rpy={rpy}")
    return q


def _shelf_top_z(local_center_z: float) -> float:
    return SHELF_ORIGIN_Z + local_center_z + SHELF_THICKNESS / 2.0


def build_block_task(regions: dict) -> BlockTask:
    configs = planning_configurations(regions)
    seeds = build_seed_points()

    q_ready = configs["Top Rack"]
    q_right_pre = _require_ik(
        "right_bin_pregrasp",
        [RIGHT_BIN_X, RIGHT_BIN_Y, BIN_PREGRASP_Z],
        RIGHT_RPY,
        q_ready,
    )
    q_right_grasp = _require_ik(
        "right_bin_grasp",
        [RIGHT_BIN_X, RIGHT_BIN_Y, BIN_GRASP_PLAN_Z],
        RIGHT_RPY,
        q_right_pre,
    )
    q_blue_above = _require_ik(
        "blue_top_shelf_above",
        [BLUE_SHELF_X, BLUE_SHELF_Y, BLUE_SHELF_ABOVE_Z],
        TOP_RPY,
        q_right_grasp,
    )
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
    q_via_front = seeds["Front to Shelve"]
    q_via_upper = configs["Top Rack"]
    q_red_bin_pre = _require_ik(
        "red_right_bin_pregrasp",
        [RIGHT_BIN_X, RIGHT_BIN_Y, BIN_PREGRASP_Z],
        RIGHT_RPY,
        q_via_upper,
    )
    q_far = seeds["Front to Shelve"]

    waypoint_names = [
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
        q_ready,
        q_right_pre,
        q_right_grasp,
        q_blue_above,
        q_left_pre,
        q_left_grasp,
        q_via_front,
        q_via_upper,
        q_red_bin_pre,
        q_far,
    ]

    blue_top = _shelf_top_z(EXTERIOR_TOP_LOCAL_Z)
    blue_xyz = [BLUE_SHELF_X, BLUE_SHELF_Y, blue_top + BOOK_HALF_HEIGHT]

    return BlockTask(
        waypoint_names=waypoint_names,
        sequence=sequence,
        blue_initial=make_pose([RIGHT_BIN_X, RIGHT_BIN_Y, BIN_FLOOR_Z + BOOK_HALF_HEIGHT]),
        blue_final=make_pose(blue_xyz),
        red_initial=make_pose([LEFT_BIN_X, LEFT_BIN_Y, BIN_FLOOR_Z + BOOK_HALF_HEIGHT]),
        red_final=make_pose([RIGHT_BIN_X, RIGHT_BIN_Y, BIN_FLOOR_Z + BOOK_HALF_HEIGHT]),
        blue_grasp_idx=2,
        red_grasp_idx=5,
        blue_close_timing_idx=1,
        blue_place_timing_idx=2,
        red_close_timing_idx=4,
        red_place_timing_idx=7,
    )


build_book_task = build_block_task


def combine_segments_with_waits(
    segment_results,
    wait: float,
) -> tuple[PiecewisePolynomial, list[SegmentTiming]]:
    knots_out: list[np.ndarray] = []
    times_out: list[float] = []
    timings: list[SegmentTiming] = []
    offset = 0.0

    for i, segment in enumerate(segment_results):
        traj = segment.trajectory
        local_times = np.asarray(traj.get_segment_times()) - traj.start_time()
        knots = traj.vector_values(traj.get_segment_times())

        if times_out:
            local_times = local_times[1:]
            knots = knots[:, 1:]

        seg_start = offset
        for t, q in zip(local_times + offset, knots.T):
            times_out.append(float(t))
            knots_out.append(q)

        seg_end = float(times_out[-1])
        wait_end = seg_end
        if wait > 0.0 and i < len(segment_results) - 1:
            wait_end = seg_end + wait
            times_out.append(wait_end)
            knots_out.append(knots_out[-1])

        timings.append(
            SegmentTiming(
                name=f"segment_{i + 1}",
                start_s=seg_start,
                end_s=seg_end,
                wait_end_s=wait_end,
            )
        )
        offset = wait_end

    traj_out = PiecewisePolynomial.FirstOrderHold(times_out, np.vstack(knots_out).T)
    return traj_out, timings


def _rigid_to_arrays(X: RigidTransform) -> tuple[np.ndarray, np.ndarray]:
    return X.rotation().matrix(), X.translation()


def _relative_pose(X_WG: RigidTransform, X_WB: RigidTransform) -> tuple[np.ndarray, np.ndarray]:
    R_WG, p_WG = _rigid_to_arrays(X_WG)
    R_WB, p_WB = _rigid_to_arrays(X_WB)
    R_GB = R_WG.T @ R_WB
    p_GB = R_WG.T @ (p_WB - p_WG)
    return R_GB, p_GB


def _compose_pose(X_WG: RigidTransform, R_GB: np.ndarray, p_GB: np.ndarray) -> RigidTransform:
    R_WG, p_WG = _rigid_to_arrays(X_WG)
    return RigidTransform(RotationMatrix(R_WG @ R_GB), R_WG @ p_GB + p_WG)


def _interp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def _action_progress(t: float, start: float, end: float) -> float:
    if end <= start:
        return 1.0
    return float(np.clip((t - start) / (end - start), 0.0, 1.0))


def gripper_opening_at(t: float, timings: list[SegmentTiming], task: BlockTask) -> float:
    close_blue = (
        timings[task.blue_close_timing_idx].end_s,
        timings[task.blue_close_timing_idx].wait_end_s,
    )
    open_blue = (
        timings[task.blue_place_timing_idx].end_s,
        timings[task.blue_place_timing_idx].wait_end_s,
    )
    close_red = (
        timings[task.red_close_timing_idx].end_s,
        timings[task.red_close_timing_idx].wait_end_s,
    )
    open_red = (
        timings[task.red_place_timing_idx].end_s,
        timings[task.red_place_timing_idx].wait_end_s,
    )

    if t < close_blue[0]:
        return WSG_OPEN
    if t < close_blue[1]:
        return _interp(WSG_OPEN, WSG_CLOSED, _action_progress(t, *close_blue))
    if t < open_blue[0]:
        return WSG_CLOSED
    if t < open_blue[1]:
        return _interp(WSG_CLOSED, WSG_OPEN, _action_progress(t, *open_blue))
    if t < close_red[0]:
        return WSG_OPEN
    if t < close_red[1]:
        return _interp(WSG_OPEN, WSG_CLOSED, _action_progress(t, *close_red))
    if t < open_red[0]:
        return WSG_CLOSED
    if t < open_red[1]:
        return _interp(WSG_CLOSED, WSG_OPEN, _action_progress(t, *open_red))
    return WSG_OPEN


def augment_with_wsg_positions(
    traj: PiecewisePolynomial,
    timings: list[SegmentTiming],
    task: BlockTask,
) -> PiecewisePolynomial:
    event_times = [timing.end_s for timing in timings] + [timing.wait_end_s for timing in timings]
    times = sorted({float(t) for t in list(traj.get_segment_times()) + event_times})
    knots: list[np.ndarray] = []
    for t in times:
        arm_q = np.squeeze(traj.value(t))
        opening = gripper_opening_at(t, timings, task)
        knots.append(np.concatenate([arm_q, [-opening, opening]]))
    return PiecewisePolynomial.FirstOrderHold(times, np.vstack(knots).T)


def block_pose_at(
    t: float,
    traj: PiecewisePolynomial,
    timings: list[SegmentTiming],
    task: BlockTask,
    blue_rel: tuple[np.ndarray, np.ndarray],
    red_rel: tuple[np.ndarray, np.ndarray],
) -> tuple[RigidTransform, RigidTransform]:
    q = np.squeeze(traj.value(t))
    X_WG = forward_kinematics([q])[0]

    blue_attach_t = timings[task.blue_close_timing_idx].wait_end_s
    blue_release_t = timings[task.blue_place_timing_idx].wait_end_s
    red_attach_t = timings[task.red_close_timing_idx].wait_end_s
    red_release_t = timings[task.red_place_timing_idx].wait_end_s

    if t < blue_attach_t:
        X_blue = task.blue_initial
    elif t < blue_release_t:
        X_blue = _compose_pose(X_WG, *blue_rel)
    else:
        X_blue = task.blue_final

    if t < red_attach_t:
        X_red = task.red_initial
    elif t < red_release_t:
        X_red = _compose_pose(X_WG, *red_rel)
    else:
        X_red = task.red_final

    return X_blue, X_red


def visualize_pick_place(
    meshcat,
    traj: PiecewisePolynomial,
    timings: list[SegmentTiming],
    task: BlockTask,
    *,
    show_line: bool = False,
    ghost_configs: list[np.ndarray] | None = None,
    fps: int = 32,
    alpha: float = 0.25,
) -> None:
    ghost_configs = ghost_configs or []

    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())
    plant = MultibodyPlant(time_step=0.0)
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    inspector = scene_graph.model_inspector()

    parser = Parser(plant, scene_graph)
    _add_gcs_package(parser)
    directives = LoadModelDirectives(str(ACTUATED_DIRECTIVES))
    models = ProcessModelDirectives(directives, plant, parser)
    iiwa, wsg, *_ = models

    if ghost_configs:
        _lower_alpha(
            plant,
            inspector,
            [iiwa.model_instance, wsg.model_instance],
            alpha,
            scene_graph,
        )

    iiwa_file = str(
        REPO_ROOT
        / "manipulation"
        / "vendor"
        / "drake"
        / "manipulation"
        / "models"
        / "iiwa_description"
        / "urdf"
        / "iiwa14_spheres_collision.urdf"
    )
    wsg_file = str(REPO_ROOT / "manipulation" / "models" / "schunk_wsg_50_welded_fingers.sdf")

    for i, q in enumerate(ghost_configs):
        new_iiwa = parser.AddModels(iiwa_file)[0]
        new_wsg = parser.AddModels(wsg_file)[0]
        plant.RenameModelInstance(new_iiwa, f"vis_iiwa_{i}")
        plant.RenameModelInstance(new_wsg, f"vis_wsg_{i}")
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base", new_iiwa))
        plant.WeldFrames(
            plant.GetFrameByName("iiwa_link_7", new_iiwa),
            plant.GetFrameByName("body", new_wsg),
            RigidTransform(rpy=RollPitchYaw([np.pi / 2.0, 0, 0]), p=[0, 0, 0.114]),
        )
        _lower_alpha(plant, inspector, [new_iiwa, new_wsg], alpha, scene_graph)
        joint_idx = 0
        for joint_index in plant.GetJointIndices(new_iiwa):
            joint = plant.get_mutable_joint(joint_index)
            if isinstance(joint, RevoluteJoint):
                joint.set_default_angle(q[joint_idx])
                joint_idx += 1

    plant.Finalize()

    from pydrake.all import MultibodyPositionToGeometryPose

    to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant))
    builder.Connect(to_pose.get_output_port(), scene_graph.get_source_pose_port(plant.get_source_id()))

    playback_traj = augment_with_wsg_positions(traj, timings, task)
    traj_system = builder.AddSystem(TrajectorySource(playback_traj))
    mux = builder.AddSystem(Multiplexer([9 for _ in range(1 + len(ghost_configs))]))
    builder.Connect(traj_system.get_output_port(), mux.get_input_port(0))
    for i, q in enumerate(ghost_configs):
        ghost_pos = builder.AddSystem(
            ConstantVectorSource(np.concatenate([q, [-WSG_OPEN, WSG_OPEN]]))
        )
        builder.Connect(ghost_pos.get_output_port(), mux.get_input_port(1 + i))
    builder.Connect(mux.get_output_port(), to_pose.get_input_port())

    params = MeshcatVisualizerParams()
    params.delete_on_initialization_event = False
    params.role = Role.kIllustration
    meshcat_viz = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)
    meshcat.Delete()

    meshcat.SetObject("blocks/blue", DrakeBox(*BOOK_DIMS), Rgba(0.05, 0.20, 1.00, 1.0))
    meshcat.SetObject("blocks/red", DrakeBox(*BOOK_DIMS), Rgba(0.95, 0.15, 0.10, 1.0))

    if show_line:
        times = np.linspace(traj.start_time(), traj.end_time(), 500)
        poses = forward_kinematics(traj.vector_values(times).T.tolist())
        pointcloud = PointCloud(len(poses))
        pointcloud.mutable_xyzs()[:] = np.array([X.translation() for X in poses]).T
        meshcat.SetObject(
            "paths/neural_gcs",
            pointcloud,
            0.015,
            rgba=Rgba(1.0, 0.75, 0.0, 1.0),
        )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    meshcat_viz.StartRecording()
    simulator.AdvanceTo(playback_traj.end_time())

    recording = meshcat.get_mutable_recording()
    recording.set_loop_mode(MeshcatAnimation.LoopMode.kLoopOnce)
    recording.set_repetitions(1)
    recording.set_clamp_when_finished(True)
    recording.set_autoplay(True)

    blue_grasp_pose = forward_kinematics([task.sequence[task.blue_grasp_idx]])[0]
    red_grasp_pose = forward_kinematics([task.sequence[task.red_grasp_idx]])[0]
    blue_rel = _relative_pose(blue_grasp_pose, task.blue_initial)
    red_rel = _relative_pose(red_grasp_pose, task.red_initial)

    for t in np.linspace(
        traj.start_time(),
        traj.end_time(),
        int(np.ceil(traj.end_time() * fps)) + 1,
    ):
        frame = recording.frame(t)
        X_blue, X_red = block_pose_at(t, traj, timings, task, blue_rel, red_rel)
        recording.SetTransform(frame, "blocks/blue", X_blue)
        recording.SetTransform(frame, "blocks/red", X_red)

    meshcat_viz.PublishRecording()


def main() -> None:
    args = parse_args()
    eval_demo.MAX_PATHS = int(args.max_paths)
    eval_demo.MAX_ROUNDING_TRIALS = int(args.max_trials)

    from manipulation.paths import manipulation_models_hint, manipulation_models_ready

    if not manipulation_models_ready():
        print(manipulation_models_hint(), file=sys.stderr)
        sys.exit(1)
    if not args.regions.exists():
        print(f"Missing IRIS regions: {args.regions}", file=sys.stderr)
        print("Run: python3 scripts/iiwa_shelf_scenes.py --generate-regions --regions-only", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_dir = f"checkpoints/manipulation_{args.planner}"
    flow_ckpt = args.flow_ckpt or f"{ckpt_dir}/manipulation_{args.planner}_flow_gnn.ckpt"
    ranknet_ckpt = args.ranknet_ckpt or f"{ckpt_dir}/manipulation_{args.planner}_ranknet.ckpt"

    print("Building IIWA shelf scene...")
    plant, _, _, _ = build_shelf_plant()
    print(f"Loading regions: {args.regions}")
    regions = load_regions(args.regions)
    task = build_block_task(regions)

    print("Block task (kinematic attach/release):")
    for i, name in enumerate(task.waypoint_names, start=1):
        print(f"  {i}. {name}")
    print(f"Planner family: {args.planner}")
    print(f"Device: {device}")

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
    facet_dim = facet_dim_for_first_segment(args, regions, task.sequence, plant, device)
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
        cfg=RankNetConfig(
            d_model=args.d_model,
            num_layers=args.ranker_layers,
            num_heads=args.ranker_heads,
        ),
        device=device,
    )
    print(f"Loaded flow: {flow_ckpt}")
    print(f"Loaded ranknet: {ranknet_ckpt}")

    print("\nPlanning with neural GCS...")
    neural = eval_demo.plan_circle(
        planner=args.planner,
        mode="neural",
        regions=regions,
        sequence=task.sequence,
        plant=plant,
        seed=args.seed,
        speed=args.speed,
        flow_model=flow_model,
        ranker=ranker,
        device=device,
    )
    print(
        f"  success={neural.success} cost={neural.cost:.4f} "
        f"length={neural.path_length:.3f} elapsed={neural.elapsed_s:.2f}s"
    )
    if not neural.success:
        print("Neural GCS plan failed.", file=sys.stderr)
        sys.exit(1)

    traj, timings = combine_segments_with_waits(neural.segments, wait=args.action_wait)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"book_stacking_neural_{args.planner}_seed{args.seed}"
    summary = {
        "planner": args.planner,
        "task": "blue_right_bin_to_top_shelf_then_red_left_bin_to_right_bin",
        "mode": "kinematic_attach_release",
        "seed": args.seed,
        "device": str(device),
        "flow_ckpt": str(flow_ckpt),
        "ranknet_ckpt": str(ranknet_ckpt),
        "waypoints": task.waypoint_names,
        "events": [
            {
                "event": "close_gripper_pick_blue",
                "start_s": timings[task.blue_close_timing_idx].end_s,
                "end_s": timings[task.blue_close_timing_idx].wait_end_s,
            },
            {
                "event": "open_gripper_place_blue",
                "start_s": timings[task.blue_place_timing_idx].end_s,
                "end_s": timings[task.blue_place_timing_idx].wait_end_s,
            },
            {
                "event": "close_gripper_pick_red",
                "start_s": timings[task.red_close_timing_idx].end_s,
                "end_s": timings[task.red_close_timing_idx].wait_end_s,
            },
            {
                "event": "open_gripper_place_red",
                "start_s": timings[task.red_place_timing_idx].end_s,
                "end_s": timings[task.red_place_timing_idx].wait_end_s,
            },
        ],
        "neural": eval_demo.plan_to_json(neural),
    }
    json_path = args.output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary -> {json_path}")

    label = "live Meshcat preview" if args.live else "Meshcat HTML"
    print(f"Rendering kinematic pick-and-place ({label})...")
    meshcat = StartMeshcat()
    print(f"Meshcat URL: {meshcat.web_url()}")
    visualize_pick_place(
        meshcat,
        traj,
        timings,
        task,
        show_line=args.show_line,
        ghost_configs=[],
        fps=args.fps,
    )

    if args.live:
        print("Live preview ready — press Play in the Meshcat tab.")
        input("Press Enter to exit...")
        return

    html_path = args.output_dir / f"{stem}.html"
    print("Writing StaticHtml (this can take a few minutes)...")
    html_path.write_text(eval_demo.patch_meshcat_html_play_once(meshcat.StaticHtml()))
    print(f"Saved HTML -> {html_path}")


if __name__ == "__main__":
    main()
