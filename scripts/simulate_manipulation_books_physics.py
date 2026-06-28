#!/usr/bin/env python3
"""Experimental physics execution for the neural GCS two-book task.

This script keeps neural GCS as the 7-DOF arm planner, then executes the plan in
a dynamic Drake simulation with an inverse-dynamics controlled IIWA, an actuated
WSG, and free-floating book bodies with frictional contact. Unlike
``visualize_manipulation_books_neural.py``, the books are not kinematically
attached to the gripper.

This is intentionally a prototype: contact grasping is sensitive to object
placement, gains, friction, and gripper timing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ConstantVectorSource,
    DiagramBuilder,
    InverseDynamicsController,
    LoadModelDirectives,
    MeshcatAnimation,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    PiecewisePolynomial,
    ProcessModelDirectives,
    RigidTransform,
    Role,
    RollPitchYaw,
    SchunkWsgPositionController,
    Simulator,
    StartMeshcat,
    StateInterpolatorWithDiscreteDerivative,
    TrajectorySource,
)

from manipulation.iiwa_helpers import build_shelf_plant
from manipulation.paths import DEFAULT_OUTPUT_DIR, DEFAULT_REGIONS_PATH, iiwa_urdf_path, register_package_maps
from manipulation.shelf_gcs import load_regions
from planning_through_contact.model.hparams import DecoderHParams, EncoderHParams
from planning_through_contact.model.ranknet import RankNetConfig
from scripts.visualize_manipulation_books_neural import (
    ACTUATED_DIRECTIVES,
    BOOK_DIMS,
    WSG_OPEN,
    build_book_task,
    combine_segments_with_waits,
    facet_dim_for_first_segment,
)


_eval_spec = importlib.util.spec_from_file_location(
    "eval_manipulation_circle_demo",
    REPO_ROOT / "scripts" / "eval_manipulation_circle_demo.py",
)
eval_demo = importlib.util.module_from_spec(_eval_spec)
assert _eval_spec.loader is not None
sys.modules["eval_manipulation_circle_demo"] = eval_demo
_eval_spec.loader.exec_module(eval_demo)


BOOK_MODEL_TEMPLATE = """<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{name}">
    <pose>{pose}</pose>
    <link name="book_body">
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{ixx}</ixx>
          <iyy>{iyy}</iyy>
          <izz>{izz}</izz>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyz>0</iyz>
        </inertia>
      </inertial>
      <visual name="visual">
        <geometry>
          <box>
            <size>{sx} {sy} {sz}</size>
          </box>
        </geometry>
        <material>
          <diffuse>{rgba}</diffuse>
        </material>
      </visual>
      <collision name="collision">
        <geometry>
          <box>
            <size>{sx} {sy} {sz}</size>
          </box>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>1.5</mu>
              <mu2>1.5</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
    </link>
    <static>0</static>
  </model>
</sdf>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Physics execution prototype for neural GCS book stacking.")
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
    parser.add_argument("--max-paths", type=int, default=eval_demo.MAX_PATHS)
    parser.add_argument("--max-trials", type=int, default=eval_demo.MAX_ROUNDING_TRIALS)
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
    parser.add_argument("--wsg-force-limit", type=float, default=60.0)
    return parser.parse_args()


def book_sdf(name: str, X_WB: RigidTransform, rgba: str) -> str:
    sx, sy, sz = BOOK_DIMS
    mass = 0.12
    ixx = mass * (sy * sy + sz * sz) / 12.0
    iyy = mass * (sx * sx + sz * sz) / 12.0
    izz = mass * (sx * sx + sy * sy) / 12.0
    rpy = RollPitchYaw(X_WB.rotation()).vector()
    xyz = X_WB.translation()
    pose = " ".join(f"{v:.8f}" for v in [*xyz, *rpy])
    return BOOK_MODEL_TEMPLATE.format(
        name=name,
        pose=pose,
        mass=mass,
        ixx=ixx,
        iyy=iyy,
        izz=izz,
        sx=sx,
        sy=sy,
        sz=sz,
        rgba=rgba,
    )


def wsg_command_trajectory(times, timings, *, opened: float = 0.10, closed: float = 0.012):
    sample_times = [float(t) for t in times]
    values = []
    close_right = (timings[0].end_s, timings[0].wait_end_s)
    open_right = (timings[1].end_s, timings[1].wait_end_s)
    close_left = (timings[3].end_s, timings[3].wait_end_s)
    open_left = (timings[4].end_s, timings[4].wait_end_s)

    def interp(a, b, start, end, t):
        if end <= start:
            return b
        u = np.clip((t - start) / (end - start), 0.0, 1.0)
        return a + (b - a) * u

    for t in sample_times:
        if t < close_right[0]:
            width = opened
        elif t < close_right[1]:
            width = interp(opened, closed, *close_right, t)
        elif t < open_right[0]:
            width = closed
        elif t < open_right[1]:
            width = interp(closed, opened, *open_right, t)
        elif t < close_left[0]:
            width = opened
        elif t < close_left[1]:
            width = interp(opened, closed, *close_left, t)
        elif t < open_left[0]:
            width = closed
        elif t < open_left[1]:
            width = interp(closed, opened, *open_left, t)
        else:
            width = opened
        values.append([width])
    return PiecewisePolynomial.FirstOrderHold(sample_times, np.asarray(values).T)


def make_controller_plant(time_step: float):
    # Reusing the finalized planning plant as a controller plant would include
    # unactuated environment models. Build only the IIWA instead.
    from pydrake.all import MultibodyPlant

    controller_plant = MultibodyPlant(time_step=time_step)
    parser = Parser(controller_plant)
    register_package_maps(parser)
    parser.AddModels(str(iiwa_urdf_path()))
    controller_plant.WeldFrames(controller_plant.world_frame(), controller_plant.GetFrameByName("base"))
    controller_plant.Finalize()
    return controller_plant


def load_models_with_books(plant, scene_graph, task):
    parser = Parser(plant, scene_graph)
    register_package_maps(parser)
    directives = LoadModelDirectives(str(ACTUATED_DIRECTIVES))
    models = ProcessModelDirectives(directives, plant, parser)
    right_book = parser.AddModelsFromString(
        book_sdf("right_book", task.right_initial, "0.15 0.35 0.95 1.0"), "sdf"
    )[0]
    left_book = parser.AddModelsFromString(
        book_sdf("left_book", task.left_initial, "0.95 0.25 0.15 1.0"), "sdf"
    )[0]
    return models, right_book, left_book


def main() -> None:
    args = parse_args()
    eval_demo.MAX_PATHS = int(args.max_paths)
    eval_demo.MAX_ROUNDING_TRIALS = int(args.max_trials)

    if not args.regions.exists():
        print(f"Missing IRIS regions: {args.regions}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_dir = f"checkpoints/manipulation_{args.planner}"
    flow_ckpt = args.flow_ckpt or f"{ckpt_dir}/manipulation_{args.planner}_flow_gnn.ckpt"
    ranknet_ckpt = args.ranknet_ckpt or f"{ckpt_dir}/manipulation_{args.planner}_ranknet.ckpt"

    print("Planning neural GCS arm trajectory...")
    planning_plant, _, _, _ = build_shelf_plant()
    regions = load_regions(args.regions)
    task = build_book_task(regions)

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
    facet_dim = facet_dim_for_first_segment(args, regions, task.sequence, planning_plant, device)
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

    neural = eval_demo.plan_circle(
        planner=args.planner,
        mode="neural",
        regions=regions,
        sequence=task.sequence,
        plant=planning_plant,
        seed=args.seed,
        speed=args.speed,
        flow_model=flow_model,
        ranker=ranker,
        device=device,
    )
    if not neural.success:
        print("Neural GCS planning failed.", file=sys.stderr)
        sys.exit(1)

    arm_traj, timings = combine_segments_with_waits(neural.segments, wait=args.action_wait)
    command_times = sorted(set(float(t) for t in list(arm_traj.get_segment_times()) + [x.end_s for x in timings] + [x.wait_end_s for x in timings]))
    wsg_traj = wsg_command_trajectory(command_times, timings)

    print("Building physics simulation...")
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=args.time_step)
    models, right_book, left_book = load_models_with_books(plant, scene_graph, task)
    iiwa_instance = plant.GetModelInstanceByName("iiwa")
    wsg_instance = plant.GetModelInstanceByName("wsg")
    plant.Finalize()

    controller_plant = make_controller_plant(args.time_step)
    iiwa_controller = builder.AddSystem(
        InverseDynamicsController(
            controller_plant,
            kp=[400.0] * 7,
            ki=[0.0] * 7,
            kd=[60.0] * 7,
            has_reference_acceleration=False,
        )
    )
    builder.Connect(plant.get_state_output_port(iiwa_instance), iiwa_controller.get_input_port_estimated_state())
    builder.Connect(iiwa_controller.get_output_port_control(), plant.get_actuation_input_port(iiwa_instance))

    arm_source = builder.AddSystem(TrajectorySource(arm_traj))
    desired_state = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(7, args.time_step, suppress_initial_transient=True)
    )
    builder.Connect(arm_source.get_output_port(), desired_state.get_input_port())
    builder.Connect(desired_state.get_output_port(), iiwa_controller.get_input_port_desired_state())

    wsg_controller = builder.AddSystem(SchunkWsgPositionController())
    builder.Connect(wsg_controller.get_generalized_force_output_port(), plant.get_actuation_input_port(wsg_instance))
    builder.Connect(plant.get_state_output_port(wsg_instance), wsg_controller.get_state_input_port())
    wsg_source = builder.AddSystem(TrajectorySource(wsg_traj))
    builder.Connect(wsg_source.get_output_port(), wsg_controller.get_desired_position_input_port())
    force_source = builder.AddSystem(ConstantVectorSource([args.wsg_force_limit]))
    builder.Connect(force_source.get_output_port(), wsg_controller.get_force_limit_input_port())

    params = MeshcatVisualizerParams()
    params.delete_on_initialization_event = False
    params.role = Role.kIllustration
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    plant.SetPositions(plant_context, iiwa_instance, np.squeeze(arm_traj.value(0.0)))
    plant.SetVelocities(plant_context, iiwa_instance, np.zeros(7))
    plant.SetPositions(plant_context, wsg_instance, [-WSG_OPEN, WSG_OPEN])
    plant.SetVelocities(plant_context, wsg_instance, [0.0, 0.0])
    plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("book_body", right_book), task.right_initial)
    plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("book_body", left_book), task.left_initial)

    print(f"Simulating to t={arm_traj.end_time():.2f}s...")
    visualizer.StartRecording()
    simulator.AdvanceTo(arm_traj.end_time())
    recording = meshcat.get_mutable_recording()
    recording.set_loop_mode(MeshcatAnimation.LoopMode.kLoopOnce)
    recording.set_repetitions(1)
    recording.set_clamp_when_finished(True)
    recording.set_autoplay(True)
    visualizer.PublishRecording()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"book_stacking_physics_{args.planner}_seed{args.seed}"
    html_path = args.output_dir / f"{stem}.html"
    html_path.write_text(eval_demo.patch_meshcat_html_play_once(meshcat.StaticHtml()))
    summary_path = args.output_dir / f"{stem}.json"
    summary_path.write_text(
        json.dumps(
            {
                "planner": args.planner,
                "seed": args.seed,
                "physics": True,
                "time_step": args.time_step,
                "wsg_force_limit": args.wsg_force_limit,
                "html": str(html_path),
                "neural": eval_demo.plan_to_json(neural),
            },
            indent=2,
        )
    )
    print(f"Saved HTML -> {html_path}")
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
