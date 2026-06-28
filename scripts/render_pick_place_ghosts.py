#!/usr/bin/env python3
"""Frozen multi-pose IIWA figure for pick-and-place (non-convex neural GCS).

Each snapshot is an arm copy at a chosen trajectory time with its own opacity
(1 = opaque, 0 = invisible). The last pose (by time) is the full-detail primary
robot; earlier times are drawn as semi-transparent orange link spheres (glTF arm
meshes do not support Meshcat opacity).

Examples:
  python3 scripts/render_pick_place_ghosts.py --times 0 1 2 3
  python3 scripts/render_pick_place_ghosts.py --pose 0:0.25 --pose 5:0.55 --pose 12:1.0
  python3 scripts/render_pick_place_ghosts.py --times 0 5 12 --alphas 0.25 0.55 1.0
  python3 scripts/render_pick_place_ghosts.py --poses-file paper_snapshots/manipulation/poses.json
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
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ConstantVectorSource,
    DiagramBuilder,
    InverseDynamicsController,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MultibodyPlant,
    MultibodyPositionToGeometryPose,
    Multiplexer,
    Parser,
    Role,
    SceneGraph,
    SchunkWsgPositionController,
    Sphere,
    StartMeshcat,
    StateInterpolatorWithDiscreteDerivative,
    TrajectorySource,
)
from pydrake.geometry import Box as DrakeBox, Rgba
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import LoadModelDirectives, ProcessModelDirectives
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder as SimBuilder
from pydrake.systems.primitives import ConstantVectorSource as SimConstant

from manipulation.paths import iiwa_urdf_path, register_package_maps

_MESHCAT_FINISH_RESET_BUG = "animation.setLoopMode(LoopMode.WRAP);"
_MESHCAT_FINISH_RESET_FIX = (
    "animation.setLoopMode(LoopMode.ONCE);"
    "animation.setRepetitions(1);"
    "animation.setClampWhenFinished(true);"
    "animation.setAutoplay(true);"
)


def patch_meshcat_html_play_once(html: str) -> str:
    return html.replace(_MESHCAT_FINISH_RESET_BUG, _MESHCAT_FINISH_RESET_FIX, 1)

_demo_spec = importlib.util.spec_from_file_location(
    "demonstrate_pick_and_place",
    REPO_ROOT / "scripts" / "demonstrate_pick_and_place.py",
)
demo = importlib.util.module_from_spec(_demo_spec)
assert _demo_spec.loader is not None
sys.modules["demonstrate_pick_and_place"] = demo
_demo_spec.loader.exec_module(demo)


@dataclass(frozen=True)
class PoseSpec:
    time: float
    alpha: float = 1.0
    eef_only: bool = False


def parse_pose_specs(args: argparse.Namespace) -> list[PoseSpec]:
    """Resolve pose times and per-pose opacity from CLI flags."""
    if args.poses_file is not None:
        payload = json.loads(Path(args.poses_file).read_text())
        specs = [
            PoseSpec(
                float(entry["t"]),
                float(entry.get("alpha", 1.0)),
                bool(entry.get("eef_only", False)),
            )
            for entry in payload
        ]
    elif args.poses:
        specs = []
        for item in args.poses:
            parts = item.split(":")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid --pose {item!r}; expected TIME:ALPHA (e.g. 2.5:0.4)."
                )
            t, alpha = float(parts[0]), float(parts[1])
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"Opacity must be in [0, 1]; got {alpha} for t={t}.")
            specs.append(PoseSpec(t, alpha, t in {float(x) for x in args.eef_only}))
    else:
        times = [float(t) for t in args.times]
        if args.alphas is None:
            alphas = [1.0] * len(times)
        elif len(args.alphas) != len(times):
            raise ValueError(
                f"--alphas length ({len(args.alphas)}) must match --times ({len(times)})."
            )
        else:
            alphas = [float(a) for a in args.alphas]
        eef_only = {float(t) for t in args.eef_only}
        specs = [PoseSpec(t, a, t in eef_only) for t, a in zip(times, alphas)]

    specs = sorted(specs, key=lambda s: s.time)
    if len(specs) < 1:
        raise ValueError("At least one pose is required.")
    for spec in specs:
        if not 0.0 <= spec.alpha <= 1.0:
            raise ValueError(f"Opacity must be in [0, 1]; got {spec.alpha} for t={spec.time}.")
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render frozen multi-pose pick-and-place figure in Meshcat.",
    )
    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[0.0, 2.0, 10.0],
        help="Trajectory times (seconds) for arm poses.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=None,
        help="Opacity per --times entry in [0, 1] (default: all 1.0).",
    )
    parser.add_argument(
        "--pose",
        action="append",
        dest="poses",
        metavar="TIME:ALPHA",
        help="Pose as trajectory time and opacity, e.g. 2.5:0.35. Repeat per snapshot.",
    )
    parser.add_argument(
        "--poses-file",
        type=Path,
        default=None,
        help='JSON list of {"t", "alpha", "eef_only"?} objects.',
    )
    parser.add_argument(
        "--display-time",
        type=float,
        default=None,
        help="Time for block layout (default: last pose time).",
    )
    parser.add_argument(
        "--eef-only",
        type=float,
        nargs="*",
        default=[],
        help="For these pose times, show only the gripper (hide the arm links).",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Write a self-contained frozen Meshcat HTML (optional).",
    )
    parser.add_argument(
        "--live",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep Meshcat open for camera adjustment (default: on).",
    )
    parser.add_argument(
        "--planner",
        choices=("convex", "nonlinear"),
        default="nonlinear",
        help="GCS planner: nonlinear = non-convex (default).",
    )
    parser.add_argument("--mode", choices=("neural", "vanilla"), default="neural")
    parser.add_argument("--regions", type=Path, default=demo.DEFAULT_REGIONS_PATH)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--speed", type=float, default=1.5)
    parser.add_argument("--action-wait", type=float, default=1.5)
    parser.add_argument("--device", default="auto", help="Torch device: auto, cuda, or cpu.")
    parser.add_argument("--flow-ckpt", default=None)
    parser.add_argument("--ranknet-ckpt", default=None)
    return parser.parse_args()


def sample_arm_wsg(arm_traj, wsg_traj, t: float) -> tuple[np.ndarray, np.ndarray]:
    q_arm = np.asarray(arm_traj.value(t), dtype=float).reshape(-1)
    wsg_span = float(np.asarray(wsg_traj.value(t)).reshape(-1)[0])
    q_wsg = np.array([-wsg_span / 2.0, wsg_span / 2.0])
    return q_arm, q_wsg


_ACTUATED_WSG_PACKAGE_URL = (
    "package://manipulation_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf"
)
_X_IIWA7_WSG = RigidTransform(
    rpy=RollPitchYaw(np.deg2rad([90.0, 0.0, 90.0])),
    p=[0.0, 0.0, 0.09],
)


def simulate_blocks_at_time(args, arm_traj, wsg_traj, extras, phase_times, t: float):
    close_s = phase_times["blue_close_s"]
    open_s = phase_times["blue_open_s"]
    red_close_s = phase_times["red_close_s"]
    red_open_s = phase_times["red_open_s"]
    blue_release_s = demo.wsg_release_s(open_s)
    red_release_s = demo.wsg_release_s(red_open_s)

    builder = SimBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=args.time_step)
    parser = Parser(plant, scene_graph)
    register_package_maps(parser)
    demo.book_viz.ProcessModelDirectives(
        demo.book_viz.LoadModelDirectives(str(demo.ACTUATED_DIRECTIVES)), plant, parser,
    )
    demo.weld_exterior_top_shelf(plant)
    blue_block = demo.add_block(plant, name="blue_block", color=np.array([0.05, 0.20, 1.00, 1.0]))
    red_block = demo.add_block(plant, name="red_block", color=np.array([0.95, 0.15, 0.10, 1.0]))
    plant.Finalize()

    iiwa = plant.GetModelInstanceByName("iiwa")
    wsg = plant.GetModelInstanceByName("wsg")
    iiwa_controller = builder.AddSystem(
        InverseDynamicsController(
            demo.make_controller_plant(),
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
        builder.AddSystem(SimConstant([demo.WSG_FORCE_N])).get_output_port(),
        wsg_controller.get_force_limit_input_port(),
    )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    q0, wsg0 = sample_arm_wsg(arm_traj, wsg_traj, arm_traj.start_time())
    plant.SetPositions(plant_context, iiwa, q0)
    plant.SetPositions(plant_context, wsg, wsg0)
    plant.SetFreeBodyPose(plant_context, blue_block, extras["block_initial"])
    plant.SetFreeBodyPose(plant_context, red_block, extras["red_initial"])

    simulator.set_target_realtime_rate(0.0)
    shelf_z = demo.shelf_top_z()
    demo.advance_physics_with_blocks(
        simulator,
        plant,
        wsg,
        [
            {
                "block": blue_block,
                "attach_s": demo.wsg_attach_s(close_s),
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
                "attach_s": demo.wsg_attach_s(red_close_s),
                "release_s": red_release_s,
                "lock_after_open_s": args.red_lock_after_open_s,
                "shelf_surface_z": shelf_z,
                "X_gripper_block": None,
                "frozen": False,
                "X_frozen": None,
                "resting_since": None,
            },
        ],
        tf=min(float(t), float(arm_traj.end_time())),
        time_step=args.time_step,
    )

    plant_context = plant.GetMyContextFromRoot(context)
    return {
        "blue": plant.EvalBodyPoseInWorld(plant_context, blue_block),
        "red": plant.EvalBodyPoseInWorld(plant_context, red_block),
    }


def _add_arm_copy(parser, plant, name: str) -> tuple[int, int]:
    new_iiwa = parser.AddModels(str(iiwa_urdf_path()))[0]
    wsg_path = parser.package_map().ResolveUrl(_ACTUATED_WSG_PACKAGE_URL)
    new_wsg = parser.AddModels(wsg_path)[0]
    plant.RenameModelInstance(new_iiwa, f"vis_iiwa_{name}")
    plant.RenameModelInstance(new_wsg, f"vis_wsg_{name}")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base", new_iiwa))
    plant.WeldFrames(
        plant.GetFrameByName("iiwa_link_7", new_iiwa),
        plant.GetFrameByName("body", new_wsg),
        _X_IIWA7_WSG,
    )
    return new_iiwa, new_wsg


def _meshcat_geom_path(prefix: str, frame_name: str, geom_name: str) -> str:
    frame_path = frame_name.replace("::", "/")
    root = prefix if frame_path == "world" else f"{prefix}/{frame_path}"
    return f"{root}/{geom_name.replace('::', '/')}"


_GHOST_LINK_RGBA = Rgba(0.92, 0.42, 0.08, 1.0)  # IIWA orange; alpha applied per pose
_GHOST_LINK_RADIUS = 0.045


def _hide_model_instance_geometries(meshcat, plant, inspector, model_instance, *, prefix: str) -> None:
    """Hide all illustration geometries for a model instance."""
    for body_id in plant.GetBodyIndices(model_instance):
        frame_id = plant.GetBodyFrameIdOrThrow(body_id)
        frame_name = inspector.GetName(frame_id)
        for g_id in inspector.GetGeometries(frame_id, Role.kIllustration):
            path = _meshcat_geom_path(prefix, frame_name, inspector.GetName(g_id))
            if meshcat.HasPath(path):
                meshcat.SetProperty(path, "visible", False)


def _draw_ghost_arm_proxy(
    meshcat,
    plant,
    plant_context,
    model_instances,
    *,
    layer_key: str,
    alpha: float,
) -> None:
    """Semi-transparent sphere proxy per link (glTF meshes ignore Meshcat opacity)."""
    if alpha >= 0.999:
        return
    rgba = Rgba(_GHOST_LINK_RGBA.r(), _GHOST_LINK_RGBA.g(), _GHOST_LINK_RGBA.b(), float(alpha))
    for model in model_instances:
        for body_id in plant.GetBodyIndices(model):
            body = plant.get_body(body_id)
            path = f"ghost_proxy/{layer_key}/{body.name()}"
            meshcat.SetObject(path, Sphere(_GHOST_LINK_RADIUS), rgba)
            meshcat.SetTransform(
                path,
                plant.EvalBodyPoseInWorld(plant_context, body),
            )


def render_figure(
    meshcat,
    arm_traj,
    wsg_traj,
    pose_specs: list[PoseSpec],
    block_poses: dict,
) -> None:
    samples = [
        (spec, *sample_arm_wsg(arm_traj, wsg_traj, spec.time)) for spec in pose_specs
    ]
    primary_spec, q_primary_arm, q_primary_wsg = samples[-1]
    q_primary = np.concatenate([q_primary_arm, q_primary_wsg])
    ghost_samples = samples[:-1]

    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())
    plant = MultibodyPlant(time_step=0.0)
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    parser = Parser(plant, scene_graph)
    register_package_maps(parser)
    ProcessModelDirectives(LoadModelDirectives(str(demo.ACTUATED_DIRECTIVES)), plant, parser)
    iiwa_primary = plant.GetModelInstanceByName("iiwa")
    wsg_primary = plant.GetModelInstanceByName("wsg")

    ghost_arms: list[tuple[int, int, PoseSpec]] = []
    for i, (spec, q_arm, q_wsg) in enumerate(ghost_samples):
        ghost_iiwa, ghost_wsg = _add_arm_copy(parser, plant, str(i))
        ghost_arms.append((ghost_iiwa, ghost_wsg, spec))
        _ = q_arm, q_wsg  # positions supplied via mux below

    plant.Finalize()
    ghost_qs = [
        np.concatenate([q_arm, q_wsg]) for _, q_arm, q_wsg in ghost_samples
    ]
    to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant))
    builder.Connect(to_pose.get_output_port(), scene_graph.get_source_pose_port(plant.get_source_id()))

    mux = builder.AddSystem(Multiplexer([9] * (1 + len(ghost_qs))))
    builder.Connect(
        builder.AddSystem(ConstantVectorSource(q_primary)).get_output_port(),
        mux.get_input_port(0),
    )
    for i, q_arm in enumerate(ghost_qs):
        builder.Connect(
            builder.AddSystem(ConstantVectorSource(q_arm)).get_output_port(),
            mux.get_input_port(1 + i),
        )
    builder.Connect(mux.get_output_port(), to_pose.get_input_port())

    params = MeshcatVisualizerParams()
    params.delete_on_initialization_event = False
    params.role = Role.kIllustration
    meshcat_viz = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)
    meshcat.Delete()
    meshcat.SetProperty("/Grid", "visible", False)
    meshcat.SetProperty("/Axes", "visible", False)

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.CreateDefaultContext()
    plant.SetPositions(plant_context, iiwa_primary, q_primary_arm)
    plant.SetPositions(plant_context, wsg_primary, q_primary_wsg)
    for (ghost_iiwa, ghost_wsg, _), q in zip(ghost_arms, ghost_qs):
        plant.SetPositions(plant_context, ghost_iiwa, q[:7])
        plant.SetPositions(plant_context, ghost_wsg, q[7:9])
    meshcat_viz.ForcedPublish(meshcat_viz.GetMyContextFromRoot(context))

    inspector = scene_graph.model_inspector()
    prefix = params.prefix
    for layer_idx, (ghost_iiwa, ghost_wsg, spec) in enumerate(ghost_arms):
        if spec.alpha >= 0.999:
            continue
        _hide_model_instance_geometries(
            meshcat, plant, inspector, ghost_iiwa, prefix=prefix,
        )
        _hide_model_instance_geometries(
            meshcat, plant, inspector, ghost_wsg, prefix=prefix,
        )
        proxy_models = [ghost_wsg] if spec.eef_only else [ghost_iiwa, ghost_wsg]
        _draw_ghost_arm_proxy(
            meshcat,
            plant,
            plant_context,
            proxy_models,
            layer_key=str(layer_idx),
            alpha=spec.alpha,
        )

    if primary_spec.eef_only:
        _hide_model_instance_geometries(meshcat, plant, inspector, iiwa_primary, prefix=prefix)

    meshcat.SetObject("blocks/blue", DrakeBox(*demo.BLOCK_DIMS), Rgba(0.05, 0.20, 1.00, 1.0))
    meshcat.SetObject("blocks/red", DrakeBox(*demo.BLOCK_DIMS), Rgba(0.95, 0.15, 0.10, 1.0))
    meshcat.SetTransform("blocks/blue", block_poses["blue"])
    meshcat.SetTransform("blocks/red", block_poses["red"])


def main() -> None:
    args = parse_args()
    args.time_step = 0.002
    args.blue_lock_after_open_s = 1.2
    args.red_lock_after_open_s = 1.0
    args.d_model = 128
    args.num_layers = 4
    args.num_heads = 4
    args.ffn_hidden_mult = 2
    args.dropout_p = 0.1
    args.decoder_hidden = "256,256"
    args.decoder_dropout_p = 0.1
    args.pointnet_hidden = 64
    args.ranker_layers = 3
    args.ranker_heads = 4

    pose_specs = parse_pose_specs(args)
    pose_times = [spec.time for spec in pose_specs]
    display_time = float(
        args.display_time if args.display_time is not None else pose_times[-1]
    )
    if args.poses_file is not None and args.poses:
        raise ValueError("Use only one of --poses-file and --pose.")
    if args.poses_file is not None and args.alphas is not None:
        raise ValueError("Use only one of --poses-file and --alphas.")
    if args.poses and args.alphas is not None:
        raise ValueError("Use only one of --pose and --alphas.")
    eef_only_times = {float(t) for t in args.eef_only}
    unknown = sorted(eef_only_times.difference(pose_times))
    if unknown:
        raise ValueError(
            f"--eef-only times must be a subset of pose times; unknown: {unknown}"
        )

    planner_label = "non-convex" if args.planner == "nonlinear" else "convex"
    mode_label = "neural" if args.mode == "neural" else "vanilla"
    device = demo.resolve_torch_device(args.device)
    if args.device not in ("cpu",) and device.type == "cpu":
        print(f"Warning: CUDA unavailable ({args.device!r} requested); using CPU.")
    print(
        f"Planning ({planner_label} {mode_label} GCS, seed={args.seed}, "
        f"device={device.type})..."
    )
    args.device = device.type
    arm_traj, wsg_traj, extras, phase_times = demo.build_physics_trajectories(args)
    t_min, t_max = float(arm_traj.start_time()), float(arm_traj.end_time())
    print(f"  trajectory: {t_min:.1f}s → {t_max:.1f}s")
    for t in pose_times + [display_time]:
        if t < t_min - 1e-6 or t > t_max + 1e-6:
            raise ValueError(f"Time {t:.3f}s outside trajectory [{t_min:.3f}, {t_max:.3f}]")

    print(f"Simulating blocks at t={display_time:.2f}s, rendering {len(pose_specs)} poses:")
    for spec in pose_specs:
        print(f"  t={spec.time:6.2f}s  alpha={spec.alpha:.2f}" + ("  (eef only)" if spec.eef_only else ""))
    block_poses = simulate_blocks_at_time(
        args, arm_traj, wsg_traj, extras, phase_times, display_time,
    )

    meshcat = StartMeshcat()
    print(f"Meshcat: {meshcat.web_url()}")
    render_figure(
        meshcat,
        arm_traj,
        wsg_traj,
        pose_specs,
        block_poses,
    )

    if args.output_html is not None:
        out_path = args.output_html.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(patch_meshcat_html_play_once(meshcat.StaticHtml()))
        print(f"Saved frozen HTML → {out_path}")

    if args.live:
        print("Adjust the camera in Meshcat, screenshot, then press Enter to quit.")
        input()


if __name__ == "__main__":
    main()
