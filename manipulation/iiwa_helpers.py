"""IIWA shelf-scene helpers (adapted from gcs/reproduction/prm_comparison/helpers.py)."""

from __future__ import annotations

import numpy as np

from pydrake.all import MultibodyPositionToGeometryPose
from pydrake.geometry import (
    IllustrationProperties,
    MeshcatAnimation,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    RoleAssign,
    SceneGraph,
)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.parsing import LoadModelDirectives, Parser, ProcessModelDirectives
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.tree import RevoluteJoint
from pydrake.solvers import Solve
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource, Multiplexer, TrajectorySource
from pydrake.trajectories import PiecewisePolynomial

from manipulation.paths import (
    DEFAULT_DIRECTIVES,
    register_package_maps,
    iiwa_urdf_path,
    wsg_sdf_path,
)


def _add_gcs_package(parser: Parser) -> None:
    register_package_maps(parser)


def build_shelf_plant(*, meshcat=None, directives_path=None):
    """Build the IIWA + shelf + bins + table scene."""
    if directives_path is None:
        directives_path = DEFAULT_DIRECTIVES

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant, scene_graph)
    _add_gcs_package(parser)

    directives = LoadModelDirectives(str(directives_path))
    ProcessModelDirectives(directives, plant, parser)
    plant.Finalize()

    meshcat_viz = None
    if meshcat is not None:
        params = MeshcatVisualizerParams()
        params.delete_on_initialization_event = False
        params.role = Role.kIllustration
        meshcat_viz = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)

    diagram = builder.Build()
    return plant, scene_graph, diagram, meshcat_viz


def inverse_kinematics(q0, translation, rpy):
    plant, _, diagram, _ = build_shelf_plant()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    gripper_frame = plant.GetBodyByName("body").body_frame()
    ik = InverseKinematics(plant, plant_context)
    ik.AddPositionConstraint(
        gripper_frame, [0, 0, 0], plant.world_frame(), translation, translation,
    )
    ik.AddOrientationConstraint(
        gripper_frame, RotationMatrix(), plant.world_frame(),
        RotationMatrix(RollPitchYaw(*rpy)), 0.001,
    )

    prog = ik.get_mutable_prog()
    q = ik.q()
    prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
    prog.SetInitialGuess(q, q0)
    result = Solve(ik.prog())
    if not result.is_success():
        return None
    return result.GetSolution(q)


def forward_kinematics(q_list):
    plant, _, diagram, _ = build_shelf_plant()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    poses = []
    body = plant.GetBodyByName("body")
    for q in q_list:
        plant.SetPositions(plant_context, q)
        poses.append(plant.EvalBodyPoseInWorld(plant_context, body))
    return poses


def make_traj(path, speed=2.0):
    t_breaks = [0.0]
    movement = np.sqrt(np.sum(np.square(path.T[1:, :] - path.T[:-1, :]), axis=1))
    for s in movement / speed:
        t_breaks.append(s + t_breaks[-1])
    return PiecewisePolynomial.FirstOrderHold(t_breaks, path)


def combine_trajectory(traj_list, wait=2.0):
    knot_list = []
    time_list = []
    for traj in traj_list:
        knots = traj.vector_values(traj.get_segment_times()).T
        knot_list.append(knots)
        duration = traj.end_time() - traj.start_time()
        offset = time_list[-1][-1] + 0.1 if time_list else 0.0
        time_list.append(np.linspace(offset, duration + offset, knots.shape[0]))
        if wait > 0.0:
            knot_list.append(knot_list[-1][-1, :])
            time_list.append(np.array([time_list[-1][-1] + wait]))
    path = np.vstack(knot_list).T
    time_break = np.hstack(time_list)
    return PiecewisePolynomial.FirstOrderHold(time_break, path)


def trajectory_length(trajectory, weights=None):
    knots = trajectory.vector_values(trajectory.get_segment_times())
    if weights is None:
        weights = np.ones(knots.shape[0])
    length = 0.0
    for ii in range(knots.shape[1] - 1):
        length += np.sqrt(np.square(knots[:, ii + 1] - knots[:, ii]).dot(weights))
    return length


def _lower_alpha(plant, inspector, model_instances, alpha, scene_graph):
    for model in model_instances:
        for body_id in plant.GetBodyIndices(model):
            frame_id = plant.GetBodyFrameIdOrThrow(body_id)
            for g_id in inspector.GetGeometries(frame_id, Role.kIllustration):
                prop = inspector.GetIllustrationProperties(g_id)
                if prop is None or not prop.HasProperty("phong", "diffuse"):
                    continue
                new_props = IllustrationProperties(prop)
                phong = prop.GetProperty("phong", "diffuse")
                phong.set(phong.r(), phong.g(), phong.b(), alpha)
                new_props.UpdateProperty("phong", "diffuse", phong)
                scene_graph.AssignRole(
                    plant.get_source_id(), g_id, new_props, RoleAssign.kReplace,
                )


def visualize_trajectory(
    meshcat,
    traj_list,
    *,
    show_line=False,
    ghost_configs=None,
    alpha=0.3,
    plan_wait=2.0,
):
    """Play back joint-space trajectory(s) in Meshcat with optional ghost robots."""
    if not isinstance(traj_list, list):
        traj_list = [traj_list]
    ghost_configs = ghost_configs or []
    combined_traj = combine_trajectory(traj_list, wait=plan_wait)

    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())
    plant = MultibodyPlant(time_step=0.0)
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    inspector = scene_graph.model_inspector()

    parser = Parser(plant, scene_graph)
    _add_gcs_package(parser)
    directives = LoadModelDirectives(str(DEFAULT_DIRECTIVES))
    models = ProcessModelDirectives(directives, plant, parser)
    iiwa, wsg, *_ = models

    if ghost_configs:
        _lower_alpha(plant, inspector, [iiwa.model_instance, wsg.model_instance], alpha, scene_graph)

    iiwa_file = str(iiwa_urdf_path())
    wsg_file = str(wsg_sdf_path())

    for i, q in enumerate(ghost_configs):
        new_iiwa = parser.AddModels(iiwa_file)[0]
        new_wsg = parser.AddModels(str(wsg_file))[0]
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

    to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant))
    builder.Connect(to_pose.get_output_port(), scene_graph.get_source_pose_port(plant.get_source_id()))

    traj_system = builder.AddSystem(TrajectorySource(combined_traj))
    mux = builder.AddSystem(Multiplexer([7 for _ in range(1 + len(ghost_configs))]))
    builder.Connect(traj_system.get_output_port(), mux.get_input_port(0))
    for i, q in enumerate(ghost_configs):
        ghost_pos = builder.AddSystem(ConstantVectorSource(q))
        builder.Connect(ghost_pos.get_output_port(), mux.get_input_port(1 + i))
    builder.Connect(mux.get_output_port(), to_pose.get_input_port())

    params = MeshcatVisualizerParams()
    params.delete_on_initialization_event = False
    params.role = Role.kIllustration
    meshcat_viz = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)
    meshcat.SetProperty("/Grid", "visible", True)
    meshcat.SetProperty("/Lights/AmbientLight/<object>", "intensity", 0.85)
    meshcat.Delete()

    if show_line:
        from pydrake.geometry import Rgba

        colors = [(0, 0, 1, 1), (1, 0.75, 0, 1), (1, 0.25, 0, 1)]
        for i, traj in enumerate(traj_list):
            times = np.linspace(traj.start_time(), traj.end_time(), 500)
            poses = forward_kinematics(traj.vector_values(times).T.tolist())
            vertices = np.array([X.translation() for X in poses]).T
            meshcat.SetLine(
                f"paths/{i}",
                vertices,
                line_width=4.0,
                rgba=Rgba(*colors[i % len(colors)]),
            )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    meshcat_viz.StartRecording()
    simulator.AdvanceTo(combined_traj.end_time())
    recording = meshcat.get_mutable_recording()
    recording.set_loop_mode(MeshcatAnimation.LoopMode.kLoopOnce)
    recording.set_repetitions(1)
    recording.set_clamp_when_finished(True)
    recording.set_autoplay(True)
    meshcat_viz.PublishRecording()
