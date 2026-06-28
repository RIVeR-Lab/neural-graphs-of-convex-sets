"""Meshcat playback of planar pushing trajectories on the IIWA tabletop scene."""

from __future__ import annotations

import logging
import msgpack
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LoadModelDirectives,
    Meshcat,
    MeshcatVisualizer,
    Parser,
    ProcessModelDirectives,
    Rgba,
    RigidTransform,
    StartMeshcat,
)
from pydrake.geometry import Role
from pydrake.multibody.plant import ContactModel

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingContactMode,
    PlanarPushingTrajectory,
)
from planning_through_contact.simulation.planar_pushing.inverse_kinematics import (
    solve_pusher_ik_for_playback,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.sim_utils import (
    ConfigureParser,
    GetSliderUrl,
    LoadRobotOnly,
    models_folder,
    package_xml_file,
)
from planning_through_contact.visualize.colors import COLORS, CRIMSON, EMERALDGREEN

logger = logging.getLogger(__name__)

# Same nominal iiwa posture used by check_this_out run_ik_only_playback.py.
_CHECK_THIS_OUT_IK_JOINTS = np.array(
    [0.0776, 1.0562, 0.3326, -1.3048, 2.7515, -0.8441, 0.5127]
)

_MESHCAT_FINISH_RESET_BUG = (
    "if(this.actions.every((A=>A.paused))){this.pause();for(let A of this.actions)A.reset()}"
)
_MESHCAT_FINISH_RESET_FIX = "if(this.actions.every((A=>A.paused))){this.pause()}"

_PUSHER_NON_CONTACT_PATH = "pusher_non_contact_viz"
_PUSHER_CONTACT_PATH = "pusher_contact_viz"
_PUSHER_NON_CONTACT_RGBA = list(EMERALDGREEN.diffuse())
_PUSHER_CONTACT_RGBA = list(CRIMSON.diffuse())
_GOAL_OVERLAY_RGBA = list(COLORS["aquamarine4"].diffuse())
_GOAL_LINE_WIDTH = 2.0
_GOAL_DASH_LENGTH = 0.012
_GOAL_DASH_GAP = 0.008

# Default tabletop view: elevated front-left, arm in upper frame, full table visible.
_PLAYBACK_CAMERA_TARGET = [-0.20, 0.0, 0.08]
_PLAYBACK_CAMERA_WORLD = [-0.45, -1.00, 0.72]
_PAUSE_BEFORE_PLAYBACK_S = 45.0
_OVERLAY_FONT_SCALE = 1.0
TimeOverlayPosition = Literal["center", "lower_right"]

_MESHCAT_DEFAULT_CAMERA_LINE = (
    "viewer.set_property(['Cameras', 'default', 'rotated', '<object>'],\n"
    '                        "position", [0.0, 1.0, 3.0])'
)

_PLAYBACK_TIME_OVERLAY_HTML = """
<div id="playback-time-overlay">
  <span id="playback-time">t = 0.00 s</span>
</div>
"""

_VIEWER_ANIMATE_BLOCK = """    viewer.animate = function() {
      viewer.animator.update();
      if (viewer.needs_render) {
        viewer.render();
      }
    }"""

_PLAYBACK_TIME_OVERLAY_CSS_CENTER = f"""
    #playback-time-overlay {{
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      z-index: 1000;
      padding: {6 * _OVERLAY_FONT_SCALE}px {10 * _OVERLAY_FONT_SCALE}px;
      border: 1px solid rgba(0, 0, 0, 0.2);
      border-radius: {4 * _OVERLAY_FONT_SCALE}px;
      background: rgba(255, 255, 255, 0.92);
      color: #111;
      font: {24 * _OVERLAY_FONT_SCALE}px/1.2 sans-serif;
      font-weight: 600;
      pointer-events: none;
      white-space: nowrap;
    }}
"""

_PLAYBACK_TIME_OVERLAY_CSS_LOWER_RIGHT = f"""
    #playback-time-overlay {{
      position: fixed;
      bottom: 38%;
      right: 22%;
      z-index: 1000;
      padding: {6 * _OVERLAY_FONT_SCALE}px {10 * _OVERLAY_FONT_SCALE}px;
      border: 1px solid rgba(0, 0, 0, 0.2);
      border-radius: {4 * _OVERLAY_FONT_SCALE}px;
      background: rgba(255, 255, 255, 0.92);
      color: #111;
      font: {24 * _OVERLAY_FONT_SCALE}px/1.2 sans-serif;
      font-weight: 600;
      pointer-events: none;
      white-space: nowrap;
    }}
"""


def _playback_time_overlay_css(position: TimeOverlayPosition) -> str:
    if position == "lower_right":
        return _PLAYBACK_TIME_OVERLAY_CSS_LOWER_RIGHT
    return _PLAYBACK_TIME_OVERLAY_CSS_CENTER


def patch_meshcat_html_play_once(
    html: str,
    *,
    camera_orbit_position: list[float],
    camera_target: list[float],
    pause_before_playback_s: float = _PAUSE_BEFORE_PLAYBACK_S,
    time_overlay_position: TimeOverlayPosition = "center",
) -> str:
    html = html.replace(_MESHCAT_FINISH_RESET_BUG, _MESHCAT_FINISH_RESET_FIX, 1)
    html = patch_meshcat_html_camera(html, camera_orbit_position, camera_target)
    return inject_playback_overlays(
        html,
        pause_before_playback_s=pause_before_playback_s,
        time_overlay_position=time_overlay_position,
    )


def patch_meshcat_html_camera(
    html: str,
    orbit_position: list[float],
    target: list[float],
) -> str:
    orbit_js = ", ".join(f"{value:.6g}" for value in orbit_position)
    target_js = ", ".join(f"{value:.6g}" for value in target)
    replacement = (
        "viewer.set_property(['Cameras', 'default', 'rotated', '<object>'],\n"
        f'                        "position", [{orbit_js}])\n'
        f"    viewer.set_camera_target([{target_js}])"
    )
    return html.replace(_MESHCAT_DEFAULT_CAMERA_LINE, replacement, 1)


def _configure_playback_camera(meshcat: Meshcat) -> tuple[list[float], list[float]]:
    """Set the demo camera and return orbit/target values for StaticHtml patching."""
    meshcat.SetCameraTarget(_PLAYBACK_CAMERA_TARGET)
    meshcat.SetCameraPose(
        camera_in_world=_PLAYBACK_CAMERA_WORLD,
        target_in_world=_PLAYBACK_CAMERA_TARGET,
    )
    packed = meshcat._GetPackedProperty(
        "/Cameras/default/rotated/<object>",
        "position",
    )
    orbit_position = msgpack.unpackb(bytes(packed), raw=False)["value"]
    return orbit_position, _PLAYBACK_CAMERA_TARGET


def _viewer_animate_with_time_hook(pause_before_playback_s: float) -> str:
    pause_s = f"{float(pause_before_playback_s):.6g}"
    return f"""    viewer.animate = function() {{
      viewer.animator.update();
      if (viewer.needs_render) {{
        viewer.render();
      }}
      _update_playback_time_label();
    }}
    function _update_playback_time_label() {{
      var el = document.getElementById("playback-time");
      if (!el || !viewer.animator) return;
      var t = Math.max(0, viewer.animator.time - {pause_s});
      el.textContent = "t = " + t.toFixed(2) + " s";
    }}"""


def inject_playback_overlays(
    html: str,
    *,
    pause_before_playback_s: float = _PAUSE_BEFORE_PLAYBACK_S,
    time_overlay_position: TimeOverlayPosition = "center",
) -> str:
    css = _playback_time_overlay_css(time_overlay_position)
    html = html.replace("</style>", f"{css}\n  </style>", 1)
    html = html.replace("</body>", f"{_PLAYBACK_TIME_OVERLAY_HTML}\n</body>", 1)
    if _VIEWER_ANIMATE_BLOCK not in html:
        raise RuntimeError("Could not inject playback time hook into Meshcat HTML.")
    return html.replace(
        _VIEWER_ANIMATE_BLOCK,
        _viewer_animate_with_time_hook(pause_before_playback_s),
        1,
    )


def inject_contact_mode_legend(html: str) -> str:
    """Backward-compatible alias (legend only, no time hook)."""
    return inject_playback_overlays(html)


def _pusher_in_contact(traj: PlanarPushingTrajectory, t: float) -> bool:
    return traj.get_mode(float(t)) != PlanarPushingContactMode.NO_CONTACT


def _hide_drake_pusher_visual(plant, scene_graph) -> None:
    """Drop the built-in red pusher illustration so MeshcatVisualizer won't redraw it."""
    pusher_body = plant.GetBodyByName("pusher")
    for geom_id in plant.GetVisualGeometriesForBody(pusher_body):
        scene_graph.RemoveRole(plant.get_source_id(), geom_id, Role.kIllustration)


def _setup_pusher_contact_overlay(meshcat: Meshcat, plant, scene_graph) -> RigidTransform:
    """Draw green/red pusher overlays; visibility toggles with contact mode."""
    pusher_body = plant.GetBodyByName("pusher")
    inspector = scene_graph.model_inspector()
    geom_id = plant.GetVisualGeometriesForBody(pusher_body)[0]
    shape = inspector.GetShape(geom_id)
    X_BG = inspector.GetPoseInFrame(geom_id)
    meshcat.SetObject(
        _PUSHER_NON_CONTACT_PATH,
        shape,
        rgba=Rgba(*_PUSHER_NON_CONTACT_RGBA),
    )
    meshcat.SetObject(
        _PUSHER_CONTACT_PATH,
        shape,
        rgba=Rgba(*_PUSHER_CONTACT_RGBA),
    )
    meshcat.SetProperty(_PUSHER_NON_CONTACT_PATH, "visible", True)
    meshcat.SetProperty(_PUSHER_CONTACT_PATH, "visible", False)
    return X_BG


def _update_pusher_contact_viz(
    meshcat: Meshcat,
    plant,
    plant_context,
    pusher_body,
    X_BG: RigidTransform,
    in_contact: bool,
    *,
    time_in_recording: float,
) -> None:
    X_WG = plant.EvalBodyPoseInWorld(plant_context, pusher_body).multiply(X_BG)
    t_rec = float(time_in_recording)
    for path in (_PUSHER_NON_CONTACT_PATH, _PUSHER_CONTACT_PATH):
        meshcat.SetTransform(path, X_WG, time_in_recording=t_rec)
    meshcat.SetProperty(
        _PUSHER_NON_CONTACT_PATH,
        "visible",
        not in_contact,
        time_in_recording=t_rec,
    )
    meshcat.SetProperty(
        _PUSHER_CONTACT_PATH,
        "visible",
        in_contact,
        time_in_recording=t_rec,
    )


def _append_dashed_edge(
    start: np.ndarray,
    end: np.ndarray,
    *,
    dash_length: float,
    gap_length: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        return starts, ends

    unit = direction / length
    pos = 0.0
    drawing = True
    while pos < length - 1e-9:
        seg_len = dash_length if drawing else gap_length
        next_pos = min(pos + seg_len, length)
        if drawing:
            starts.append(start + unit * pos)
            ends.append(start + unit * next_pos)
        pos = next_pos
        drawing = not drawing
    return starts, ends


def _dashed_line_segments(
    points: list[np.ndarray],
    *,
    dash_length: float = _GOAL_DASH_LENGTH,
    gap_length: float = _GOAL_DASH_GAP,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 3xN start/end arrays for a closed dashed polyline."""
    all_starts: list[np.ndarray] = []
    all_ends: list[np.ndarray] = []
    for i in range(len(points)):
        edge_starts, edge_ends = _append_dashed_edge(
            points[i],
            points[(i + 1) % len(points)],
            dash_length=dash_length,
            gap_length=gap_length,
        )
        all_starts.extend(edge_starts)
        all_ends.extend(edge_ends)
    if not all_starts:
        return np.zeros((3, 0)), np.zeros((3, 0))
    return np.column_stack(all_starts), np.column_stack(all_ends)


def _table_surface_z_world(plant, plant_context) -> float:
    """World z of the table top face (TableTop link origin is the upper surface)."""
    table_body = plant.GetBodyByName("TableTop")
    z = plant.EvalBodyPoseInWorld(plant_context, table_body).translation()[2]
    return float(z) + 0.001


def _draw_goal_dashed_overlay(
    meshcat: Meshcat,
    traj: PlanarPushingTrajectory,
    *,
    table_z: float,
) -> None:
    """Dashed slider (and pusher) goal outline, matching the 2D video and legend."""
    target = traj.target_slider_planar_pose
    p_WB = target.pos()
    R_WB = target.rot_matrix()[:2, :2]

    slider_points = []
    for vertex in traj.config.slider_geometry.vertices:
        v_W = p_WB + R_WB.dot(vertex)
        slider_points.append(
            np.array([float(v_W[0, 0]), float(v_W[1, 0]), table_z], dtype=float)
        )

    goal_rgba = Rgba(*_GOAL_OVERLAY_RGBA)
    ss, es = _dashed_line_segments(slider_points)
    meshcat.SetLineSegments(
        "goal_pose/slider_outline",
        ss,
        es,
        line_width=_GOAL_LINE_WIDTH,
        rgba=goal_rgba,
    )

    target_pusher = traj.target_pusher_planar_pose
    if target_pusher is None:
        return

    radius = float(traj.config.dynamics_config.pusher_radius)
    center = target_pusher.pos().flatten()
    num_samples = 64
    circle_points = [
        np.array(
            [
                center[0] + radius * np.cos(2.0 * np.pi * i / num_samples),
                center[1] + radius * np.sin(2.0 * np.pi * i / num_samples),
                table_z,
            ],
            dtype=float,
        )
        for i in range(num_samples)
    ]
    ss, es = _dashed_line_segments(circle_points)
    meshcat.SetLineSegments(
        "goal_pose/pusher_outline",
        ss,
        es,
        line_width=_GOAL_LINE_WIDTH,
        rgba=goal_rgba,
    )


def _planar_pusher_to_world_pose(
    pusher_planar: PlanarPose, pusher_z_offset: float
) -> RigidTransform:
    """Map planar (x, y) to the pusher_end target used by sim and check_this_out IK."""
    return pusher_planar.to_pose(pusher_z_offset)


# Extra radial gap so the 3D cylinder clears the box mesh (2D plan is disk tangency;
# playback IK slack + corner geometry can clip slightly).
_PUSHER_SLIDER_VIZ_CLEARANCE = 0.003


def _closest_point_on_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[float, float]:
    abx, aby = bx - ax, by - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq < 1e-16:
        return ax, ay
    t = float(np.clip(((px - ax) * abx + (py - ay) * aby) / ab_len_sq, 0.0, 1.0))
    return ax + t * abx, ay + t * aby


def _closest_point_on_slider_boundary(
    lx: float,
    ly: float,
    geom,
) -> tuple[float, float]:
    """Closest point on the slider outline in the slider body frame."""
    if isinstance(geom, Box2d):
        hx = float(geom.width) / 2
        hy = float(geom.height) / 2
        return float(np.clip(lx, -hx, hx)), float(np.clip(ly, -hy, hy))

    best_x, best_y = lx, ly
    best_dist_sq = float("inf")
    vertices = geom.vertices
    for i in range(len(vertices)):
        v0 = vertices[i].reshape(2)
        v1 = vertices[(i + 1) % len(vertices)].reshape(2)
        cx, cy = _closest_point_on_segment(lx, ly, v0[0], v0[1], v1[0], v1[1])
        dist_sq = (lx - cx) ** 2 + (ly - cy) ** 2
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_x, best_y = cx, cy
    return best_x, best_y


def _pusher_planar_for_playback(
    traj: PlanarPushingTrajectory,
    t: float,
    *,
    clearance: float = _PUSHER_SLIDER_VIZ_CLEARANCE,
) -> PlanarPose:
    """Nudge the pusher outward when near the slider so the cylinder does not clip."""
    pusher = traj.get_pusher_planar_pose(t)
    geom = traj.config.slider_geometry
    if traj.get_mode(float(t)) != PlanarPushingContactMode.NO_CONTACT and not isinstance(
        geom, Box2d
    ):
        return pusher

    slider = traj.get_slider_planar_pose(t)

    r = float(traj.config.dynamics_config.pusher_radius)
    c, s = np.cos(slider.theta), np.sin(slider.theta)
    dx, dy = pusher.x - slider.x, pusher.y - slider.y
    lx = c * dx + s * dy
    ly = -s * dx + c * dy

    clx, cly = _closest_point_on_slider_boundary(lx, ly, geom)
    ox, oy = lx - clx, ly - cly
    dist = float(np.hypot(ox, oy))
    min_dist = r + clearance

    if dist >= min_dist:
        return pusher

    if dist < 1e-8:
        ox, oy = lx, ly
        dist = float(np.hypot(ox, oy))
        if dist < 1e-8:
            ox, oy = 1.0, 0.0
            dist = 1.0

    scale = min_dist / dist
    lx = clx + ox * scale
    ly = cly + oy * scale
    wx = slider.x + c * lx - s * ly
    wy = slider.y + s * lx + c * ly
    return PlanarPose(wx, wy, pusher.theta)


def _solve_manipulator_q(
    robot_plant,
    pusher_planar: PlanarPose,
    *,
    pusher_z_offset: float,
    q_nominal: np.ndarray,
    downward_angle_tol: float = 0.05,
) -> np.ndarray:
    world_pose = _planar_pusher_to_world_pose(pusher_planar, pusher_z_offset)
    return solve_pusher_ik_for_playback(
        robot_plant,
        world_pose,
        q_nominal,
        q_nominal,
        downward_angle_tol=downward_angle_tol,
    )


def _style_meshcat(meshcat: Meshcat) -> None:
    meshcat.SetProperty("/Grid", "visible", False)
    meshcat.SetProperty("/Axes", "visible", False)
    meshcat.SetProperty("/Background", "top_color", [0.92, 0.92, 0.92])
    meshcat.SetProperty("/Background", "bottom_color", [0.92, 0.92, 0.92])


def _get_floating_base_body(plant, model_instance):
    if hasattr(plant, "GetUniqueFloatingBaseBodyOrThrow"):
        return plant.GetUniqueFloatingBaseBodyOrThrow(model_instance)
    return plant.GetUniqueFreeBaseBodyOrThrow(model_instance)


def _configure_viz_parser(parser: Parser) -> None:
    """Match check_this_out viz plant setup, plus local drake_models for IIWA."""
    parser.package_map().AddPackageXml(filename=package_xml_file)
    parser.package_map().PopulateFromFolder(str(models_folder))
    ConfigureParser(parser)


def _build_playback_scene(
    traj: PlanarPushingTrajectory,
    sim_config: PlanarPushingSimConfig,
    meshcat: Meshcat,
):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant, scene_graph)
    _configure_viz_parser(parser)
    directives = LoadModelDirectives(
        f"{models_folder}/{sim_config.scene_directive_name}"
    )
    ProcessModelDirectives(directives, plant, parser)
    (slider_model,) = parser.AddModels(url=GetSliderUrl(sim_config))
    plant.Finalize()
    _hide_drake_pusher_visual(plant, scene_graph)

    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    iiwa_model = plant.GetModelInstanceByName("iiwa")
    slider_body = _get_floating_base_body(plant, slider_model)
    slider_half_height = 0.025

    _draw_goal_dashed_overlay(
        meshcat,
        traj,
        table_z=_table_surface_z_world(plant, plant_context),
    )

    pusher_body = plant.GetBodyByName("pusher")
    pusher_geom_pose_in_body = _setup_pusher_contact_overlay(meshcat, plant, scene_graph)

    return (
        diagram,
        context,
        plant,
        plant_context,
        iiwa_model,
        slider_body,
        slider_half_height,
        pusher_body,
        pusher_geom_pose_in_body,
    )


def _publish_playback_frame(
    meshcat: Meshcat,
    traj: PlanarPushingTrajectory,
    diagram,
    context,
    plant,
    plant_context,
    robot_plant,
    iiwa_model,
    slider_body,
    slider_half_height: float,
    pusher_body,
    pusher_geom_pose_in_body: RigidTransform,
    t: float,
    *,
    recording_time: float,
    pusher_z_offset: float,
    q_start: np.ndarray,
) -> None:
    pusher_planar = _pusher_planar_for_playback(traj, float(t))
    q_iiwa = _solve_manipulator_q(
        robot_plant,
        pusher_planar,
        pusher_z_offset=pusher_z_offset,
        q_nominal=q_start,
    )
    plant.SetPositions(plant_context, iiwa_model, q_iiwa)

    slider_planar = traj.get_slider_planar_pose(float(t))
    X_WS = slider_planar.to_pose(slider_half_height, z_axis_is_positive=True)
    plant.SetFreeBodyPose(plant_context, slider_body, X_WS)

    context.SetTime(float(recording_time))
    diagram.ForcedPublish(context)
    _update_pusher_contact_viz(
        meshcat,
        plant,
        plant_context,
        pusher_body,
        pusher_geom_pose_in_body,
        _pusher_in_contact(traj, t),
        time_in_recording=recording_time,
    )


def _record_playback_animation(
    meshcat: Meshcat,
    traj: PlanarPushingTrajectory,
    diagram,
    context,
    plant,
    plant_context,
    robot_plant,
    iiwa_model,
    slider_body,
    slider_half_height: float,
    pusher_body,
    pusher_geom_pose_in_body: RigidTransform,
    *,
    times: np.ndarray,
    q_start: np.ndarray,
    pusher_z_offset: float,
    pause_before_playback_s: float = _PAUSE_BEFORE_PLAYBACK_S,
) -> None:
    """Record motion with an initial hold at the start pose."""
    start_t = float(traj.start_time)
    frame_kwargs = dict(
        meshcat=meshcat,
        traj=traj,
        diagram=diagram,
        context=context,
        plant=plant,
        plant_context=plant_context,
        robot_plant=robot_plant,
        iiwa_model=iiwa_model,
        slider_body=slider_body,
        slider_half_height=slider_half_height,
        pusher_body=pusher_body,
        pusher_geom_pose_in_body=pusher_geom_pose_in_body,
        pusher_z_offset=pusher_z_offset,
        q_start=q_start,
    )

    meshcat.StartRecording(set_visualizations_while_recording=False)
    plant.SetPositions(plant_context, iiwa_model, q_start)
    _publish_playback_frame(
        **frame_kwargs,
        t=start_t,
        recording_time=0.0,
    )
    if pause_before_playback_s > 0.0:
        _publish_playback_frame(
            **frame_kwargs,
            t=start_t,
            recording_time=pause_before_playback_s,
        )
    for t in times:
        _publish_playback_frame(
            **frame_kwargs,
            t=float(t),
            recording_time=pause_before_playback_s + (float(t) - start_t),
        )
    meshcat.StopRecording()
    meshcat.PublishRecording()


def render_ik_only_playback(
    traj: PlanarPushingTrajectory,
    out_path: Union[str, Path],
    *,
    pusher_z_offset: float = 0.025,
    fps: float = 30.0,
    default_joint_positions: Optional[np.ndarray] = None,
    meshcat: Optional[Meshcat] = None,
    time_overlay_position: TimeOverlayPosition = "center",
) -> Path:
    """
    Exact check_this_out ``run_ik_only_playback.py`` behavior:

    per-frame full-orientation ``solve_ik`` + slider teleport.
    Works for hardware-demo-style trajectories; asserts on dataset plans parked at (-0.3, 0).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if default_joint_positions is None:
        default_joint_positions = _CHECK_THIS_OUT_IK_JOINTS.copy()

    slider = traj.config.dynamics_config.slider
    logger.info("Rendering check_this_out IK-only playback for slider '%s'", slider.name)

    sim_config = PlanarPushingSimConfig(
        slider=slider,
        dynamics_config=traj.config.dynamics_config,
        contact_model=ContactModel.kHydroelastic,
        pusher_start_pose=traj.initial_pusher_planar_pose,
        slider_start_pose=traj.initial_slider_planar_pose,
        slider_goal_pose=traj.target_slider_planar_pose,
        time_step=1e-3,
        scene_directive_name="planar_pushing_iiwa_plant_hydroelastic.yaml",
        use_hardware=False,
        pusher_z_offset=pusher_z_offset,
        default_joint_positions=default_joint_positions,
    )

    robot_plant = LoadRobotOnly(sim_config, "iiwa_controller_plant.yaml")
    start_pusher = _pusher_planar_for_playback(traj, traj.start_time)
    q_start = _solve_manipulator_q(
        robot_plant,
        start_pusher,
        pusher_z_offset=pusher_z_offset,
        q_nominal=default_joint_positions,
    )
    sim_config.default_joint_positions = q_start
    logger.info("Manipulator start pusher pose (matches 2D video): %s", start_pusher)
    owns_meshcat = meshcat is None
    if meshcat is None:
        meshcat = StartMeshcat()
    _style_meshcat(meshcat)
    camera_orbit_position, camera_target = _configure_playback_camera(meshcat)

    (
        diagram,
        context,
        plant,
        plant_context,
        iiwa_model,
        slider_body,
        slider_half_height,
        pusher_body,
        pusher_geom_pose_in_body,
    ) = _build_playback_scene(traj, sim_config, meshcat)

    dt = 1.0 / fps
    times = np.arange(traj.start_time, traj.end_time, dt)
    logger.info(
        "Recording %d frames at %.1f fps (solve_ik) with %.1fs intro hold",
        len(times),
        fps,
        _PAUSE_BEFORE_PLAYBACK_S,
    )

    _record_playback_animation(
        meshcat,
        traj,
        diagram,
        context,
        plant,
        plant_context,
        robot_plant,
        iiwa_model,
        slider_body,
        slider_half_height,
        pusher_body,
        pusher_geom_pose_in_body,
        times=times,
        q_start=q_start,
        pusher_z_offset=pusher_z_offset,
    )

    html = patch_meshcat_html_play_once(
        meshcat.StaticHtml(),
        camera_orbit_position=camera_orbit_position,
        camera_target=camera_target,
        pause_before_playback_s=_PAUSE_BEFORE_PLAYBACK_S,
        time_overlay_position=time_overlay_position,
    )
    out_path.write_text(html)
    logger.info("Saved check_this_out IK playback to %s", out_path)

    if owns_meshcat:
        del meshcat

    return out_path


def render_diff_ik_playback(
    traj: PlanarPushingTrajectory,
    out_path: Union[str, Path],
    *,
    pusher_z_offset: float = 0.025,
    fps: float = 30.0,
    meshcat: Optional[Meshcat] = None,
    time_overlay_position: TimeOverlayPosition = "center",
) -> Path:
    """
    Tabletop playback: per-frame playback IK tracks the planned pusher (x, y) path
    (same samples as the 2D ``prediction.mp4``) while the slider is teleported.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    slider = traj.config.dynamics_config.slider
    logger.info("Rendering diff-IK tabletop playback for slider '%s'", slider.name)

    sim_config = PlanarPushingSimConfig.from_traj(traj, pusher_z_offset=pusher_z_offset)

    robot_plant = LoadRobotOnly(sim_config, "iiwa_controller_plant.yaml")
    start_pusher = _pusher_planar_for_playback(traj, traj.start_time)
    q_start = _solve_manipulator_q(
        robot_plant,
        start_pusher,
        pusher_z_offset=pusher_z_offset,
        q_nominal=sim_config.default_joint_positions,
    )
    sim_config.default_joint_positions = q_start
    logger.info("Manipulator start pusher pose (matches 2D video): %s", start_pusher)
    owns_meshcat = meshcat is None
    if meshcat is None:
        meshcat = StartMeshcat()
    _style_meshcat(meshcat)
    camera_orbit_position, camera_target = _configure_playback_camera(meshcat)

    (
        diagram,
        context,
        plant,
        plant_context,
        iiwa_model,
        slider_body,
        slider_half_height,
        pusher_body,
        pusher_geom_pose_in_body,
    ) = _build_playback_scene(traj, sim_config, meshcat)

    dt = 1.0 / fps
    times = np.arange(traj.start_time, traj.end_time, dt)
    logger.info(
        "Recording %d frames at %.1f fps (playback IK) with %.1fs intro hold",
        len(times),
        fps,
        _PAUSE_BEFORE_PLAYBACK_S,
    )

    _record_playback_animation(
        meshcat,
        traj,
        diagram,
        context,
        plant,
        plant_context,
        robot_plant,
        iiwa_model,
        slider_body,
        slider_half_height,
        pusher_body,
        pusher_geom_pose_in_body,
        times=times,
        q_start=q_start,
        pusher_z_offset=pusher_z_offset,
    )

    html = patch_meshcat_html_play_once(
        meshcat.StaticHtml(),
        camera_orbit_position=camera_orbit_position,
        camera_target=camera_target,
        pause_before_playback_s=_PAUSE_BEFORE_PLAYBACK_S,
        time_overlay_position=time_overlay_position,
    )
    out_path.write_text(html)
    logger.info("Saved playback-IK tabletop animation to %s", out_path)

    if owns_meshcat:
        del meshcat

    return out_path


def render_tabletop_playback(
    traj: PlanarPushingTrajectory,
    out_path: Union[str, Path],
    *,
    mode: Literal["diff_ik", "check_this_out_ik"] = "diff_ik",
    **kwargs,
) -> Path:
    if mode == "check_this_out_ik":
        return render_ik_only_playback(traj, out_path, **kwargs)
    return render_diff_ik_playback(traj, out_path, **kwargs)
