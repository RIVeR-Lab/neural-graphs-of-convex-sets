#!/usr/bin/env python3
"""Frozen multi-pose IIWA figure from planar pushing motion HTML.

Uses the same trajectories as ``demo_planar_pushing_neural_motion.py`` HTML
outputs (companion ``.json`` / ``_traj.pkl`` next to each ``.html``).

Mirrors ``render_pick_place_ghosts.py``: stack arm copies at ``--times``,
optional ``--pusher-only`` times (hide arm, show pusher only).

Examples:
  python3 scripts/render_planar_pushing_ghosts.py \\
      --html planning_through_contact/results/motion_demo/motion_sugar_box_plan639_seed17.html
  python3 scripts/render_planar_pushing_ghosts.py --body tee --no-live \\
      --output-html planning_through_contact/results/motion_demo/motion_tee_plan3_seed17_ghosts.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LoadModelDirectives,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    ProcessModelDirectives,
    RigidTransform,
    Role,
    StartMeshcat,
)
from pydrake.geometry import Meshcat

from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.sim_utils import (
    GetSliderUrl,
    LoadRobotOnly,
    models_folder,
)
from planning_through_contact.visualize.ik_only_playback import (
    _CHECK_THIS_OUT_IK_JOINTS,
    _configure_playback_camera,
    _configure_viz_parser,
    _draw_goal_dashed_overlay,
    _get_floating_base_body,
    patch_meshcat_html_play_once,
    _pusher_planar_for_playback,
    _solve_manipulator_q,
    _style_meshcat,
    _table_surface_z_world,
)

_IIWA7_URL = "package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf"
_PUSHER_URL = "package://planning_through_contact/pusher_floating_hydroelastic.sdf"
_IIWA_WELD_ON_TABLE = RigidTransform(np.array([-0.38, -0.15, 0.02]))
_PUSHER_WELD_ON_EE = RigidTransform(np.array([0.0, 0.0, 0.075]))
_SLIDER_HALF_HEIGHT = 0.025

_MOTION_DEMO_DIR = REPO_ROOT / "planning_through_contact/results/motion_demo"
DEFAULT_HTML = {
    "sugar_box": _MOTION_DEMO_DIR / "motion_sugar_box_plan639_seed17.html",
    "tee": _MOTION_DEMO_DIR / "motion_tee_plan3_seed17.html",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen multi-pose figure from planar pushing motion HTML.",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Motion demo HTML from demo_planar_pushing_neural_motion.py.",
    )
    parser.add_argument(
        "--body",
        choices=("sugar_box", "tee"),
        default="sugar_box",
        help="Default motion HTML when --html is omitted (website demos).",
    )
    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=None,
        help="Trajectory times (seconds) for arm poses (default: 0, ⅓, ⅔, end).",
    )
    parser.add_argument(
        "--display-time",
        type=float,
        default=None,
        help="Slider layout time (default: last --times entry).",
    )
    parser.add_argument(
        "--pusher-only",
        type=float,
        nargs="*",
        default=[],
        help="For these pose times, show only the pusher (hide arm links).",
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
        "--pusher-z-offset",
        type=float,
        default=0.025,
        help="Pusher tip z offset used by playback IK.",
    )
    return parser.parse_args()


def resolve_traj_from_motion_html(html_path: Path) -> tuple[Path, Path]:
    """Return (html_path, traj_pkl) for a motion-demo HTML file."""
    html_path = html_path.resolve()
    if not html_path.is_file():
        raise FileNotFoundError(f"Motion HTML not found: {html_path}")

    json_path = html_path.with_suffix(".json")
    if json_path.is_file():
        payload = json.loads(json_path.read_text())
        traj_pkl = Path(payload["traj_pkl"])
        if traj_pkl.is_file():
            return html_path, traj_pkl

    traj_pkl = html_path.with_name(html_path.stem + "_traj.pkl")
    if traj_pkl.is_file():
        return html_path, traj_pkl

    raise FileNotFoundError(
        f"No trajectory pickle for {html_path.name}. Expected {json_path.name} "
        f"with traj_pkl or {traj_pkl.name}."
    )


def _meshcat_geom_path(prefix: str, frame_name: str, geom_name: str) -> str:
    frame_path = frame_name.replace("::", "/")
    root = prefix if frame_path == "world" else f"{prefix}/{frame_path}"
    return f"{root}/{geom_name.replace('::', '/')}"


def _hide_model_instance_geometries(
    meshcat: Meshcat,
    plant,
    inspector,
    model_instance,
    *,
    prefix: str,
) -> None:
    for body_id in plant.GetBodyIndices(model_instance):
        frame_id = plant.GetBodyFrameIdOrThrow(body_id)
        frame_name = inspector.GetName(frame_id)
        for geom_id in inspector.GetGeometries(frame_id, Role.kIllustration):
            path = _meshcat_geom_path(prefix, frame_name, inspector.GetName(geom_id))
            if meshcat.HasPath(path):
                meshcat.SetProperty(path, "visible", False)


def _add_iiwa_pusher_copy(parser: Parser, plant, suffix: str) -> tuple[int, int]:
    parser.SetAutoRenaming(True)
    (iiwa,) = parser.AddModelsFromUrl(_IIWA7_URL)
    (pusher,) = parser.AddModelsFromUrl(_PUSHER_URL)
    parser.SetAutoRenaming(False)
    plant.RenameModelInstance(iiwa, f"vis_iiwa_{suffix}")
    plant.RenameModelInstance(pusher, f"vis_pusher_{suffix}")

    table_body = plant.GetBodyByName("TableTop")
    iiwa_base = plant.GetFrameByName("iiwa_link_0", iiwa)
    plant.WeldFrames(table_body.body_frame(), iiwa_base, _IIWA_WELD_ON_TABLE)

    ee = plant.GetFrameByName("iiwa_link_7", iiwa)
    pusher_frame = plant.GetFrameByName("pusher", pusher)
    plant.WeldFrames(ee, pusher_frame, _PUSHER_WELD_ON_EE)
    return iiwa, pusher


def _sample_iiwa_q(
    traj: PlanarPushingTrajectory,
    robot_plant,
    t: float,
    *,
    q_start: np.ndarray,
    pusher_z_offset: float,
) -> np.ndarray:
    pusher_planar = _pusher_planar_for_playback(traj, float(t))
    return _solve_manipulator_q(
        robot_plant,
        pusher_planar,
        pusher_z_offset=pusher_z_offset,
        q_nominal=q_start,
    )


def render_figure(
    meshcat: Meshcat,
    traj: PlanarPushingTrajectory,
    pose_times: list[float],
    display_time: float,
    *,
    pusher_only_times: set[float],
    pusher_z_offset: float,
) -> tuple[list[float], list[float]]:
    sim_config = PlanarPushingSimConfig.from_traj(traj, pusher_z_offset=pusher_z_offset)
    robot_plant = LoadRobotOnly(sim_config, "iiwa_controller_plant.yaml")
    q_start = _sample_iiwa_q(
        traj, robot_plant, traj.start_time, q_start=_CHECK_THIS_OUT_IK_JOINTS,
        pusher_z_offset=pusher_z_offset,
    )

    samples = [
        (t, _sample_iiwa_q(traj, robot_plant, t, q_start=q_start, pusher_z_offset=pusher_z_offset))
        for t in pose_times
    ]
    q_primary = samples[-1][1]
    ghost_qs = [q for _, q in samples[:-1]]

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant, scene_graph)
    _configure_viz_parser(parser)
    directives = LoadModelDirectives(
        f"{models_folder}/{sim_config.scene_directive_name}"
    )
    ProcessModelDirectives(directives, plant, parser)
    (slider_model,) = parser.AddModels(url=GetSliderUrl(sim_config))

    iiwa_primary = plant.GetModelInstanceByName("iiwa")
    ghost_arms: list[tuple[int, int]] = []
    for i, _ in enumerate(ghost_qs):
        ghost_arms.append(_add_iiwa_pusher_copy(parser, plant, str(i)))

    plant.Finalize()

    params = MeshcatVisualizerParams()
    params.delete_on_initialization_event = False
    params.role = Role.kIllustration
    meshcat_viz = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)
    diagram = builder.Build()

    _style_meshcat(meshcat)
    meshcat.Delete()

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    slider_body = _get_floating_base_body(plant, slider_model)

    plant.SetPositions(plant_context, iiwa_primary, q_primary)
    for (iiwa_i, _), q_i in zip(ghost_arms, ghost_qs):
        plant.SetPositions(plant_context, iiwa_i, q_i)

    slider_planar = traj.get_slider_planar_pose(float(display_time))
    plant.SetFreeBodyPose(
        plant_context,
        slider_body,
        slider_planar.to_pose(_SLIDER_HALF_HEIGHT, z_axis_is_positive=True),
    )

    meshcat_viz.ForcedPublish(meshcat_viz.GetMyContextFromRoot(context))

    _draw_goal_dashed_overlay(
        meshcat,
        traj,
        table_z=_table_surface_z_world(plant, plant_context),
    )
    camera_orbit, camera_target = _configure_playback_camera(meshcat)

    inspector = scene_graph.model_inspector()
    prefix = params.prefix
    if pose_times[-1] in pusher_only_times:
        _hide_model_instance_geometries(
            meshcat, plant, inspector, iiwa_primary, prefix=prefix,
        )
    for i, ghost_time in enumerate(pose_times[:-1]):
        if ghost_time in pusher_only_times:
            iiwa_i, _ = ghost_arms[i]
            _hide_model_instance_geometries(
                meshcat, plant, inspector, iiwa_i, prefix=prefix,
            )

    return camera_orbit, camera_target


def main() -> None:
    args = parse_args()
    html_path = (args.html or DEFAULT_HTML[args.body]).resolve()
    html_path, traj_path = resolve_traj_from_motion_html(html_path)

    traj = PlanarPushingTrajectory.load(str(traj_path))
    t_min, t_max = float(traj.start_time), float(traj.end_time)
    if args.times is None:
        pose_times = [t_min, t_min + (t_max - t_min) / 3, t_min + 2 * (t_max - t_min) / 3, t_max]
    else:
        pose_times = sorted(set(float(t) for t in args.times))
    display_time = float(
        args.display_time if args.display_time is not None else pose_times[-1]
    )
    pusher_only_times = {float(t) for t in args.pusher_only}
    unknown = sorted(pusher_only_times.difference(pose_times))
    if unknown:
        raise ValueError(
            f"--pusher-only times must be a subset of --times; unknown: {unknown}"
        )

    print(f"Motion HTML: {html_path.name}")
    print(f"Trajectory:  {traj_path.name} ({t_min:.1f}s → {t_max:.1f}s)")
    for t in pose_times + [display_time]:
        if t < t_min - 1e-6 or t > t_max + 1e-6:
            raise ValueError(f"Time {t:.3f}s outside trajectory [{t_min:.3f}, {t_max:.3f}]")

    print(
        f"Rendering {len(pose_times)} poses (slider at t={display_time:.2f}s)..."
    )
    meshcat = StartMeshcat()
    print(f"Meshcat: {meshcat.web_url()}")
    camera_orbit, camera_target = render_figure(
        meshcat,
        traj,
        pose_times,
        display_time,
        pusher_only_times=pusher_only_times,
        pusher_z_offset=args.pusher_z_offset,
    )

    if args.output_html is not None:
        out_path = args.output_html.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        html = patch_meshcat_html_play_once(
            meshcat.StaticHtml(),
            camera_orbit_position=camera_orbit,
            camera_target=camera_target,
        )
        out_path.write_text(html)
        print(f"Saved frozen HTML → {out_path}")

    if args.live:
        print("Adjust the camera in Meshcat, screenshot, then press Enter to quit.")
        input()


if __name__ == "__main__":
    main()
