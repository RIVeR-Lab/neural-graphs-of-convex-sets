#!/usr/bin/env python3
"""Render motion-demo assets: 2D figures/video + Meshcat playback HTML.

2D outputs (same contact colors as Meshcat: green non-contact, red contact):
  {stem}_trajectory.pdf, {stem}_panel_*.pdf, {stem}_legend.pdf, {stem}_prediction.mp4

3D output:
  {stem}.html  — IIWA tabletop playback with centered time overlay

Examples:
  python3 scripts/render_planar_pushing_paper_figure.py \\
      --html planning_through_contact/results/motion_demo/motion_sugar_box_plan639_seed17.html
  python3 scripts/render_planar_pushing_paper_figure.py --body tee --skip-3d --no-live
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.visualize.ik_only_playback import render_tabletop_playback
from planning_through_contact.visualize.planar_pushing import (
    make_traj_figure,
    visualize_planar_pushing_trajectory,
)

_MOTION_DEMO_DIR = REPO_ROOT / "planning_through_contact/results/motion_demo"
DEFAULT_HTML = {
    "sugar_box": _MOTION_DEMO_DIR / "motion_sugar_box_plan639_seed17.html",
    "tee": _MOTION_DEMO_DIR / "motion_tee_plan3_seed17.html",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render 2D motion figures + Meshcat HTML from a motion demo traj.",
    )
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--body", choices=("sugar_box", "tee"), default="sugar_box")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: next to motion HTML).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Meshcat HTML path (default: motion HTML path).",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--check_this_out_ik",
        action="store_true",
        help="Use check_this_out per-frame solve_ik (default: diff-IK playback).",
    )
    parser.add_argument("--skip-2d", action="store_true")
    parser.add_argument("--skip-3d", action="store_true")
    parser.add_argument("--num-contact-frames", type=int, default=4)
    parser.add_argument("--num-non-collision-frames", type=int, default=8)
    parser.add_argument(
        "--time-overlay",
        choices=("center", "lower-right"),
        default=None,
        help="Time label position (default: center for box, lower-right for tee).",
    )
    parser.add_argument(
        "--live",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def resolve_traj_from_motion_html(html_path: Path) -> tuple[Path, Path]:
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
    raise FileNotFoundError(f"No trajectory pickle for {html_path.name}")


def render_2d_assets(
    traj: PlanarPushingTrajectory,
    stem: Path,
    *,
    num_contact_frames: int,
    num_non_collision_frames: int,
) -> None:
    print("Rendering 2D trajectory figure (panels + legend + combined PDF)...")
    make_traj_figure(
        traj,
        filename=str(stem),
        split_on_mode_type=True,
        start_end_legend=False,
        plot_lims=None,
        plot_knot_points=False,
        plot_forces=False,
        show_start_pose=False,
        show_goal_pusher=False,
        num_contact_frames=num_contact_frames,
        num_non_collision_frames=num_non_collision_frames,
        save_individual_panels=True,
        show_contact_legend=True,
    )
    combined = stem.parent / f"{stem.name}_trajectory.pdf"
    panels = sorted(stem.parent.glob(f"{stem.name}_panel_*.pdf"))
    legend = stem.parent / f"{stem.name}_legend.pdf"
    print(f"  combined → {combined}")
    for panel in panels:
        print(f"  panel    → {panel}")
    print(f"  legend   → {legend}")

    mp4_path = stem.parent / f"{stem.name}_prediction.mp4"
    print(f"Rendering 2D prediction video → {mp4_path.name}...")
    animation_lims = traj.get_pos_limits(buffer=0.12)
    visualize_planar_pushing_trajectory(
        traj,
        save=True,
        filename=str(stem.parent / f"{stem.name}_prediction"),
        visualize_knot_points=False,
        lims=animation_lims,
        show_contact_legend=True,
    )
    print(f"  video    → {mp4_path}")


def main() -> None:
    args = parse_args()
    html_path = (args.html or DEFAULT_HTML[args.body]).resolve()
    html_path, traj_path = resolve_traj_from_motion_html(html_path)
    out_dir = args.output_dir or html_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / html_path.stem
    meshcat_out = (args.output or html_path).resolve()

    traj = PlanarPushingTrajectory.load(str(traj_path))
    print(f"Trajectory: {traj_path.name}  ({traj.end_time:.2f}s)")

    if not args.skip_2d:
        render_2d_assets(
            traj,
            stem,
            num_contact_frames=args.num_contact_frames,
            num_non_collision_frames=args.num_non_collision_frames,
        )

    if not args.skip_3d:
        mode = "check_this_out_ik" if args.check_this_out_ik else "diff_ik"
        overlay_pos = args.time_overlay
        if overlay_pos is None:
            overlay_pos = "lower-right" if args.body == "tee" else "center"
        time_overlay_position = "lower_right" if overlay_pos == "lower-right" else "center"
        print(f"Rendering Meshcat playback ({mode}) → {meshcat_out.name}")
        render_tabletop_playback(
            traj,
            meshcat_out,
            fps=args.fps,
            mode=mode,
            time_overlay_position=time_overlay_position,
        )
        print(f"  meshcat  → {meshcat_out}")

    if args.live and not args.skip_3d:
        print("Pause Meshcat in the browser to screenshot; time is in the overlay.")
        input("Press Enter to quit…")


if __name__ == "__main__":
    main()
