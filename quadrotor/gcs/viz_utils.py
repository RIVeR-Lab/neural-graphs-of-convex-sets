"""Meshcat helpers for multi-trajectory quadrotor visualization."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from pydrake.geometry import Rgba, SceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.trajectories import Trajectory

from quadrotor.helpers import TrailFlatnessInverter

try:
    from underactuated.uav_environment import _QuadrotorGeometry
except ImportError:  # pragma: no cover - optional at import time
    _QuadrotorGeometry = None


TRAIL_OPACITY_MIN = 0.18
TRAIL_OPACITY_MAX = 0.55


def trail_lags(trail_seconds: float, trail_poses: int) -> list[float]:
    """Simulation-time lags for evenly spaced trail ghosts (newest first)."""
    if trail_seconds <= 0 or trail_poses <= 0:
        return []
    step = trail_seconds / trail_poses
    return [step * (i + 1) for i in range(trail_poses)]


def trail_opacities(trail_poses: int) -> list[float]:
    """Opacity for each trail ghost; newest (smallest lag) is most opaque."""
    if trail_poses <= 0:
        return []
    if trail_poses == 1:
        return [TRAIL_OPACITY_MAX]
    return [
        TRAIL_OPACITY_MIN + (TRAIL_OPACITY_MAX - TRAIL_OPACITY_MIN) * (1.0 - i / (trail_poses - 1))
        for i in range(trail_poses)
    ]


def add_quadrotor_pose_trail(
    builder: DiagramBuilder,
    scene_graph: SceneGraph,
    play_traj: Trajectory,
    *,
    trail_seconds: float = 5.8,
    trail_poses: int = 6,
) -> list[str]:
    """Add ghost quadrotors showing poses from the trailing ``trail_seconds`` window."""
    if _QuadrotorGeometry is None:
        raise ImportError(
            "Pose trail requires underactuated.uav_environment._QuadrotorGeometry"
        )
    if trail_seconds <= 0 or trail_poses <= 0:
        return []

    prefixes: list[str] = []
    for i, lag in enumerate(trail_lags(trail_seconds, trail_poses)):
        prefix = f"trail_{i}_"
        prefixes.append(prefix)
        trail_sys = builder.AddSystem(TrailFlatnessInverter(play_traj, lag_s=lag))
        _QuadrotorGeometry.AddToBuilder(
            builder,
            trail_sys.get_output_port(0),
            scene_graph,
            prefix,
        )
    return prefixes


def apply_trail_opacities(meshcat, trail_prefixes: Sequence[str]) -> None:
    """Fade trail ghost meshes after the first SceneGraph publish."""
    opacities = trail_opacities(len(trail_prefixes))
    for prefix, opacity in zip(trail_prefixes, opacities):
        root = f"/drake/{prefix}skydio_2"
        for path in (root, f"{root}/base_link"):
            try:
                meshcat.SetProperty(path, "modulated_opacity", float(opacity))
            except Exception:
                pass


class DelayedTrajectory:
    """Hold the start pose for delay, then play traj at the given speed factor."""

    def __init__(self, traj: Trajectory, delay: float, *, speed: float = 1.0):
        self._traj = traj
        self._delay = float(delay)
        self._speed = float(speed)
        self._t0 = float(traj.start_time())
        self._start = np.squeeze(traj.value(self._t0))

    def _inner_t(self, t: float) -> float:
        if t < self._delay:
            return self._t0
        return self._t0 + self._speed * (t - self._delay)

    def value(self, t):
        return self._traj.value(self._inner_t(np.squeeze(t)))

    def EvalDerivative(self, t, derivative_order=1):
        t_scalar = float(np.squeeze(t))
        if t_scalar < self._delay:
            dim = self._start.shape[0]
            if derivative_order == 0:
                return self._start.reshape(-1, 1)
            return np.zeros((dim, 1))
        return self._traj.EvalDerivative(self._inner_t(t_scalar), derivative_order) * (
            self._speed**derivative_order
        )

    def start_time(self):
        return 0.0

    def end_time(self):
        motion_duration = self._traj.end_time() - self._t0
        return self._delay + motion_duration / self._speed

    def rows(self):
        return self._traj.rows()

    def cols(self):
        return self._traj.cols()


def trajectory_polyline(traj: Trajectory, n_samples: int = 200) -> np.ndarray:
    """Return (3, n_samples) vertices for a Meshcat line."""
    times = np.linspace(traj.start_time(), traj.end_time(), n_samples)
    pts = np.array([np.squeeze(traj.value(t))[:3] for t in times], dtype=np.float64)
    return pts.T


def add_trajectory_traces(meshcat, trajectories: Sequence[Trajectory], colors: Sequence[Rgba], prefix: str) -> None:
    for i, (traj, color) in enumerate(zip(trajectories, colors)):
        meshcat.SetLine(
            f"{prefix}/trace_{i}",
            trajectory_polyline(traj),
            line_width=4.0,
            rgba=color,
        )
