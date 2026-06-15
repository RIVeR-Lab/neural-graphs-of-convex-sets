"""Meshcat helpers for multi-trajectory quadrotor visualization."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from pydrake.geometry import Rgba
from pydrake.trajectories import Trajectory


class DelayedTrajectory:
    """Shift a trajectory forward in time by holding the start pose until delay."""

    def __init__(self, traj: Trajectory, delay: float):
        self._traj = traj
        self._delay = float(delay)
        self._t0 = float(traj.start_time())
        self._start = np.squeeze(traj.value(self._t0))

    def _inner_t(self, t: float) -> float:
        if t < self._delay:
            return self._t0
        return float(t - self._delay)

    def value(self, t):
        return self._traj.value(self._inner_t(np.squeeze(t)))

    def EvalDerivative(self, t, derivative_order=1):
        t_scalar = float(np.squeeze(t))
        if t_scalar < self._delay:
            dim = self._start.shape[0]
            if derivative_order == 0:
                return self._start.reshape(-1, 1)
            return np.zeros((dim, 1))
        return self._traj.EvalDerivative(self._inner_t(t_scalar), derivative_order)

    def start_time(self):
        return 0.0

    def end_time(self):
        return self._delay + self._traj.end_time() - self._t0

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
