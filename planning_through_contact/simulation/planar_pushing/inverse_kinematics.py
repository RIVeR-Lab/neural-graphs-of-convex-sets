import numpy as np
import numpy.typing as npt
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.all import InverseKinematics
from pydrake.multibody.plant import MultibodyPlant
from pydrake.solvers import Solve


def solve_ik(
    plant: MultibodyPlant,
    pose: RigidTransform,
    default_joint_positions: npt.NDArray[np.float64],
    disregard_angle: bool = False,
) -> npt.NDArray[np.float64]:
    """Full 6-DoF IK for the pusher end frame (used on hardware / sim)."""
    ik = InverseKinematics(plant, with_joint_limits=True)  # type: ignore
    pusher_frame = plant.GetFrameByName("pusher_end")
    EPS = 1e-3

    ik.AddPositionConstraint(
        pusher_frame,
        np.zeros(3),
        plant.world_frame(),
        pose.translation() - np.ones(3) * EPS,
        pose.translation() + np.ones(3) * EPS,
    )

    if disregard_angle:
        z_unit_vec = np.array([0, 0, 1])
        ik.AddAngleBetweenVectorsConstraint(
            pusher_frame,
            z_unit_vec,
            plant.world_frame(),
            -z_unit_vec,
            0.0,
            EPS,
        )
    else:
        ik.AddOrientationConstraint(
            pusher_frame,
            RotationMatrix(),
            plant.world_frame(),
            pose.rotation(),
            EPS,
        )

    prog = ik.get_mutable_prog()
    q = ik.q()
    prog.AddQuadraticErrorCost(np.identity(len(q)), default_joint_positions, q)
    prog.SetInitialGuess(q, default_joint_positions)

    result = Solve(ik.prog())
    assert result.is_success()
    return result.GetSolution(q)


def solve_pusher_ik_for_playback(
    plant: MultibodyPlant,
    pose: RigidTransform,
    q_guess: npt.NDArray[np.float64],
    q_nominal: npt.NDArray[np.float64] | None = None,
    downward_angle_tol: float = 0.5,
) -> npt.NDArray[np.float64]:
    """
    Playback IK: position constraint plus a relaxed "pusher points down" constraint.

    Full 6-DoF IK is infeasible for many planar targets; unconstrained position-only
    IK often collapses the arm. We regularize toward ``q_nominal`` and prefer
    configurations where the pusher z-axis is roughly vertical.
    """
    if q_nominal is None:
        q_nominal = q_guess

    pusher_frame = plant.GetFrameByName("pusher_end")
    eps = 1e-3
    z_unit_vec = np.array([0.0, 0.0, 1.0])
    pos_lb = pose.translation() - np.ones(3) * eps
    pos_ub = pose.translation() + np.ones(3) * eps

    for angle_tol in (downward_angle_tol, 1.5, 2.5):
        ik = InverseKinematics(plant, with_joint_limits=True)  # type: ignore
        ik.AddPositionConstraint(
            pusher_frame,
            np.zeros(3),
            plant.world_frame(),
            pos_lb,
            pos_ub,
        )
        ik.AddAngleBetweenVectorsConstraint(
            pusher_frame,
            z_unit_vec,
            plant.world_frame(),
            -z_unit_vec,
            0.0,
            angle_tol,
        )
        prog = ik.get_mutable_prog()
        q = ik.q()
        prog.AddQuadraticErrorCost(np.identity(len(q)), q_nominal, q)
        prog.SetInitialGuess(q, q_guess)
        result = Solve(ik.prog())
        if result.is_success():
            return result.GetSolution(q)

    ik = InverseKinematics(plant, with_joint_limits=True)  # type: ignore
    ik.AddPositionConstraint(
        pusher_frame,
        np.zeros(3),
        plant.world_frame(),
        pos_lb,
        pos_ub,
    )
    prog = ik.get_mutable_prog()
    q = ik.q()
    prog.AddQuadraticErrorCost(np.identity(len(q)), q_nominal, q)
    prog.SetInitialGuess(q, q_guess)
    result = Solve(ik.prog())
    if result.is_success():
        return result.GetSolution(q)
    return q_guess.copy()
