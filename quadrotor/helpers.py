import numpy as np
import os
import pickle
import time

from pydrake.solvers import MosekSolver
from pydrake.systems.framework import LeafSystem

from quadrotor.gcs.bezier import BezierGCS
from quadrotor.gcs.rounding import randomForwardPathSearch
from quadrotor.building_generation import generate_grid_world, compile_sdf


class FlatnessInverter(LeafSystem):
    """Converts a flat-output position trajectory into full quadrotor state."""

    def __init__(self, traj, animator=None, t_offset=0):
        LeafSystem.__init__(self)
        self.traj = traj
        self.animator = animator
        self.t_offset = t_offset
        self.port = self.DeclareVectorOutputPort(
            "state", 12, self.DoCalcState, {self.time_ticket()})

    def DoCalcState(self, context, output):
        t = context.get_time() + self.t_offset - 1e-4

        q = np.squeeze(self.traj.value(t))
        q_dot = np.squeeze(self.traj.EvalDerivative(t))
        q_ddot = np.squeeze(self.traj.EvalDerivative(t, 2))

        fz = np.sqrt(q_ddot[0]**2 + q_ddot[1]**2 + (q_ddot[2] + 9.81)**2)
        r = np.arcsin(-q_ddot[1] / fz)
        p = np.arcsin(q_ddot[0] / fz)

        output.set_value(np.concatenate((q, [r, p, 0], q_dot, np.zeros(3))))

        if self.animator is not None:
            from pydrake.math import RigidTransform
            frame = self.animator.frame(context.get_time())
            self.animator.SetProperty(
                frame, "/Cameras/default/rotated/<object>", "position", [-2.5, 4, 2.5])
            self.animator.SetTransform(frame, "/drake", RigidTransform(-q))


def generate_buildings(save_location, num_buildings, building_shape=(3, 3), seed=None):
    start = np.array([-1, -1])
    goal = np.array([2, 1])

    for ii in range(num_buildings):
        file_location = save_location + "/room_" + str(ii).zfill(3)
        if not os.path.exists(file_location):
            os.makedirs(file_location)

        rng_seed = seed + ii if seed is not None else None
        grid, indoor_edges, outdoor_edges = generate_grid_world(
            shape=building_shape, start=start, goal=goal, seed=rng_seed)
        regions = compile_sdf(
            file_location + "/building.sdf", grid, start, goal, indoor_edges, outdoor_edges)
        with open(file_location + '/regions.reg', 'wb') as f:
            pickle.dump(regions, f)


def build_bezier_gcs(regions, solver):
    order = 7
    continuity = 4
    vel_limit = 10 * np.ones(3)
    hdot_min = 1e-3
    weights = {"time": 1., "norm": 1.}
    max_paths = 10
    max_trials = 100
    rounding_seed = 0

    gcs = BezierGCS(regions, order, continuity, hdot_min=hdot_min, full_dim_overlap=True)
    gcs.addTimeCost(weights["time"])
    gcs.addPathLengthCost(weights["norm"])
    gcs.addVelocityLimits(-vel_limit, vel_limit)
    gcs.setPaperSolverOptions()
    gcs.setSolver(solver)
    gcs.setRoundingStrategy(randomForwardPathSearch, max_paths=max_paths, max_trials=max_trials, seed=rounding_seed)

    return gcs


def plan_through_building(regions, start_pose, goal_pose, solver=None, verbose=False):
    if solver is None:
        solver = MosekSolver()

    t0 = time.time()
    gcs = build_bezier_gcs(regions, solver)
    gcs.addSourceTarget(start_pose, goal_pose, zero_deriv_boundary=3)
    setup_time = time.time() - t0

    t1 = time.time()
    b_traj, results_dict = gcs.SolvePath(rounding=True, verbose=verbose, preprocessing=True)
    solve_time = time.time() - t1

    results_dict["setup_time"] = setup_time
    results_dict["solve_time"] = solve_time

    return b_traj, results_dict
