import os
from datetime import datetime
from typing import List, Literal, Optional, Tuple

import numpy as np

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.collision_geometry.vertex_defined_geometry import (
    VertexDefinedGeometry,
)
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    BoxWorkspace,
    ContactConfig,
    ContactCost,
    NonCollisionCost,
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    PlanarPushingWorkspace,
    PlanarSolverParams,
    SliderPusherSystemConfig,
)
from planning_through_contact.planning.planar.utils import (
    get_plan_start_and_goals_to_point,
)


def create_output_folder(
    output_dir: str, slider_type: str, traj_number: Optional[int]
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    folder_name = f"{output_dir}/run_{get_time_as_str()}_{slider_type}"
    if traj_number is not None:
        folder_name += f"_traj_{traj_number}"
    os.makedirs(folder_name, exist_ok=True)

    return folder_name


def get_time_as_str() -> str:
    current_time = datetime.now()
    # For example, YYYYMMDDHHMMSS format
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")
    return formatted_time


def get_box() -> RigidBody:
    mass = 0.1
    box_geometry = Box2d(width=0.2, height=0.2)
    slider = RigidBody("box", box_geometry, mass)
    return slider


def get_tee() -> RigidBody:
    mass = 0.1
    body = RigidBody("t_pusher", TPusher2d(), mass)
    return body


def get_sugar_box() -> RigidBody:
    mass = 0.1
    box_geometry = Box2d(width=0.106, height=0.185)
    slider = RigidBody("sugar_box", box_geometry, mass)
    return slider


def get_four_corner_slider() -> RigidBody:
    vertices = [[-0.4, -1], [-1, 0.5], [1, 0.7], [1, -0.6]]
    scale = 1 / 10
    vertices = [np.array(v) * scale for v in vertices]
    geometry = VertexDefinedGeometry(vertices)
    return RigidBody("convex_4_corners", geometry, mass=0.1)


def get_five_corner_slider() -> RigidBody:
    vertices = [[-0.4, -0.4], [-0.4, 0.4], [0.4, 0.4], [0.4, -0.4], [0, -1.0]]
    scale = 1 / 10
    vertices = [np.array(v) * scale for v in vertices]
    geometry = VertexDefinedGeometry(vertices)
    return RigidBody("convex_4_corners", geometry, mass=0.1)


def get_triangle() -> RigidBody:
    vertices = [[-1, -1], [-1, 1], [1, 1]]
    scale = 1 / 10
    vertices = [np.array(v) * scale for v in vertices]
    geometry = VertexDefinedGeometry(vertices)
    return RigidBody("triangle", geometry, mass=0.1)


def get_default_contact_cost() -> ContactCost:
    contact_cost = ContactCost(
        keypoint_arc_length=10.0,
        force_regularization=100000.0,  # NOTE: This is multiplied by 1e-4 because we have forces in other units in the optimization problem
        keypoint_velocity_regularization=100.0,
        trace=None,
        mode_transition_cost=None,
        time=1.0,
    )
    return contact_cost


def get_default_non_collision_cost() -> NonCollisionCost:
    non_collision_cost = NonCollisionCost(
        distance_to_object=0.1,
        pusher_velocity_regularization=10.0,
        pusher_arc_length=10.0,
    )
    return non_collision_cost


def get_default_plan_config(
    slider_type: Literal["box", "sugar_box", "tee"] = "box",
    pusher_radius: float = 0.015,
    time_contact: float = 2.0,
    time_non_collision: float = 4.0,
    workspace: Optional[PlanarPushingWorkspace] = None,
    use_case: Literal["normal"] = "normal",
) -> PlanarPlanConfig:
    if slider_type == "box":
        slider = get_box()
    elif slider_type == "sugar_box":
        slider = get_sugar_box()
    elif slider_type == "convex_4":
        slider = get_four_corner_slider()
    elif slider_type == "convex_5":
        slider = get_five_corner_slider()
    elif slider_type == "triangle":
        slider = get_triangle()
    elif slider_type == "tee":
        slider = get_tee()
    else:
        raise NotImplementedError(f"Slider type {slider_type} not supported")

    slider_pusher_config = SliderPusherSystemConfig(
        slider=slider,
        pusher_radius=pusher_radius,
        friction_coeff_slider_pusher=0.1,
        friction_coeff_table_slider=0.5,
        integration_constant=0.3,
    )
    contact_cost = get_default_contact_cost()
    non_collision_cost = get_default_non_collision_cost()
    buffer_to_corners = 0.0
    contact_config = ContactConfig(
        cost=contact_cost, lam_min=buffer_to_corners, lam_max=1 - buffer_to_corners
    )

    time_contact = 4.0
    time_non_collision = 2.0

    num_knot_points_non_collision = 3
    num_knot_points_contact = 3

    plan_cfg = PlanarPlanConfig(
        dynamics_config=slider_pusher_config,
        num_knot_points_contact=num_knot_points_contact,
        num_knot_points_non_collision=num_knot_points_non_collision,
        use_band_sparsity=True,
        contact_config=contact_config,
        non_collision_cost=non_collision_cost,
        continuity_on_pusher_velocity=True,
        allow_teleportation=False,
        time_in_contact=time_contact,
        time_non_collision=time_non_collision,
        workspace=workspace,
    )

    return plan_cfg


def get_default_solver_params(
    debug: bool = False, clarabel: bool = False
) -> PlanarSolverParams:
    solver_params = PlanarSolverParams(
        measure_solve_time=debug,
        rounding_steps=100,
        print_flows=False,
        solver="mosek" if not clarabel else "clarabel",
        print_solver_output=debug,
        save_solver_output=False,
        print_rounding_details=debug,
        print_path=False,
        print_cost=debug,
        assert_result=False,
        assert_nan_values=True,
        nonl_round_major_feas_tol=1e-5,
        nonl_round_minor_feas_tol=1e-5,
        nonl_round_opt_tol=1e-5,
    )
    return solver_params


def get_default_experiment_plans(
    seed: int, num_trajs: int, config: PlanarPlanConfig, workspace_size: float = 0.6
) -> List[PlanarPushingStartAndGoal]:
    """
    Generates a collection of random initial configurations with the origin as the target
    configuration.
    """
    workspace = PlanarPushingWorkspace(
        slider=BoxWorkspace(
            width=workspace_size,
            height=workspace_size,
            center=np.array([0.0, 0.0]),
            buffer=0,
        ),
    )

    plans = get_plan_start_and_goals_to_point(
        seed,
        num_trajs,
        workspace,
        config,
        (0.0, 0.0),
        limit_rotations=False,
    )

    return plans
