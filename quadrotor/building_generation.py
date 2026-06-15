import numpy as np
import matplotlib.pyplot as plt
import lxml.etree as ET
from pathlib import Path

from pydrake.geometry.optimization import HPolyhedron
from pydrake.math import RigidTransform, RollPitchYaw

MODELS_DIR = Path(__file__).parent / "models"

DEFAULT_GROW_PROBABILITY = 0.7
DEFAULT_START_INDOOR = False
DEFAULT_TREE_PROBABILITY = 0.7

DEFAULT_INDOOR_OPTIONS = {
    "models/room_gen/half_wall_horizontal.sdf": 0.5,
    "models/room_gen/half_wall_horizontal_mirror.sdf": 0.5,
    "models/room_gen/half_wall_vertical.sdf": 0.25,
    "models/room_gen/wall_with_center_door_internal.sdf": 0.5,
    "": 0.25,
}
DEFAULT_WALL_OPTIONS = {
    "models/room_gen/just_wall.sdf": 1.0,
    "models/room_gen/wall_with_center_door.sdf": 0.1,
    "models/room_gen/wall_with_left_window.sdf": 0.05,
    "models/room_gen/wall_with_right_window.sdf": 0.05,
    "models/room_gen/wall_with_windows.sdf": 0.02,
}

# Mostly-indoor scenes with doors/windows on exterior and interior walls.
MOSTLY_INDOOR_GROW_PROBABILITY = 1.0
MOSTLY_INDOOR_START_INDOOR = True
MOSTLY_INDOOR_TREE_PROBABILITY = 0.0
DOORS_WINDOWS_INDOOR_OPTIONS = {
    "models/room_gen/wall_with_center_door_internal.sdf": 1.0,
    "models/room_gen/half_wall_horizontal.sdf": 0.15,
    "models/room_gen/half_wall_horizontal_mirror.sdf": 0.15,
    "models/room_gen/half_wall_vertical.sdf": 0.1,
    "": 0.05,
}
DOORS_WINDOWS_WALL_OPTIONS = {
    "models/room_gen/just_wall.sdf": 0.05,
    "models/room_gen/wall_with_center_door.sdf": 0.35,
    "models/room_gen/wall_with_left_window.sdf": 0.2,
    "models/room_gen/wall_with_right_window.sdf": 0.2,
    "models/room_gen/wall_with_windows.sdf": 0.2,
}


def diagonal_start_goal(shape):
    """Grid indices for opposite corners (bottom-left, top-right)."""
    return np.array([0, 0]), np.array([shape[0] - 1, shape[1] - 1])


def generate_grid_world(
    shape,
    start,
    goal,
    seed=None,
    *,
    grow_probability=DEFAULT_GROW_PROBABILITY,
    start_indoor=DEFAULT_START_INDOOR,
):
    if seed is not None:
        np.random.seed(seed)

    grid = np.zeros(shape) - 1
    grid[start[0], start[1]] = 1 if start_indoor else 0
    grid[goal[0], goal[1]] = 1

    growth_queue = [goal]
    while len(growth_queue) > 0:
        here = growth_queue.pop(0)
        for dx, dy in zip([-1, 1, 0, 0], [0, 0, -1, 1]):
            target = here + np.array([dx, dy])
            if target[0] < 0 or target[1] < 0 or target[0] >= grid.shape[0] or target[1] >= grid.shape[1]:
                continue
            if grid[target[0], target[1]] >= 0:
                continue
            grow = np.random.random() < grow_probability
            if grow:
                grid[target[0], target[1]] = 1
                growth_queue.append(target)
            else:
                grid[target[0], target[1]] = 0

    indoor_edges = []
    outdoor_edges = []
    for i in range(-1, grid.shape[0]):
        for j in range(-1, grid.shape[1]):
            first = np.array([i, j])
            if i < 0 or j < 0:
                first_state = 0
            else:
                first_state = grid[i, j]
            for dx, dy in zip([1, 0], [0, 1]):
                second = first + np.array([dx, dy])
                if second[0] < 0 or second[1] < 0 or second[0] >= grid.shape[0] or second[1] >= grid.shape[1]:
                    second_state = 0
                else:
                    second_state = grid[second[0], second[1]]
                midpoint = (first + second) / 2.
                wall_endpoints = [midpoint + np.array([-dy, dx]) / 2., midpoint + np.array([dy, -dx]) / 2.]
                if first_state > 0.5 and np.isclose(first_state, second_state):
                    indoor_edges.append(wall_endpoints)
                elif first_state > 0.5 and second_state < 0.5:
                    outdoor_edges.append(wall_endpoints)
                elif first_state < 0.5 and second_state > 0.5:
                    outdoor_edges.append(wall_endpoints[::-1])

    return grid, indoor_edges, outdoor_edges


def draw_grid_world(grid, start, goal, indoor_edges, outdoor_edges):
    plt.figure(dpi=150).set_size_inches(5, 5)
    grid_plot = np.clip(grid, 0, 1)
    plt.imshow(grid_plot.T, cmap="binary", vmin=0, vmax=1)
    plt.scatter(np.ones(1) * start[0], np.ones(1) * start[1], s=100, marker="+", label="start")
    plt.scatter(np.ones(1) * goal[0], np.ones(1) * goal[1], s=100, marker="*", label="goal")
    for e1, e2 in indoor_edges:
        plt.plot([e1[0], e2[0]], [e1[1], e2[1]], linestyle="--", c="red")
    for e1, e2 in outdoor_edges:
        plt.arrow(e1[0], e1[1], (e2 - e1)[0], (e2 - e1)[1], linestyle="-", color="orange", head_width=0.1)
    plt.xlim([-2, grid.shape[0]])
    plt.ylim([-2, grid.shape[1]])
    plt.legend()


def compile_sdf(
    output_file,
    grid,
    start,
    goal,
    indoor_edges,
    outdoor_edges,
    seed=None,
    *,
    indoor_options=None,
    wall_options=None,
    tree_probability=DEFAULT_TREE_PROBABILITY,
):
    if seed is not None:
        np.random.seed(seed)

    if indoor_options is None:
        indoor_options = DEFAULT_INDOOR_OPTIONS
    if wall_options is None:
        wall_options = DEFAULT_WALL_OPTIONS

    root_item = ET.Element('sdf', version="1.5", nsmap={'drake': 'drake.mit.edu'})
    model_item = ET.SubElement(root_item, "model", name="building")

    def include_static_sdf_at_pose(name, uri, tf):
        include_item = ET.SubElement(model_item, "include")
        name_item = ET.SubElement(include_item, "name")
        name_item.text = name
        uri_item = ET.SubElement(include_item, "uri")
        uri_item.text = str(MODELS_DIR / uri.replace("models/", "")) if uri else uri
        static_item = ET.SubElement(include_item, "static")
        static_item.text = "True"
        pose_item = ET.SubElement(include_item, "pose")
        xyz = tf.translation()
        rpy = RollPitchYaw(tf.rotation()).vector()
        pose_item.text = "%f %f %f %f %f %f" % (*xyz, *rpy)

    quad_radius = 0.2
    wall_offset = 0.125 + quad_radius
    z_min = quad_radius
    z_max = 3 - quad_radius
    x_cells, y_cells = grid.shape

    regions = [
        HPolyhedron.MakeBox([-2.5, -2.5, z_min], [x_cells * 5 + 7.5, 2.5 - wall_offset, z_max]),
        HPolyhedron.MakeBox([-2.5, 2.5 - wall_offset, z_min], [2.5 - wall_offset, y_cells * 5 + 2.5 + wall_offset, z_max]),
        HPolyhedron.MakeBox([x_cells * 5 + 2.5 + wall_offset, 2.5 - wall_offset, z_min], [x_cells * 5 + 7.5, y_cells * 5 + 2.5 + wall_offset, z_max]),
        HPolyhedron.MakeBox([-2.5, y_cells * 5 + 2.5 + wall_offset, z_min], [x_cells * 5 + 7.5, y_cells * 5 + 7.5, z_max]),
    ]

    for i in range(-1, x_cells + 1):
        for j in range(-1, y_cells + 1):
            xy = (np.array([i, j]) - start) * 5
            tf = RigidTransform(p=np.r_[xy, 0])
            if i >= 0 and j >= 0 and i < x_cells and j < y_cells and grid[i, j] > 0.5:
                include_static_sdf_at_pose("floor_%05d_%05d" % (i, j), "models/room_gen/floor_indoor.sdf", tf)
                include_static_sdf_at_pose("ceiling_%05d_%05d" % (i, j), "models/room_gen/ceiling.sdf", tf)
                regions.append(HPolyhedron.MakeBox(
                    [xy[0] - (2.5 - wall_offset), xy[1] - (2.5 - wall_offset), z_min],
                    [xy[0] + (2.5 - wall_offset), xy[1] + (2.5 - wall_offset), z_max]))
            else:
                include_static_sdf_at_pose("floor_%05d_%05d" % (i, j), "models/room_gen/floor_outdoor.sdf", tf)
                if i < 0 or j < 0 or i == x_cells or j == y_cells:
                    continue
                lb = [xy[0] - 2.5, xy[1] - 2.5, z_min]
                ub = [xy[0] + 2.5, xy[1] + 2.5, z_max]
                if i == 0:
                    lb[0] -= wall_offset
                if j == 0:
                    lb[1] -= wall_offset
                if i == x_cells - 1:
                    ub[0] += wall_offset
                if j == y_cells - 1:
                    ub[1] += wall_offset
                if i > 0 and j >= 0 and j < y_cells and grid[i - 1, j] > 0.5:
                    lb[0] += wall_offset
                if j > 0 and i >= 0 and i < x_cells and grid[i, j - 1] > 0.5:
                    lb[1] += wall_offset
                if i < x_cells - 1 and j >= 0 and j < y_cells and grid[i + 1, j] > 0.5:
                    ub[0] -= wall_offset
                if j < y_cells - 1 and i >= 0 and i < x_cells and grid[i, j + 1] > 0.5:
                    ub[1] -= wall_offset

                if np.random.random() < 1 - tree_probability:
                    regions.append(HPolyhedron.MakeBox(lb, ub))
                    continue
                else:
                    tree_pose = xy + 3.0 * np.random.rand(2) - 1.5
                    tf_tree = RigidTransform(p=np.r_[tree_pose, 0])
                    include_static_sdf_at_pose("tree_%05d_%05d" % (i, j), "models/room_gen/tree.sdf", tf_tree)
                    regions.append(HPolyhedron.MakeBox(lb, [ub[0], tree_pose[1] - 0.5, ub[2]]))
                    regions.append(HPolyhedron.MakeBox([lb[0], tree_pose[1] - 0.5, lb[2]], [tree_pose[0] - 0.5, tree_pose[1] + 0.5, ub[2]]))
                    regions.append(HPolyhedron.MakeBox([tree_pose[0] + 0.5, tree_pose[1] - 0.5, lb[2]], [ub[0], tree_pose[1] + 0.5, ub[2]]))
                    regions.append(HPolyhedron.MakeBox([lb[0], tree_pose[1] + 0.5, lb[2]], ub))

    door_width = 1.25 - 2 * quad_radius
    door_height = 2 - quad_radius
    window_width = 1.5 - 2 * quad_radius
    window_offset = 1.25
    window_z_min = 0.75 + quad_radius
    window_z_max = 2.25 - quad_radius
    half_wall_offset = 1.25

    key_options = list(wall_options.keys())
    probs = np.array(list(wall_options.values()))
    probs = probs / np.sum(probs)
    np.random.shuffle(outdoor_edges)
    for k, (e1, e2) in enumerate(outdoor_edges):
        sdf_key = np.random.choice(key_options, p=probs)
        while k == 0 and "door" not in sdf_key and "window" not in sdf_key:
            sdf_key = np.random.choice(key_options, p=probs)

        delta = e2 - e1
        theta = np.arctan2(delta[0], delta[1])
        midpoint = (e1 + e2) / 2.
        midpoint = (midpoint - start) * 5

        if "door" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + door_width / 2.0 * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + door_width / 2.0 * np.cos(theta))
            regions.append(HPolyhedron.MakeBox(
                [midpoint[0] - dx, midpoint[1] - dy, z_min],
                [midpoint[0] + dx, midpoint[1] + dy, door_height]))
        elif "left_window" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + window_width / 2.0 * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + window_width / 2.0 * np.cos(theta))
            regions.append(HPolyhedron.MakeBox(
                [midpoint[0] - dx + window_offset * np.sin(theta), midpoint[1] - dy + window_offset * np.cos(theta), window_z_min],
                [midpoint[0] + dx + window_offset * np.sin(theta), midpoint[1] + dy + window_offset * np.cos(theta), window_z_max]))
        elif "right_window" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + window_width / 2.0 * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + window_width / 2.0 * np.cos(theta))
            regions.append(HPolyhedron.MakeBox(
                [midpoint[0] - dx - window_offset * np.sin(theta), midpoint[1] - dy - window_offset * np.cos(theta), window_z_min],
                [midpoint[0] + dx - window_offset * np.sin(theta), midpoint[1] + dy - window_offset * np.cos(theta), window_z_max]))
        elif "windows" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + window_width / 2.0 * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + window_width / 2.0 * np.cos(theta))
            regions.append(HPolyhedron.MakeBox(
                [midpoint[0] - dx + window_offset * np.sin(theta), midpoint[1] - dy + window_offset * np.cos(theta), window_z_min],
                [midpoint[0] + dx + window_offset * np.sin(theta), midpoint[1] + dy + window_offset * np.cos(theta), window_z_max]))
            regions.append(HPolyhedron.MakeBox(
                [midpoint[0] - dx - window_offset * np.sin(theta), midpoint[1] - dy - window_offset * np.cos(theta), window_z_min],
                [midpoint[0] + dx - window_offset * np.sin(theta), midpoint[1] + dy - window_offset * np.cos(theta), window_z_max]))

        tf = RigidTransform(p=np.r_[midpoint, 0], rpy=RollPitchYaw(0, 0, -theta))
        include_static_sdf_at_pose("outer_wall_%05d" % k, sdf_key, tf)

    key_options = list(indoor_options.keys())
    probs = np.array(list(indoor_options.values()))
    probs = probs / np.sum(probs)
    np.random.shuffle(indoor_edges)
    for k, (e1, e2) in enumerate(indoor_edges):
        sdf_key = np.random.choice(key_options, p=probs)
        delta = e2 - e1
        theta = np.arctan2(*delta)
        midpoint = (e1 + e2) / 2.
        midpoint = (midpoint - start) * 5

        if sdf_key == "":
            dx = np.abs(wall_offset * np.cos(theta) + (2.5 - wall_offset) * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + (2.5 - wall_offset) * np.cos(theta))
            regions.append(HPolyhedron.MakeBox(
                [midpoint[0] - dx, midpoint[1] - dy, z_min],
                [midpoint[0] + dx, midpoint[1] + dy, z_max]))
            continue
        elif "door" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + door_width / 2.0 * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + door_width / 2.0 * np.cos(theta))
            regions.append(HPolyhedron.MakeBox(
                [midpoint[0] - dx, midpoint[1] - dy, z_min],
                [midpoint[0] + dx, midpoint[1] + dy, door_height]))
        elif "mirror" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + (1.25 - wall_offset) * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + (1.25 - wall_offset) * np.cos(theta))
            regions.append(HPolyhedron.MakeBox(
                [midpoint[0] - dx - half_wall_offset * np.sin(theta), midpoint[1] - dy + half_wall_offset * np.cos(theta), z_min],
                [midpoint[0] + dx - half_wall_offset * np.sin(theta), midpoint[1] + dy + half_wall_offset * np.cos(theta), z_max]))
        elif "horizontal" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + (1.25 - wall_offset) * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + (1.25 - wall_offset) * np.cos(theta))
            regions.append(HPolyhedron.MakeBox(
                [midpoint[0] - dx + half_wall_offset * np.sin(theta), midpoint[1] - dy - half_wall_offset * np.cos(theta), z_min],
                [midpoint[0] + dx + half_wall_offset * np.sin(theta), midpoint[1] + dy - half_wall_offset * np.cos(theta), z_max]))
        elif "vertical" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + (2.5 - wall_offset) * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + (2.5 - wall_offset) * np.cos(theta))
            regions.append(HPolyhedron.MakeBox(
                [midpoint[0] - dx, midpoint[1] - dy, 1.7],
                [midpoint[0] + dx, midpoint[1] + dy, z_max]))

        tf = RigidTransform(p=np.r_[midpoint, 0], rpy=RollPitchYaw(0, 0, theta))
        include_static_sdf_at_pose("inner_wall_%05d" % k, sdf_key, tf)

    tf = RigidTransform(p=np.r_[(start - start) * 5, 0])
    include_static_sdf_at_pose("start_indicator", "models/room_gen/start.sdf", tf)
    tf = RigidTransform(p=np.r_[(goal - start) * 5, 0])
    include_static_sdf_at_pose("goal_indicator", "models/room_gen/target.sdf", tf)

    tree = ET.ElementTree(root_item)
    tree.write(output_file, pretty_print=True)

    return regions
