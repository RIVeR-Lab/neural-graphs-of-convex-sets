"""Chance-constrained inflation of 3D box obstacles, then free-space region carving."""

from __future__ import annotations

import numpy as np
from pydrake.geometry.optimization import HPolyhedron
from scipy.stats import norm

CELL_SIZE = 5.0
HALF_CELL = CELL_SIZE / 2.0
ROOM_HEIGHT = 3.0
WALL_HALF_THICKNESS = 0.125
QUAD_RADIUS = 0.2


def inflate_box_faces(
    box: tuple[float, float, float, float, float, float],
    sigma_k: np.ndarray,
    delta_k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inflate all six faces of an axis-aligned box obstacle.

    Returns (A, b_infl) where the inflated solid is ⋂_i {p : A[i] @ p < b_infl[i]}.
    """
    x0, x1, y0, y1, z0, z1 = box
    A = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    b = np.array([x1, -x0, y1, -y0, z1, -z0], dtype=float)
    ell = A.shape[0]
    delta_i = delta_k / ell
    q = float(norm.ppf(1.0 - delta_i))
    sigma_face = np.sqrt(np.einsum("ij,jk,ik->i", A, sigma_k, A))
    b_infl = b + sigma_face * q
    return A, b_infl


def inflated_box_bounds(A: np.ndarray, b_infl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover axis-aligned bounds of an inflated box from its halfspace form."""
    lb = np.array([-b_infl[1], -b_infl[3], -b_infl[5]], dtype=float)
    ub = np.array([b_infl[0], b_infl[2], b_infl[4]], dtype=float)
    return lb, ub


def nominal_box_bounds(box: tuple[float, float, float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    x0, x1, y0, y1, z0, z1 = box
    return np.array([x0, y0, z0], dtype=float), np.array([x1, y1, z1], dtype=float)


def wall_box_from_edge(
    e1: np.ndarray,
    e2: np.ndarray,
    start: np.ndarray,
    *,
    wall_half_thickness: float = WALL_HALF_THICKNESS,
    z0: float = 0.0,
    z1: float = ROOM_HEIGHT,
) -> tuple[float, float, float, float, float, float]:
    """Thin 3D box for a wall segment between two grid-edge endpoints."""
    delta = e2 - e1
    midpoint = (e1 + e2) / 2.0
    tangent = np.array([delta[1], -delta[0]], dtype=float)
    if np.linalg.norm(tangent) < 1e-9:
        tangent = np.array([1.0, 0.0])
    tangent = tangent / np.linalg.norm(tangent) * 0.5
    p1 = (midpoint + tangent - start) * CELL_SIZE
    p2 = (midpoint - tangent - start) * CELL_SIZE
    normal = np.array([delta[0], delta[1]], dtype=float)
    if np.linalg.norm(normal) < 1e-9:
        normal = np.array([1.0, 0.0])
    normal = normal / np.linalg.norm(normal) * wall_half_thickness
    cx, cy = midpoint[0] - start[0], midpoint[1] - start[1]
    cx *= CELL_SIZE
    cy *= CELL_SIZE
    xs = sorted([p1[0], p2[0], cx - normal[0], cx + normal[0]])
    ys = sorted([p1[1], p2[1], cy - normal[1], cy + normal[1]])
    return (xs[0], xs[-1], ys[0], ys[-1], z0, z1)


def extract_wall_obstacles(
    grid: np.ndarray,
    start: np.ndarray,
    indoor_edges: list,
    outdoor_edges: list,
    *,
    include_perimeter: bool = False,
    include_cell_walls: bool = True,
) -> list[tuple[float, float, float, float, float, float]]:
    """Collect individual 3D wall box obstacles for a grid world."""
    obstacles: list[tuple[float, float, float, float, float, float]] = []
    x_cells, y_cells = grid.shape

    for e1, e2 in indoor_edges + outdoor_edges:
        obstacles.append(wall_box_from_edge(np.asarray(e1), np.asarray(e2), start))

    z0, z1 = 0.0, ROOM_HEIGHT
    if include_perimeter:
        x_min = -HALF_CELL
        y_min = -HALF_CELL
        x_max = (x_cells - 1) * CELL_SIZE + HALF_CELL
        y_max = (y_cells - 1) * CELL_SIZE + HALF_CELL
        t = WALL_HALF_THICKNESS
        obstacles.extend(
            [
                (x_min - 5.0, x_max + 5.0, y_min - t, y_min + t, z0, z1),
                (x_min - t, x_min + t, y_min - 5.0, y_max + 5.0, z0, z1),
                (x_max - t, x_max + t, y_min - 5.0, y_max + 5.0, z0, z1),
                (x_min - 5.0, x_max + 5.0, y_max - t, y_max + t, z0, z1),
            ]
        )

    if include_cell_walls:
        t = WALL_HALF_THICKNESS
        for i in range(x_cells):
            for j in range(y_cells):
                xy = (np.array([i, j]) - start) * CELL_SIZE
                indoor = grid[i, j] > 0.5
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < x_cells and 0 <= nj < y_cells:
                        neighbor_indoor = grid[ni, nj] > 0.5
                        if indoor == neighbor_indoor:
                            continue
                    elif indoor:
                        pass
                    else:
                        continue
                    if di == -1:
                        x0, x1 = xy[0] - HALF_CELL - t, xy[0] - HALF_CELL + t
                        y0, y1 = xy[1] - HALF_CELL, xy[1] + HALF_CELL
                    elif di == 1:
                        x0, x1 = xy[0] + HALF_CELL - t, xy[0] + HALF_CELL + t
                        y0, y1 = xy[1] - HALF_CELL, xy[1] + HALF_CELL
                    elif dj == -1:
                        x0, x1 = xy[0] - HALF_CELL, xy[0] + HALF_CELL
                        y0, y1 = xy[1] - HALF_CELL - t, xy[1] - HALF_CELL + t
                    else:
                        x0, x1 = xy[0] - HALF_CELL, xy[0] + HALF_CELL
                        y0, y1 = xy[1] + HALF_CELL - t, xy[1] + HALF_CELL + t
                    obstacles.append((x0, x1, y0, y1, z0, z1))

    return obstacles


def inflate_obstacles(
    obstacles: list[tuple[float, float, float, float, float, float]],
    sigma_k: np.ndarray,
    delta_k: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Inflate every obstacle box; returns list of (A, b_infl) halfspace pairs."""
    return [inflate_box_faces(box, sigma_k, delta_k) for box in obstacles]


def _clip_box_from_obstacle(
    lb: np.ndarray,
    ub: np.ndarray,
    obs_lb: np.ndarray,
    obs_ub: np.ndarray,
    clearance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink a candidate free-space box so it stays outside one inflated obstacle."""
    lb = lb.copy()
    ub = ub.copy()
    blo = obs_lb.copy()
    bhi = obs_ub.copy()

    if ub[0] <= blo[0] or lb[0] >= bhi[0] or ub[1] <= blo[1] or lb[1] >= bhi[1] or ub[2] <= blo[2] or lb[2] >= bhi[2]:
        return lb, ub

    x_overlap = min(ub[0], bhi[0]) - max(lb[0], blo[0])
    y_overlap = min(ub[1], bhi[1]) - max(lb[1], blo[1])
    z_overlap = min(ub[2], bhi[2]) - max(lb[2], blo[2])
    axis = int(np.argmin([x_overlap, y_overlap, z_overlap]))

    if axis == 0:
        if (blo[0] + bhi[0]) / 2.0 <= (lb[0] + ub[0]) / 2.0:
            lb[0] = bhi[0] + clearance
        else:
            ub[0] = blo[0] - clearance
    elif axis == 1:
        if (blo[1] + bhi[1]) / 2.0 <= (lb[1] + ub[1]) / 2.0:
            lb[1] = bhi[1] + clearance
        else:
            ub[1] = blo[1] - clearance
    else:
        if (blo[2] + bhi[2]) / 2.0 <= (lb[2] + ub[2]) / 2.0:
            lb[2] = bhi[2] + clearance
        else:
            ub[2] = blo[2] - clearance
    return lb, ub


def _clip_box_from_extra_inflation(
    lb: np.ndarray,
    ub: np.ndarray,
    obs_lb_nom: np.ndarray,
    obs_ub_nom: np.ndarray,
    obs_lb_infl: np.ndarray,
    obs_ub_infl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink a nominal free-space box by the extra obstacle inflation only."""
    if not (
        np.any(obs_lb_infl < obs_lb_nom - 1e-9)
        or np.any(obs_ub_infl > obs_ub_nom + 1e-9)
    ):
        return lb, ub
    return _clip_box_from_obstacle(lb, ub, obs_lb_infl, obs_ub_infl, clearance=0.0)


def tighten_box_for_inflation(
    lb: np.ndarray,
    ub: np.ndarray,
    nominal_obstacles: list[tuple[np.ndarray, np.ndarray]],
    inflated_obstacles: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Apply only the incremental tightening from obstacle inflation."""
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    for (A, b_nom), (_, b_infl) in zip(nominal_obstacles, inflated_obstacles):
        obs_lb_nom, obs_ub_nom = inflated_box_bounds(A, b_nom)
        obs_lb_infl, obs_ub_infl = inflated_box_bounds(A, b_infl)
        lb, ub = _clip_box_from_extra_inflation(
            lb, ub, obs_lb_nom, obs_ub_nom, obs_lb_infl, obs_ub_infl,
        )
    if np.any(lb >= ub):
        return None
    return lb, ub


def carve_free_box(
    lb: np.ndarray,
    ub: np.ndarray,
    inflated_obstacles: list[tuple[np.ndarray, np.ndarray]],
    clearance: float = QUAD_RADIUS,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Carve a free-space box around the union of inflated obstacles."""
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    for A, b_infl in inflated_obstacles:
        obs_lb, obs_ub = inflated_box_bounds(A, b_infl)
        lb, ub = _clip_box_from_obstacle(lb, ub, obs_lb, obs_ub, clearance)
    if np.any(lb >= ub):
        return None
    return lb, ub


def tighten_regions(
    regions: list[HPolyhedron],
    margin: float,
) -> list[HPolyhedron]:
    """Uniformly shrink compile_sdf free-space boxes by the obstacle inflation margin."""
    if margin <= 0.0:
        return list(regions)

    tightened: list[HPolyhedron] = []
    for region in regions:
        A, b = region.A(), region.b()
        lb = np.full(3, -np.inf)
        ub = np.full(3, np.inf)
        for i in range(A.shape[0]):
            row = A[i]
            nz = np.nonzero(row)[0]
            if len(nz) != 1:
                continue
            dim = nz[0]
            if row[dim] > 0:
                ub[dim] = min(ub[dim], b[i] / row[dim])
            else:
                lb[dim] = max(lb[dim], b[i] / row[dim])
        if np.any(np.isinf(lb)) or np.any(np.isinf(ub)) or np.any(ub <= lb):
            tightened.append(region)
            continue
        carved = tighten_box_uniform(lb, ub, margin)
        if carved is not None:
            tightened.append(HPolyhedron.MakeBox(*carved))
    return tightened


def inflation_margin(sigma: float, delta_k: float, num_faces: int = 6) -> float:
    """Uniform free-space tightening from isotropic obstacle-face inflation."""
    return sigma * float(norm.ppf(1.0 - delta_k))


def tighten_box_uniform(
    lb: np.ndarray,
    ub: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Shrink a free-space box uniformly by the obstacle inflation margin."""
    if margin <= 0.0:
        return lb, ub
    lb2 = np.asarray(lb, dtype=float) + margin
    ub2 = np.asarray(ub, dtype=float) - margin
    if np.any(lb2 >= ub2):
        return None
    return lb2, ub2


def _append_region_box(
    regions: list[HPolyhedron],
    lb,
    ub,
    *,
    inflation_margin_m: float = 0.0,
) -> None:
    lb_arr = np.asarray(lb, dtype=float)
    ub_arr = np.asarray(ub, dtype=float)
    if inflation_margin_m > 0.0:
        tightened = tighten_box_uniform(lb_arr, ub_arr, inflation_margin_m)
        if tightened is None:
            return
        lb_arr, ub_arr = tightened
    if np.all(lb_arr < ub_arr):
        regions.append(HPolyhedron.MakeBox(lb_arr, ub_arr))


def build_regions_from_obstacles(
    grid: np.ndarray,
    start: np.ndarray,
    indoor_edges: list,
    outdoor_edges: list,
    *,
    sigma: float = 0.0,
    delta_k: float = 0.05,
    seed: int | None = None,
    tree_probability: float = 0.7,
    indoor_options: dict | None = None,
    wall_options: dict | None = None,
) -> list[HPolyhedron]:
    """
    Build GCS free-space convex sets after inflating 3D box obstacles.

    Obstacle faces are inflated by inflation_margin(sigma, delta_k). Free-space
    candidate boxes follow compile_sdf topology and are tightened uniformly by
    that margin, preserving door/passage connectivity.
    """
    from quadrotor.building_generation import DEFAULT_INDOOR_OPTIONS, DEFAULT_WALL_OPTIONS

    if seed is not None:
        np.random.seed(seed)

    if indoor_options is None:
        indoor_options = DEFAULT_INDOOR_OPTIONS
    if wall_options is None:
        wall_options = DEFAULT_WALL_OPTIONS

    wall_offset = WALL_HALF_THICKNESS + QUAD_RADIUS
    margin_m = inflation_margin(sigma, delta_k)
    z_min = QUAD_RADIUS
    z_max = ROOM_HEIGHT - QUAD_RADIUS
    x_cells, y_cells = grid.shape
    regions: list[HPolyhedron] = []

    def _try_box(lb, ub):
        _append_region_box(regions, lb, ub, inflation_margin_m=margin_m)

    exterior = [
        ([-2.5, -2.5, z_min], [x_cells * CELL_SIZE + 7.5, 2.5 - wall_offset, z_max]),
        ([-2.5, 2.5 - wall_offset, z_min], [2.5 - wall_offset, y_cells * CELL_SIZE + 2.5 + wall_offset, z_max]),
        (
            [x_cells * CELL_SIZE + 2.5 + wall_offset, 2.5 - wall_offset, z_min],
            [x_cells * CELL_SIZE + 7.5, y_cells * CELL_SIZE + 2.5 + wall_offset, z_max],
        ),
        ([-2.5, y_cells * CELL_SIZE + 2.5 + wall_offset, z_min], [x_cells * CELL_SIZE + 7.5, y_cells * CELL_SIZE + 7.5, z_max]),
    ]
    for lb, ub in exterior:
        _try_box(lb, ub)

    for i in range(-1, x_cells + 1):
        for j in range(-1, y_cells + 1):
            xy = (np.array([i, j]) - start) * CELL_SIZE
            if i >= 0 and j >= 0 and i < x_cells and j < y_cells and grid[i, j] > 0.5:
                half = HALF_CELL - wall_offset
                if half > 0:
                    _try_box(
                        [xy[0] - half, xy[1] - half, z_min],
                        [xy[0] + half, xy[1] + half, z_max],
                    )
            else:
                if i < 0 or j < 0 or i == x_cells or j == y_cells:
                    continue
                lb = [xy[0] - HALF_CELL, xy[1] - HALF_CELL, z_min]
                ub = [xy[0] + HALF_CELL, xy[1] + HALF_CELL, z_max]
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
                    _try_box(lb, ub)
                else:
                    tree_pose = xy + 3.0 * np.random.rand(2) - 1.5
                    _try_box(lb, [ub[0], tree_pose[1] - 0.5, ub[2]])
                    _try_box([lb[0], tree_pose[1] - 0.5, lb[2]], [tree_pose[0] - 0.5, tree_pose[1] + 0.5, ub[2]])
                    _try_box([tree_pose[0] + 0.5, tree_pose[1] - 0.5, lb[2]], [ub[0], tree_pose[1] + 0.5, ub[2]])
                    _try_box([lb[0], tree_pose[1] + 0.5, lb[2]], ub)

    door_width = 1.25 - 2 * QUAD_RADIUS
    door_height = 2 - QUAD_RADIUS
    window_width = 1.5 - 2 * QUAD_RADIUS
    window_offset = 1.25
    window_z_min = 0.75 + QUAD_RADIUS
    window_z_max = 2.25 - QUAD_RADIUS
    half_wall_offset = 1.25

    key_options = list(wall_options.keys())
    probs = np.array(list(wall_options.values()), dtype=float)
    probs = probs / np.sum(probs)
    shuffled_outdoor = list(outdoor_edges)
    np.random.shuffle(shuffled_outdoor)
    for k, (e1, e2) in enumerate(shuffled_outdoor):
        sdf_key = np.random.choice(key_options, p=probs)
        while k == 0 and "door" not in sdf_key and "window" not in sdf_key:
            sdf_key = np.random.choice(key_options, p=probs)

        delta = np.asarray(e2) - np.asarray(e1)
        theta = np.arctan2(delta[0], delta[1])
        midpoint = (np.asarray(e1) + np.asarray(e2)) / 2.0
        midpoint = (midpoint - start) * CELL_SIZE

        if "door" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + door_width / 2.0 * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + door_width / 2.0 * np.cos(theta))
            _try_box(
                [midpoint[0] - dx, midpoint[1] - dy, z_min],
                [midpoint[0] + dx, midpoint[1] + dy, door_height],
            )
        elif "left_window" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + window_width / 2.0 * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + window_width / 2.0 * np.cos(theta))
            _try_box(
                [
                    midpoint[0] - dx + window_offset * np.sin(theta),
                    midpoint[1] - dy + window_offset * np.cos(theta),
                    window_z_min,
                ],
                [
                    midpoint[0] + dx + window_offset * np.sin(theta),
                    midpoint[1] + dy + window_offset * np.cos(theta),
                    window_z_max,
                ],
            )
        elif "right_window" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + window_width / 2.0 * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + window_width / 2.0 * np.cos(theta))
            _try_box(
                [
                    midpoint[0] - dx - window_offset * np.sin(theta),
                    midpoint[1] - dy - window_offset * np.cos(theta),
                    window_z_min,
                ],
                [
                    midpoint[0] + dx - window_offset * np.sin(theta),
                    midpoint[1] + dy - window_offset * np.cos(theta),
                    window_z_max,
                ],
            )
        elif "windows" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + window_width / 2.0 * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + window_width / 2.0 * np.cos(theta))
            _try_box(
                [
                    midpoint[0] - dx + window_offset * np.sin(theta),
                    midpoint[1] - dy + window_offset * np.cos(theta),
                    window_z_min,
                ],
                [
                    midpoint[0] + dx + window_offset * np.sin(theta),
                    midpoint[1] + dy + window_offset * np.cos(theta),
                    window_z_max,
                ],
            )
            _try_box(
                [
                    midpoint[0] - dx - window_offset * np.sin(theta),
                    midpoint[1] - dy - window_offset * np.cos(theta),
                    window_z_min,
                ],
                [
                    midpoint[0] + dx - window_offset * np.sin(theta),
                    midpoint[1] + dy - window_offset * np.cos(theta),
                    window_z_max,
                ],
            )

    key_options = list(indoor_options.keys())
    probs = np.array(list(indoor_options.values()), dtype=float)
    probs = probs / np.sum(probs)
    shuffled_indoor = list(indoor_edges)
    np.random.shuffle(shuffled_indoor)
    for e1, e2 in shuffled_indoor:
        sdf_key = np.random.choice(key_options, p=probs)
        delta = np.asarray(e2) - np.asarray(e1)
        theta = np.arctan2(delta[0], delta[1])
        midpoint = (np.asarray(e1) + np.asarray(e2)) / 2.0
        midpoint = (midpoint - start) * CELL_SIZE

        if sdf_key == "":
            dx = np.abs(wall_offset * np.cos(theta) + (HALF_CELL - wall_offset) * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + (HALF_CELL - wall_offset) * np.cos(theta))
            _try_box(
                [midpoint[0] - dx, midpoint[1] - dy, z_min],
                [midpoint[0] + dx, midpoint[1] + dy, z_max],
            )
        elif "door" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + door_width / 2.0 * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + door_width / 2.0 * np.cos(theta))
            _try_box(
                [midpoint[0] - dx, midpoint[1] - dy, z_min],
                [midpoint[0] + dx, midpoint[1] + dy, door_height],
            )
        elif "mirror" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + (1.25 - wall_offset) * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + (1.25 - wall_offset) * np.cos(theta))
            _try_box(
                [
                    midpoint[0] - dx - half_wall_offset * np.sin(theta),
                    midpoint[1] - dy + half_wall_offset * np.cos(theta),
                    z_min,
                ],
                [
                    midpoint[0] + dx - half_wall_offset * np.sin(theta),
                    midpoint[1] + dy + half_wall_offset * np.cos(theta),
                    z_max,
                ],
            )
        elif "horizontal" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + (1.25 - wall_offset) * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + (1.25 - wall_offset) * np.cos(theta))
            _try_box(
                [
                    midpoint[0] - dx + half_wall_offset * np.sin(theta),
                    midpoint[1] - dy - half_wall_offset * np.cos(theta),
                    z_min,
                ],
                [
                    midpoint[0] + dx + half_wall_offset * np.sin(theta),
                    midpoint[1] + dy - half_wall_offset * np.cos(theta),
                    z_max,
                ],
            )
        elif "vertical" in sdf_key:
            dx = np.abs(wall_offset * np.cos(theta) + (HALF_CELL - wall_offset) * np.sin(theta))
            dy = np.abs(wall_offset * np.sin(theta) + (HALF_CELL - wall_offset) * np.cos(theta))
            _try_box(
                [midpoint[0] - dx, midpoint[1] - dy, 1.7],
                [midpoint[0] + dx, midpoint[1] + dy, z_max],
            )

    return regions
