"""
Generate a random building, sample random start/goal poses, run vanilla GCS,
visualize with Drake meshcat, and export a self-contained HTML file.
"""

import argparse
import numpy as np

from pydrake.examples import QuadrotorGeometry
from pydrake.geometry import MeshcatVisualizer, StartMeshcat
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.parsing import Parser
from pydrake.solvers import MosekSolver
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

from quadrotor.building_generation import generate_grid_world, compile_sdf, MODELS_DIR
from quadrotor.helpers import build_bezier_gcs, FlatnessInverter

DELTA = 0.3   # margin from box boundaries when sampling
D_MIN = 3     # minimum graph-hop distance between start and goal regions


def box_bounds(region):
    """Extract axis-aligned lower/upper bounds from an HPolyhedron."""
    A, b = region.A(), region.b()
    lb = np.full(3, -np.inf)
    ub = np.full(3, np.inf)
    for i in range(A.shape[0]):
        row = A[i]
        nz = np.nonzero(row)[0]
        if len(nz) == 1:
            dim = nz[0]
            if row[dim] > 0:
                ub[dim] = min(ub[dim], b[i] / row[dim])
            else:
                lb[dim] = max(lb[dim], b[i] / row[dim])
    return lb, ub


def region_volume(lb, ub):
    dims = ub - lb
    dims = np.where(np.isinf(dims), 1.0, dims)
    return float(np.prod(np.maximum(dims, 0)))


def sample_point_in_region(region, rng):
    """Sample uniformly inside region with DELTA margin from each face."""
    lb, ub = box_bounds(region)
    lo = lb + DELTA
    hi = ub - DELTA
    if np.any(lo >= hi):
        return None
    return rng.uniform(lo, hi)


def build_adjacency(regions):
    """Build adjacency list (region index → list of overlapping region indices)."""
    n = len(regions)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if regions[i].IntersectsWith(regions[j]):
                adj[i].append(j)
                adj[j].append(i)
    return adj


def bfs_distance(adj, src):
    """BFS from src, return dict of {region_idx: hop_distance}."""
    dist = {src: 0}
    queue = [src]
    while queue:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                queue.append(nb)
    return dist


def sample_start_goal(regions, rng, max_attempts=200):
    """
    Sample start and goal poses following the PDF Section 2 procedure:
    - Pick region proportional to volume, sample uniformly inside with margin δ.
    - Goal must be in a different region, same connected component,
      graph distance >= D_MIN hops.
    - Returns (start_pose, goal_pose, start_region_idx, goal_region_idx).
    """
    bounds = [box_bounds(r) for r in regions]
    vols = np.array([region_volume(lb, ub) for lb, ub in bounds])
    vols = np.maximum(vols, 0)
    if vols.sum() == 0:
        raise RuntimeError("All regions have zero volume.")
    probs = vols / vols.sum()

    print("  Building region adjacency graph...")
    adj = build_adjacency(regions)

    for attempt in range(max_attempts):
        # Sample start region
        s_idx = rng.choice(len(regions), p=probs)
        s_pt = sample_point_in_region(regions[s_idx], rng)
        if s_pt is None:
            continue

        # BFS from start region
        dist = bfs_distance(adj, s_idx)

        # Filter candidate goal regions: different region, reachable, >= D_MIN hops
        candidates = [i for i, d in dist.items() if i != s_idx and d >= D_MIN]
        if not candidates:
            continue

        # Sample goal region proportional to volume among candidates
        cand_vols = vols[candidates]
        if cand_vols.sum() == 0:
            continue
        cand_probs = cand_vols / cand_vols.sum()
        g_idx = candidates[rng.choice(len(candidates), p=cand_probs)]
        g_pt = sample_point_in_region(regions[g_idx], rng)
        if g_pt is None:
            continue

        print(f"  Start region {s_idx} @ {s_pt.round(2)}, "
              f"goal region {g_idx} @ {g_pt.round(2)} "
              f"(graph dist={dist[g_idx]} hops, attempt {attempt+1})")
        return s_pt, g_pt, s_idx, g_idx

    raise RuntimeError(f"Could not find valid start/goal after {max_attempts} attempts.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for both building generation and query sampling.")
    parser.add_argument("--query-seed", type=int, default=None,
                        help="Separate RNG seed for start/goal sampling (default: same as --seed).")
    parser.add_argument("--grid", type=int, nargs=2, default=[3, 3], metavar=("ROWS", "COLS"))
    parser.add_argument("--output", type=str, default="quadrotor_demo.html")
    default_sdf = str(MODELS_DIR / "room_gen" / "building.sdf")
    parser.add_argument("--sdf", type=str, default=default_sdf)
    args = parser.parse_args()

    query_seed = args.query_seed if args.query_seed is not None else args.seed
    rng = np.random.default_rng(query_seed)

    shape = tuple(args.grid)
    grid_start = np.array([-1, -1])
    grid_goal = np.array([2, 1])

    # --- Build environment ---
    print(f"Generating {shape[0]}x{shape[1]} building (seed={args.seed})...")
    grid, indoor_edges, outdoor_edges = generate_grid_world(
        shape=shape, start=grid_start, goal=grid_goal, seed=args.seed)
    regions = compile_sdf(
        args.sdf, grid, grid_start, grid_goal, indoor_edges, outdoor_edges, seed=args.seed)
    print(f"  {len(regions)} convex regions.")

    # --- Sample start/goal ---
    print(f"Sampling start/goal (query_seed={query_seed})...")
    start_pose, goal_pose, s_idx, g_idx = sample_start_goal(regions, rng)

    # --- Solve GCS ---
    print("Solving GCS...")
    solver = MosekSolver()
    gcs = build_bezier_gcs(regions, solver)
    gcs.addSourceTarget(start_pose, goal_pose, zero_deriv_boundary=3)
    traj, results = gcs.SolvePath(rounding=True, verbose=False, preprocessing=True)

    if traj is None:
        print("GCS failed — no trajectory found. Try a different seed.")
        return

    cost = results.get("rounded_cost", float("nan"))
    print(f"  Cost={cost:.3f}, duration={traj.start_time():.2f}s → {traj.end_time():.2f}s")

    # --- Drake visualization ---
    print("Building Drake diagram...")
    meshcat = StartMeshcat()
    meshcat.SetProperty("/Grid", "visible", False)
    meshcat.SetProperty("/Axes", "visible", False)
    meshcat.SetProperty("/Lights/AmbientLight/<object>", "intensity", 0.8)
    meshcat.SetProperty("/Lights/PointLightNegativeX/<object>", "intensity", 0)
    meshcat.SetProperty("/Lights/PointLightPositiveX/<object>", "intensity", 0)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

    parser_drake = Parser(plant, scene_graph)
    parser_drake.AddModels(args.sdf)
    plant.Finalize()

    meshcat_viz = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    animator = meshcat_viz.StartRecording()
    traj_system = builder.AddSystem(FlatnessInverter(traj, animator))
    QuadrotorGeometry.AddToBuilder(builder, traj_system.get_output_port(0), scene_graph)

    diagram = builder.Build()

    print("Simulating...")
    meshcat.Delete()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(0.0)
    simulator.AdvanceTo(traj.end_time() + 0.05)
    meshcat_viz.PublishRecording()

    html = meshcat.StaticHtml()
    with open(args.output, "w") as f:
        f.write(html)
    print(f"Saved to {args.output} — open it in your browser.")


if __name__ == "__main__":
    main()
