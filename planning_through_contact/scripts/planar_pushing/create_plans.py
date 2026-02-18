import argparse

from planning_through_contact.experiments.utils import (
    create_output_folder,
    get_default_experiment_plans,
    get_default_plan_config,
    get_default_solver_params,
)
from planning_through_contact.planning.planar.utils import create_plan

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        help="Random seed for generating trajectories",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--traj",
        help="(Optional) specify a specific trajectory number to generate, with the given random seed.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--body",
        help="Which slider body to use.",
        type=str,
        default="sugar_box",
    )
    parser.add_argument(
        "--num",
        help="Number of trajectories to generate",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--save_relaxed",
        help="Also save the relaxed trajectory (which may not be feasible).",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        help="Debug mode. Will print additional information, including solver output.",
        action="store_true",
    )
    parser.add_argument(
        "--interpolate",
        help="Interpolate trajectory in video (does not impact the plans themselves).",
        action="store_true",
    )
    parser.add_argument(
        "--output_dir",
        help="High-level output directory for plans. If the folder doesn't exist, it will be created. A timestamped \
        subfolder is created within 'output_dir'.",
        type=str,
        default="trajectories",
    )
    parser.add_argument(
        "--save_graph_edge_labels",
        default=True,
        help="Save a graph visualization plus a 0/1 label for every directed edge (1 on chosen path, else 0).",
        action="store_true",
    )
    args = parser.parse_args()
    seed = args.seed
    traj_number = args.traj
    debug = args.debug
    rounding = True
    interpolate = args.interpolate
    slider_type = args.body
    num_trajs = args.num
    output_dir = args.output_dir
    save_graph_edge_labels = args.save_graph_edge_labels
    save_relaxed = args.save_relaxed or debug

    if not debug:
        import logging

        # Set the logging level to WARNING to suppress INFO-level messages
        logging.getLogger("drake").setLevel(logging.WARNING)

    pusher_radius = 0.015

    use_case = "normal"

    print("For help on how to use this script, run `--help`")

    print(
        f'Generating {num_trajs} different random trajectories for slider type "{slider_type}"'
    )
    if not debug:
        print("For more detailed information, run with the `--debug` flag.")

    print("")

    config = get_default_plan_config(
        slider_type=slider_type,
        pusher_radius=pusher_radius,
        use_case=use_case,
    )
    solver_params = get_default_solver_params(debug, clarabel=False)
    folder_name = create_output_folder(output_dir, slider_type, traj_number)
    plans = get_default_experiment_plans(seed, num_trajs, config)

    print("")
    print(f"Output folder: {folder_name}")

    if traj_number is not None:
        plans_to_plan_for = [plans[traj_number]]
    else:
        plans_to_plan_for = plans

    print("Generating plans...")
    from tqdm import tqdm

    for idx, plan in enumerate(tqdm(plans_to_plan_for)):
        create_plan(
            plan,
            config,
            solver_params,
            output_folder=folder_name,
            debug=debug,
            output_name=f"traj_{idx}",
            save_video=True,
            save_traj=True,
            animation_lims=None,
            interpolate_video=interpolate,
            do_rounding=rounding,
            save_relaxed=save_relaxed,
            save_graph_edge_labels=save_graph_edge_labels,
        )
