"""Default checkpoint locations for quadrotor models."""

from __future__ import annotations

from pathlib import Path

CHECKPOINTS_ROOT = Path("quadrotor/checkpoints")
PLANNERS = ("convex", "nonconvex")


def planner_tag(planner: str) -> str:
    if planner not in PLANNERS:
        raise ValueError(f"planner must be one of {PLANNERS}, got {planner!r}")
    return f"quadrotor_{planner}"


def planner_ckpt_dir(planner: str, *, root: Path | str = CHECKPOINTS_ROOT) -> Path:
    return Path(root) / planner_tag(planner)


def flow_ckpt_path(
    planner: str,
    *,
    ckpt_dir: Path | str | None = None,
    root: Path | str = CHECKPOINTS_ROOT,
) -> Path:
    directory = Path(ckpt_dir) if ckpt_dir is not None else planner_ckpt_dir(planner, root=root)
    tag = planner_tag(planner)
    return directory / f"{tag}_flow_gnn.ckpt"


def ranknet_ckpt_path(
    planner: str,
    *,
    ckpt_dir: Path | str | None = None,
    root: Path | str = CHECKPOINTS_ROOT,
) -> Path:
    directory = Path(ckpt_dir) if ckpt_dir is not None else planner_ckpt_dir(planner, root=root)
    tag = planner_tag(planner)
    return directory / f"{tag}_ranknet.ckpt"


def add_planner_checkpoint_args(parser, *, default_planner: str = "convex") -> None:
  """Register --planner, --ckpt_dir, --flow_ckpt, --ranknet_ckpt on an ArgumentParser."""
  parser.add_argument("--planner", choices=PLANNERS, default=default_planner)
  parser.add_argument(
      "--ckpt_dir",
      default=None,
      help="Per-planner checkpoint directory (default: quadrotor/checkpoints/quadrotor_<planner>).",
  )
  parser.add_argument("--flow_ckpt", default=None, help="Override flow GNN checkpoint file.")
  parser.add_argument("--ranknet_ckpt", default=None, help="Override RankNet checkpoint file.")


def resolve_flow_ckpt(args) -> str:
    if getattr(args, "flow_ckpt", None):
        return str(args.flow_ckpt)
    return str(flow_ckpt_path(args.planner, ckpt_dir=args.ckpt_dir))


def resolve_ranknet_ckpt(args) -> str:
    if getattr(args, "ranknet_ckpt", None):
        return str(args.ranknet_ckpt)
    return str(ranknet_ckpt_path(args.planner, ckpt_dir=args.ckpt_dir))
