from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VALID_BODIES = ("sugar_box", "tee")


def validate_body(body: str) -> str:
    if body not in VALID_BODIES:
        raise ValueError(f"body must be one of {VALID_BODIES}, got {body!r}")
    return body


def data_dir_for_body(body: str, root: str | Path = "planning_through_contact/dataset/data") -> Path:
    return Path(root) / validate_body(body)


@dataclass(frozen=True)
class BodyDatasetPaths:
    body: str
    data_dir: Path
    h5_path: Path
    node_features_csv: Path
    global_features_csv: Path


def dataset_paths_for_body(
    body: str,
    *,
    data_root: str | Path = "planning_through_contact/dataset/data",
    data_dir: str | Path | None = None,
) -> BodyDatasetPaths:
    body = validate_body(body)
    resolved_data_dir = Path(data_dir) if data_dir is not None else data_dir_for_body(body, data_root)
    return BodyDatasetPaths(
        body=body,
        data_dir=resolved_data_dir,
        h5_path=resolved_data_dir / "gcs_solutions.h5",
        node_features_csv=resolved_data_dir / "node_features.csv",
        global_features_csv=resolved_data_dir / "global_features.csv",
    )


def flow_checkpoint_name(body: str) -> str:
    return f"{validate_body(body)}_flow"


def ranker_checkpoint_name(body: str) -> str:
    return f"{validate_body(body)}_ranker"
