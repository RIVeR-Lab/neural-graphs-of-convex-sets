"""Resolve paths for IIWA shelf manipulation assets."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIPULATION_ROOT = REPO_ROOT / "manipulation"
SCENE_MODELS_ROOT = MANIPULATION_ROOT / "scene_models"
VENDOR_MODELS_ROOT = MANIPULATION_ROOT / "vendor" / "drake" / "manipulation" / "models"
IIWA_MODELS_ROOT = VENDOR_MODELS_ROOT / "iiwa_description"

DEFAULT_DIRECTIVES = MANIPULATION_ROOT / "models" / "iiwa14_spheres_collision_welded_gripper.yaml"
DEFAULT_REGIONS_PATH = MANIPULATION_ROOT / "data" / "IRIS.reg"
DEFAULT_DATASET_DIR = MANIPULATION_ROOT / "dataset"
DEFAULT_CHECKPOINTS_ROOT = MANIPULATION_ROOT / "checkpoints"
DEFAULT_OUTPUT_DIR = MANIPULATION_ROOT / "results" / "shelf_viz"


def iiwa_urdf_path() -> Path:
    return IIWA_MODELS_ROOT / "urdf" / "iiwa14_spheres_collision.urdf"


def wsg_sdf_path() -> Path:
    return MANIPULATION_ROOT / "models" / "schunk_wsg_50_welded_fingers.sdf"


def manipulation_models_ready() -> bool:
    return iiwa_urdf_path().is_file()


def manipulation_models_hint() -> str:
    return (
        "IIWA manipulation models not found.\n"
        "Run: bash scripts/setup_iiwa_models.sh\n"
        f"Expected: {iiwa_urdf_path()}"
    )


def register_package_maps(parser) -> None:
    """Register manipulation scene + robot package paths for model directives."""
    parser.package_map().Add("manipulation_scene", str(SCENE_MODELS_ROOT))
    if not manipulation_models_ready():
        raise FileNotFoundError(manipulation_models_hint())
    parser.package_map().Add("manipulation_models", str(VENDOR_MODELS_ROOT))
    parser.package_map().Add("manipulation_assets", str(MANIPULATION_ROOT / "models"))
