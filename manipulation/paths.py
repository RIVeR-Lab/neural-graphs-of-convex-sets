"""Resolve paths for IIWA shelf manipulation assets."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GCS_ROOT = REPO_ROOT / "gcs"
MANIPULATION_ROOT = REPO_ROOT / "manipulation"
VENDOR_MODELS_ROOT = MANIPULATION_ROOT / "vendor" / "drake" / "manipulation" / "models"
IIWA_MODELS_ROOT = VENDOR_MODELS_ROOT / "iiwa_description"

DEFAULT_DIRECTIVES = MANIPULATION_ROOT / "models" / "iiwa14_spheres_collision_welded_gripper.yaml"
DEFAULT_REGIONS_PATH = MANIPULATION_ROOT / "data" / "IRIS.reg"
DEFAULT_OUTPUT_DIR = MANIPULATION_ROOT / "results" / "shelf_viz"


def gcs_dir() -> Path:
    return GCS_ROOT


def find_gcs_model(relative_path: str) -> Path:
    assert relative_path.startswith("models/")
    return GCS_ROOT / relative_path


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
    """Register gcs + manipulation_models package paths for model directives."""
    parser.package_map().Add("gcs", str(gcs_dir()))
    if not manipulation_models_ready():
        raise FileNotFoundError(manipulation_models_hint())
    parser.package_map().Add("manipulation_models", str(VENDOR_MODELS_ROOT))
    parser.package_map().Add("manipulation_assets", str(MANIPULATION_ROOT / "models"))
