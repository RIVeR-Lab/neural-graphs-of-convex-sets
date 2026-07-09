"""Inflate shelf/bin/table collision boxes for manipulation IRIS scenes."""

from __future__ import annotations

import shutil
from pathlib import Path

import lxml.etree as ET

from manipulation.paths import INFLATED_SCENE_ROOT, SCENE_MODELS_ROOT

SCENE_MODELS = ("bin", "shelves", "table")


def _inflate_collision_boxes_in_tree(root: ET.Element, margin: float) -> None:
    for collision in root.iter("collision"):
        for size_el in collision.findall(".//size"):
            if size_el.text is None:
                continue
            parts = [float(x) for x in size_el.text.split()]
            if len(parts) != 3:
                continue
            size_el.text = " ".join(f"{p + 2.0 * margin:.8g}" for p in parts)


def inflate_sdf_file(src: Path, dst: Path, margin: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(str(src))
    _inflate_collision_boxes_in_tree(tree.getroot(), margin)
    tree.write(str(dst), pretty_print=True)


def prepare_inflated_scene_models(margin: float) -> Path:
    if margin <= 0.0:
        raise ValueError("margin must be positive for inflated scene models")

    if INFLATED_SCENE_ROOT.exists():
        shutil.rmtree(INFLATED_SCENE_ROOT)

    for name in SCENE_MODELS:
        src_dir = SCENE_MODELS_ROOT / name
        dst_dir = INFLATED_SCENE_ROOT / name
        shutil.copytree(src_dir, dst_dir)
        sdf_name = "table_wide.sdf" if name == "table" else f"{name}.sdf"
        inflate_sdf_file(dst_dir / sdf_name, dst_dir / sdf_name, margin)

    return INFLATED_SCENE_ROOT


def inflated_directives_yaml(nominal_yaml: Path, margin: float) -> Path:
    if margin <= 0.0:
        return nominal_yaml

    prepare_inflated_scene_models(margin)
    text = nominal_yaml.read_text()
    replacements = {
        "package://manipulation_scene/shelves/shelves.sdf":
            "package://manipulation_scene_inflated/shelves/shelves.sdf",
        "package://manipulation_scene/bin/bin.sdf":
            "package://manipulation_scene_inflated/bin/bin.sdf",
        "package://manipulation_scene/table/table_wide.sdf":
            "package://manipulation_scene_inflated/table/table_wide.sdf",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    out = nominal_yaml.parent / f"{nominal_yaml.stem}_inflated.yaml"
    out.write_text(text)
    return out
