"""
Print a compact tree view of a generated GCS dataset HDF5 file.

Example:
  python -m planning_through_contact.dataset.inspect_h5 \
    planning_through_contact/dataset/data/sugar_box/gcs_solutions.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _format_attrs(attrs: Any) -> str:
    if len(attrs) == 0:
        return ""
    pairs = [f"{k}={attrs[k]!r}" for k in attrs.keys()]
    return " attrs: " + ", ".join(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "h5_path",
        type=str,
        nargs="?",
        default="planning_through_contact/dataset/data/sugar_box/gcs_solutions.h5",
    )
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--max_items", type=int, default=20)
    args = parser.parse_args()

    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Missing dependency `h5py`.") from e

    h5_path = Path(args.h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    def visit(name: str, obj: Any) -> None:
        depth = 0 if name == "" else name.count("/")
        if depth > args.max_depth:
            return

        leaf = "/" if name == "" else name
        indent = "  " * depth
        if isinstance(obj, h5py.Group):
            keys = list(obj.keys())
            print(f"{indent}{leaf}/ group children={len(keys)}{_format_attrs(obj.attrs)}")
            if len(keys) > args.max_items:
                print(f"{indent}  ... showing first {args.max_items} children")
        else:
            print(
                f"{indent}{leaf} dataset shape={obj.shape} dtype={obj.dtype}"
                f"{_format_attrs(obj.attrs)}"
            )

    with h5py.File(h5_path, "r") as h5:
        visit("", h5)
        h5.visititems(visit)


if __name__ == "__main__":
    main()
