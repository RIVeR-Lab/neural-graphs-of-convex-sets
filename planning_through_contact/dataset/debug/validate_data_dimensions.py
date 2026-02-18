"""
Validate and print dimensions of stored data (H5 + node features + plan index).

Symbols follow ICRA2027_Ananya.pdf:
  - x_v ∈ R^(11 + 2*NF_max): node features (type 2, contact-geom 5, normals 4, transition 2*NF_max)
  - g = [g_pose (6) || g_entry (num_entries)] ∈ R^g_dim: global conditioning
  - y ∈ {0,1}^E: binary edge labels
  - φ* ∈ R^E: relaxed optimal edge flows
  - N: number of nodes, E: number of edges

Run from repo root, e.g.:
  python planning_through_contact/dataset/debug/validate_data_dimensions.py
  python planning_through_contact/dataset/debug/validate_data_dimensions.py --h5_path path/to/solutions.h5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    import h5py  # type: ignore
except ModuleNotFoundError:
    h5py = None  # type: ignore


def _decode_str(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8")
    return str(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate data dimensions (PDF symbols).")
    parser.add_argument(
        "--h5_path",
        type=str,
        default="planning_through_contact/dataset/data/gcs_solutions.h5",
        help="Solutions HDF5.",
    )
    parser.add_argument(
        "--node_features_csv",
        type=str,
        default="planning_through_contact/dataset/data/node_features.csv",
        help="Static node features CSV.",
    )
    parser.add_argument(
        "--plan_index_csv",
        type=str,
        default="planning_through_contact/dataset/data/global_features.csv",
        help="Plan index CSV for expected g_dim (default: same dir as data).",
    )
    args = parser.parse_args()

    h5_path = Path(args.h5_path)
    node_csv = Path(args.node_features_csv)
    plan_index = Path(args.plan_index_csv)

    if not node_csv.exists():
        print(f"Error: node features not found: {node_csv}")
        return
    if not h5_path.exists():
        print(f"Error: H5 not found: {h5_path}")
        return

    # ---- Node features (PDF: x_v ∈ R^(11 + 2*NF_max)) ----
    with node_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("Error: empty node features CSV")
            return
        x_cols = [c for c in reader.fieldnames if c.startswith("x_")]
        rows = list(reader)

    N_static = len(rows)
    x_dim = len(x_cols)
    # PDF: 11 + 2*NF_max => NF_max = (x_dim - 11) / 2 if consistent
    nf_max_implied = (x_dim - 11) // 2 if x_dim >= 11 else None

    print("=" * 60)
    print("Data dimension validation (ICRA2027_Ananya.pdf symbols)")
    print("=" * 60)
    print()
    print("1. Node features (node_features.csv)")
    print(f"   N_static (vertices in CSV)     = {N_static}")
    print(f"   x_dim (x_0..x_{{x_dim-1}})        = {x_dim}   [PDF: x_v ∈ R^(11 + 2*NF_max)]")
    if nf_max_implied is not None and (x_dim - 11) % 2 == 0:
        print(f"   implied NF_max                 = {nf_max_implied}")
    print()

    # ---- Plan index: expected g_dim (PDF: g_pose 6 + g_entry num_entries) ----
    expected_g_dim = None
    num_entries = None
    if plan_index.exists():
        with plan_index.open(newline="") as f:
            r = csv.DictReader(f)
            if r.fieldnames:
                g_entry_cols = sorted([k for k in r.fieldnames if k.startswith("g_entry_")])
                num_entries = len(g_entry_cols)
                expected_g_dim = 6 + num_entries  # g_pose (6) + g_entry
        print("2. Plan index (global_features.csv) – expected g")
        print(f"   num_entries (g_entry_* cols)   = {num_entries}   [PDF: g_entry one-hot]")
        print(f"   expected g_dim                = 6 + num_entries = {expected_g_dim}   [PDF: g ∈ R^g_dim]")
        print()
    else:
        print("2. Plan index: not found at default path (use --plan_index_csv to point to global_features.csv)")
        print()

    # ---- H5 samples ----
    if h5py is None:
        print("Error: h5py required. pip install h5py")
        return

    print("3. H5 samples (per-sample dimensions)")
    with h5py.File(h5_path, "r") as h5:
        if "samples" not in h5:
            print(f"   Error: no 'samples' group in {h5_path}")
            return
        samples = h5["samples"]
        keys = sorted(samples.keys(), key=lambda k: int(k) if k.isdigit() else 0)
        g_dims = []
        for k in keys:
            grp = samples[k]
            plan_id = int(grp.attrs.get("plan_id", k))
            split = _decode_str(grp.attrs.get("split", ""))
            g = grp["g"]
            g_len = g.shape[0] if g.ndim >= 1 else 0
            g_dims.append(g_len)
            edge_u = grp["edge_u"]
            E = edge_u.shape[0] if edge_u.ndim >= 1 else 0
            y = grp["y"]
            phi_star = grp["phi_star"]
            print(f"   plan_id={plan_id}  split={split!r}  g ∈ R^{g_len}  |E|={E}  y ∈ {{0,1}}^{E}  φ* ∈ R^{E}")

        print()
        print("4. Consistency")
        if g_dims:
            unique_g = set(g_dims)
            if len(unique_g) == 1:
                print(f"   g_dim: all samples have g_dim = {g_dims[0]}  OK")
            else:
                print(f"   g_dim: MISMATCH  values = {sorted(unique_g)}  (all should match expected g_dim)")
            if expected_g_dim is not None and unique_g != {expected_g_dim}:
                print(f"   expected g_dim from plan index = {expected_g_dim}")
        print("=" * 60)


if __name__ == "__main__":
    main()
