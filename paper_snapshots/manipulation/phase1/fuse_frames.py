#!/usr/bin/env python3
"""Equal-weight blend of selected phase1 frames (default: 00001, 00011, 00016)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_FRAMES = ("00001.png", "00011.png", "00016.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", nargs="*", default=list(DEFAULT_FRAMES))
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    paths = [ROOT / f for f in args.frames]
    imgs = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) for p in paths]
    fused = np.stack(imgs, axis=0).mean(axis=0).astype(np.uint8)

    stem = "_".join(p.stem for p in paths)
    out = args.output or ROOT / f"fused_{stem}.png"
    Image.fromarray(fused).save(out)
    pct = 100.0 / len(paths)
    print(f"Saved {out} ({len(paths)} frames @ {pct:.1f}% each)")


if __name__ == "__main__":
    main()
