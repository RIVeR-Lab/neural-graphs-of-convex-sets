#!/usr/bin/env python3
"""Record a Meshcat static HTML animation to MP4 at a fixed 16:9 viewport."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Meshcat HTML to MP4.")
    parser.add_argument("html", type=Path, help="Path to Meshcat static HTML file.")
    parser.add_argument("output", type=Path, help="Output MP4 path.")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--duration", type=float, required=True, help="Recording length in seconds.")
    parser.add_argument("--warmup", type=float, default=1.0, help="Seconds to wait before recording.")
    parser.add_argument("--tail", type=float, default=0.5, help="Extra seconds after animation ends.")
    parser.add_argument(
        "--trim-start",
        type=float,
        default=0.0,
        help="Skip this many seconds from the start of the captured video.",
    )
    args = parser.parse_args()

    html_path = args.html.resolve()
    if not html_path.exists():
        raise FileNotFoundError(html_path)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Playwright is required. Install with:\n"
            "  pip install playwright && playwright install chromium"
        ) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_ms = int((args.warmup + args.trim_start + args.duration + args.tail) * 1000)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        with tempfile.TemporaryDirectory(prefix="meshcat_video_") as tmpdir:
            context = browser.new_context(
                viewport={"width": args.width, "height": args.height},
                record_video_dir=tmpdir,
                record_video_size={"width": args.width, "height": args.height},
            )
            page = context.new_page()
            page.goto(html_path.as_uri(), wait_until="load")
            page.wait_for_timeout(total_ms)
            video_path = page.video.path() if page.video else None
            context.close()
            browser.close()

            if video_path is None:
                raise RuntimeError("Playwright did not produce a video recording.")

            webm_path = Path(video_path)
            trimmed_webm = webm_path.with_name("trimmed.webm")
            start_s = max(0.0, args.warmup - 0.15 + args.trim_start)
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{start_s:.3f}",
                    "-i",
                    str(webm_path),
                    "-t",
                    f"{args.duration:.3f}",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "18",
                    str(args.output),
                ],
                check=True,
            )
            print(f"Saved MP4 → {args.output}")


if __name__ == "__main__":
    main()
