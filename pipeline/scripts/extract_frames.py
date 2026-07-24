"""Extract sharp frames from a capture video for upload to the Kaggle pipeline.

Runs ffmpeg at a fixed fps, then drops the blurriest frames (Laplacian
variance) until at most --max-frames remain. Output is a folder of JPEGs you
upload to Kaggle as a Dataset.

Usage:
    python pipeline/scripts/extract_frames.py capture.mp4 --out frames/ --fps 2 --max-frames 60

Requires: ffmpeg on PATH, `pip install opencv-python`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2


def sharpness(path: Path) -> float:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return float(cv2.Laplacian(img, cv2.CV_64F).var()) if img is not None else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=Path("frames"))
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--min-sharpness", type=float, default=5.0)
    args = parser.parse_args()

    if not args.video.exists():
        sys.exit(f"video not found: {args.video}")
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH (install with: brew install ffmpeg)")

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(args.video),
             "-vf", f"fps={args.fps}", f"{tmp}/%05d.jpg"],
            check=True,
        )
        frames = sorted(Path(tmp).glob("*.jpg"))
        print(f"extracted {len(frames)} frames at {args.fps} fps")

        scored = [(p, sharpness(p)) for p in frames]
        scored = [s for s in scored if s[1] >= args.min_sharpness]
        if len(scored) > args.max_frames:
            cutoff = sorted(s[1] for s in scored)[len(scored) - args.max_frames]
            scored = [s for s in scored if s[1] >= cutoff][: args.max_frames]

        args.out.mkdir(parents=True, exist_ok=True)
        for existing in args.out.glob("*.jpg"):
            existing.unlink()
        for i, (p, _) in enumerate(scored):
            shutil.copy(p, args.out / f"{i:05d}.jpg")

    print(f"kept {len(scored)} sharp frames -> {args.out}/")


if __name__ == "__main__":
    main()
