"""Build a training whitelist of ALL RE10K scenes that currently have frames.

Flash3D's RE10K reader uses env var RE10K_TRAIN_WHITELIST -> JSON list of scene keys.
The stock train_full_frames.json lists only the original 5,533 scenes. As the YouTube
re-download adds scenes, we regenerate the whitelist to include everything present so
training uses the maximal available data.

A scene is 'present' if its frame dir exists and has >= min_frames JPGs AND it has pcl
(sparse point cloud) metadata unpacked (train pipeline needs both). We check frames here;
the pcl set is a superset (65K) so frames are the binding constraint.

Usage:
    python build_present_whitelist.py --out /root/projects/flash3d/splits/train_present.json
"""

import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", default="/home/data/RealEstate10K/train")
    ap.add_argument(
        "--meta_dir", default="/home/data/RealEstate10K/RealEstate10K/train"
    )
    ap.add_argument("--out", default="/root/projects/flash3d/splits/train_present.json")
    ap.add_argument("--min_frames", type=int, default=20)
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    present = []
    n_checked = 0
    for d in sorted(frames_dir.iterdir()):
        if not d.is_dir():
            continue
        n_checked += 1
        # fast count: first min_frames jpgs
        cnt = 0
        for f in d.iterdir():
            if f.suffix == ".jpg":
                cnt += 1
                if cnt >= args.min_frames:
                    break
        if cnt >= args.min_frames:
            present.append(d.name)

    json.dump(present, open(args.out, "w"))
    print(
        f"checked {n_checked} dirs, present(>= {args.min_frames} frames) = {len(present)}"
    )
    print(f"wrote whitelist -> {args.out}")


if __name__ == "__main__":
    main()
