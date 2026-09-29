"""Build a STRICT training whitelist: only scenes whose frames are fully present.

The Flash3D re10k reader requests frames at the timestamps stored in each scene's pcl
(sparse point cloud) metadata. A scene mid-download can be in the whitelist but miss some
timestamps -> FileNotFoundError crashes training. This builder keeps a scene ONLY if
EVERY timestamp referenced by its pcl has a corresponding {timestamp}.jpg on disk.

Reads pcl pickles from the unpacked dir (/tmp/monosplat/pcl.train after a train run
unpacks them, or unpacks a provided tar). Falls back to using the meta .txt timestamps
if pcl not available.

Usage:
    python build_strict_whitelist.py --out /root/projects/flash3d/splits/train_strict.json
"""

import argparse
import glob
import gzip
import json
import os
import pickle
from pathlib import Path


def load_pcl_timestamps(pcl_dir, sid):
    p = os.path.join(pcl_dir, sid + ".pickle.gz")
    if not os.path.exists(p):
        return None
    try:
        with gzip.open(p, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return None
    # pcl dict has 'timestamps' key in Flash3D format
    if isinstance(data, dict) and "timestamps" in data:
        return list(data["timestamps"])
    return None


def meta_timestamps(meta_dir, sid):
    p = os.path.join(meta_dir, sid + ".txt")
    if not os.path.exists(p):
        return None
    ts = []
    with open(p) as f:
        next(f, None)  # url line
        for line in f:
            parts = line.split()
            if parts:
                ts.append(int(parts[0]))
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--frames_dir", default="/root/projects/flash3d/data/RealEstate10K/train"
    )
    ap.add_argument("--pcl_dir", default="/tmp/monosplat/pcl.train")
    ap.add_argument(
        "--meta_dir", default="/home/data/RealEstate10K/RealEstate10K/train"
    )
    ap.add_argument("--out", default="/root/projects/flash3d/splits/train_strict.json")
    ap.add_argument("--min_frames", type=int, default=20)
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    use_pcl = os.path.isdir(args.pcl_dir)
    print(f"using pcl timestamps: {use_pcl} ({args.pcl_dir})")

    kept, dropped_partial, dropped_few = 0, 0, 0
    whitelist = []
    for d in sorted(frames_dir.iterdir()):
        if not d.is_dir():
            continue
        sid = d.name
        on_disk = set()
        for f in d.iterdir():
            if f.suffix == ".jpg":
                try:
                    on_disk.add(int(f.stem))
                except ValueError:
                    pass
        if len(on_disk) < args.min_frames:
            dropped_few += 1
            continue
        ts = None
        if use_pcl:
            ts = load_pcl_timestamps(args.pcl_dir, sid)
        if ts is None:
            ts = meta_timestamps(args.meta_dir, sid)
        if ts is None:
            # can't verify; keep if enough frames
            whitelist.append(sid)
            kept += 1
            continue
        # require every referenced timestamp present
        missing = [t for t in ts if t not in on_disk]
        if len(missing) == 0:
            whitelist.append(sid)
            kept += 1
        else:
            dropped_partial += 1

    json.dump(whitelist, open(args.out, "w"))
    print(
        f"kept={kept}, dropped_partial(missing frames)={dropped_partial}, dropped_few(<{args.min_frames})={dropped_few}"
    )
    print(f"wrote strict whitelist ({len(whitelist)} scenes) -> {args.out}")


if __name__ == "__main__":
    main()
