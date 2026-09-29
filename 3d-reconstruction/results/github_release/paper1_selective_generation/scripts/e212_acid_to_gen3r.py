"""E-212: convert ACID scenes (RealEstate10K-format camera .txt + jpg frames)
into Gen3R scene-folder format (transforms.json + images/), so the SAME
pipeline (Flash3D evidence -> Gen3R baseline -> selective injection) runs on a
SECOND dataset (outdoor aerial) for a generalization experiment.

ACID meta line (RealEstate10K convention, 19 fields):
  timestamp  fx fy cx cy  k1 k2  <12 values = world-to-camera 3x4 row-major>
fx,fy,cx,cy are normalized by image width/height. We build a 4x4
camera-to-world matrix (invert the 3x4 w2c) and denormalized intrinsics, and
write a transforms.json matching RE10K (per-frame w,h,fl_x,fl_y,cx,cy,
file_path,transform_matrix).

Frames are copied/symlinked into <out>/<scene>/images/frameNNNNN.png-equivalent
(we keep jpg, transforms.json points at them).

Run on server (needs numpy, PIL). No GPU.
"""

import os
import json
import glob
import shutil
import argparse
import numpy as np
from PIL import Image


def parse_meta(txt_path):
    lines = open(txt_path).read().strip().split("\n")
    # first line is the youtube url
    rows = []
    for ln in lines[1:]:
        p = ln.split()
        if len(p) < 19:
            continue
        ts = p[0]
        fx, fy, cx, cy = map(float, p[1:5])
        # p[5],p[6] distortion (ignore)
        m = np.array(list(map(float, p[7:19])), dtype=np.float64).reshape(
            3, 4
        )  # w2c 3x4
        rows.append((ts, fx, fy, cx, cy, m))
    return rows


def w2c_to_c2w(m34):
    w2c = np.eye(4)
    w2c[:3, :4] = m34
    return np.linalg.inv(w2c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", default="/home/data/ACID/acid/test")
    ap.add_argument("--frames_dir", default="/home/data/ACID_frames/test")
    ap.add_argument("--out", default="/home/data/E-212_acid_gen3r")
    ap.add_argument("--n_frames", type=int, default=49)
    ap.add_argument("--min_frames", type=int, default=49)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scenes = sorted(
        os.path.basename(p)[:-4]
        for p in glob.glob(os.path.join(args.meta_dir, "*.txt"))
    )
    done = 0
    for sid in scenes:
        if done >= args.limit:
            break
        fdir = os.path.join(args.frames_dir, sid)
        if not os.path.isdir(fdir):
            continue
        jpgs = sorted(glob.glob(os.path.join(fdir, "*.jpg")))
        if len(jpgs) < args.min_frames:
            continue
        meta = parse_meta(os.path.join(args.meta_dir, sid + ".txt"))
        # align meta timestamps to available jpgs
        ts_to_jpg = {os.path.basename(j)[:-4]: j for j in jpgs}
        meta = [r for r in meta if r[0] in ts_to_jpg]
        if len(meta) < args.min_frames:
            continue
        meta = meta[: args.n_frames]
        # image size
        W, H = Image.open(ts_to_jpg[meta[0][0]]).size
        scene_out = os.path.join(args.out, f"acid_{sid}")
        img_out = os.path.join(scene_out, "images")
        os.makedirs(img_out, exist_ok=True)
        frames = []
        for i, (ts, fx, fy, cx, cy, m34) in enumerate(meta):
            c2w = w2c_to_c2w(m34)
            dst = os.path.join(img_out, f"frame{i:05d}.jpg")
            if not os.path.exists(dst):
                shutil.copy(ts_to_jpg[ts], dst)
            frames.append(
                {
                    "w": W,
                    "h": H,
                    "fl_x": fx * W,
                    "fl_y": fy * H,
                    "cx": cx * W,
                    "cy": cy * H,
                    "file_path": f"images/frame{i:05d}.jpg",
                    "transform_matrix": c2w.tolist(),
                }
            )
        json.dump(
            {"frames": frames}, open(os.path.join(scene_out, "transforms.json"), "w")
        )
        done += 1
        print(f"{sid[:16]} frames={len(frames)} WxH={W}x{H}")
    print(f"\nconverted {done} ACID scenes -> {args.out}")


if __name__ == "__main__":
    main()
