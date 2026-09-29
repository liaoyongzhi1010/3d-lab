"""E-149: augment the frame-level quality-probe diagnostic dataset.

Adds camera translation/rotation plus interactions derived from released PSNR
quality probes. Camera motion is observable, but f3d_vis, f3d_inv, vis_frac and
their interactions depend on GT-derived evaluation artifacts. Consequently the
combined E-145b feature set is a diagnostic upper bound, not an inference-time
feature set.
"""

import json
import os
import argparse
import numpy as np

NF = 49


def load_cams(scene_dir):
    tf = json.load(open(os.path.join(scene_dir, "transforms.json")))
    frames = tf["frames"][:NF]
    mats = [np.asarray(f["transform_matrix"], dtype=np.float64) for f in frames]
    return mats


def rot_angle_deg(R0, Rk):
    R = R0[:3, :3].T @ Rk[:3, :3]
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="frame dataset json (e144/e148)")
    ap.add_argument("--data", default="/home/data/gen3r_re10k/re10k")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = json.load(open(args.frames))
    cam_cache = {}
    eps = 1e-6
    n_ok = 0
    for r in rows:
        sid = r["sid"]
        scene_dir = os.path.join(args.data, sid)
        if sid not in cam_cache:
            try:
                cam_cache[sid] = load_cams(scene_dir)
            except Exception:
                cam_cache[sid] = None
        mats = cam_cache[sid]
        k = r["frame_idx"]
        if mats is not None and k < len(mats):
            m0, mk = mats[0], mats[k]
            r["cam_trans"] = float(np.linalg.norm(mk[:3, 3] - m0[:3, 3]))
            r["cam_rot_deg"] = rot_angle_deg(m0, mk)
            n_ok += 1
        else:
            r["cam_trans"] = 0.0
            r["cam_rot_deg"] = 0.0
        vf = r.get("vis_frac") or 0.9
        r["disocc_frac"] = 1.0 - vf
        r["f3d_vi_ratio"] = float(r["f3d_inv"]) / max(float(r["f3d_vis"]), eps)
        r["f3d_vi_prod"] = float(r["f3d_vis"]) * vf
    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"saved {args.out} rows={len(rows)} cam_ok={n_ok}")


if __name__ == "__main__":
    main()
