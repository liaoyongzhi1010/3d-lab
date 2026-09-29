"""E-144: build the frame-level quality-probe diagnostic dataset.

Aligns Gen3R per-frame invisible PSNR (baseline vs adaptive2) with per-frame
Flash3D quality probes. One row is one (scene, frame). Both pipelines iterate
from frame 1 and skip frames with too few visible or invisible pixels.

The JSON column names are retained for released-data compatibility. They are
GT-derived quality probes, not inference-time features. In particular, the
historical ``vis_gap`` column is exactly ``f3d_vis - base_inv``; it is not a
visible-region gap because per-frame baseline-visible PSNR was unavailable.
"""

import json
import os
import argparse

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
EXP = os.path.join(RESULTS, "expanded")


def build(gen3r_json, probe_json, split):
    g = json.load(open(gen3r_json))["per_scene"]
    bl = g["baseline"]
    ad = g.get("adaptive2", {})
    probe = {r["scene"]: r for r in json.load(open(probe_json))}
    rows = []
    for scene in bl:
        sid = scene.replace("test_", "")
        b = bl[scene]
        a = ad.get(scene, {})
        p = probe.get(sid) or probe.get(scene)
        if p is None:
            continue
        b_pf = b.get("per_frame_invis_psnr")
        a_pf = a.get("per_frame_invis_psnr")
        if not b_pf or not a_pf:
            continue
        valid = [
            f
            for f in p["frames"]
            if f["f3d_inv_psnr"] is not None and f["f3d_vis_psnr"] is not None
        ]
        n = min(len(b_pf), len(a_pf), len(valid))
        if n == 0:
            continue
        for j in range(n):
            binv = b_pf[j]
            tinv = a_pf[j]
            pf = valid[j]
            delta = tinv - binv
            rows.append(
                {
                    "sid": sid,
                    "split": split,
                    "frame_idx": pf["frame"],
                    "base_inv": binv,
                    "teacher_inv": tinv,
                    "delta": delta,
                    "label": int(delta > 0.1),
                    "f3d_vis": pf["f3d_vis_psnr"],
                    "f3d_inv": pf["f3d_inv_psnr"],
                    "vis_frac": pf["vis_frac"],
                    "vis_gap": pf["f3d_vis_psnr"] - binv,
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gen3r", required=True, help="Gen3R e012_vesg(.partial).json with per_frame"
    )
    ap.add_argument("--probe", required=True, help="per-frame probe json (e143)")
    ap.add_argument("--split", default="train5")
    ap.add_argument("--out", default=os.path.join(EXP, "E-144_frame_dataset.json"))
    args = ap.parse_args()
    rows = build(args.gen3r, args.probe, args.split)
    json.dump(rows, open(args.out, "w"), indent=2)
    pos = sum(r["label"] for r in rows)
    scenes = len(set(r["sid"] for r in rows))
    print(f"saved {args.out} frame_rows={len(rows)} scenes={scenes} positive={pos}")


if __name__ == "__main__":
    main()
