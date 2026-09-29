"""E-606: SS baseline vs gated on the DISOCC protocol (easy/hard), target-frame eval.

Protocol aligned to SS Table 1 (E-605 validated Flash3D easy 16.24 / hard 12.67).
For each disocc_scenes/* we run the full Scene-Splatter pipeline in mode ss and grounded,
then evaluate the LAST trajectory frame (the disocclusion target) of the final render_video
vs gt_target.png. Standard metrics, 5% crop, VGG-LPIPS. Split easy/hard.

Goal: show gated > SS > Flash3D on hard (large-disocclusion) set.

Run (viewcrafter env, background):
  cd /root/projects/Scene-Splatter
  nohup /root/miniconda3/envs/viewcrafter/bin/python _e606_disocc_batch.py \
    --bands hard --modes ss grounded > /home/data/E-606_disocc/hard.log 2>&1 &
"""

import os
import re
import sys
import glob
import json
import time
import argparse
import subprocess
import numpy as np

SS_ROOT = "/root/projects/Scene-Splatter"
SCENES_DIR = os.path.join(SS_ROOT, "disocc_scenes")
PY = "/root/miniconda3/envs/viewcrafter/bin/python"
OUT = "/home/data/E-606_disocc"
os.makedirs(OUT, exist_ok=True)


def set_cfg(img, cam, mode):
    p = os.path.join(SS_ROOT, "configs", "config.yaml")
    s = open(p).read()
    s = re.sub(r'image_path: ".*"', f'image_path: "{img}"', s)
    s = re.sub(r'camera_path: ".*"', f'camera_path: "{cam}"', s)
    s = re.sub(r'fusion_mode: ".*"', f'fusion_mode: "{mode}"', s)
    open(p, "w").write(s)


def run_one(scene_dir, mode):
    img = os.path.join(scene_dir, "images", "0.png")
    cam = os.path.join(scene_dir, "cameras", "camera0.pickle.gz")
    set_cfg(img, cam, mode)
    before = set(glob.glob(os.path.join(SS_ROOT, "results", "2026-*")))
    t0 = time.time()
    r = subprocess.run(
        [PY, "scenesplatter.py"], cwd=SS_ROOT, capture_output=True, text=True
    )
    after = set(glob.glob(os.path.join(SS_ROOT, "results", "2026-*")))
    new = sorted(after - before)
    if not new:
        print(
            f"  [{os.path.basename(scene_dir)}/{mode}] FAILED:\n{r.stderr[-800:]}",
            flush=True,
        )
        return None
    return new[-1], time.time() - t0


def eval_target(result_dir, scene_dir):
    import imageio.v2 as iio, math
    from PIL import Image
    import torch, lpips

    global _LP
    try:
        _LP
    except NameError:
        _LP = lpips.LPIPS(net="vgg").cuda().eval()

    def crop5(x):
        H, W = x.shape[:2]
        ch, cw = int(math.ceil(0.05 * H)), int(math.ceil(0.05 * W))
        return x[ch : H - ch, cw : W - cw]

    def psnr(a, b):
        v = ((a - b) ** 2).mean()
        return 100.0 if v < 1e-10 else float(-10 * np.log10(v))

    def lp(a, b):
        pa = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
        pb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
        with torch.no_grad():
            return float(_LP(pa, pb).item())

    rv = sorted(glob.glob(os.path.join(result_dir, "render_video_*.mp4")))
    if not rv:
        return None
    frames = [np.asarray(f).astype(np.float32) / 255 for f in iio.get_reader(rv[-1])]
    pred = frames[-1]  # target = last trajectory frame
    H, W = pred.shape[:2]
    gt = (
        np.asarray(
            Image.open(os.path.join(scene_dir, "images", "gt_target.png"))
            .convert("RGB")
            .resize((W, H))
        ).astype(np.float32)
        / 255
    )
    pc, gc = crop5(pred), crop5(gt)
    return {"psnr": psnr(pc, gc), "ssim": None, "lpips": lp(pc, gc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", nargs="+", default=["hard"])
    ap.add_argument("--modes", nargs="+", default=["ss", "grounded"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    scenes = []
    for sd in sorted(glob.glob(os.path.join(SCENES_DIR, "*/"))):
        name = os.path.basename(sd.rstrip("/"))
        band = "easy" if name.startswith("easy") else "hard"
        if band in a.bands:
            scenes.append((name, band, sd))
    if a.limit:
        scenes = scenes[: a.limit]
    results = {}
    for name, band, sd in scenes:
        results[name] = {"band": band}
        for mode in a.modes:
            print(f"\n=== {name} [{band}] mode={mode} ===", flush=True)
            out = run_one(sd, mode)
            if out is None:
                continue
            rdir, dt = out
            ev = eval_target(rdir, sd)
            results[name][mode] = {"dir": rdir, "time_s": dt, **(ev or {})}
            print(
                f"  [{name}/{mode}] {dt / 60:.1f}min PSNR={ev['psnr']:.2f} LPIPS={ev['lpips']:.3f}",
                flush=True,
            )
            json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)

    print("\n===== E-606 SUMMARY (target-frame, disocc protocol) =====")
    for band in a.bands:
        for mode in a.modes:
            ps = [
                results[s][mode]["psnr"]
                for s in results
                if results[s].get("band") == band and mode in results[s]
            ]
            ls = [
                results[s][mode]["lpips"]
                for s in results
                if results[s].get("band") == band and mode in results[s]
            ]
            if ps:
                print(
                    f"  {band}/{mode:9s}: PSNR {np.mean(ps):.2f} LPIPS {np.mean(ls):.3f} (n={len(ps)})"
                )
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    print(f"saved {OUT}/results.json")


if __name__ == "__main__":
    main()
