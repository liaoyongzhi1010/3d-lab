"""E-603: batch SS baseline vs grounded (Paper1) across scenes, standard whole-image metrics.

Runs, for each scene, the full Scene-Splatter pipeline in two modes (ss baseline, grounded)
by editing configs/config.yaml and invoking scenesplatter.py, then evaluates the final
render_video vs GT (all frames, 5% crop, VGG-LPIPS) and vs flash3d_video.

Produces the core Paper1 evidence table: does geometric-consistency grounding
consistently reduce the generation-induced quality loss vs the SS baseline?

Run (viewcrafter env, background):
  cd /root/projects/Scene-Splatter
  nohup /root/miniconda3/envs/viewcrafter/bin/python _e603_batch_grounded.py \
    --scenes test_0a9f2831a3e73de8 test_5dbf866479511338 test_249fd0890d439aa9 \
             test_5f75672448394958 test_70c12e8b01c13d84 \
    > /home/data/E-603_batch/run.log 2>&1 &
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
SCENES_DIR = os.path.join(SS_ROOT, "gen3r_ss_scenes")
PY = "/root/miniconda3/envs/viewcrafter/bin/python"
OUT = "/home/data/E-603_batch"
os.makedirs(OUT, exist_ok=True)


def set_cfg(img, cam, mode):
    p = os.path.join(SS_ROOT, "configs", "config.yaml")
    s = open(p).read()
    s = re.sub(r'image_path: ".*"', f'image_path: "{img}"', s)
    s = re.sub(r'camera_path: ".*"', f'camera_path: "{cam}"', s)
    s = re.sub(r'fusion_mode: ".*"', f'fusion_mode: "{mode}"', s)
    open(p, "w").write(s)


def run_one(scene, mode):
    sd = os.path.join(SCENES_DIR, scene)
    img = os.path.join(sd, "images", "0.png")
    cam = os.path.join(sd, "cameras", "camera0.pickle.gz")
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
            f"  [{scene}/{mode}] no result dir! stderr tail:\n{r.stderr[-500:]}",
            flush=True,
        )
        return None
    return new[-1], time.time() - t0


def evaluate(result_dir, scene):
    import imageio.v2 as iio, math
    from PIL import Image
    import torch, lpips

    global _LP
    try:
        _LP
    except NameError:
        _LP = lpips.LPIPS(net="vgg").cuda().eval()
    sd = os.path.join(SCENES_DIR, scene)

    def rd(p):
        return [np.asarray(f).astype(np.float32) / 255 for f in iio.get_reader(p)]

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
    ss = rd(rv[-1])
    f3p = os.path.join(result_dir, "flash3d_video.mp4")
    f3 = rd(f3p) if os.path.exists(f3p) else None
    H, W = ss[0].shape[:2]
    n = min(len(ss), 49)

    def gtf(k):
        return (
            np.asarray(
                Image.open(os.path.join(sd, "images", f"gt_frame_{k:03d}.png"))
                .convert("RGB")
                .resize((W, H))
            ).astype(np.float32)
            / 255
        )

    sp, sl, fp, fl = [], [], [], []
    for k in range(n):
        gt = crop5(gtf(k))
        s = crop5(ss[k])
        sp.append(psnr(s, gt))
        sl.append(lp(s, gt))
        if f3 is not None and k < len(f3):
            f = crop5(f3[k])
            fp.append(psnr(f, gt))
            fl.append(lp(f, gt))
    return {
        "psnr": float(np.mean(sp)),
        "lpips": float(np.mean(sl)),
        "f3d_psnr": float(np.mean(fp)) if fp else None,
        "f3d_lpips": float(np.mean(fl)) if fl else None,
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", default=["ss", "grounded"])
    a = ap.parse_args()
    results = {}
    for scene in a.scenes:
        results[scene] = {}
        for mode in a.modes:
            print(f"\n=== running {scene} mode={mode} ===", flush=True)
            out = run_one(scene, mode)
            if out is None:
                continue
            rdir, dt = out
            ev = evaluate(rdir, scene)
            results[scene][mode] = {"dir": rdir, "time_s": dt, **(ev or {})}
            print(
                f"  [{scene}/{mode}] {dt / 60:.1f}min PSNR={ev['psnr']:.2f} LPIPS={ev['lpips']:.3f} "
                f"(F3D {ev['f3d_psnr']:.2f}/{ev['f3d_lpips']:.3f})",
                flush=True,
            )
            json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)

    # summary
    print("\n===== E-603 SUMMARY =====")
    for mode in a.modes:
        ps = [results[s][mode]["psnr"] for s in results if mode in results[s]]
        ls = [results[s][mode]["lpips"] for s in results if mode in results[s]]
        if ps:
            print(
                f"  {mode:10s}: PSNR {np.mean(ps):.2f} LPIPS {np.mean(ls):.3f} (n={len(ps)})"
            )
    f3 = [
        results[s]["ss"]["f3d_psnr"]
        for s in results
        if "ss" in results[s] and results[s]["ss"].get("f3d_psnr")
    ]
    if f3:
        print(f"  Flash3D   : PSNR {np.mean(f3):.2f} (n={len(f3)})")
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    print(f"saved {OUT}/results.json")


if __name__ == "__main__":
    main()
