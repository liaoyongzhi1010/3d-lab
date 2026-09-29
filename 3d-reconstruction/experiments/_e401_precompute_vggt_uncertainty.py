"""V2-2: precompute per-pixel VGGT uncertainty map for each E-064 teacher-cache
sample, to be used as an OracleGS-style uncertainty weight in student distillation.

For each npz (teacher_rgb, hole, base_rgb, ...):
  - run VGGT on the teacher completion image (the generative proposal, 256x384)
  - take depth_conf as per-pixel geometric confidence
  - normalize to U in [0,1] (high = confident/reliable, low = uncertain/hallucination)
  - resize U back to (256,384) and save alongside (keyed by npz basename)

OracleGS analogy: teacher_rgb = generative proposal I'; VGGT = MVS oracle;
U = validated per-pixel reliability. The student's teacher-imitation loss will be
weighted by U so it learns to imitate the teacher ONLY where the oracle trusts it.

Run (flash3d venv):
  python _e401_precompute_vggt_uncertainty.py --cache /home/data/E-064_teacher_cache \
      --out /home/data/E-401_vggt_unc --limit 0
"""

import sys, os, glob, argparse
from pathlib import Path

sys.path.insert(0, "/root/projects/vggt")
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cuda:0"


def to_vggt_input(rgb_uint8, size=518):
    x = torch.from_numpy(rgb_uint8).float().permute(2, 0, 1)[None] / 255.0
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/home/data/E-064_teacher_cache")
    ap.add_argument("--out", default="/home/data/E-401_vggt_unc")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.cache, "*.npz")))
    if args.limit:
        files = files[: args.limit]
    print(f"samples: {len(files)}", flush=True)

    from vggt.models.vggt import VGGT

    print("loading VGGT-1B...", flush=True)
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(DEVICE).eval()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    done = 0
    for f in files:
        name = Path(f).stem
        outp = os.path.join(args.out, name + ".npy")
        if os.path.exists(outp):
            done += 1
            continue
        d = np.load(f, allow_pickle=True)
        teacher = d["teacher_rgb"]  # (256,384,3) uint8
        base = d["base_rgb"]
        H, W = teacher.shape[:2]
        # two-view VGGT: base (source-like) + teacher (proposal) -> conf on proposal
        xt = to_vggt_input(teacher).to(DEVICE)
        xb = to_vggt_input(base).to(DEVICE)
        imgs = torch.stack([xb[0], xt[0]], 0)[None]  # [1,2,3,518,518]
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            agg, ps_idx = model.aggregator(imgs)
            depth_map, depth_conf = model.depth_head(agg, imgs, ps_idx)
        # depth_conf: [1,2,518,518] -> take the teacher view (index 1)
        conf_t = depth_conf[0, 1].float().cpu().numpy()  # (518,518)
        # normalize per-image to [0,1] (robust min-max on 5/95 pct)
        lo, hi = np.percentile(conf_t, 5), np.percentile(conf_t, 95)
        U = np.clip((conf_t - lo) / (hi - lo + 1e-6), 0, 1)
        # resize to (H,W)
        U_t = torch.from_numpy(U)[None, None]
        U_hw = F.interpolate(U_t, size=(H, W), mode="bilinear", align_corners=False)[
            0, 0
        ].numpy()
        np.save(outp, U_hw.astype(np.float32))
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{len(files)}", flush=True)

    print(f"saved {done} maps to {args.out}", flush=True)


if __name__ == "__main__":
    main()
