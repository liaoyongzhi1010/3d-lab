"""Paper2 P2-2: precompute per-sample teacher reliability for gate-aware
distillation. For each E-064 teacher-cache npz, compute the teacher's hole-region
PSNR gain over baseline vs GROUND TRUTH. Samples where the teacher actually hurts
(gain<0) are the ones a naive student wastes capacity imitating.

gate-aware distillation = train the student only on (or up-weighting) samples the
teacher completes reliably. This is the Paper2 positive contribution: a cleaner
distillation target -> better held-out completion.

Output: JSON {npz_basename: teacher_gain_db}
"""

import glob, json, os, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/home/data/E-064_teacher_cache")
    ap.add_argument("--out", default="/home/data/E-410_teacher_quality.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache, "*.npz")))
    q = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        t = d["teacher_rgb"].astype(np.float32) / 255
        g = d["gt_rgb"].astype(np.float32) / 255
        b = d["base_rgb"].astype(np.float32) / 255
        h = d["hole"].astype(bool)
        if h.sum() < 16:
            continue
        te = ((t - g) ** 2)[h].mean()
        be = ((b - g) ** 2)[h].mean()
        t_psnr = 10 * np.log10(1 / (te + 1e-9))
        b_psnr = 10 * np.log10(1 / (be + 1e-9))
        q[os.path.basename(f)] = float(t_psnr - b_psnr)
    json.dump(q, open(args.out, "w"))
    arr = np.array(list(q.values()))
    print(
        f"saved {args.out} n={len(q)} mean_gain={arr.mean():.3f} frac_help={(arr > 0.5).mean():.3f}"
    )


if __name__ == "__main__":
    main()
