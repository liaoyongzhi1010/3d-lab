"""SV3D-Eval-Suite : cross-method GT consistency check.

Before you compare two methods' FID/KID (or any cross-method metric), you MUST
prove they were scored on the SAME image set with the SAME ground truth. This
tool verifies:
  1. both export dirs contain the SAME set of row-directories (names identical), and
  2. for a random sample of rows, the GT frames are pixel-identical (GT-MAE ~ 0).

If either check fails, the two methods used different (scene, src, tgt) triples
or different GT -> any cross-method FID is meaningless (this is exactly the
wide700 failure mode where common-scene GT-MAE was 60~97).

Directory layout expected (row-level unique naming, see docs/USAGE.md):
  <root>/<row_dir>/gt/00X.png   with 000=src 001=tgt5 002=tgt10 003=tgt_rand

Usage:
  python metrics/check_gt_consistency.py \
    --a /home/data/E-052_official/flash3d_present_imgs2 \
    --b /home/data/E-052_official/catsplat_present_imgs2 \
    --sample 200
Exit code 0 = comparable, 1 = NOT comparable.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

FRAMES = ("000", "001", "002", "003")


def dir_names(root):
    return sorted(p.name for p in Path(root).iterdir() if p.is_dir())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="method A export dir")
    ap.add_argument("--b", required=True, help="method B export dir")
    ap.add_argument("--sample", type=int, default=200, help="#row-dirs to pixel-check")
    ap.add_argument("--tol", type=float, default=1e-3, help="max allowed GT-MAE")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    na, nb = dir_names(args.a), dir_names(args.b)
    print(f"[gt-check] A={args.a}\n           #dirs={len(na)}")
    print(f"[gt-check] B={args.b}\n           #dirs={len(nb)}")

    set_a, set_b = set(na), set(nb)
    common = sorted(set_a & set_b)
    only_a, only_b = set_a - set_b, set_b - set_a
    names_identical = na == nb
    print(f"[gt-check] dir names identical: {names_identical}")
    print(
        f"[gt-check] common={len(common)}  only-in-A={len(only_a)}  only-in-B={len(only_b)}"
    )
    if only_a:
        print("           e.g. only-in-A:", sorted(only_a)[:3])
    if only_b:
        print("           e.g. only-in-B:", sorted(only_b)[:3])

    if not common:
        print("VERDICT: NOT COMPARABLE (no shared row-dirs)")
        sys.exit(1)

    random.seed(args.seed)
    samp = random.sample(common, min(args.sample, len(common)))
    bad, checked, maxmae = 0, 0, 0.0
    for nm in samp:
        for fr in FRAMES:
            fa = Path(args.a) / nm / "gt" / f"{fr}.png"
            fb = Path(args.b) / nm / "gt" / f"{fr}.png"
            if not (fa.exists() and fb.exists()):
                continue
            a = np.asarray(Image.open(fa).convert("RGB"), np.float32)
            b = np.asarray(Image.open(fb).convert("RGB"), np.float32)
            m = np.abs(a - b).mean() if a.shape == b.shape else 999.0
            maxmae = max(maxmae, m)
            checked += 1
            if m > args.tol:
                bad += 1
                if bad <= 5:
                    print(f"           MISMATCH {nm}/{fr} GT-MAE={m:.4f}")

    print(f"[gt-check] sampled {len(samp)} dirs, checked {checked} GT frames")
    print(f"[gt-check] max GT-MAE={maxmae:.6f}  mismatches(>{args.tol})={bad}")

    ok = names_identical and bad == 0
    print(
        "VERDICT:",
        "COMPARABLE (same set, GT pixel-identical)"
        if ok
        else "NOT COMPARABLE (see above)",
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
