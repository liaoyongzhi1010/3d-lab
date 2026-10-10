"""E-205: temporal consistency (flicker) metric for Paper 1.

Generative disocclusion content tends to flicker across adjacent target views;
injecting a consistent feed-forward geometry prior should stabilize it. We measure
the second-order temporal difference energy in the disoccluded region:

    TC(seq) = mean_t || X[t+1] - 2 X[t] + X[t-1] ||_1   over invisible pixels

Lower is more temporally stable. We report baseline (Gen3R) vs ours (adaptive2)
over scenes with full per-frame renders (teacher_npy). This is a geometry/temporal
consistency proxy computable from renders alone (no depth needed), complementing
the per-frame PSNR/LPIPS/SSIM.

Run (any env with numpy):
  python e205_temporal_consistency.py --teacher_dir <dir> --out <json>
"""

import os, glob, json, argparse
import numpy as np


def load_stack(path):
    a = np.load(path).astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    return np.clip(a, 0, 1)  # [F,3,H,W]


def resize_mask_to(m, H, W):
    from PIL import Image

    im = Image.fromarray((m * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
    return np.asarray(im).astype(np.float32) / 255.0


def tc_invisible(stack, vis, n):
    F_ = min(len(stack), len(vis), n)
    H, W = stack.shape[2], stack.shape[3]
    vals = []
    for t in range(1, F_ - 1):
        inv = 1.0 - resize_mask_to(vis[t], H, W)
        m = inv > 0.5
        if m.sum() < 100:
            continue
        d2 = np.abs(stack[t + 1] - 2 * stack[t] + stack[t - 1])  # [3,H,W]
        d2 = d2.mean(0)  # [H,W]
        vals.append(float((d2 * m).sum() / m.sum()))
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir", required=True)
    ap.add_argument("--data", default="/home/data/gen3r_re10k/re10k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_frames", type=int, default=49)
    args = ap.parse_args()
    rows = []
    sids = sorted(
        os.path.basename(p)[len("adaptive2_") : -4]
        for p in glob.glob(os.path.join(args.teacher_dir, "adaptive2_*.npy"))
    )
    for sid in sids:
        try:
            base = load_stack(os.path.join(args.teacher_dir, f"baseline_{sid}.npy"))
            ours = load_stack(os.path.join(args.teacher_dir, f"adaptive2_{sid}.npy"))
            vis = np.load(os.path.join(args.data, sid, "visibility.npy")).astype(
                np.float32
            )
        except Exception as e:
            print("skip", sid[:16], e)
            continue
        tb = tc_invisible(base, vis, args.n_frames)
        to = tc_invisible(ours, vis, args.n_frames)
        if tb is None or to is None:
            continue
        rows.append({"sid": sid, "tc_base": tb, "tc_ours": to})
        print(
            f"{sid[:16]} TC base={tb:.5f} ours={to:.5f} {'ours-stabler' if to < tb else 'base-stabler'}",
            flush=True,
        )
    tb = np.mean([r["tc_base"] for r in rows])
    to = np.mean([r["tc_ours"] for r in rows])
    wins = sum(1 for r in rows if r["tc_ours"] < r["tc_base"])
    print(
        f"\n[SUMMARY] n={len(rows)}  TC baseline={tb:.5f}  ours={to:.5f}  "
        f"reduction={100 * (tb - to) / tb:.1f}%  ours-stabler in {wins}/{len(rows)}"
    )
    json.dump(
        {
            "rows": rows,
            "tc_base": float(tb),
            "tc_ours": float(to),
            "reduction_pct": float(100 * (tb - to) / tb),
            "wins": wins,
            "n": len(rows),
        },
        open(args.out, "w"),
        indent=2,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
