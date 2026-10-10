"""E-204: region-separated LPIPS + SSIM for Paper 1 (perceptual metrics).

Top venues expect LPIPS/SSIM, not only PSNR. Using the full per-frame teacher
renders dumped by --save_teacher_npy (gt/baseline/adaptive2, [F,3,560,560]) plus
visibility.npy, we compute visible- and invisible-region LPIPS (VGG, oracle
GT-substitution) and SSIM, per scene, for baseline (Gen3R) and ours (adaptive2).

Region handling:
  - LPIPS: oracle GT-substitution — only the measured region differs from GT,
    the other region is copied from GT, so the perceptual metric reflects only
    the region of interest (standard trick for masked LPIPS).
  - SSIM: computed on the full frame but weighted by the region mask
    (mean over masked pixels of the per-pixel SSIM map).

Run (Gen3R env, has lpips + torch):
  python e204_region_perceptual.py --teacher_dir <dir_with_npy> --out <json>
"""

import os, glob, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

RES = 560


def to_t(a):
    x = torch.from_numpy(np.ascontiguousarray(a.astype(np.float32)))
    return x


def load_stack(path):
    a = np.load(path).astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    return np.clip(a, 0, 1)


def resize_mask(m, hw):
    im = torch.from_numpy(m.astype(np.float32))[None, None]
    im = F.interpolate(im, size=hw, mode="nearest")[0, 0]
    return im


def ssim_map(x, y, C1=0.01**2, C2=0.03**2):
    # x,y: [3,H,W] in [0,1]; simple 11x11 uniform-window SSIM
    import torch.nn.functional as FF

    k = 11
    pad = k // 2
    w = torch.ones(3, 1, k, k, device=x.device) / (k * k)
    mux = FF.conv2d(x[None], w, padding=pad, groups=3)
    muy = FF.conv2d(y[None], w, padding=pad, groups=3)
    mux2, muy2, muxy = mux * mux, muy * muy, mux * muy
    sx = FF.conv2d(x[None] * x[None], w, padding=pad, groups=3) - mux2
    sy = FF.conv2d(y[None] * y[None], w, padding=pad, groups=3) - muy2
    sxy = FF.conv2d(x[None] * y[None], w, padding=pad, groups=3) - muxy
    s = ((2 * mux * muy + C1) * (2 * sxy + C2)) / ((mux2 + muy2 + C1) * (sx + sy + C2))
    return s[0].mean(0)  # [H,W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir", required=True)
    ap.add_argument("--data", default="/home/data/gen3r_re10k/re10k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_frames", type=int, default=49)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import lpips

    lp = lpips.LPIPS(net="vgg").to(dev).eval()

    rows = []
    sids = sorted(
        os.path.basename(p)[len("adaptive2_") : -4]
        for p in glob.glob(os.path.join(args.teacher_dir, "adaptive2_*.npy"))
    )
    for sid in sids:
        try:
            gt = load_stack(os.path.join(args.teacher_dir, f"gt_{sid}.npy"))
            base = load_stack(os.path.join(args.teacher_dir, f"baseline_{sid}.npy"))
            ours = load_stack(os.path.join(args.teacher_dir, f"adaptive2_{sid}.npy"))
            vis = np.load(os.path.join(args.data, sid, "visibility.npy")).astype(
                np.float32
            )
        except Exception as e:
            print("skip", sid[:16], e)
            continue
        Fn = min(len(gt), len(base), len(ours), len(vis), args.n_frames)
        acc = {
            k: []
            for k in [
                "b_vis_lp",
                "b_inv_lp",
                "o_vis_lp",
                "o_inv_lp",
                "b_vis_ss",
                "b_inv_ss",
                "o_vis_ss",
                "o_inv_ss",
            ]
        }
        for i in range(1, Fn):
            g = to_t(gt[i]).to(dev)
            b = to_t(base[i]).to(dev)
            o = to_t(ours[i]).to(dev)
            vm = (resize_mask(vis[i], g.shape[1:]).to(dev) > 0.5).float()
            im = 1.0 - vm
            if vm.sum() < 100 or im.sum() < 100:
                continue
            for name, pred in [("b", b), ("o", o)]:
                # oracle GT-substitution for LPIPS
                cg_v = g.clone()
                cg_v[:, vm > 0.5] = pred[:, vm > 0.5]
                cg_i = g.clone()
                cg_i[:, im > 0.5] = pred[:, im > 0.5]
                with torch.no_grad():
                    acc[f"{name}_vis_lp"].append(
                        float(lp(cg_v[None] * 2 - 1, g[None] * 2 - 1).item())
                    )
                    acc[f"{name}_inv_lp"].append(
                        float(lp(cg_i[None] * 2 - 1, g[None] * 2 - 1).item())
                    )
                    sm = ssim_map(pred, g)
                    acc[f"{name}_vis_ss"].append(float((sm * vm).sum() / vm.sum()))
                    acc[f"{name}_inv_ss"].append(float((sm * im).sum() / im.sum()))
        row = {"sid": sid}
        for k, v in acc.items():
            row[k] = float(np.mean(v)) if v else None
        rows.append(row)
        print(
            f"{sid[:16]} inv LPIPS b={row['b_inv_lp']:.4f} o={row['o_inv_lp']:.4f} "
            f"| inv SSIM b={row['b_inv_ss']:.4f} o={row['o_inv_ss']:.4f}",
            flush=True,
        )
    json.dump(rows, open(args.out, "w"), indent=2)

    # summary
    def mean(k):
        vs = [r[k] for r in rows if r.get(k) is not None]
        return float(np.mean(vs)) if vs else None

    summ = {k: mean(k) for k in rows[0] if k != "sid"} if rows else {}
    print("\n[SUMMARY]")
    print(
        f"invisible LPIPS: baseline {summ.get('b_inv_lp'):.4f} -> ours {summ.get('o_inv_lp'):.4f}"
    )
    print(
        f"invisible SSIM : baseline {summ.get('b_inv_ss'):.4f} -> ours {summ.get('o_inv_ss'):.4f}"
    )
    print(
        f"visible   LPIPS: baseline {summ.get('b_vis_lp'):.4f} -> ours {summ.get('o_vis_lp'):.4f}"
    )
    print(
        f"visible   SSIM : baseline {summ.get('b_vis_ss'):.4f} -> ours {summ.get('o_vis_ss'):.4f}"
    )
    json.dump({"rows": rows, "summary": summ}, open(args.out, "w"), indent=2)
    print(f"saved {args.out} n={len(rows)}")


if __name__ == "__main__":
    main()
