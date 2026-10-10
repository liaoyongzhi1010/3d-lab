"""
Paper A (D015) — uncertainty-quality evaluator on held-out (scene-disjoint) RE10K.

Loads a trained UncertaintyModel checkpoint, renders mean RGB + predicted per-pixel variance to novel
views, and computes (single-image inference, target used only as GT):

  Recon (parity vs Flash3D): PSNR/SSIM/LPIPS on the 5%-cropped merged render, overall + region-split.
  Uncertainty quality:
    - Spearman rank corr between predicted variance and true squared error (per image, averaged).
    - AUSE (Area Under Sparsification Error) vs the error-oracle, for OUR variance and two baselines:
        (a) RANDOM uncertainty, (b) NEGATIVE rendered opacity (low opacity = uncertain).
      AUSE lower = better; ours must beat both baselines.
    - Regression calibration: expected-vs-observed coverage of the Gaussian predictive interval at
      several z; report calibration error (mean |observed-expected|).
    - NLL (Gaussian) on held-out.
    - Region-separated mean variance (visible / occluded / oof): expect occ,oof > visible.

Everything is region-separated with the SAME method-agnostic forward-warp masks used in training/eval.

Usage (server):
  python /root/sv3d-lab/paper_a_explicit3d/eval_uncertainty.py \
     --ckpt /home/data/sv3d-lab/runs/unc_L2/ckpt_final.pt --n_scenes 72 --scene_start 0 \
     --split .../test_files_wide700.txt --out /home/data/sv3d-lab/evaluations/unc_L2 --viz 6
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, "/root/sv3d-lab")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.uncertainty_model import UncertaintyModel, UncertaintyConfig  # noqa: E402
from train_paper_a import build_cfg, load_re10k, to_device  # noqa: E402
from trainer_lib import compute_region_masks, crop5, psnr, SSIM  # noqa: E402


def spearman(a, b):
    """Spearman rank correlation between two 1D tensors."""
    a = a.flatten().float()
    b = b.flatten().float()
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / (ra.std() + 1e-8)
    rb = (rb - rb.mean()) / (rb.std() + 1e-8)
    return float((ra * rb).mean())


def sparsification_error(err, unc, n_bins=20):
    """AUSE: for uncertainty `unc` and per-pixel error `err` (1D), remove the most-uncertain
    fraction progressively and record mean error of the REMAINING pixels; compare to the oracle
    (remove by true error). Returns area between the two curves (sparsification error).
    Also returns the random-removal reference curve area for context."""
    err = err.flatten().float()
    unc = unc.flatten().float()
    N = err.numel()
    fracs = np.linspace(0, 0.99, n_bins)
    # order by DESC uncertainty (remove most uncertain first)
    order_unc = torch.argsort(unc, descending=True)
    order_oracle = torch.argsort(err, descending=True)
    err_by_unc = err[order_unc]
    err_by_oracle = err[order_oracle]
    curve_unc, curve_oracle = [], []
    for f in fracs:
        k = int(f * N)
        rem_unc = err_by_unc[k:]
        rem_oracle = err_by_oracle[k:]
        curve_unc.append(float(rem_unc.mean()) if rem_unc.numel() > 0 else 0.0)
        curve_oracle.append(float(rem_oracle.mean()) if rem_oracle.numel() > 0 else 0.0)
    curve_unc = np.array(curve_unc)
    curve_oracle = np.array(curve_oracle)
    # normalize by the no-removal error so AUSE is scale-free & comparable across images
    norm = curve_unc[0] + 1e-8
    ause = float(np.trapz((curve_unc - curve_oracle) / norm, fracs))
    return ause


def calibration_coverage(err_abs, sigma, zs=(0.5, 1.0, 1.5, 2.0, 2.5)):
    """Observed vs expected coverage of a zero-mean Gaussian predictive interval.
    err_abs, sigma: 1D tensors (per pixel). Returns (mean |obs-exp| calibration error, table)."""
    from math import erf, sqrt

    err_abs = err_abs.flatten().float()
    sigma = sigma.flatten().float().clamp(min=1e-4)
    table = []
    cerr = []
    for z in zs:
        expected = erf(z / sqrt(2.0))  # P(|N(0,1)| <= z)
        observed = float((err_abs <= z * sigma).float().mean())
        table.append({"z": z, "expected": expected, "observed": observed})
        cerr.append(abs(observed - expected))
    return float(np.mean(cerr)), table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_scenes", type=int, default=72)
    ap.add_argument("--scene_start", type=int, default=0)
    ap.add_argument("--novel_frames", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--viz", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    cfg = build_cfg(args.novel_frames, args.split)
    Re10KDataset = load_re10k()
    ds = Re10KDataset(cfg, split="test")
    s0 = args.scene_start
    ds._seq_key_src_idx_pairs = ds._seq_key_src_idx_pairs[s0 : s0 + args.n_scenes]
    ds.length = len(ds._seq_key_src_idx_pairs)

    from torch.utils.data import DataLoader
    from datasets.util import custom_collate
    from common.data.robust_dataset import RobustDataset

    ds = RobustDataset(ds)
    loader = DataLoader(ds, 1, shuffle=False, num_workers=2, collate_fn=custom_collate)

    model = UncertaintyModel(cfg, UncertaintyConfig(freeze_visible=True)).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd["model"], strict=False)
    model.eval()
    ssim = SSIM().to(device)
    try:
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

        lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
    except Exception:
        lpips = None

    pad = cfg.dataset.pad_border_aug
    agg = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
        "spearman": [],
        "ause_ours": [],
        "ause_random": [],
        "ause_opacity": [],
        "calib_err": [],
        "nll": [],
        "var_vis": [],
        "var_occ": [],
        "var_oof": [],
        "err_vis": [],
        "err_occ": [],
        "err_oof": [],
    }
    calib_tables = []
    viz_done = 0

    for si, batch in enumerate(loader):
        inputs = to_device(batch, device)
        target_ids = [f for f in args.novel_frames if ("color", f, 0) in inputs]
        if not target_ids:
            continue
        with torch.no_grad():
            out = model(inputs, target_ids)

        H = inputs["color", 0, 0].shape[2]
        W = inputs["color", 0, 0].shape[3]
        depth_full = out[("depth", 0)]
        dpp = depth_full[0, 0]
        dsrc = dpp[pad : dpp.shape[0] - pad, pad : dpp.shape[1] - pad] if pad else dpp
        K0 = inputs[("K_src", 0)][0]

        # opacity-as-uncertainty baseline: render opacity (use rgb=1 payload -> alpha) -> low = uncertain
        for fid in target_ids:
            gt = inputs[("color", fid, 0)][0]
            pred = out[("render", fid)][0]
            var = out[("var", fid)][0]  # (1,H,W)
            T = out[("cam_T_cam", 0, fid)][0]
            vis_m, occ_m, oof_m = compute_region_masks(dsrc, K0, T, H, W, device)

            predc, gtc, varc = crop5(pred), crop5(gt), crop5(var)
            p = psnr(predc, gtc)
            if p is not None:
                agg["psnr"].append(p)
            agg["ssim"].append(float(ssim(predc.unsqueeze(0), gtc.unsqueeze(0)).mean()))
            if lpips is not None:
                agg["lpips"].append(
                    float(
                        lpips(
                            predc.unsqueeze(0).clamp(0, 1) * 2 - 1,
                            gtc.unsqueeze(0).clamp(0, 1) * 2 - 1,
                        )
                    )
                )

            err = ((pred - gt) ** 2).mean(dim=0, keepdim=True)  # (1,H,W) per-pixel MSE
            errc = crop5(err)
            # uncertainty quality (on cropped image, flattened)
            agg["spearman"].append(spearman(varc, errc))
            agg["ause_ours"].append(sparsification_error(errc, varc))
            agg["ause_random"].append(sparsification_error(errc, torch.rand_like(varc)))
            # opacity baseline: accumulated alpha (confidence); uncertainty = 1 - alpha.
            alpha = out[("alpha", fid)][0]  # (1,H,W)
            agg["ause_opacity"].append(sparsification_error(errc, crop5(1.0 - alpha)))

            ce, tbl = calibration_coverage(errc.sqrt(), varc.sqrt())
            agg["calib_err"].append(ce)
            calib_tables.append(tbl)
            nll = (
                0.5
                * (errc / varc.clamp(min=1e-6) + torch.log(varc.clamp(min=1e-6))).mean()
            )
            agg["nll"].append(float(nll))

            if vis_m.sum() > 10:
                agg["var_vis"].append(float(var[0][vis_m].mean()))
                agg["err_vis"].append(float(err[0][vis_m].mean()))
            if occ_m.sum() > 10:
                agg["var_occ"].append(float(var[0][occ_m].mean()))
                agg["err_occ"].append(float(err[0][occ_m].mean()))
            if oof_m.sum() > 10:
                agg["var_oof"].append(float(var[0][oof_m].mean()))
                agg["err_oof"].append(float(err[0][oof_m].mean()))

            if viz_done < args.viz and fid == target_ids[-1]:
                _save_viz(
                    args.out, si, fid, gt, pred, var[0], err[0], vis_m, occ_m, oof_m
                )
                viz_done += 1

    summary = {k: (float(np.mean(v)) if v else None) for k, v in agg.items()}
    summary["n_scenes"] = ds.length
    summary["n_targets"] = len(agg["psnr"])
    summary["ckpt"] = args.ckpt
    # headline deltas
    if summary["ause_ours"] is not None and summary["ause_random"] is not None:
        summary["ause_gain_vs_random"] = summary["ause_random"] - summary["ause_ours"]
        summary["ause_gain_vs_opacity"] = summary["ause_opacity"] - summary["ause_ours"]
    with open(os.path.join(args.out, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "calib_tables.json"), "w") as f:
        json.dump(calib_tables[:20], f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def _save_viz(outdir, si, fid, gt, pred, var, err, vis_m, occ_m, oof_m):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def chw(t):
        return t.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)

    def hw(t):
        return t.detach().cpu().numpy()

    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    ax[0].imshow(chw(gt))
    ax[0].set_title("GT")
    ax[1].imshow(chw(pred))
    ax[1].set_title("pred (mean)")
    v = hw(var)
    ax[2].imshow(v, cmap="viridis")
    ax[2].set_title(f"pred variance [{v.min():.3f},{v.max():.3f}]")
    e = hw(err)
    ax[3].imshow(e, cmap="viridis")
    ax[3].set_title(f"true sq-error [{e.min():.3f},{e.max():.3f}]")
    reg = np.zeros_like(hw(vis_m).astype(float))
    reg[hw(occ_m)] = 0.5
    reg[hw(oof_m)] = 1.0
    ax[4].imshow(reg, cmap="magma")
    ax[4].set_title("regions (occ=.5 oof=1)")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    p = os.path.join(outdir, f"viz_scene{si:02d}_f{fid}.png")
    plt.savefig(p, dpi=80, bbox_inches="tight")
    plt.close()
    print(f"saved {p}", flush=True)


if __name__ == "__main__":
    main()
