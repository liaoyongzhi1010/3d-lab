"""
Paper A — L1/L2 evaluator. Loads a trained checkpoint, runs over the (overfit or pilot) scene set,
computes region-separated PSNR/SSIM + causal deletion Δ averaged over scenes & targets. Also dumps
a few visualizations (merged / vis-only / hidden-only / GT + masks) for personal inspection.

This is the DECISIVE L1 verdict (per D009), NOT the noisy per-step training log.

Usage (server):
  python /root/sv3d-lab/paper_a_explicit3d/eval_paper_a.py \
     --ckpt /home/data/sv3d-lab/runs/paper_a_L1/ckpt_final.pt --n_scenes 20 \
     --out /home/data/sv3d-lab/evaluations/paper_a_L1 --viz 4
"""

import os
import sys
import json
import math
import argparse
import importlib.util

import numpy as np
import torch

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, "/root/sv3d-lab")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import PaperAModel, PaperAConfig
from train_paper_a import build_cfg, load_re10k, to_device
from trainer_lib import compute_region_masks, crop5, psnr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_scenes", type=int, default=20)
    ap.add_argument("--scene_start", type=int, default=0)
    ap.add_argument("--n_queries", type=int, default=16384)
    ap.add_argument("--novel_frames", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument(
        "--split",
        default="/root/projects/flash3d/splits/re10k_mine_filtered/test_files_wide.txt",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--viz", type=int, default=4)
    ap.add_argument("--latent_dim", type=int, default=0)
    ap.add_argument("--latent_mode", choices=["global", "spatial"], default="global")
    ap.add_argument("--z_mode", default="prior", choices=["prior", "prior_mean"])
    ap.add_argument(
        "--best_of_k", type=int, default=1, help="CVAE: report best-of-K prior samples"
    )
    ap.add_argument(
        "--anchor_mode",
        choices=["canonical", "frustum", "visible"],
        default="canonical",
    )
    ap.add_argument("--k_hidden", type=int, default=2)
    ap.add_argument("--opacity_init", type=float, default=0.3)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    cfg = build_cfg(args.novel_frames, args.split)
    Re10K = load_re10k()
    ds = Re10K(cfg, split="test")
    ds._seq_key_src_idx_pairs = ds._seq_key_src_idx_pairs[
        args.scene_start : args.scene_start + args.n_scenes
    ]
    ds.length = len(ds._seq_key_src_idx_pairs)

    from torch.utils.data import DataLoader
    from datasets.util import custom_collate

    loader = DataLoader(ds, 1, shuffle=False, num_workers=0, collate_fn=custom_collate)

    model = PaperAModel(
        cfg,
        PaperAConfig(
            n_queries=args.n_queries,
            anchor_mode=args.anchor_mode,
            latent_dim=args.latent_dim,
            latent_mode=args.latent_mode,
            k_hidden=args.k_hidden,
            opacity_init=args.opacity_init,
        ),
    ).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd["model"], strict=False)
    model.eval()

    pad = cfg.dataset.pad_border_aug

    agg = {
        "psnr_hidden_merged": [],
        "psnr_hidden_vis": [],
        "deletion_delta": [],
        "psnr_all_merged": [],
        "psnr_occ_merged": [],
        "psnr_occ_vis": [],
        "psnr_oof_merged": [],
        "psnr_oof_vis": [],
        "hidden_opacity": [],
        "psnr_visible_merged": [],
        "_wsum": [],
        "_wden": [],
    }
    per_target = []
    viz_saved = 0

    for si, inputs in enumerate(loader):
        inputs = to_device(inputs, device)
        target_ids = [f for f in args.novel_frames if ("color", f, 0) in inputs]
        if not target_ids:
            continue
        with torch.no_grad():
            K = max(1, args.best_of_k) if args.latent_dim > 0 else 1
            zmode = args.z_mode if args.latent_dim > 0 else None
            outs = [
                model(inputs, target_ids, hidden_ray_fid=target_ids[-1], z_mode=zmode)
                for _ in range(K)
            ]
        out = outs[0]
        depth_p = out[("depth", 0)][0, 0]
        depth_src = (
            depth_p[pad : depth_p.shape[0] - pad, pad : depth_p.shape[1] - pad]
            if pad
            else depth_p
        )
        K0 = inputs[("K_src", 0)][0]
        H = inputs["color", 0, 0].shape[2]
        W = inputs["color", 0, 0].shape[3]
        agg["hidden_opacity"].append(
            float(out["hidden_gaussians"][0]["opacity"].mean())
        )

        for fid in target_ids:
            gt = inputs[("color", fid, 0)][0]
            vonly = out[("render_vis", fid)][0]
            T = out[("cam_T_cam", 0, fid)][0]
            vis_m, occ_m, oof_m = compute_region_masks(depth_src, K0, T, H, W, device)
            hid_m = occ_m | oof_m
            hfrac = float(hid_m.float().mean())
            # best-of-K: pick the sample whose merged render best matches GT in the hidden region
            best_pm, best_merged, best_honly = (
                None,
                out[("render_merged", fid)][0],
                out[("render_hidden", fid)][0],
            )
            for o in outs:
                mcand = o[("render_merged", fid)][0]
                pcand = psnr(mcand, gt, hid_m)
                if pcand is not None and (best_pm is None or pcand > best_pm):
                    best_pm, best_merged, best_honly = (
                        pcand,
                        mcand,
                        o[("render_hidden", fid)][0],
                    )
            merged = best_merged
            honly = best_honly
            pm = best_pm if best_pm is not None else psnr(merged, gt, hid_m)
            pv = psnr(vonly, gt, hid_m)
            # visible-region quality (merged vs GT in VISIBLE region) — must not degrade
            pvis_merged = psnr(merged, gt, vis_m)
            if pvis_merged is not None:
                agg["psnr_visible_merged"].append(pvis_merged)
            if pm is not None and pv is not None:
                agg["psnr_hidden_merged"].append(pm)
                agg["psnr_hidden_vis"].append(pv)
                agg["deletion_delta"].append(pm - pv)
                per_target.append(
                    {
                        "scene": si,
                        "fid": fid,
                        "hidden_frac": hfrac,
                        "delta": pm - pv,
                        "hid_merged": pm,
                        "hid_vis": pv,
                    }
                )
                # hidden-frac-weighted delta (area-weighted; more faithful to actual hidden pixels)
                agg["_wsum"].append((pm - pv) * hfrac)
                agg["_wden"].append(hfrac)
            for name, m in [("occ", occ_m), ("oof", oof_m)]:
                a = psnr(merged, gt, m)
                b = psnr(vonly, gt, m)
                if a is not None:
                    agg[f"psnr_{name}_merged"].append(a)
                    agg[f"psnr_{name}_vis"].append(b)
            agg["psnr_all_merged"].append(
                psnr(crop5(merged.unsqueeze(0)), crop5(gt.unsqueeze(0)))
            )

            # save viz for the WIDEST target (last), where hidden regions are substantial
            if viz_saved < args.viz and fid == target_ids[-1]:
                _save_viz(args.out, si, fid, gt, merged, vonly, honly, hid_m)
                viz_saved += 1

    summary = {
        k: (float(np.mean([x for x in v if x is not None])) if len(v) else None)
        for k, v in agg.items()
        if not k.startswith("_")
    }
    # area-weighted deletion delta (weights each measurement by its hidden-pixel fraction)
    wsum = sum(agg["_wsum"])
    wden = sum(agg["_wden"])
    summary["deletion_delta_area_weighted"] = float(wsum / wden) if wden > 0 else None
    # delta restricted to targets with meaningful hidden regions (>=5% of image)
    big = [d["delta"] for d in per_target if d["hidden_frac"] >= 0.05]
    summary["deletion_delta_hidden_ge5pct"] = float(np.mean(big)) if big else None
    summary["n_targets_hidden_ge5pct"] = len(big)
    summary["n_scenes"] = args.n_scenes
    summary["ckpt"] = args.ckpt
    with open(os.path.join(args.out, "per_target.json"), "w") as f:
        json.dump(per_target, f, indent=2)
    with open(os.path.join(args.out, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def _save_viz(out_dir, si, fid, gt, merged, vonly, honly, hid_m):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def t2i(t):
        return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    ax[0].imshow(t2i(gt))
    ax[0].set_title("GT")
    ax[1].imshow(t2i(merged))
    ax[1].set_title("merged")
    ax[2].imshow(t2i(vonly))
    ax[2].set_title("vis-only")
    ax[3].imshow(t2i(honly))
    ax[3].set_title("hidden-only")
    ax[4].imshow(hid_m.detach().cpu().numpy(), cmap="gray")
    ax[4].set_title("hidden mask")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    p = os.path.join(out_dir, f"viz_scene{si:02d}_f{fid}.png")
    plt.savefig(p, dpi=80, bbox_inches="tight")
    plt.close()
    print("saved", p)


if __name__ == "__main__":
    main()
