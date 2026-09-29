"""
Diagnostic: decompose a hold-out-view render into its components to attribute artifacts.
Layout per row: [ GT | Flash3D (visible-only) | hidden-only | Ours (merged) ]
- If the "comb/venetian-blind" streaks on foreground appear ALREADY in the visible-only panel,
  they are Flash3D's wide-baseline stretching artifact (we inherit, do not cause).
- If the noise/mottling appears in the hidden-only panel, it is our hidden Gaussians' texture.

Run on server (flash3d venv):
  python /root/sv3d-lab/paper_a_explicit3d/oracle/make_component_viz.py \
     --scene_ids 6 --steps 500 --holdout_frame 3 \
     --split splits/re10k_mine_filtered/test_files_wide700.txt \
     --out /home/data/sv3d-lab/evaluations/component_wide
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, "/root/sv3d-lab/paper_a_explicit3d/oracle")
sys.path.insert(0, "/root/sv3d-lab")

from oracle_core import render_gaussians_relpose, merge_gaussians, psnr, crop5
from run_oracle import build_cfg
import hidden_opt_oracle as hoo
from PIL import Image


def _u8(img):
    return (
        (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    )


def save_strip(imgs, path):
    h = imgs[0].shape[1]
    gap = np.full((h, 6, 3), 255, dtype=np.uint8)
    parts = []
    for i, im in enumerate(imgs):
        parts.append(_u8(im))
        if i < len(imgs) - 1:
            parts.append(gap)
    Image.fromarray(np.concatenate(parts, axis=1)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_ids", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--split", default="splits/re10k_mine_filtered/test_files_wide700.txt"
    )
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--loss", default="l2", choices=["l2", "l1", "lpips"])
    ap.add_argument("--holdout_frame", type=int, default=3)
    ap.add_argument("--lr_xyz", type=float, default=2e-4)
    ap.add_argument("--lr_scale", type=float, default=5e-3)
    ap.add_argument("--lr_rot", type=float, default=1e-3)
    ap.add_argument("--lr_opacity", type=float, default=5e-2)
    ap.add_argument("--lr_rgb", type=float, default=1e-2)
    ap.add_argument("--w_lpips", type=float, default=1.0)
    ap.add_argument("--w_null", type=float, default=10.0)
    ap.add_argument("--w_null_alpha", type=float, default=5.0)
    ap.add_argument("--footprint", type=float, default=0.15)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--opacity_init", type=float, default=0.05)
    ap.add_argument("--appearance_lr_mult", type=float, default=3.0)
    # regularization knobs (0 = off, = current paper config)
    ap.add_argument("--w_opa_sparse", type=float, default=0.0)
    ap.add_argument("--w_scale", type=float, default=0.0)
    ap.add_argument("--max_scale", type=float, default=0.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--curve_steps", type=int, nargs="+", default=[])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    print("Loading UniDepth v1...")
    unidepth = (
        torch.hub.load(
            "lpiccinelli-eth/UniDepth",
            "UniDepth",
            version="v1",
            backbone="vitl14",
            pretrained=True,
            trust_repo=True,
        )
        .to(device)
        .eval()
    )
    try:
        import lpips as lpips_lib

        lpips_fn = lpips_lib.LPIPS(net="vgg").to(device).eval()
    except Exception as e:
        print(f"lpips unavailable ({e})")
        lpips_fn = None

    cfg = build_cfg(args.split)
    from datasets.re10k import Re10KDataset

    ds = Re10KDataset(cfg, split="test")
    print(f"Dataset length: {len(ds)}")

    manifest = []
    for sid in args.scene_ids:
        if sid >= len(ds):
            print(f"scene {sid} out of range, skip")
            continue
        r = hoo.process_scene(unidepth, ds[sid], device, args, lpips_fn)
        g_vis = r["_g_vis"]
        hidden = r["_hidden_module"]
        K = r["_K"]
        H, W = r["_HW"]
        eval_frames = r["_eval_frames"]
        tgt = r["_tgt"]
        with torch.no_grad():
            gh = hidden.gauss()
            g_merged = merge_gaussians([g_vis, gh])
            for f in eval_frames:
                gt = tgt[f]["gt"]
                T_rel = tgt[f]["T_rel"]
                vis = render_gaussians_relpose(g_vis, K, T_rel, H, W, device)[
                    "render"
                ].clamp(0, 1)
                hidonly = render_gaussians_relpose(gh, K, T_rel, H, W, device)[
                    "render"
                ].clamp(0, 1)
                merged = render_gaussians_relpose(g_merged, K, T_rel, H, W, device)[
                    "render"
                ].clamp(0, 1)
                p_flash = psnr(crop5(vis), crop5(gt))
                p_ours = psnr(crop5(merged), crop5(gt))
                fn = f"scene{sid}_holdout{f}_COMPONENTS_flash{p_flash:.2f}_ours{p_ours:.2f}.png"
                save_strip([gt, vis, hidonly, merged], os.path.join(args.out, fn))
                manifest.append(
                    {
                        "scene": sid,
                        "holdout_frame": f,
                        "psnr_flash3d": p_flash,
                        "psnr_ours": p_ours,
                        "delta": p_ours - p_flash,
                        "final_hidden_opacity_mean": r["final_hidden_opacity_mean"],
                        "n_hidden": r["n_hidden"],
                        "file": fn,
                        "layout": "GT | Flash3D(vis-only) | hidden-only | Ours(merged)",
                    }
                )
                print(
                    f"  {fn}  opacity={r['final_hidden_opacity_mean']:.3f} n_hidden={r['n_hidden']}"
                )
        del r, g_vis, hidden
        torch.cuda.empty_cache()

    with open(os.path.join(args.out, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)
    print("\n=== COMPONENT MANIFEST (GT | Flash3D | hidden-only | Ours) ===")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
