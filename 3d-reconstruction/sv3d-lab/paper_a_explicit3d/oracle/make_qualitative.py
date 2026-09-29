"""
Qualitative side-by-side generator for Paper A (leakage-free hold-out view).

For fixed scene ids, per scene: optimize hidden Gaussians on the OPT frames (holdout protocol), then render
the HELD-OUT (never-optimized) view three ways and save a horizontal triptych:
    [ GT | Flash3D (visible-only) | Ours (visible + hidden) ]
plus the per-image PSNR against GT in the filename. No cherry-picking: scene ids are fixed via --scene_ids.

Run on server (flash3d venv), AFTER headline runs finish (uses the GPU):
  cd /root/projects/flash3d && source .venv/bin/activate
  export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python /root/sv3d-lab/paper_a_explicit3d/oracle/make_qualitative.py \
      --scene_ids 0 1 2 3 4 5 --steps 500 --holdout_frame 3 \
      --split splits/re10k_mine_filtered/test_files_wide700.txt \
      --out /home/data/sv3d-lab/evaluations/qualitative_wide
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


def _to_uint8(img):
    a = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return a


def save_triptych(gt, flash3d, ours, path):
    h = gt.shape[1]
    gap = np.full((h, 6, 3), 255, dtype=np.uint8)
    strip = np.concatenate(
        [_to_uint8(gt), gap, _to_uint8(flash3d), gap, _to_uint8(ours)], axis=1
    )
    Image.fromarray(strip).save(path)


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
    # Keep defaults identical to E018/E021 unless explicitly overridden.
    ap.add_argument("--w_opa_sparse", type=float, default=0.0)
    ap.add_argument("--w_scale", type=float, default=0.0)
    ap.add_argument("--max_scale", type=float, default=0.0)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--curve_steps", type=int, nargs="+", default=[])
    ap.add_argument("--verbose", action="store_true")
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
                flash3d = render_gaussians_relpose(g_vis, K, T_rel, H, W, device)[
                    "render"
                ].clamp(0, 1)
                ours = render_gaussians_relpose(g_merged, K, T_rel, H, W, device)[
                    "render"
                ].clamp(0, 1)
                p_flash = psnr(crop5(flash3d), crop5(gt))
                p_ours = psnr(crop5(ours), crop5(gt))
                fn = f"scene{sid}_holdout{f}_flash{p_flash:.2f}_ours{p_ours:.2f}.png"
                save_triptych(gt, flash3d, ours, os.path.join(args.out, fn))
                manifest.append(
                    {
                        "scene": sid,
                        "holdout_frame": f,
                        "psnr_flash3d": p_flash,
                        "psnr_ours": p_ours,
                        "delta": p_ours - p_flash,
                        "file": fn,
                    }
                )
                print(f"  {fn}")
        del r, g_vis, hidden
        torch.cuda.empty_cache()

    with open(os.path.join(args.out, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)
    print("\n=== QUALITATIVE MANIFEST (layout: GT | Flash3D | Ours) ===")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
