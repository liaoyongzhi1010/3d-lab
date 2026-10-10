"""
D030 oracle-teacher generator: for each training scene, run the E014 per-scene hidden-Gaussian
optimizer (hidden_opt_oracle.process_scene) and CACHE the optimized G*_hidden (source-frame Gaussians)
to disk. These concrete self-consistent 3D solutions become the DISTILLATION TARGETS for the amortized
feed-forward head (train_paper_a --distill_dir), converting the un-learnable multi-modal RGB regression
(which collapsed 4x: E006/E008/E009/E015) into a learnable one-teacher-per-scene regression.

Saved per scene (npz): xyz(N,3) scaling(N,3) rotation(N,4) opacity(N,1) rgb(N,3) in SOURCE cam frame,
plus the scene key + src/target frame ids + per-target T_rel + quality (deletion_delta) for filtering.

Run on server (flash3d venv):
  cd /root/projects/flash3d && source .venv/bin/activate
  export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python /root/sv3d-lab/paper_a_explicit3d/oracle/gen_oracle_teacher.py \
      --n_scenes 600 --steps 400 --loss l2 --split splits/re10k_mine_filtered/test_files_wide700.txt \
      --out /home/data/sv3d-lab/teachers/wide700_l2
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

import hidden_opt_oracle as H
from run_oracle import build_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_scenes", type=int, default=600)
    ap.add_argument("--scene_start", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--split", default="splits/re10k_mine_filtered/test_files_wide700.txt"
    )
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--loss", default="l2", choices=["l2", "l1", "lpips"])
    ap.add_argument("--w_lpips", type=float, default=1.0)
    ap.add_argument("--w_null", type=float, default=5.0)
    ap.add_argument("--w_null_alpha", type=float, default=5.0)
    ap.add_argument("--footprint", type=float, default=0.15)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--opacity_init", type=float, default=0.05)
    ap.add_argument("--lr_xyz", type=float, default=2e-4)
    ap.add_argument("--lr_scale", type=float, default=5e-3)
    ap.add_argument("--lr_rot", type=float, default=1e-3)
    ap.add_argument("--lr_opacity", type=float, default=5e-2)
    ap.add_argument("--lr_rgb", type=float, default=1e-2)
    ap.add_argument(
        "--opacity_keep",
        type=float,
        default=0.02,
        help="drop teacher Gaussians below this opacity to shrink cache",
    )
    ap.add_argument(
        "--min_deletion",
        type=float,
        default=0.2,
        help="skip scenes whose oracle deletion_delta < this (bad teacher)",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    print("Loading UniDepth v1...", flush=True)
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
    lpips_fn = None
    if args.loss == "lpips":
        import lpips as lpips_lib

        lpips_fn = lpips_lib.LPIPS(net="vgg").to(device).eval()

    cfg = build_cfg(args.split)
    from datasets.re10k import Re10KDataset

    ds = Re10KDataset(cfg, split="test")
    n = min(args.n_scenes, len(ds) - args.scene_start)
    print(
        f"Dataset length {len(ds)}; generating teachers for {n} scenes from {args.scene_start}",
        flush=True,
    )

    manifest = []
    kept = 0
    deltas = []
    for i in range(args.scene_start, args.scene_start + n):
        try:
            inputs = ds[i]
            r = H.process_scene(unidepth, inputs, device, args, lpips_fn)
        except Exception as e:
            print(f"scene {i} FAILED: {e}", flush=True)
            continue
        # oracle quality = mean hidden-region deletion delta across targets
        d_hidden = [
            m["delta_hidden"]
            for m in r["per_target"].values()
            if m.get("delta_hidden") is not None
        ]
        d_overall = [m["delta_overall"] for m in r["per_target"].values()]
        q = float(np.mean(d_hidden)) if d_hidden else 0.0
        q_ov = float(np.mean(d_overall)) if d_overall else 0.0
        deltas.append(q)
        if q < args.min_deletion:
            if args.verbose:
                print(
                    f"  scene {i} skipped (deletion {q:.3f} < {args.min_deletion})",
                    flush=True,
                )
            continue

        hidden = r["_hidden_module"]
        with torch.no_grad():
            g = hidden.gauss()
            opa = g["opacity"].squeeze(-1)
            keep = opa >= args.opacity_keep
            if keep.sum() < 16:
                keep = torch.ones_like(opa, dtype=torch.bool)
            xyz = g["xyz"][keep].cpu().numpy().astype(np.float32)
            scaling = g["scaling"][keep].cpu().numpy().astype(np.float32)
            rotation = g["rotation"][keep].cpu().numpy().astype(np.float32)
            opacity = g["opacity"][keep].cpu().numpy().astype(np.float32)
            rgb = g["rgb_direct"][keep].cpu().numpy().astype(np.float32)

        # scene key: Re10KDataset stores (seq_key, src_idx) pairs
        try:
            seq_key, src_idx = ds._seq_key_src_idx_pairs[i]
        except Exception:
            seq_key, src_idx = f"idx{i}", -1
        fn = os.path.join(args.out, f"teacher_{i:06d}.npz")
        np.savez_compressed(
            fn,
            xyz=xyz,
            scaling=scaling,
            rotation=rotation,
            opacity=opacity,
            rgb=rgb,
            seq_key=str(seq_key),
            src_idx=int(src_idx) if isinstance(src_idx, (int, np.integer)) else -1,
            deletion_hidden=q,
            deletion_overall=q_ov,
            n_gauss=xyz.shape[0],
        )
        manifest.append(
            {
                "idx": i,
                "file": os.path.basename(fn),
                "seq_key": str(seq_key),
                "src_idx": int(src_idx)
                if isinstance(src_idx, (int, np.integer))
                else -1,
                "n_gauss": int(xyz.shape[0]),
                "deletion_hidden": q,
                "deletion_overall": q_ov,
            }
        )
        kept += 1
        if (i + 1) % 10 == 0:
            print(
                f"[{i + 1}] kept={kept} mean_deletion={np.mean(deltas):.3f} "
                f"last_n_gauss={xyz.shape[0]}",
                flush=True,
            )

    with open(os.path.join(args.out, "manifest.json"), "w") as fp:
        json.dump(
            {
                "n_kept": kept,
                "n_seen": n,
                "mean_deletion_hidden": float(np.mean(deltas)) if deltas else 0.0,
                "args": vars(args),
                "scenes": manifest,
            },
            fp,
            indent=2,
        )
    print(
        f"\nDONE: kept {kept}/{n} teachers, mean hidden deletion {np.mean(deltas):.3f} "
        f"-> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
