"""
Oracle sanity gate (pre-registered ORACLE_PREREGISTRATION §4.1 gate #2).
MUST pass before any P1-P7 oracle number is trusted:
  1. G_vis (source depth backprojected) rendered back to SOURCE view -> PSNR >= 30 dB.
  2. G_vis rendered to a TARGET view (RE10K pose) -> not random (sanity check pose/scale).

This isolates whether source geometry/scale/K/render is correct BEFORE building hidden Gaussians.
Uses Flash3D Re10KDataset (whitelist: RE10K reader) + UniDepth + render_predicted.

Run on server:
  cd /root/projects/flash3d && source .venv/bin/activate
  export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python /root/sv3d-lab/paper_a_explicit3d/oracle/sanity_gate.py --n_scenes 5
"""

import sys
import argparse
import math
import torch

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, "/root/sv3d-lab/paper_a_explicit3d/oracle")

from oracle_core import (
    backproject,
    make_gaussians,
    set_scale_from_depth,
    render_gaussians_relpose,
    psnr,
    crop5,
)


def build_cfg():
    """Minimal hydra-free cfg to instantiate Re10KDataset for eval."""
    from omegaconf import OmegaConf
    import hydra
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir="/root/projects/flash3d/configs", version_base=None
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=layered_re10k",
                "+dataset.crop_border=true",
                "dataset.test_split_path=splits/re10k_mine_filtered/test_files_present.txt",
                "model.depth.version=v1",
            ],
        )
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_scenes", type=int, default=5)
    args = ap.parse_args()
    device = "cuda"

    print("Loading UniDepth...")
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

    print("Building dataset...")
    cfg = build_cfg()
    from datasets.re10k import Re10KDataset

    ds = Re10KDataset(cfg, split="test")
    print(f"Dataset length: {len(ds)}")

    src_psnrs, tgt_psnrs = [], []
    for i in range(min(args.n_scenes, len(ds))):
        inputs = ds[i]
        # source frame keys
        color_src = inputs[("color", 0, 0)].to(device)  # (3,H,W)
        K_src = inputs[("K_tgt", 0)].to(device)  # pixel K (3x3) for target-res render
        c2w_src = inputs[("T_c2w", 0)].to(device)
        w2c_src = inputs[("T_w2c", 0)].to(device)
        H, W = color_src.shape[1:]

        with torch.no_grad():
            d = unidepth.infer(color_src.unsqueeze(0), intrinsics=K_src.unsqueeze(0))
            depth_src = d["depth"].squeeze()

        # Gaussians live in SOURCE camera frame (Flash3D convention): no c2w applied.
        inv_K = torch.linalg.inv(K_src)
        pts_cam = backproject(depth_src, inv_K, device)  # source-camera coords
        rgb = color_src.permute(1, 2, 0).reshape(-1, 3)
        g_vis = make_gaussians(pts_cam, rgb, K_src[0, 0].item())
        set_scale_from_depth(g_vis, depth_src.reshape(-1), K_src[0, 0].item())

        # gate 1: render back to source (relative pose = identity)
        T_id = torch.eye(4, device=device)
        out_src = render_gaussians_relpose(g_vis, K_src, T_id, H, W, device)
        p_src = psnr(crop5(out_src["render"].clamp(0, 1)), crop5(color_src))
        src_psnrs.append(p_src)

        # gate 2: render to target frame 1 (tgt5) via relative pose src->tgt
        if ("color", 1, 0) in inputs:
            color_t = inputs[("color", 1, 0)].to(device)
            c2w_t = inputs[("T_c2w", 1)].to(device)
            w2c_t = inputs[("T_w2c", 1)].to(device)
            # cam_T_cam(0, 1) = w2c_t @ c2w_src  (source-cam -> target-cam)
            T_rel = w2c_t @ c2w_src
            out_t = render_gaussians_relpose(g_vis, K_src, T_rel, H, W, device)
            p_t = psnr(crop5(out_t["render"].clamp(0, 1)), crop5(color_t))
            tgt_psnrs.append(p_t)
            print(
                f"scene {i}: src_recon={p_src:.2f}dB  tgt5_visonly={p_t:.2f}dB  "
                f"n_gauss={g_vis['xyz'].shape[0]}"
            )
        else:
            print(
                f"scene {i}: src_recon={p_src:.2f}dB  (no tgt)  "
                f"n_gauss={g_vis['xyz'].shape[0]}"
            )

    import numpy as np

    print("\n=== SANITY GATE RESULTS ===")
    print(f"src self-recon PSNR: mean={np.mean(src_psnrs):.2f}dB (GATE: >=30dB)")
    if tgt_psnrs:
        print(
            f"tgt5 vis-only PSNR:  mean={np.mean(tgt_psnrs):.2f}dB "
            f"(should be well above random ~10dB; ~Flash3D tgt5=28.7 is upper ref)"
        )
    gate1 = np.mean(src_psnrs) >= 30.0
    print(f"\nGATE 1 (src recon >=30dB): {'PASS' if gate1 else 'FAIL'}")
    if not gate1:
        print(
            "  -> source geometry/scale/K/render is WRONG. Fix before oracle. "
            "Check: K convention (K_tgt vs K_src), c2w vs w2c, pixel-center, depth scale."
        )


if __name__ == "__main__":
    main()
