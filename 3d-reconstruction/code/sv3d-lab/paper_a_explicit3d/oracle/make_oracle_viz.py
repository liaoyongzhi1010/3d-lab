"""
Generate >=10 fixed oracle visualizations (pre-registration ORACLE_PREREGISTRATION §3 requirement).
For deterministic scene ids: source, GT target, vis-only render, merged render, hidden mask,
and hidden-only render. Saved to /home/data/sv3d-lab/visualizations/oracle/.

Run on server:
  cd /root/projects/flash3d && source .venv/bin/activate
  export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python /root/sv3d-lab/paper_a_explicit3d/oracle/make_oracle_viz.py --n_scenes 12
"""

import os
import sys
import argparse
import torch
import torchvision.utils as vutils

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, "/root/sv3d-lab/paper_a_explicit3d/oracle")
sys.path.insert(0, "/root/sv3d-lab")

from oracle_core import (
    backproject,
    make_gaussians,
    set_scale_from_depth,
    merge_gaussians,
    render_gaussians_relpose,
)
from common.geometry.visibility import visibility_partition
from run_oracle import build_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_scenes", type=int, default=12)
    ap.add_argument("--split", default="splits/re10k_mine_filtered/test_files_wide.txt")
    ap.add_argument("--out", default="/home/data/sv3d-lab/visualizations/oracle")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

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
    cfg = build_cfg(args.split)
    from datasets.re10k import Re10KDataset

    ds = Re10KDataset(cfg, split="test")

    for i in range(min(args.n_scenes, len(ds))):
        inputs = ds[i]
        color_src = inputs[("color", 0, 0)].to(device)
        K = inputs[("K_tgt", 0)].to(device)
        c2w_src = inputs[("T_c2w", 0)].to(device)
        w2c_src = inputs[("T_w2c", 0)].to(device)
        H, W = color_src.shape[1:]
        inv_K = torch.linalg.inv(K)
        with torch.no_grad():
            depth_src = unidepth.infer(
                color_src.unsqueeze(0), intrinsics=K.unsqueeze(0)
            )["depth"].squeeze()
        pts_src = backproject(depth_src, inv_K, device)
        rgb_src = color_src.permute(1, 2, 0).reshape(-1, 3)
        g_vis = make_gaussians(pts_src, rgb_src, K[0, 0].item())
        set_scale_from_depth(g_vis, depth_src.reshape(-1), K[0, 0].item(), 0.15)

        # widest target = frame 3
        f = 3 if ("color", 3, 0) in inputs else 1
        color_t = inputs[("color", f, 0)].to(device)
        c2w_t = inputs[("T_c2w", f)].to(device)
        w2c_t = inputs[("T_w2c", f)].to(device)
        with torch.no_grad():
            depth_t = unidepth.infer(color_t.unsqueeze(0), intrinsics=K.unsqueeze(0))[
                "depth"
            ].squeeze()
        pts_t_cam = backproject(depth_t, inv_K, device)
        T_ts = w2c_src @ c2w_t
        N = pts_t_cam.shape[0]
        homo = torch.cat([pts_t_cam, torch.ones(N, 1, device=device)], 1)
        pts_in_src = (T_ts @ homo.T).T[:, :3]
        rgb_t = color_t.permute(1, 2, 0).reshape(-1, 3)

        T_st = w2c_t @ c2w_src  # src->tgt
        visible, occluded, oof = visibility_partition(
            depth_src, K, T_st, H, W, device, dilate=2
        )
        hidden_pix = (occluded | oof).reshape(-1)

        g_hidden = make_gaussians(
            pts_in_src[hidden_pix], rgb_t[hidden_pix], K[0, 0].item()
        )
        set_scale_from_depth(
            g_hidden, pts_in_src[hidden_pix][:, 2].clamp(min=0.1), K[0, 0].item(), 0.15
        )
        g_merged = merge_gaussians([g_vis, g_hidden])

        merged = render_gaussians_relpose(g_merged, K, T_st, H, W, device)[
            "render"
        ].clamp(0, 1)
        visonly = render_gaussians_relpose(g_vis, K, T_st, H, W, device)[
            "render"
        ].clamp(0, 1)
        hidden_only = render_gaussians_relpose(g_hidden, K, T_st, H, W, device)[
            "render"
        ].clamp(0, 1)

        mask_rgb = torch.zeros(3, H, W, device=device)
        mask_rgb[0][occluded] = 1.0  # red = occluded
        mask_rgb[2][oof] = 1.0  # blue = oof
        mask_rgb[1][visible] = 0.5  # green-ish = visible

        panel = torch.stack(
            [color_src, color_t, visonly, merged, hidden_only, mask_rgb], 0
        )
        vutils.save_image(
            panel, os.path.join(args.out, f"scene{i:02d}_panel.png"), nrow=3
        )
        print(
            f"scene {i}: saved panel (src|tgt|vis|merged|hidden|mask), "
            f"hidden_frac={hidden_pix.float().mean():.3f}"
        )

    print(f"\nSaved {min(args.n_scenes, len(ds))} panels to {args.out}")


if __name__ == "__main__":
    main()
