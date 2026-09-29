"""
Paper A — trainer (Phase 4). Fresh implementation (D009). Does NOT copy Flash3D's trainer,
losses, gate thresholds, masks, or hyperparameters.

Losses (D009, region-first + causal + cross-view):
  1. photometric on MERGED render, region-weighted (hidden regions weighted up):
       L1 + SSIM + LPIPS(vgg, after warmup), with per-pixel weight = 1 + w_hidden * hidden_mask.
  2. cross-view consistency: the SAME hidden Gaussians render to ALL target views; each is supervised
     against its GT, so consistency is enforced by sharing one 3D set (no extra term needed at v1,
     but we ALSO add a small penalty that hidden-only renders agree with (GT - vis) residual where
     hidden region is defined — the "load-bearing" signal).
  3. deletion-aware / causal regularizer: encourage hidden-only render to explain the residual the
     visible render leaves in the hidden region (so hidden Gaussians are load-bearing), and an
     opacity sparsity prior so hidden Gaussians don't smear over visible regions.
  4. gaussian scale reg (avoid degenerate huge splats).

Region masks: method-agnostic forward-warp partition (common/geometry/visibility.py).

Run on server (see AGENTS.md env):
  python /root/sv3d-lab/paper_a_explicit3d/train_paper_a.py --level L0 ...
"""

import os
import sys
import json
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, "/root/sv3d-lab")

from common.geometry.visibility import visibility_partition  # noqa: E402


def crop5(img):
    # img (B,3,H,W) or (3,H,W)
    if img.dim() == 4:
        _, _, H, W = img.shape
    else:
        _, H, W = img.shape
    import math

    y0, y1 = int(math.ceil(0.05 * H)), int(math.floor(0.95 * H))
    x0, x1 = int(math.ceil(0.05 * W)), int(math.floor(0.95 * W))
    return img[..., y0:y1, x0:x1]


def psnr(pred, gt, mask=None, eps=1e-10):
    if mask is not None:
        if mask.sum() < 10:
            return None
        pred = pred[..., mask]
        gt = gt[..., mask]
    mse = ((pred - gt) ** 2).mean()
    return (-10 * torch.log10(mse + eps)).item()


class SSIM(nn.Module):
    """SSIM (Flash3D-style local windowed). Fresh minimal implementation."""

    def __init__(self):
        super().__init__()
        self.mu_pool = nn.AvgPool2d(3, 1)
        self.sig_pool = nn.AvgPool2d(3, 1)
        self.pad = nn.ReflectionPad2d(1)
        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        x = self.pad(x)
        y = self.pad(y)
        mu_x = self.mu_pool(x)
        mu_y = self.mu_pool(y)
        sig_x = self.sig_pool(x**2) - mu_x**2
        sig_y = self.sig_pool(y**2) - mu_y**2
        sig_xy = self.sig_pool(x * y) - mu_x * mu_y
        n = (2 * mu_x * mu_y + self.C1) * (2 * sig_xy + self.C2)
        d = (mu_x**2 + mu_y**2 + self.C1) * (sig_x + sig_y + self.C2)
        return torch.clamp((1 - n / d) / 2, 0, 1)


def compute_region_masks(depth_src, K, T_src2tgt, H, W, device):
    """visible/occluded/oof (H,W) bool for one target. hidden = occluded|oof."""
    vis, occ, oof = visibility_partition(depth_src, K, T_src2tgt, H, W, device)
    return vis, occ, oof


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["L0", "L1", "L2"], default="L0")
    ap.add_argument("--n_scenes", type=int, default=2)
    ap.add_argument("--scene_start", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_queries", type=int, default=2048)
    ap.add_argument(
        "--w_hidden", type=float, default=4.0, help="hidden-region photometric weight"
    )
    ap.add_argument("--w_lpips", type=float, default=0.25)
    ap.add_argument("--lpips_after", type=int, default=200)
    ap.add_argument("--w_causal", type=float, default=1.0)
    ap.add_argument("--w_opa_sparse", type=float, default=1e-3)
    ap.add_argument("--w_scale_reg", type=float, default=0.1)
    ap.add_argument(
        "--w_null",
        type=float,
        default=0.0,
        help="source-null constraint weight (D027/4b)",
    )
    ap.add_argument("--freeze_visible", action="store_true")
    ap.add_argument(
        "--anchor_mode",
        choices=["canonical", "frustum", "visible"],
        default="canonical",
    )
    ap.add_argument("--behind_offset", type=float, default=0.3)
    ap.add_argument(
        "--k_hidden",
        type=int,
        default=2,
        help="hidden Gaussians per source pixel (visible anchor)",
    )
    ap.add_argument(
        "--opacity_init",
        type=float,
        default=0.3,
        help="moderate opacity init (anti-collapse)",
    )
    ap.add_argument(
        "--latent_dim", type=int, default=0, help=">0 enables CVAE hidden head (D013)"
    )
    ap.add_argument(
        "--latent_mode",
        choices=["global", "spatial"],
        default="global",
        help="CVAE latent granularity: global pooled z (D013 r1) or per-cell grid (D013 r2)",
    )
    ap.add_argument(
        "--w_kl", type=float, default=1e-3, help="beta for KL(q||p) in CVAE"
    )
    ap.add_argument(
        "--kl_anneal", type=int, default=2000, help="linear KL warmup steps"
    )
    ap.add_argument("--novel_frames", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument(
        "--split",
        default="/root/projects/flash3d/splits/re10k_mine_filtered/test_files_wide.txt",
    )
    ap.add_argument("--out", default="/home/data/sv3d-lab/runs/paper_a_L0")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    return ap
