"""
Paper A Phase 4b — Visible-Anchored Hidden Gaussian Head.

Design (METHOD_CHARTER_4b.md, post-D027): unlike the collapse-prone canonical-query head
(hidden_head.py, E006: opacity->0 on held-out, LPIPS did NOT rescue), this head spawns hidden
Gaussians ANCHORED to visible source pixels — exactly the property that keeps Flash3D's own
per-pixel offset layers from collapsing. Each source pixel emits K_hidden extra Gaussians that are:
  - placed BEHIND the visible surface along the pixel ray (occluded volume), and/or pushed toward
    the frustum edge (beyond-FOV), via a predicted non-negative along-ray offset;
  - INITIALIZED to inherit the source pixel's color (residual-learned), so a "do nothing" prediction
    already renders plausible continuation instead of gray => the L2 mean-collapse optimum (emit
    nothing) is removed by construction;
  - opacity initialized MODERATE (not near-transparent); the old opacity_bias=-2 is the exact knob
    L2 exploited to emit nothing (decision_log:735).

Hidden Gaussians live in the SOURCE camera frame (same as G_vis) so the unified render is trivial
via render_gaussians_relpose. This module predicts params only; rendering/merging is in PaperAModel.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtish(nn.Module):
    """Lightweight residual conv block over the source feature map."""

    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pw1 = nn.Conv2d(dim, dim * 2, 1)
        self.pw2 = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x):
        r = x
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = F.gelu(x)
        x = self.pw2(x)
        return r + x


class VisibleAnchoredHiddenHead(nn.Module):
    """Predicts K_hidden hidden Gaussians per source pixel, anchored behind/beyond the visible
    surface. Output is at the (unpadded) source resolution flattened to (B, H*W*K, ...).

    Args:
        feat_dim:   channel dim of the source feature map (before projection).
        k_hidden:   hidden Gaussians spawned per source pixel.
        dim:        conv width.
        n_blocks:   number of residual conv blocks.
        scale_lambda / scale_bias: match Flash3D activated-scale convention.
        opacity_init: initial opacity (0..1) BEFORE inv-sigmoid bias; moderate (e.g. 0.3), NOT ~0.
        behind_min / behind_max: along-ray behind-surface offset range (as a fraction of surface depth).
    """

    def __init__(
        self,
        feat_dim,
        k_hidden=2,
        dim=128,
        n_blocks=3,
        scale_lambda=0.01,
        scale_bias=0.02,
        opacity_init=0.3,
        behind_min=0.05,
        behind_max=1.5,
        max_lateral=0.15,
    ):
        super().__init__()
        self.k = k_hidden
        self.scale_lambda = scale_lambda
        self.behind_min = behind_min
        self.behind_max = behind_max
        self.max_lateral = max_lateral

        self.proj = nn.Conv2d(feat_dim, dim, 1)
        self.blocks = nn.ModuleList([ConvNeXtish(dim) for _ in range(n_blocks)])
        self.norm_out = nn.GroupNorm(1, dim)
        # per pixel, per hidden Gaussian: behind(1) lateral(2) log_scale(3) rot(4) opacity(1) drgb(3)=14
        self.per = 1 + 2 + 3 + 4 + 1 + 3
        self.head = nn.Conv2d(dim, self.k * self.per, 1)

        # init: head near-zero so predictions start at the anchor defaults (inherit source color,
        # moderate opacity, behind_min offset). This is the anti-collapse init.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.scale_bias_raw = math.log(max(scale_bias / scale_lambda, 1e-6))
        self.opacity_bias_raw = math.log(opacity_init / (1 - opacity_init))

    def forward(self, feat_map, depth_src, rgb_src, inv_K_src, out_hw):
        """
        feat_map:  (B, C, Hf, Wf) source features.
        depth_src: (B, H, W) source metric depth (unpadded, matches out_hw).
        rgb_src:   (B, 3, H, W) source RGB (unpadded), for color anchoring.
        inv_K_src: (B, 3, 3) source inverse intrinsics (pixel).
        out_hw:    (H, W) output resolution to place hidden Gaussians at.
        Returns dict of (B, N, ...) hidden Gaussian params in the SOURCE camera frame, N=H*W*k.
        """
        B = feat_map.shape[0]
        H, W = out_hw
        device = feat_map.device

        x = self.proj(feat_map)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_out(x)
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        out = self.head(x)  # (B, k*per, H, W)
        out = out.view(B, self.k, self.per, H, W)

        # per-pixel ray directions in source frame (unit), from inv_K
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        ones = torch.ones_like(xs)
        pix = torch.stack([xs, ys, ones], 0).reshape(3, -1)  # (3, HW)
        rays = torch.einsum("bij,jk->bik", inv_K_src, pix)  # (B, 3, HW)
        ray_dir = rays / rays.norm(dim=1, keepdim=True).clamp(min=1e-6)  # (B,3,HW)
        ray_dir = ray_dir.view(B, 3, H, W)
        # surface point along each ray = ray * depth (camera z-scaled ray => use rays*depth)
        surf = rays.view(B, 3, H, W) * depth_src.unsqueeze(
            1
        )  # (B,3,H,W) source-frame surface pts

        # split raw params
        behind_raw = out[:, :, 0]  # (B,k,H,W)
        lateral_raw = out[:, :, 1:3]  # (B,k,2,H,W)
        scale_raw = out[:, :, 3:6]
        rot_raw = out[:, :, 6:10]
        opa_raw = out[:, :, 10]
        drgb_raw = out[:, :, 11:14]

        # behind-surface offset: in [behind_min, behind_max] * depth, monotone via sigmoid
        frac = self.behind_min + (self.behind_max - self.behind_min) * torch.sigmoid(
            behind_raw
        )
        behind = frac * depth_src.unsqueeze(
            1
        )  # (B,k,H,W) distance along ray beyond surface

        # lateral wiggle in the ray's tangent plane (small); build 2 tangent vectors
        up = torch.tensor([0.0, 1.0, 0.0], device=device).view(1, 3, 1, 1)
        t1 = torch.cross(ray_dir, up.expand_as(ray_dir), dim=1)
        t1 = t1 / t1.norm(dim=1, keepdim=True).clamp(min=1e-6)
        t2 = torch.cross(ray_dir, t1, dim=1)
        lat = torch.tanh(lateral_raw) * self.max_lateral  # (B,k,2,H,W)

        xyz_list = []
        rgb_list = []
        scl_list = []
        rot_list = []
        opa_list = []
        for j in range(self.k):
            # placement: surface + ray_dir*behind + tangential wiggle (scaled by depth)
            disp = (
                ray_dir * behind[:, j : j + 1]
                + t1 * (lat[:, j, 0:1] * depth_src.unsqueeze(1))
                + t2 * (lat[:, j, 1:2] * depth_src.unsqueeze(1))
            )  # (B,3,H,W)
            xyz = surf + disp  # (B,3,H,W)
            xyz_list.append(xyz.flatten(2).transpose(1, 2))  # (B,HW,3)

            scl = torch.exp(scale_raw[:, j] + self.scale_bias_raw) * self.scale_lambda
            scl_list.append(scl.flatten(2).transpose(1, 2))  # (B,HW,3)
            rq = F.normalize(rot_raw[:, j], dim=1)  # (B,4,H,W)
            rot_list.append(rq.flatten(2).transpose(1, 2))
            opa = torch.sigmoid(opa_raw[:, j] + self.opacity_bias_raw)  # (B,H,W)
            opa_list.append(opa.flatten(1).unsqueeze(-1))  # (B,HW,1)
            # color: inherit source pixel + learned residual (tanh, small)
            rgb = (rgb_src + 0.5 * torch.tanh(drgb_raw[:, j])).clamp(0, 1)  # (B,3,H,W)
            rgb_list.append(rgb.flatten(2).transpose(1, 2))

        xyz = torch.cat(xyz_list, dim=1)
        scaling = torch.cat(scl_list, dim=1)
        rotation = torch.cat(rot_list, dim=1)
        opacity = torch.cat(opa_list, dim=1)
        rgb = torch.cat(rgb_list, dim=1)
        return {
            "xyz": xyz,
            "scaling": scaling,
            "rotation": rotation,
            "opacity": opacity,
            "rgb": rgb,
        }
