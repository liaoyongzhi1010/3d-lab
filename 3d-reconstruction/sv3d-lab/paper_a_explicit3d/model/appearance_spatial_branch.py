"""Appearance-aware spatial guidance branch for single-image 3DGS (Paper A).

Design goal: reproduce CATSplat's proven spatial-guidance gain (backproject depth
-> point encoder -> cross-attention into image features) but (1) WITHOUT the LLaVA
text branch, and (2) with an END-TO-END trained, APPEARANCE-AWARE point encoder that
consumes RGB+XYZ per point (CATSplat uses a frozen ModelNet40 PointNet on XYZ only).

Novelty vs CATSplat:
  - CATSplat point encoder: PointNet pretrained on ModelNet40 (CAD objects), FROZEN,
    geometry-only (XYZ). Domain-mismatched to real scenes, no appearance.
  - Ours: lightweight PointNet-style encoder trained jointly on RE10K, input = (XYZ, RGB)
    per point. Scene points carry colour -> orthogonal information CATSplat never uses.

Integration point: enrich the DEEPEST ResNet feature (2048ch @ H/32 for ResNet50, i.e.
~10x14=140 tokens) via cross-attention. This is the cheapest map (few tokens) and is the
one the Gaussian/depth decoders start upsampling from, so guidance propagates to all
per-pixel Gaussian params.

This module is self-contained (no import of the Flash3D package) so it can be unit-tested
locally and dropped into models/encoder/ on the server.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointFeatureEncoder(nn.Module):
    """PointNet-style per-point encoder over (XYZ, RGB), trained end-to-end.

    Input:  points [B, N, 6]  (xyz in metric depth space + rgb in [0,1])
    Output: point tokens [B, M, d_model]  (M = num_out_tokens via learned pooling)

    Uses shared MLPs + a global feature concat (classic PointNet trick) so each point
    token is aware of global context, then a small set of learned query tokens pools the
    per-point features into M tokens for cheap cross-attention downstream.
    """

    def __init__(self, d_model=256, hidden=128, num_out_tokens=32):
        super().__init__()
        self.num_out_tokens = num_out_tokens

        # shared per-point MLP (operates on 6D xyz+rgb)
        self.mlp1 = nn.Sequential(
            nn.Linear(6, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        # after concatenating global max-pooled feature -> per-point MLP
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # learned query tokens pool variable-N points -> fixed M tokens
        self.query_tokens = nn.Parameter(torch.randn(num_out_tokens, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, points):
        # points: [B, N, 6]
        f = self.mlp1(points)  # [B, N, hidden]
        g = f.max(dim=1, keepdim=True).values  # [B, 1, hidden] global feature
        f = torch.cat([f, g.expand(-1, f.shape[1], -1)], dim=-1)  # [B, N, 2*hidden]
        f = self.mlp2(f)  # [B, N, d_model]
        # pool to M tokens via cross-attention with learned queries
        B = f.shape[0]
        q = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, M, d_model]
        pooled, _ = self.pool_attn(q, f, f)  # [B, M, d_model]
        return self.norm(pooled)


class SpatialGuidanceFusion(nn.Module):
    """Cross-attention fusion: enrich image feature map with 3D point tokens.

    image_feat: [B, C, H, W]  (deepest ResNet feature)
    point_tokens: [B, M, d_model]
    returns: enriched image_feat [B, C, H, W] (additive residual, per-channel gated).

    Stability design (after D023 showed a scalar-gated overwrite destabilised training):
      - PER-CHANNEL zero-init gate (vector [C]) instead of a single scalar -> each output
        channel independently decides how much guidance to admit; at init all zero => exact
        identity => from-scratch training starts identical to plain Flash3D.
      - img_proj_out uses default (non-zero) init so d(loss)/d(gate)!=0 (no deadlock).
      - The residual is ADDED to image_feat (never overwrites), and the whole branch is a
        pure residual side-path: at init the forward AND the backbone gradient are unchanged.
    """

    def __init__(self, img_channels, d_model=256, num_heads=8):
        super().__init__()
        self.img_channels = img_channels
        self.d_model = d_model

        self.img_proj_in = nn.Conv2d(img_channels, d_model, 1)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.img_proj_out = nn.Conv2d(d_model, img_channels, 1)
        # PER-CHANNEL zero-init gate -> identity at step0, no deadlock (img_proj_out!=0).
        self.gate = nn.Parameter(torch.zeros(img_channels))

    def forward(self, image_feat, point_tokens):
        B, C, H, W = image_feat.shape
        x = self.img_proj_in(image_feat)  # [B, d, H, W]
        x_tok = x.flatten(2).transpose(1, 2)  # [B, HW, d]

        q = self.norm_q(x_tok)
        kv = self.norm_kv(point_tokens)
        attn_out, _ = self.cross_attn(q, kv, kv)  # [B, HW, d]
        x_tok = x_tok + attn_out
        x_tok = x_tok + self.ffn(self.norm_ffn(x_tok))

        x = x_tok.transpose(1, 2).reshape(B, self.d_model, H, W)
        residual = self.img_proj_out(x)  # [B, C, H, W]
        gate = self.gate.view(1, C, 1, 1)  # [1, C, 1, 1], zero at init
        return image_feat + gate * residual


class AppearanceSpatialBranch(nn.Module):
    """Full branch: sample points from backprojected depth + RGB, encode, fuse.

    Called with:
      - depth:  [B, 1, H, W]   metric depth (first Gaussian layer depth)
      - rgb:    [B, 3, H, W]   source image (aligned to depth resolution)
      - inv_K:  [B, 3, 3] or [B,4,4] inverse intrinsics for the depth map resolution
      - image_feat: [B, C, hf, wf] deepest ResNet feature to enrich
    Returns enriched image_feat [B, C, hf, wf].
    """

    def __init__(self, img_channels, d_model=256, num_points=4096, num_out_tokens=32):
        super().__init__()
        self.num_points = num_points
        self.point_encoder = PointFeatureEncoder(
            d_model=d_model, num_out_tokens=num_out_tokens
        )
        self.fusion = SpatialGuidanceFusion(img_channels, d_model=d_model)

    @staticmethod
    def _backproject(depth, inv_K):
        """depth [B,1,H,W], inv_K [B,3,3] (or [B,4,4]) -> xyz [B, H*W, 3]."""
        B, _, H, W = depth.shape
        device = depth.device
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=depth.dtype),
            torch.arange(W, device=device, dtype=depth.dtype),
            indexing="ij",
        )
        ones = torch.ones_like(xs)
        pix = torch.stack([xs, ys, ones], dim=0).reshape(3, -1)  # [3, HW]
        pix = pix.unsqueeze(0).expand(B, -1, -1)  # [B, 3, HW]
        K3 = inv_K[:, :3, :3]
        rays = torch.bmm(K3, pix)  # [B, 3, HW]
        d = depth.reshape(B, 1, -1)  # [B, 1, HW]
        xyz = rays * d  # [B, 3, HW]
        return xyz.transpose(1, 2)  # [B, HW, 3]

    def forward(self, depth, rgb, inv_K, image_feat):
        B, _, H, W = depth.shape
        xyz = self._backproject(depth, inv_K)  # [B, HW, 3]
        rgb_flat = rgb.reshape(B, 3, -1).transpose(1, 2)  # [B, HW, 3]
        pts = torch.cat([xyz, rgb_flat], dim=-1)  # [B, HW, 6]

        # random subsample to num_points for efficiency (uniform, on-GPU)
        N = pts.shape[1]
        if N > self.num_points:
            idx = torch.randperm(N, device=pts.device)[: self.num_points]
            pts = pts[:, idx, :]

        # normalise xyz per-sample (zero-mean, unit-scale) for stable point encoding;
        # keep rgb as-is (already ~[0,1])
        xyz_p, rgb_p = pts[..., :3], pts[..., 3:]
        centroid = xyz_p.mean(dim=1, keepdim=True)
        xyz_p = xyz_p - centroid
        scale = xyz_p.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        xyz_p = xyz_p / scale
        pts = torch.cat([xyz_p, rgb_p], dim=-1)

        point_tokens = self.point_encoder(pts)  # [B, M, d]
        return self.fusion(image_feat, point_tokens)
