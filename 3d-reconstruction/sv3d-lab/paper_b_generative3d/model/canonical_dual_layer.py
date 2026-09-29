"""Source-Conditioned Canonical Dual-Layer 3DGS model.

KEY INVARIANT: build_scene(I_src, K_src, z) produces a SINGLE GaussianScene.
Changing/adding target cameras does NOT change the scene. P_tgt is only for rendering.

Difference from D009/D010 HiddenGaussianHead:
  - NO target camera / Plücker raymap in scene generation (canonical queries only)
  - Canonical teacher pseudo-GT supervision (not just render loss)
  - Visibility routing: Free gaussians penalized on Anchor-visible rays
  - Source-only CVAE latent z (one sample = one shared scene for all views)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, n_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, kv):
        q2 = self.norm_q(q)
        kv2 = self.norm_kv(kv)
        q = q + self.attn(q2, kv2, kv2, need_weights=False)[0]
        q = q + self.mlp(self.norm2(q))
        return q


class SelfAttnBlock(nn.Module):
    def __init__(self, dim, n_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x2 = self.norm(x)
        x = x + self.attn(x2, x2, x2, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class LatentEncoder(nn.Module):
    """Source-only CVAE encoder: F_src -> (mu, logvar) for z."""

    def __init__(self, feat_dim=256, latent_dim=64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)

    def forward(self, feat_map):
        # feat_map: [B, C, H, W]
        pooled = self.pool(feat_map).flatten(1)  # [B, C]
        return self.fc_mu(pooled), self.fc_logvar(pooled)

    def sample(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std


class CanonicalFreeGenerator(nn.Module):
    """Generate M canonical free Gaussians from source features + latent z.

    Uses LEARNED SPATIAL QUERIES (not target-dependent). Queries attend to source
    features via cross-attention, then decode into Gaussian parameters.
    """

    def __init__(
        self,
        n_free=4096,
        feat_dim=256,
        latent_dim=64,
        dim=256,
        n_cross=3,
        n_self=2,
        n_heads=4,
    ):
        super().__init__()
        self.n_free = n_free
        self.queries = nn.Parameter(torch.randn(1, n_free, dim) * 0.02)
        self.latent_proj = nn.Linear(latent_dim, dim)
        self.cross_blocks = nn.ModuleList(
            [CrossAttnBlock(dim, n_heads) for _ in range(n_cross)]
        )
        self.self_blocks = nn.ModuleList(
            [SelfAttnBlock(dim, n_heads) for _ in range(n_self)]
        )
        # Decode to Gaussian params: xyz(3) + scale(3) + rot(4) + opacity(1) + color(3) = 14
        self.head = nn.Linear(dim, 14)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feat_tokens, z):
        """
        feat_tokens: [B, N_src, dim] source feature tokens (flattened spatial)
        z: [B, latent_dim]
        Returns: dict with xyz, scales, rotations, opacity, color_rgb  (all [B, M, ...])
        """
        B = feat_tokens.shape[0]
        q = self.queries.expand(B, -1, -1)  # [B, M, dim]
        # Inject latent z as additive bias to queries
        z_proj = self.latent_proj(z).unsqueeze(1)  # [B, 1, dim]
        q = q + z_proj
        # Cross-attend to source features
        for blk in self.cross_blocks:
            q = blk(q, feat_tokens)
        # Self-attend among free queries
        for blk in self.self_blocks:
            q = blk(q)
        # Decode
        params = self.head(q)  # [B, M, 14]
        xyz = params[:, :, 0:3]
        scales = params[:, :, 3:6].exp().clamp(max=0.5)
        rot_raw = params[:, :, 6:10]
        rotations = F.normalize(rot_raw, dim=-1)
        opacity = torch.sigmoid(params[:, :, 10:11])
        color_rgb = torch.sigmoid(params[:, :, 11:14])
        return {
            "xyz": xyz,
            "scales": scales,
            "rotations": rotations,
            "opacity": opacity,
            "color_rgb": color_rgb,
        }


class CanonicalDualLayer(nn.Module):
    """Source-Conditioned Canonical Dual-Layer 3DGS.

    Invariant: build_scene(I_src, K_src, z) -> GaussianScene (independent of target cameras).
    """

    def __init__(
        self,
        anchor_backbone,  # Flash3DBackbone (frozen or partial-trainable)
        feat_proj_dim=256,
        latent_dim=64,
        n_free=4096,
        n_cross=3,
        n_self=2,
    ):
        super().__init__()
        self.anchor = anchor_backbone
        self.feat_proj = nn.Conv2d(2048, feat_proj_dim, 1)  # project source features
        self.latent_enc = LatentEncoder(feat_proj_dim, latent_dim)
        self.free_gen = CanonicalFreeGenerator(
            n_free=n_free,
            feat_dim=feat_proj_dim,
            latent_dim=latent_dim,
            dim=feat_proj_dim,
            n_cross=n_cross,
            n_self=n_self,
        )

    def build_scene(self, inputs, z: Optional[torch.Tensor] = None):
        """Build a SINGLE shared GaussianScene from source image only.

        Args:
            inputs: Flash3D-format dict with source image + intrinsics
            z: optional latent (if None, encode from source and sample)

        Returns:
            scene: dict with 'anchor' and 'free' gaussian dicts + 'z', 'mu', 'logvar'
        """
        # Anchor layer (may be frozen)
        gauss_out = self.anchor.extract_source_gaussians(inputs)
        anchor = {
            "xyz": gauss_out["xyz"][0],
            "scales": gauss_out["scales"][0],
            "rotations": gauss_out["rotations"][0],
            "opacity": gauss_out["opacity"][0],
            "color_rgb": gauss_out["color_rgb"][0],
        }

        # Source features -> project -> tokens
        src_feat = gauss_out["source_features"][0]  # [2048, hf, wf]
        feat_proj = self.feat_proj(src_feat.unsqueeze(0))  # [1, dim, hf, wf]
        B, C, hf, wf = feat_proj.shape
        feat_tokens = feat_proj.flatten(2).transpose(1, 2)  # [B, hf*wf, dim]

        # Latent
        mu, logvar = self.latent_enc(feat_proj)
        if z is None:
            z = self.latent_enc.sample(mu, logvar)

        # Free layer (canonical, target-independent)
        free = self.free_gen(feat_tokens, z)

        return {
            "anchor": anchor,
            "free": {k: v[0] for k, v in free.items()},  # unbatch
            "z": z,
            "mu": mu,
            "logvar": logvar,
        }

    def merge_scene(self, scene):
        """Merge anchor + free into one flat Gaussian dict for rendering."""
        a = scene["anchor"]
        f = scene["free"]
        return {
            "xyz": torch.cat([a["xyz"], f["xyz"]], dim=0),
            "scales": torch.cat([a["scales"], f["scales"]], dim=0),
            "rotations": torch.cat([a["rotations"], f["rotations"]], dim=0),
            "opacity": torch.cat([a["opacity"], f["opacity"]], dim=0),
            "color_rgb": torch.cat([a["color_rgb"], f["color_rgb"]], dim=0),
        }

    def visibility_routing_loss(self, scene, inputs, K_src, H=256, W=384, margin=0.1):
        """Penalize Free gaussians that are redundant with / occlude the Anchor surface.

        Mechanism (differentiable, source-frame): the Anchor is per-pixel gaussians in
        source-cam coords, so its z-values form a source depth map. Project each Free
        gaussian into the source camera; a Free point is "illegal" (should be low opacity)
        if it projects INSIDE the source image AND its depth is <= anchor_depth + margin
        (i.e. it sits in front of or on the visible surface, polluting anchor-visible rays).
        A Free point is "legal" (hidden geometry) if it projects outside the source image
        OR sits clearly behind the anchor surface.

        Loss = mean( Free.opacity * illegal_mask_soft ). Fully differentiable through opacity.
        """
        device = scene["free"]["xyz"].device
        anchor_xyz = scene["anchor"]["xyz"]  # [Na,3] source-cam
        free = scene["free"]
        free_xyz = free["xyz"]  # [Nf,3] source-cam
        free_op = free["opacity"].squeeze(-1)  # [Nf]

        if K_src.dim() == 3:
            K_src = K_src[0]
        fx, fy = K_src[0, 0], K_src[1, 1]
        cx, cy = K_src[0, 2], K_src[1, 2]

        # Build a coarse source depth map from anchor z (splat nearest per pixel).
        az = anchor_xyz[:, 2].clamp(min=1e-3)
        ax = anchor_xyz[:, 0] / az * fx + cx
        ay = anchor_xyz[:, 1] / az * fy + cy
        ax_l = ax.round().long().clamp(0, W - 1)
        ay_l = ay.round().long().clamp(0, H - 1)
        depth_map = torch.full((H, W), 1e4, device=device)
        flat_idx = ay_l * W + ax_l
        depth_map.view(-1).scatter_reduce_(
            0, flat_idx, az.detach(), reduce="amin", include_self=True
        )

        # Project free points into source
        fz = free_xyz[:, 2]
        fx_p = free_xyz[:, 0] / (fz + 1e-6) * fx + cx
        fy_p = free_xyz[:, 1] / (fz + 1e-6) * fy + cy
        inside = (fx_p >= 0) & (fx_p < W) & (fy_p >= 0) & (fy_p < H) & (fz > 1e-3)
        fx_c = fx_p.round().long().clamp(0, W - 1)
        fy_c = fy_p.round().long().clamp(0, H - 1)
        anchor_depth_at = depth_map[fy_c, fx_c]
        # illegal = inside image AND in front of / on the anchor surface
        in_front = fz <= (anchor_depth_at + margin)
        illegal = (inside & in_front).float()

        loss = (free_op * illegal).sum() / (illegal.sum() + 1.0)
        return loss
