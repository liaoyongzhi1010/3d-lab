"""
Paper A — HiddenGaussianHead (the novel module).

Ray-conditioned canonical-volume hidden-Gaussian predictor. NOT Flash3D's per-source-pixel
2D offset / padding. A set of learned queries cross-attend to (a) the source ResNet feature
map and (b) a target/virtual raymap embedding (Plucker rays of the wide target pose expressed
in the SOURCE camera frame). Each query decodes to a Gaussian expressed in the SOURCE camera
frame so it merges directly with the visible per-pixel Gaussians and renders with the validated
render_gaussians_relpose convention.

Design rationale (charter §Method 2 + D009):
- Queries "own" their 3D placement (canonical), conditioned on where the source frustum ends and
  where occlusion boundaries are, rather than being tied 1:1 to source pixels. This is what lets
  them populate disoccluded + beyond-frustum regions the source cannot see.
- v1 anchoring: each query predicts a direction (unit ray in source frame) + a positive depth
  along it. Directions are initialised from a learned canonical spread but free to move. This keeps
  xyz in the SOURCE frame (same frame as G_vis) so the unified render is trivial.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_plucker_raymap(K, T_rel, H, W, device):
    """Plucker coordinates (6) per pixel of a TARGET view whose pose relative to the source
    is T_rel (source->target, 4x4), expressed in the SOURCE camera frame.

    We generate the target pixel rays in the target camera frame, then map ray origin + direction
    back into the source frame via inv(T_rel) (target->source). Returns (6, H, W):
    channels [0:3]=direction (unit), [3:6]=moment (origin x direction).
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x = (xs - cx) / fx
    y = (ys - cy) / fy
    z = torch.ones_like(x)
    dirs_t = torch.stack([x, y, z], dim=0).reshape(3, -1)  # (3, HW) target frame
    dirs_t = dirs_t / dirs_t.norm(dim=0, keepdim=True).clamp(min=1e-6)

    T_t2s = torch.linalg.inv(T_rel.float())
    R = T_t2s[:3, :3]
    t = T_t2s[:3, 3:4]  # (3,1) target-cam origin in source frame
    dirs_s = R @ dirs_t  # (3, HW)
    origins_s = t.expand_as(dirs_s)  # (3, HW) all rays share the target cam center
    moment = torch.cross(origins_s, dirs_s, dim=0)  # (3, HW)
    plucker = torch.cat([dirs_s, moment], dim=0)  # (6, HW)
    return plucker.reshape(6, H, W)


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, n_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, q, kv):
        a, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + a
        q = q + self.mlp(self.norm2(q))
        return q


class SelfAttnBlock(nn.Module):
    """Self-attention over queries. To keep memory O(N) instead of O(N^2) for large N_q, we
    partition queries into `window` groups and attend WITHIN each group (queries are unordered,
    so contiguous chunking is a valid random partition). window=0 → full global attention."""

    def __init__(self, dim, n_heads=4, mlp_ratio=2.0, window=2048):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.window = window
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x):
        B, N, D = x.shape
        h = self.norm1(x)
        if self.window and N > self.window:
            g = self.window
            n_groups = (N + g - 1) // g
            pad = n_groups * g - N
            if pad:
                h = torch.cat([h, h[:, :pad]], dim=1)
            h = h.view(B * n_groups, g, D)
            a, _ = self.attn(h, h, h)
            a = a.reshape(B, n_groups * g, D)[:, :N]
        else:
            a, _ = self.attn(h, h, h)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class HiddenGaussianHead(nn.Module):
    """Predicts N_q hidden Gaussians (source-frame) from source features + target raymap.

    Args:
        feat_dim:  channel dim of the source feature map fed in (projected to `dim`).
        n_queries: number of hidden Gaussians.
        dim:       transformer width.
        depth:     number of (cross-attn + self-attn) blocks.
        scale_lambda / scale_bias: match Flash3D activated-scale convention (linear scale ~0.01).
        opacity_bias: initial opacity logit (start near-transparent so the model must earn opacity).
    """

    def __init__(
        self,
        feat_dim,
        n_queries=2048,
        dim=256,
        depth=4,
        n_heads=4,
        scale_lambda=0.01,
        scale_bias=0.02,
        opacity_bias=-2.0,
        depth_init=2.0,
        depth_range=(0.1, 50.0),
        anchor_mode="canonical",
        behind_offset=0.3,
        latent_dim=0,
        latent_mode="global",
    ):
        super().__init__()
        self.n_queries = n_queries
        self.dim = dim
        self.scale_lambda = scale_lambda
        self.depth_range = depth_range
        self.anchor_mode = anchor_mode  # "canonical" (D009) or "frustum" (D010)
        self.behind_offset = behind_offset
        self.latent_dim = latent_dim  # >0 enables CVAE latent conditioning (D013)
        # "global" (D013 round-1, single pooled z; collapsed at L2 held-out, E008) or
        # "spatial" (D013 round-2, per-cell latent grid so KL budget & best-of-K diversity are
        # LOCAL to hidden regions instead of a single scene-global code).
        self.latent_mode = latent_mode

        self.feat_proj = nn.Conv2d(feat_dim, dim, 1)
        self.ray_proj = nn.Conv2d(6, dim, 1)
        self.query = nn.Parameter(torch.randn(n_queries, dim) * 0.02)

        # CVAE latent (D013): prior p(z|source) and posterior q(z|source,target).
        if latent_dim > 0:
            if latent_mode == "global":
                # single pooled z injected into every query token
                self.prior_net = nn.Sequential(
                    nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2 * latent_dim)
                )
                self.post_net = nn.Sequential(
                    nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 2 * latent_dim)
                )
                self.z_proj = nn.Linear(latent_dim, dim)
                self.tgt_feat_proj = nn.Conv2d(feat_dim, dim, 1)
            elif latent_mode == "spatial":
                # per-cell latent GRID at feature-map resolution. prior conditions on source
                # features; posterior on source+target features (train only). Each grid cell
                # becomes an extra KV token, so queries cross-attend to spatially-localized z.
                self.prior_net = nn.Sequential(
                    nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, 2 * latent_dim, 1)
                )
                self.post_net = nn.Sequential(
                    nn.Conv2d(2 * dim, dim, 1),
                    nn.GELU(),
                    nn.Conv2d(dim, 2 * latent_dim, 1),
                )
                # project a latent cell (+ its ray positional code) into a KV token
                self.z_proj = nn.Conv2d(latent_dim, dim, 1)
                self.tgt_feat_proj = nn.Conv2d(feat_dim, dim, 1)
            else:
                raise ValueError(latent_mode)

        # canonical direction init: spread queries over a hemisphere-ish set of directions
        dirs0 = torch.randn(n_queries, 3)
        dirs0[:, 2] = dirs0[:, 2].abs() + 0.5  # bias forward (+z) but allow off-frustum
        dirs0 = dirs0 / dirs0.norm(dim=1, keepdim=True)
        self.register_buffer("dir_init", dirs0)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(CrossAttnBlock(dim, n_heads))
            self.blocks.append(SelfAttnBlock(dim, n_heads))

        self.norm_out = nn.LayerNorm(dim)
        # heads: ddir(3), log_depth(1), scale(3), rot(4), opacity(1), rgb(3)
        self.head = nn.Linear(dim, 3 + 1 + 3 + 4 + 1 + 3)
        # init last layer small so predictions start near the canonical anchors.
        # For the CVAE (latent_dim>0) use a SMALL non-zero weight so the latent can influence the
        # output from step 0 (a fully-zero head makes all z produce identical output → dead latent).
        if latent_dim > 0:
            nn.init.normal_(self.head.weight, std=1e-3)
        else:
            nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            b = self.head.bias
            # log_depth bias -> depth_init
            ld = math.log(max(depth_init, 1e-3))
            b[3] = ld
            # scale bias (pre-exp so exp*lambda ~ scale_bias)
            b[4:7] = math.log(scale_bias / scale_lambda)
            # rotation -> identity quat (w=1)
            b[7] = 1.0
            # opacity
            b[11] = opacity_bias

    def forward(
        self,
        feat_map,
        K_ray,
        T_rel_ray,
        H,
        W,
        depth_src=None,
        inv_K_src=None,
        tgt_feat_map=None,
        z_mode="prior",
        z=None,
    ):
        """
        feat_map:  (B, C, Hf, Wf) source feature map.
        K_ray:     (B, 3, 3) intrinsics for the raymap view (use a target/virtual K).
        T_rel_ray: (B, 4, 4) source->target relative pose used to build the conditioning raymap.
        H, W:      resolution to build the raymap at (can be feature-map res for cheapness).
        depth_src: (B, Hs, Ws) source depth (unpadded), needed for anchor_mode="frustum".
        inv_K_src: (B, 3, 3) source inverse intrinsics, needed for anchor_mode="frustum".
        CVAE (D013, only if latent_dim>0):
          tgt_feat_map: (B,C,Hf,Wf) TARGET feature map — used ONLY to form the posterior at TRAIN.
                        MUST be None at inference (single-image constraint).
          z_mode: "posterior" (train, needs tgt_feat_map), "prior" (sample p(z|src)),
                  "prior_mean" (deterministic prior mean), "fixed" (use provided z).
          z: (B,latent_dim) used when z_mode=="fixed".
        Returns dict of hidden Gaussian params (source frame) + (if CVAE) z/mu/logvar for KL.
        """
        B = feat_map.shape[0]
        device = feat_map.device

        f = self.feat_proj(feat_map)  # (B, dim, Hf, Wf)
        # build raymap at feature-map resolution, project, add to feature tokens
        Hf, Wf = f.shape[2], f.shape[3]
        rays = []
        for b in range(B):
            pm = build_plucker_raymap(
                K_ray[b], T_rel_ray[b], Hf, Wf, device
            )  # (6,Hf,Wf)
            rays.append(pm)
        rays = torch.stack(rays, 0)  # (B,6,Hf,Wf)
        r = self.ray_proj(rays)  # (B,dim,Hf,Wf)
        kv = (f + r).flatten(2).transpose(1, 2)  # (B, Hf*Wf, dim)

        # ---- CVAE latent (D013) ----
        z_out = {}
        z_inj = None  # global-mode per-token injection (None in spatial mode)
        if self.latent_dim > 0 and self.latent_mode == "global":
            src_glob = f.mean(dim=(2, 3))  # (B,dim) source global feature
            pri = self.prior_net(src_glob)
            mu_p, logvar_p = pri[:, : self.latent_dim], pri[:, self.latent_dim :]
            if z_mode == "posterior":
                assert tgt_feat_map is not None, (
                    "posterior needs target features (train only)"
                )
                tf = self.tgt_feat_proj(tgt_feat_map).mean(dim=(2, 3))  # (B,dim)
                post = self.post_net(torch.cat([src_glob, tf], dim=1))
                mu_q, logvar_q = post[:, : self.latent_dim], post[:, self.latent_dim :]
                std = torch.exp(0.5 * logvar_q)
                z_used = mu_q + std * torch.randn_like(std)
                z_out = {
                    "mu_q": mu_q,
                    "logvar_q": logvar_q,
                    "mu_p": mu_p,
                    "logvar_p": logvar_p,
                }
            elif z_mode == "prior":
                std = torch.exp(0.5 * logvar_p)
                z_used = mu_p + std * torch.randn_like(std)
                z_out = {"mu_p": mu_p, "logvar_p": logvar_p}
            elif z_mode == "prior_mean":
                z_used = mu_p
                z_out = {"mu_p": mu_p, "logvar_p": logvar_p}
            elif z_mode == "fixed":
                assert z is not None
                z_used = z
            else:
                raise ValueError(z_mode)
            z_inj = self.z_proj(z_used).unsqueeze(1)  # (B,1,dim)
            kv = kv + z_inj  # condition the context tokens on z
            z_out["z"] = z_used
        elif self.latent_dim > 0 and self.latent_mode == "spatial":
            # per-cell latent GRID at feature-map resolution (D013 round-2).
            fp = f  # (B,dim,Hf,Wf) projected source features (reused from above)
            pri = self.prior_net(fp)  # (B,2*ld,Hf,Wf)
            mu_p = pri[:, : self.latent_dim]
            logvar_p = pri[:, self.latent_dim :]
            if z_mode == "posterior":
                assert tgt_feat_map is not None, (
                    "posterior needs target features (train only)"
                )
                tf = self.tgt_feat_proj(tgt_feat_map)  # (B,dim,Hf,Wf)
                post = self.post_net(torch.cat([fp, tf], dim=1))
                mu_q = post[:, : self.latent_dim]
                logvar_q = post[:, self.latent_dim :]
                std = torch.exp(0.5 * logvar_q)
                z_grid = mu_q + std * torch.randn_like(std)  # (B,ld,Hf,Wf)
                z_out = {
                    "mu_q": mu_q,
                    "logvar_q": logvar_q,
                    "mu_p": mu_p,
                    "logvar_p": logvar_p,
                }
            elif z_mode == "prior":
                std = torch.exp(0.5 * logvar_p)
                z_grid = mu_p + std * torch.randn_like(std)
                z_out = {"mu_p": mu_p, "logvar_p": logvar_p}
            elif z_mode == "prior_mean":
                z_grid = mu_p
                z_out = {"mu_p": mu_p, "logvar_p": logvar_p}
            elif z_mode == "fixed":
                assert z is not None
                z_grid = z
            else:
                raise ValueError(z_mode)
            # each grid cell -> a KV token carrying its localized latent + ray position
            z_tok = self.z_proj(z_grid)  # (B,dim,Hf,Wf)
            z_tok = (z_tok + r).flatten(2).transpose(1, 2)  # (B,Hf*Wf,dim) w/ ray pos
            kv = torch.cat([kv, z_tok], dim=1)  # append latent KV tokens
            z_out["z"] = z_grid

        q = self.query.unsqueeze(0).expand(B, -1, -1)  # (B, Nq, dim)
        if z_inj is not None:
            q = q + z_inj  # global mode: also inject z into queries directly
        for blk in self.blocks:
            if isinstance(blk, CrossAttnBlock):
                q = blk(q, kv)
            else:
                q = blk(q)
        q = self.norm_out(q)
        out = self.head(q)  # (B, Nq, 15)

        ddir = out[..., 0:3]
        log_depth = out[..., 3:4]
        scale_raw = out[..., 4:7]
        rot_raw = out[..., 7:11]
        opa_raw = out[..., 11:12]
        rgb_raw = out[..., 12:15]

        dmin, dmax = self.depth_range

        if (
            self.anchor_mode == "frustum"
            and depth_src is not None
            and inv_K_src is not None
        ):
            # Anchor each query at a source-pixel backprojected 3D point, displaced BEHIND the
            # surface along its ray (to populate occluded volume) OR slightly outside frustum.
            anchors, dir_anchor = self._frustum_anchors(
                depth_src, inv_K_src, device
            )  # (B,Nq,3),(B,Nq,3)
            # residual: along-ray push behind surface (>=0 via softplus) + small free 3D offset
            behind = F.softplus(log_depth) * self.behind_offset  # (B,Nq,1) >=0
            free = 0.1 * ddir  # small lateral wiggle
            xyz = anchors + dir_anchor * behind + free
            xyz = xyz.clamp(-dmax, dmax)
        else:
            dir0 = self.dir_init.unsqueeze(0).to(device)  # (1,Nq,3)
            direction = dir0 + 0.1 * ddir
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            depth = torch.exp(log_depth).clamp(dmin, dmax)  # (B,Nq,1)
            xyz = direction * depth  # (B,Nq,3) SOURCE frame

        scaling = torch.exp(scale_raw) * self.scale_lambda
        rotation = F.normalize(rot_raw, dim=-1)
        opacity = torch.sigmoid(opa_raw)
        rgb = torch.sigmoid(rgb_raw)

        result = {
            "xyz": xyz,
            "scaling": scaling,
            "rotation": rotation,
            "opacity": opacity,
            "rgb": rgb,
        }
        if self.latent_dim > 0:
            result.update(z_out)
        return result

    def _frustum_anchors(self, depth_src, inv_K_src, device):
        """Backproject N_q source pixels (grid sample) to 3D source-frame points + their ray dirs.
        depth_src: (B,Hs,Ws). inv_K_src: (B,3,3). Returns anchors (B,Nq,3), ray dirs (B,Nq,3)."""
        B, Hs, Ws = depth_src.shape
        Nq = self.n_queries
        # deterministic near-uniform grid of Nq pixel locations
        n_side = int(math.ceil(math.sqrt(Nq)))
        ys = torch.linspace(0, Hs - 1, n_side, device=device)
        xs = torch.linspace(0, Ws - 1, n_side, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        gy = gy.reshape(-1)[:Nq]
        gx = gx.reshape(-1)[:Nq]
        anchors_all, dirs_all = [], []
        for b in range(B):
            yi = gy.long().clamp(0, Hs - 1)
            xi = gx.long().clamp(0, Ws - 1)
            d = depth_src[b][yi, xi]  # (Nq,)
            pix = torch.stack([gx, gy, torch.ones_like(gx)], 0)  # (3,Nq)
            rays = inv_K_src[b] @ pix  # (3,Nq)
            rays = rays / rays[2:3].clamp(
                min=1e-6
            )  # normalize so z=1 then scale by depth
            pts = rays * d.unsqueeze(0)  # (3,Nq) camera/source frame
            ray_dir = rays / rays.norm(dim=0, keepdim=True).clamp(min=1e-6)
            anchors_all.append(pts.T)
            dirs_all.append(ray_dir.T)
        return torch.stack(anchors_all, 0), torch.stack(dirs_all, 0)
