"""
Paper A (D017) — SpatialRefine: beat single-image RE10K NVS SOTA (CATSplat 29.09) by adding a
spatial-guided refinement head on top of the reproduced Flash3D backbone.

Design (residual learning — start = Flash3D output, learn only a correction so we never regress the
28.68 baseline):
  1. Flash3D backbone (whitelist) -> per-pixel visible Gaussians (xyz/opacity/scale/rot/rgb) +
     ResNet source features + backprojected depth points (all validated in uncertainty_model).
  2. Spatial guidance: backproject the mono-depth to a source-frame point cloud; encode each point
     with a small positional-MLP (Fourier features of xyz) -> point tokens. Cross-attend image
     feature tokens (query) to point tokens (key/value) so the per-pixel Gaussian decoder sees 3D
     spatial context (this is CATSplat's proven +0.4 dB lever, WITHOUT the VLM/text branch or the
     external PointNet dependency).
  3. Refinement head predicts RESIDUALS to the Flash3D Gaussian params:
       xyz   += d_xyz            (free signed 3D offset — fixes Flash3D depth/boundary errors)
       scale *= exp(d_logs)
       opac   = sigmoid(logit(opac0) + d_opa)
       rgb    = clamp(rgb0 + d_rgb)
     Head is zero-initialised -> at step 0 the model == Flash3D exactly (safe warm start).

Single-image inference: everything is a function of the source image only. Rendered via the validated
render convention. NVS quality is the target; no target leakage.
"""

import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/root/projects/flash3d")

from models.model import GaussianPredictor  # noqa: E402
from models.decoder.gauss_util import (  # noqa: E402
    render_predicted,
    focal2fov,
    getProjectionMatrix,
)


@dataclass
class SpatialRefineConfig:
    bg: float = 0.5
    znear: float = 0.01
    zfar: float = 100.0
    dim: int = 256  # refinement transformer width
    n_heads: int = 4
    depth: int = 3  # number of cross+self attn blocks
    n_points: int = 2048  # subsampled point tokens for spatial guidance
    n_freq: int = 8  # Fourier positional-encoding frequencies for xyz
    freeze_visible: bool = False  # if True, only the refinement head trains


class _RenderCfg:
    class model:
        renderer_w_pose = True
        max_sh_degree = 1


def _render(gauss, K, T_rel, H, W, device, bg, znear, zfar, sh_degree, override=None):
    """Render source-frame Gaussians at relative pose T_rel. If override is None, use the native SH
    path (features_dc [+features_rest]) with sh_degree, matching Flash3D render_images EXACTLY.
    Validated convention (Flash3D render_images / oracle_core.render_gaussians_relpose)."""
    fx, fy = K[0, 0].item(), K[1, 1].item()
    fovX = focal2fov(fx, W)
    fovY = focal2fov(fy, H)
    proj = getProjectionMatrix(znear, zfar, fovX, fovY, pX=0.0, pY=0.0).to(device)
    proj = proj.transpose(0, 1).float()
    wvt = T_rel.transpose(0, 1).float()
    cam_center = (-wvt[3, :3] @ wvt[:3, :3].transpose(0, 1)).float()
    full_proj = (wvt @ proj).float()
    bg_t = torch.full((3,), bg, device=device)
    return render_predicted(
        _RenderCfg(),
        gauss,
        wvt,
        full_proj,
        proj,
        cam_center,
        (fovX, fovY),
        (H, W),
        bg_t,
        max_sh_degree=(0 if override is not None else sh_degree),
        override_color=override,
    )


def fourier_encode(xyz, n_freq):
    """xyz (N,3) -> (N, 3 + 3*2*n_freq) Fourier positional features."""
    feats = [xyz]
    for i in range(n_freq):
        f = 2.0**i
        feats.append(torch.sin(f * xyz))
        feats.append(torch.cos(f * xyz))
    return torch.cat(feats, dim=-1)


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.q = nn.LayerNorm(dim)
        self.kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, x, ctx):
        h, _ = self.attn(self.q(x), self.kv(ctx), self.kv(ctx))
        x = x + h
        x = x + self.ffn(self.norm(x))
        return x


class SelfAttnBlock(nn.Module):
    def __init__(self, dim, n_heads, window=2048):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.window = window

    def forward(self, x):
        B, N, C = x.shape
        w = self.window
        xn = self.norm(x)
        if N > w:
            pad = (w - N % w) % w
            if pad:
                xn = torch.cat([xn, xn[:, :pad]], dim=1)
            xn = xn.reshape(B, -1, w, C).reshape(-1, w, C)
            h, _ = self.attn(xn, xn, xn)
            h = h.reshape(B, -1, C)[:, :N]
        else:
            h, _ = self.attn(xn, xn, xn)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class SpatialRefineModel(nn.Module):
    def __init__(self, flash3d_cfg, rcfg: SpatialRefineConfig):
        super().__init__()
        self.rcfg = rcfg
        self.fcfg = flash3d_cfg
        self.visible = GaussianPredictor(flash3d_cfg)
        self.visible.set_train()
        if rcfg.freeze_visible:
            for p in self.visible.parameters():
                p.requires_grad_(False)

        feat_dim = self._infer_feat_dim()
        dim = rcfg.dim
        self.feat_proj = nn.Conv2d(feat_dim, dim, 1)
        pt_in = 3 + 3 * 2 * rcfg.n_freq
        self.point_proj = nn.Sequential(
            nn.Linear(pt_in, dim), nn.GELU(), nn.Linear(dim, dim)
        )

        self.blocks = nn.ModuleList()
        for _ in range(rcfg.depth):
            self.blocks.append(CrossAttnBlock(dim, rcfg.n_heads))
            self.blocks.append(SelfAttnBlock(dim, rcfg.n_heads))
        self.norm_out = nn.LayerNorm(dim)
        # residual head: d_xyz(3) d_logscale(3) d_opa(1) d_rgb(3) = 10
        self.head = nn.Linear(dim, 10)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)  # zero init -> starts identical to Flash3D

    def _infer_feat_dim(self):
        enc = self.visible.models["unidepth_extended"].encoder
        return int(enc.num_ch_enc[-1])

    def load_visible_pretrained(self, ckpt_path, device="cpu"):
        self.visible.load_model(ckpt_path, device=device)

    def _source_feature_map(self, inputs):
        ext = self.visible.models["unidepth_extended"]
        color = inputs["color_aug", 0, 0]
        with torch.no_grad():
            K = inputs[("K_src", 0)] if ("K_src", 0) in inputs else None
            depth_outs = ext.unidepth.infer(color, intrinsics=K)
        if ext.cfg.model.backbone.depth_cond:
            x = torch.cat([color, depth_outs["depth"] / 20.0], dim=1)
        else:
            x = color
        feats = ext.encoder(x)
        return feats[-1]

    def _visible_gaussians_raw(self, inputs, outputs):
        """Flash3D per-pixel Gaussians as (B, N, ...) tensors, keeping differentiable graph."""
        from einops import rearrange

        gpp = self.fcfg.model.gaussians_per_pixel
        B = inputs["color", 0, 0].shape[0]
        xyz = rearrange(
            outputs["gauss_means"][:, :3, :], "(b n) c l -> b (n l) c", n=gpp
        )
        opacity = rearrange(
            outputs["gauss_opacity"], "(b n) c h w -> b (n h w) c", n=gpp
        )
        scaling = rearrange(
            outputs["gauss_scaling"], "(b n) c h w -> b (n h w) c", n=gpp
        )
        rotation = rearrange(
            outputs["gauss_rotation"], "(b n) c h w -> b (n h w) c", n=gpp
        )
        fdc = rearrange(
            outputs["gauss_features_dc"], "(b n) c h w -> b (n h w) c", n=gpp
        )
        rgb = (0.28209479177387814 * fdc + 0.5).clamp(0, 1)
        out = {
            "xyz": xyz,
            "scaling": scaling,
            "rotation": rotation,
            "opacity": opacity,
            "rgb": rgb,
            "features_dc": fdc,  # raw SH DC coeffs (N,3) for the native SH render path
        }
        if "gauss_features_rest" in outputs:
            out["features_rest"] = rearrange(
                outputs["gauss_features_rest"],
                "(b n) (sh c) h w -> b (n h w) sh c",
                c=3,
                n=gpp,
            )
        return out

    def _refine(self, inputs, outputs, feat_map):
        """Predict per-pixel residuals via spatial-guided cross-attention; apply to Flash3D gauss.
        Attention runs at LOW encoder-feature resolution (cheap), residual upsampled to per-pixel."""
        from einops import rearrange

        g = self._visible_gaussians_raw(inputs, outputs)
        B = feat_map.shape[0]
        gpp = self.fcfg.model.gaussians_per_pixel
        pad = self.fcfg.dataset.pad_border_aug
        Hf, Wf = feat_map.shape[2], feat_map.shape[3]
        Hpix = inputs["color", 0, 0].shape[2] + 2 * pad
        Wpix = inputs["color", 0, 0].shape[3] + 2 * pad

        # image feature tokens at LOW encoder resolution (Hf*Wf, e.g. ~20x28=560 tokens)
        f = self.feat_proj(feat_map)  # (B,dim,Hf,Wf)
        img_tok = rearrange(f, "b c h w -> b (h w) c")  # (B, Hf*Wf, dim)

        # spatial guidance: subsample points from Flash3D gauss xyz, encode, cross-attend
        N = g["xyz"].shape[1]
        npts = min(self.rcfg.n_points, N)
        idx = torch.randperm(N, device=g["xyz"].device)[:npts]
        pts = g["xyz"][:, idx, :].detach()  # (B,npts,3)
        pt_feat = self.point_proj(fourier_encode(pts, self.rcfg.n_freq))  # (B,npts,dim)

        x = img_tok
        for blk in self.blocks:
            if isinstance(blk, CrossAttnBlock):
                x = blk(x, pt_feat)
            else:
                x = blk(x)
        res_low = self.head(self.norm_out(x))  # (B, Hf*Wf, 10)
        # upsample residual map to per-pixel gauss resolution
        res_map = rearrange(res_low, "b (h w) c -> b c h w", h=Hf, w=Wf)
        res_map = F.interpolate(
            res_map, size=(Hpix, Wpix), mode="bilinear", align_corners=False
        )
        res = rearrange(res_map, "b c h w -> b (h w) c")  # (B, HW, 10)
        res = res.repeat(
            1, gpp, 1
        )  # (B, gpp*HW, 10) aligns with gauss ordering (n outer, hw inner)

        d_xyz = res[..., 0:3]
        d_logs = res[..., 3:6]
        d_opa = res[..., 6:7]
        d_rgb = res[..., 7:10]

        xyz = g["xyz"] + d_xyz
        scaling = g["scaling"] * torch.exp(d_logs.clamp(-3, 3))
        # opacity residual in logit space
        opa0 = g["opacity"].clamp(1e-4, 1 - 1e-4)
        opacity = torch.sigmoid(torch.logit(opa0) + d_opa)
        # color residual applied to the SH DC coefficient (so the native SH render path is preserved,
        # incl. view-dependent features_rest). d_rgb in RGB space -> DC space via /0.28209.
        C0 = 0.28209479177387814
        features_dc = g["features_dc"] + d_rgb / C0
        ref = {
            "xyz": xyz,
            "scaling": scaling,
            "rotation": g["rotation"],
            "opacity": opacity,
            "features_dc": features_dc,
            "rgb": (g["rgb"] + d_rgb).clamp(0, 1),  # kept for override fallback / viz
        }
        if "features_rest" in g:
            ref["features_rest"] = g["features_rest"]
        return ref

    def forward(self, inputs, target_frame_ids):
        inputs["target_frame_ids"] = target_frame_ids
        outputs = self.visible.models["unidepth_extended"](inputs)
        self.visible.compute_gauss_means(inputs, outputs)
        self.visible.process_gt_poses(inputs, outputs)

        B, _, H, W = inputs["color", 0, 0].shape
        device = inputs["color", 0, 0].device
        feat_map = self._source_feature_map(inputs)
        gref = self._refine(
            inputs, outputs, feat_map
        )  # refined per-pixel gaussians (B,N,..)

        out = {}
        if ("depth", 0) in outputs:
            out[("depth", 0)] = outputs[("depth", 0)].detach()
        for fid in target_frame_ids:
            T = outputs[("cam_T_cam", 0, fid)]
            K_t = (
                inputs[("K_tgt", fid)]
                if ("K_tgt", fid) in inputs
                else inputs[("K_src", 0)]
            )
            out[("cam_T_cam", 0, fid)] = T
            renders = []
            for b in range(B):
                # native SH render path (features_dc [+features_rest], max_sh_degree) — matches
                # Flash3D render_images exactly so baseline == reproduced 28.68.
                gb = {
                    "xyz": gref["xyz"][b],
                    "scaling": gref["scaling"][b],
                    "rotation": gref["rotation"][b],
                    "opacity": gref["opacity"][b],
                    "features_dc": gref["features_dc"][b].reshape(-1, 1, 3),
                }
                if "features_rest" in gref:
                    gb["features_rest"] = gref["features_rest"][b]
                r = _render(
                    gb,
                    K_t[b],
                    T[b],
                    H,
                    W,
                    device,
                    self.rcfg.bg,
                    self.rcfg.znear,
                    self.rcfg.zfar,
                    self.fcfg.model.max_sh_degree,
                )["render"]
                renders.append(r.clamp(0, 1))
            out[("render", fid)] = torch.stack(renders, 0)
        return out
