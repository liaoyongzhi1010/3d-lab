"""
Paper A (reframed, D015) — Calibrated aleatoric uncertainty for feed-forward single-image 3DGS.

Model = Flash3D visible per-pixel Gaussians (whitelist backbone) + ONE extra per-Gaussian
log-variance attribute (β = log σ²), predicted from the SAME source features, splatted through the
SAME validated rasterizer (render_gaussians_relpose convention) to give a per-pixel predicted
variance map σ²(u) at every target view. Because it is one shared 3D Gaussian set rendered to all
targets, the uncertainty is 3D-grounded and multi-view consistent (charter rule), NOT per-view 2D.

Single-image inference: σ² is a function of source features only. Target frames are used ONLY as
photometric supervision (heteroscedastic NLL), exactly like the existing L1/SSIM recon losses.

We deliberately DROP the hidden-appearance head here: E006/E008/E009 proved hidden APPEARANCE is not
learnable feed-forward from one image + photometric loss (3 collapses). Paper A's contribution is
knowing WHERE it cannot know (calibrated uncertainty); synthesizing hidden appearance is Paper B.
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
class UncertaintyConfig:
    bg: float = 0.5
    znear: float = 0.01
    zfar: float = 100.0
    freeze_visible: bool = False  # if True only the uncertainty head trains
    unc_dim: int = 128  # width of the small uncertainty MLP
    logvar_min: float = -8.0  # clamp log σ² for numerical stability
    logvar_max: float = 4.0
    logvar_init: float = -2.0  # start σ² ~ 0.14 (moderate) so NLL is well-conditioned


class _RenderCfg:
    class model:
        renderer_w_pose = True


def _render_channels(gauss, override, K, T_rel, H, W, device, bg, znear, zfar):
    """Render source-frame Gaussians at relative pose T_rel with an arbitrary per-Gaussian
    (N,3) `override` payload alpha-composited like colour. Validated convention
    (oracle_core.render_gaussians_relpose): world_view = T_rel.T, proj = getProjMtx(...).T,
    px/py NDC = 0 (re10k), bg gray. Differentiable."""
    fx, fy = K[0, 0].item(), K[1, 1].item()
    fovX = focal2fov(fx, W)
    fovY = focal2fov(fy, H)
    proj_mtrx = getProjectionMatrix(znear, zfar, fovX, fovY, pX=0.0, pY=0.0).to(device)
    proj_mtrx = proj_mtrx.transpose(0, 1).float()
    world_view_transform = T_rel.transpose(0, 1).float()
    camera_center = (
        -world_view_transform[3, :3] @ world_view_transform[:3, :3].transpose(0, 1)
    ).float()
    full_proj = (world_view_transform @ proj_mtrx).float()
    bg_t = torch.full((3,), bg, device=device)
    out = render_predicted(
        _RenderCfg(),
        gauss,
        world_view_transform,
        full_proj,
        proj_mtrx,
        camera_center,
        (fovX, fovY),
        (H, W),
        bg_t,
        max_sh_degree=0,
        override_color=override,
    )
    return out


class UncertaintyModel(nn.Module):
    """Flash3D visible Gaussians + per-Gaussian log-variance head + variance splatting."""

    def __init__(self, flash3d_cfg, ucfg: UncertaintyConfig):
        super().__init__()
        self.ucfg = ucfg
        self.fcfg = flash3d_cfg

        self.visible = GaussianPredictor(flash3d_cfg)
        self.visible.set_train()
        if ucfg.freeze_visible:
            for p in self.visible.parameters():
                p.requires_grad_(False)

        feat_dim = self._infer_feat_dim()
        gpp = flash3d_cfg.model.gaussians_per_pixel
        # per-Gaussian log-variance head: source features -> gpp log-variance maps.
        # A 1x1 conv keeps it per-pixel & cheap; output gpp channels (one per Gaussian layer).
        self.logvar_head = nn.Sequential(
            nn.Conv2d(feat_dim, ucfg.unc_dim, 1),
            nn.GELU(),
            nn.Conv2d(ucfg.unc_dim, gpp, 1),
        )
        # init last conv near zero so logvar starts at logvar_init (well-conditioned NLL)
        nn.init.zeros_(self.logvar_head[-1].weight)
        nn.init.constant_(self.logvar_head[-1].bias, ucfg.logvar_init)

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

    def _visible_gaussians(self, inputs, outputs):
        """Flat source-frame gauss dict per batch item (xyz/scaling/rotation/opacity/rgb_direct)."""
        from einops import rearrange

        gpp = self.fcfg.model.gaussians_per_pixel
        xyz = outputs["gauss_means"]  # (B*gpp,4,HW)
        B = inputs["color", 0, 0].shape[0]
        xyz = rearrange(xyz[:, :3, :], "(b n) c l -> b (n l) c", n=gpp)
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
        out = []
        for b in range(B):
            out.append(
                {
                    "xyz": xyz[b],
                    "scaling": scaling[b],
                    "rotation": rotation[b],
                    "opacity": opacity[b],
                    "rgb_direct": rgb[b],
                    "features_dc": rgb[b].reshape(-1, 1, 3),
                }
            )
        return out

    def _per_gauss_logvar(self, inputs, feat_map):
        """Predict per-Gaussian log-variance, aligned to the visible Gaussian ordering
        (b (n h w)). feat_map is at encoder resolution; upsample to padded gauss resolution."""
        from einops import rearrange

        gpp = self.fcfg.model.gaussians_per_pixel
        pad = self.fcfg.dataset.pad_border_aug
        H = inputs["color", 0, 0].shape[2] + 2 * pad
        W = inputs["color", 0, 0].shape[3] + 2 * pad
        lv = self.logvar_head(feat_map)  # (B,gpp,hf,wf)
        lv = F.interpolate(lv, size=(H, W), mode="bilinear", align_corners=False)
        lv = lv.clamp(self.ucfg.logvar_min, self.ucfg.logvar_max)
        # (B,gpp,H,W) -> (B, gpp*H*W, 1) matching rearrange "(b n) c h w -> b (n h w) c"
        lv = rearrange(lv, "b n h w -> b (n h w) 1")
        return lv

    def forward(self, inputs, target_frame_ids):
        """Returns per target fid: ('render', fid) mean RGB, ('logvar', fid) per-pixel log-variance
        map (B,1,H,W). Uncertainty predicted from source features only (single-image)."""
        inputs["target_frame_ids"] = target_frame_ids
        outputs = self.visible.models["unidepth_extended"](inputs)
        self.visible.compute_gauss_means(inputs, outputs)
        self.visible.process_gt_poses(inputs, outputs)

        B, _, H, W = inputs["color", 0, 0].shape
        device = inputs["color", 0, 0].device
        vis_list = self._visible_gaussians(inputs, outputs)

        feat_map = self._source_feature_map(inputs)
        logvar = self._per_gauss_logvar(inputs, feat_map)  # (B, Ngauss, 1)

        out = {}
        if ("depth", 0) in outputs:
            out[("depth", 0)] = outputs[("depth", 0)].detach()  # (B*gpp,1,Hpad,Wpad)
        for fid in target_frame_ids:
            T = outputs[("cam_T_cam", 0, fid)]
            K_t = (
                inputs[("K_tgt", fid)]
                if ("K_tgt", fid) in inputs
                else inputs[("K_src", 0)]
            )
            out[("cam_T_cam", 0, fid)] = T
            renders, lvmaps, alphas = [], [], []
            for b in range(B):
                g = vis_list[b]
                # mean RGB render
                rgb_out = _render_channels(
                    g,
                    g["rgb_direct"],
                    K_t[b],
                    T[b],
                    H,
                    W,
                    device,
                    self.ucfg.bg,
                    self.ucfg.znear,
                    self.ucfg.zfar,
                )
                renders.append(rgb_out["render"].clamp(0, 1))
                a = rgb_out.get("rendered_alpha", None)
                alphas.append(
                    a.reshape(1, H, W)
                    if a is not None
                    else torch.ones(1, H, W, device=device)
                )
                # variance map: alpha-composite exp(logvar) with the BACKGROUND set to the PRIOR
                # variance exp(logvar_init). Low-alpha / empty pixels then fall back to the prior
                # (moderate) variance instead of exploding. NO division by alpha (that blows up at
                # low-alpha disocclusion/frustum edges: small_sum / tiny_alpha -> huge var -> NLL
                # log-term diverges). This is the standard "variance as a compositable channel".
                var = torch.exp(logvar[b]).expand(-1, 3)  # (N,3) variance payload
                var_bg = float(torch.exp(torch.tensor(self.ucfg.logvar_init)))
                vout = _render_channels(
                    g,
                    var,
                    K_t[b],
                    T[b],
                    H,
                    W,
                    device,
                    var_bg,  # bg = prior variance (bounded fallback for empty space)
                    self.ucfg.znear,
                    self.ucfg.zfar,
                )
                vmap = vout["render"][0:1].clamp(  # (1,H,W) alpha-composited variance
                    min=float(torch.exp(torch.tensor(self.ucfg.logvar_min))),
                    max=float(torch.exp(torch.tensor(self.ucfg.logvar_max))),
                )
                lvmaps.append(vmap)
            out[("render", fid)] = torch.stack(renders, 0)
            out[("var", fid)] = torch.stack(lvmaps, 0).clamp(min=1e-6)
            out[("alpha", fid)] = torch.stack(
                alphas, 0
            )  # accumulated opacity (confidence baseline)
        out["logvar_gauss"] = logvar
        return out
