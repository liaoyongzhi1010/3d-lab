"""
Paper A — full model: visible per-pixel Gaussians (Flash3D backbone, whitelist) + hidden
Gaussians (novel HiddenGaussianHead), rendered UNIFIED in the source camera frame via the
validated render_gaussians_relpose convention.

Whitelist: Flash3D GaussianPredictor supplies G_vis (UniDepth + ResNet + per-pixel decoder). We
do NOT copy its trainer / losses / heads for the hidden path. The hidden path is fresh (D009).

The model outputs both merged and split (vis-only / hidden-only) renders so the trainer can compute
region-separated + causal-deletion losses and so eval can measure deletion Δ.
"""

import sys
import math
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn

# Flash3D upstream (whitelist) on the server path.
sys.path.insert(0, "/root/projects/flash3d")

from models.model import GaussianPredictor  # noqa: E402
from models.decoder.gauss_util import (  # noqa: E402
    render_predicted,
    focal2fov,
    getProjectionMatrix,
)

from .hidden_head import HiddenGaussianHead  # noqa: E402
from .hidden_head_anchored import VisibleAnchoredHiddenHead  # noqa: E402


@dataclass
class PaperAConfig:
    n_queries: int = 2048
    hidden_dim: int = 256
    hidden_depth: int = 4
    hidden_heads: int = 4
    scale_lambda: float = 0.01
    scale_bias: float = 0.02
    opacity_bias: float = -2.0
    depth_init: float = 2.0
    depth_range: tuple = (0.1, 50.0)
    bg: float = 0.5
    znear: float = 0.01
    zfar: float = 100.0
    freeze_visible: bool = (
        False  # if True, only train the hidden head (isolate learnability)
    )
    anchor_mode: str = (
        "canonical"  # "canonical" (D009), "frustum" (D010), or "visible" (D027/4b)
    )
    behind_offset: float = 0.3
    k_hidden: int = 2  # hidden Gaussians per source pixel (anchor_mode="visible")
    opacity_init: float = (
        0.3  # moderate init (NOT near-transparent) for the anchored head
    )
    latent_dim: int = 0  # >0 enables CVAE hidden head (D013)
    latent_mode: str = (
        "global"  # "global" (D013 r1) or "spatial" (D013 r2 per-cell grid)
    )


class _RenderCfg:
    class model:
        renderer_w_pose = True


def _render_source_frame(gauss, K, T_rel, H, W, device, bg, znear, zfar):
    """Render source-frame Gaussians to the view at relative pose T_rel (source->target).
    Uses direct RGB via override_color (gauss['rgb_direct']). Validated convention
    (oracle_core.render_gaussians_relpose): world_view = T_rel.T, proj = getProjMtx(...).T,
    px/py NDC = 0 (re10k), bg gray. Differentiable (no torch.no_grad)."""
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
        override_color=gauss["rgb_direct"],
    )
    return out


class PaperAModel(nn.Module):
    def __init__(self, flash3d_cfg, paper_cfg: PaperAConfig):
        super().__init__()
        self.pcfg = paper_cfg
        self.fcfg = flash3d_cfg

        # Visible backbone (whitelist). Provides depth + per-pixel Gaussians in source frame.
        self.visible = GaussianPredictor(flash3d_cfg)
        self.visible.set_train()  # sets _is_train, used by process_gt_poses scale estimator
        if paper_cfg.freeze_visible:
            for p in self.visible.parameters():
                p.requires_grad_(False)

        # feature dim: we tap the ResNet encoder's last feature map.
        feat_dim = self._infer_feat_dim()
        if paper_cfg.anchor_mode == "visible":
            # visible-anchored head (D027/4b): spawns k_hidden Gaussians per source pixel behind/
            # beyond the visible surface, color-anchored, moderate opacity init. Anti-collapse.
            self.hidden = VisibleAnchoredHiddenHead(
                feat_dim=feat_dim,
                k_hidden=paper_cfg.k_hidden,
                dim=paper_cfg.hidden_dim,
                n_blocks=paper_cfg.hidden_depth,
                opacity_init=paper_cfg.opacity_init,
            )
        else:
            self.hidden = HiddenGaussianHead(
                feat_dim=feat_dim,
                n_queries=paper_cfg.n_queries,
                dim=paper_cfg.hidden_dim,
                depth=paper_cfg.hidden_depth,
                n_heads=paper_cfg.hidden_heads,
                scale_lambda=paper_cfg.scale_lambda,
                scale_bias=paper_cfg.scale_bias,
                opacity_bias=paper_cfg.opacity_bias,
                depth_init=paper_cfg.depth_init,
                depth_range=paper_cfg.depth_range,
                anchor_mode=paper_cfg.anchor_mode,
                behind_offset=paper_cfg.behind_offset,
                latent_dim=paper_cfg.latent_dim,
                latent_mode=paper_cfg.latent_mode,
            )

    def _infer_feat_dim(self):
        # ResNet encoder num_ch_enc last element.
        enc = self.visible.models["unidepth_extended"].encoder
        return int(enc.num_ch_enc[-1])

    def load_visible_pretrained(self, ckpt_path, device="cpu"):
        self.visible.load_model(ckpt_path, device=device)

    # ---- visible Gaussians as a source-frame dict with rgb_direct ----
    def _visible_gaussians(self, inputs, outputs):
        """Convert Flash3D per-pixel outputs into a flat source-frame gauss dict per batch item.
        Returns list (len B) of dicts with xyz/scaling/rotation/opacity/rgb_direct."""
        gpp = self.fcfg.model.gaussians_per_pixel
        from einops import rearrange

        xyz = outputs["gauss_means"]  # (B*gpp, 4, HW) homogeneous
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
        # color: SH DC -> RGB (Flash3D SH: rgb = 0.28209*dc + 0.5)
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

    def _source_feature_map(self, inputs):
        """Run the visible encoder to get depth-conditioned ResNet features (last scale)."""
        ext = self.visible.models["unidepth_extended"]
        color = inputs["color_aug", 0, 0]
        with torch.no_grad():
            K = inputs[("K_src", 0)] if ("K_src", 0) in inputs else None
            depth_outs = ext.unidepth.infer(color, intrinsics=K)
        if ext.cfg.model.backbone.depth_cond:
            x = torch.cat([color, depth_outs["depth"] / 20.0], dim=1)
        else:
            x = color
        feats = ext.encoder(x)  # list of feature maps
        return feats[-1]

    def _target_feature_map(self, inputs, fid):
        """Encode a TARGET frame's RGB into the same ResNet feature space, for the CVAE POSTERIOR.
        TRAIN-ONLY. This is supervision, not an inference input (guarded in forward + leakage test).
        Uses the target color (unaugmented 'color'); depth-cond channel uses the target's UniDepth
        depth so the posterior sees target geometry (it's collapsed to a global vector anyway)."""
        ext = self.visible.models["unidepth_extended"]
        color_t = inputs[("color", fid, 0)]
        # pad to match source encoder input spatial size if needed (source uses color_aug padded)
        src_in = inputs["color_aug", 0, 0]
        if color_t.shape[-2:] != src_in.shape[-2:]:
            import torch.nn.functional as _F

            color_t = _F.interpolate(
                color_t, size=src_in.shape[-2:], mode="bilinear", align_corners=False
            )
        with torch.no_grad():
            K = inputs[("K_tgt", fid)] if ("K_tgt", fid) in inputs else None
            depth_outs = ext.unidepth.infer(color_t, intrinsics=K)
        if ext.cfg.model.backbone.depth_cond:
            x = torch.cat([color_t, depth_outs["depth"] / 20.0], dim=1)
        else:
            x = color_t
        feats = ext.encoder(x)
        return feats[-1]

    def forward(
        self,
        inputs,
        target_frame_ids,
        hidden_ray_fid=None,
        z_mode=None,
        post_tgt_fid=None,
    ):
        """
        inputs: Flash3D-style batch dict (color/color_aug/K_src/K_tgt/T_c2w...).
        target_frame_ids: list of novel frame ids to render.
        hidden_ray_fid: which target id's pose/K conditions the hidden raymap (default = first).
        z_mode (CVAE, D013): None (deterministic head), "posterior"(train), "prior"/"prior_mean"(infer).
        post_tgt_fid: target frame id whose features form the posterior (train only). Default last.

        Returns outputs dict with, per target frame id:
          ("render_merged", fid), ("render_vis", fid), ("render_hidden", fid)  each (B,3,H,W)
        plus the raw gauss dicts for regularization; if CVAE, outputs["z_out"] has mu/logvar for KL.
        """
        cfg = self.fcfg
        inputs["target_frame_ids"] = target_frame_ids
        # run visible backbone forward WITHOUT its own rendering (we render unified ourselves)
        outputs = self.visible.models["unidepth_extended"](inputs)
        self.visible.compute_gauss_means(inputs, outputs)
        self.visible.process_gt_poses(inputs, outputs)

        B, _, H, W = inputs["color", 0, 0].shape
        device = inputs["color", 0, 0].device
        vis_list = self._visible_gaussians(inputs, outputs)

        # hidden head conditioning
        if hidden_ray_fid is None:
            hidden_ray_fid = target_frame_ids[0]
        feat_map = self._source_feature_map(inputs)
        K_ray = (
            inputs[("K_tgt", hidden_ray_fid)]
            if ("K_tgt", hidden_ray_fid) in inputs
            else inputs[("K_src", 0)]
        )
        T_ray = outputs[("cam_T_cam", 0, hidden_ray_fid)]

        # anchors (frustum mode): unpadded source depth (first gpp layer per batch) + inv_K_src
        depth_src_anchor = None
        inv_K_src_anchor = None
        if self.pcfg.anchor_mode == "frustum":
            gpp = self.fcfg.model.gaussians_per_pixel
            pad = self.fcfg.dataset.pad_border_aug
            dp = outputs[("depth", 0)]  # (B*gpp,1,Hpad,Wpad)
            dp = dp.view(B, gpp, dp.shape[1], dp.shape[2], dp.shape[3])[
                :, 0, 0
            ]  # (B,Hpad,Wpad)
            if pad and pad > 0:
                dp = dp[:, pad : dp.shape[1] - pad, pad : dp.shape[2] - pad]
            depth_src_anchor = dp.detach()  # (B,H,W)
            inv_K_src_anchor = torch.linalg.inv(inputs[("K_src", 0)].float())

        # CVAE: posterior target features (TRAIN ONLY). Guard against inference leakage.
        tgt_feat_map = None
        if self.pcfg.latent_dim > 0 and z_mode == "posterior":
            if post_tgt_fid is None:
                post_tgt_fid = target_frame_ids[-1]
            tgt_feat_map = self._target_feature_map(inputs, post_tgt_fid)

        if self.pcfg.anchor_mode == "visible":
            # visible-anchored head: needs unpadded source depth, source RGB, inv_K_src.
            gpp = self.fcfg.model.gaussians_per_pixel
            pad = self.fcfg.dataset.pad_border_aug
            dp = outputs[("depth", 0)]  # (B*gpp,1,Hpad,Wpad)
            dp = dp.view(B, gpp, dp.shape[1], dp.shape[2], dp.shape[3])[
                :, 0, 0
            ]  # (B,Hpad,Wpad)
            if pad and pad > 0:
                dp = dp[:, pad : dp.shape[1] - pad, pad : dp.shape[2] - pad]
            depth_src_u = (
                dp  # (B,H,W) keep grad off depth? depth is from frozen-ish backbone
            )
            rgb_src_u = inputs["color", 0, 0]  # (B,3,H,W) unpadded source RGB in [0,1]
            inv_K_src = torch.linalg.inv(inputs[("K_src", 0)].float())
            hid = self.hidden(feat_map, depth_src_u, rgb_src_u, inv_K_src, (H, W))
        else:
            hid = self.hidden(
                feat_map,
                K_ray,
                T_ray,
                feat_map.shape[2],
                feat_map.shape[3],
                depth_src=depth_src_anchor,
                inv_K_src=inv_K_src_anchor,
                tgt_feat_map=tgt_feat_map,
                z_mode=(z_mode if self.pcfg.latent_dim > 0 else "prior"),
            )
        if self.pcfg.latent_dim > 0:
            outputs["z_out"] = {
                k: v
                for k, v in hid.items()
                if k in ("mu_q", "logvar_q", "mu_p", "logvar_p", "z")
            }

        hid_list = []
        for b in range(B):
            hid_list.append(
                {
                    "xyz": hid["xyz"][b],
                    "scaling": hid["scaling"][b],
                    "rotation": hid["rotation"][b],
                    "opacity": hid["opacity"][b],
                    "rgb_direct": hid["rgb"][b],
                    "features_dc": hid["rgb"][b].reshape(-1, 1, 3),
                }
            )

        outputs["vis_gaussians"] = vis_list
        outputs["hidden_gaussians"] = hid_list

        # render each target: merged, vis-only, hidden-only
        for fid in [0] + list(target_frame_ids):
            if fid == 0:
                T = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
            else:
                T = outputs[("cam_T_cam", 0, fid)]
            K_t = (
                inputs[("K_tgt", fid)]
                if ("K_tgt", fid) in inputs
                else inputs[("K_src", 0)]
            )

            merged, vonly, honly = [], [], []
            for b in range(B):
                merged.append(
                    self._render_merged(
                        vis_list[b], hid_list[b], K_t[b], T[b], H, W, device
                    )["render"]
                )
                vonly.append(
                    _render_source_frame(
                        vis_list[b],
                        K_t[b],
                        T[b],
                        H,
                        W,
                        device,
                        self.pcfg.bg,
                        self.pcfg.znear,
                        self.pcfg.zfar,
                    )["render"]
                )
                honly.append(
                    _render_source_frame(
                        hid_list[b],
                        K_t[b],
                        T[b],
                        H,
                        W,
                        device,
                        self.pcfg.bg,
                        self.pcfg.znear,
                        self.pcfg.zfar,
                    )["render"]
                )
            outputs[("render_merged", fid)] = torch.stack(merged, 0).clamp(0, 1)
            outputs[("render_vis", fid)] = torch.stack(vonly, 0).clamp(0, 1)
            outputs[("render_hidden", fid)] = torch.stack(honly, 0).clamp(0, 1)

        return outputs

    def _render_merged(self, gvis, ghid, K, T_rel, H, W, device):
        merged = {
            k: torch.cat([gvis[k], ghid[k]], dim=0)
            for k in ["xyz", "scaling", "rotation", "opacity", "rgb_direct"]
        }
        return _render_source_frame(
            merged,
            K,
            T_rel,
            H,
            W,
            device,
            self.pcfg.bg,
            self.pcfg.znear,
            self.pcfg.zfar,
        )
