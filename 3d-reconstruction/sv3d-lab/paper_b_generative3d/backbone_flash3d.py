"""Flash3D backbone wrapper for Paper B reconstruction candidates.

Loads the frozen official Flash3D GaussianPredictor, runs UniDepth extended +
compute_gauss_means, and exposes:
  - extract_source_gaussians(inputs) -> dict with (N,C) Gaussian params + features
  - render_gaussians(gauss, T_rel, K_tgt, H, W) -> rendered image (3,H,W)
  - build_source_input_from_dataloader(inputs) -> SourceInput

All backbone parameters are frozen (requires_grad=False).  Rendering follows the
validated convention from paper_a_model._render_source_frame (lines 71-100):
  znear=0.01, zfar=100, proj = getProjectionMatrix(...).T, world_view = T_rel.T,
  camera_center from world_view, full_proj = world_view @ proj, bg=0.5, SH0 override.

Does NOT introduce target leakage — only source image/pose/K enters the backbone.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

sys.path.insert(0, "/root/projects/flash3d")

from .model.reconstruction_interface import SourceInput

if TYPE_CHECKING:
    from models.model import GaussianPredictor


def _import_flash3d():
    from models.model import GaussianPredictor
    from models.decoder.gauss_util import (
        render_predicted,
        focal2fov,
        getProjectionMatrix,
    )

    return GaussianPredictor, render_predicted, focal2fov, getProjectionMatrix


class _RenderCfg:
    class model:
        renderer_w_pose = True


class Flash3DBackbone(nn.Module):
    """Frozen Flash3D per-pixel Gaussian predictor for Paper B candidates."""

    def __init__(self, flash3d_cfg, ckpt_path: str, device: str = "cuda"):
        super().__init__()
        GaussianPredictor, _, _, _ = _import_flash3d()
        self.cfg = flash3d_cfg
        self.device = device
        self.predictor = GaussianPredictor(flash3d_cfg)
        self.predictor.set_train()
        self.predictor.load_model(ckpt_path, device=device)
        for p in self.predictor.parameters():
            p.requires_grad_(False)
        self.predictor.eval()
        self._gpp = flash3d_cfg.model.gaussians_per_pixel
        self._pad = getattr(flash3d_cfg.dataset, "pad_border_aug", 32)
        self._bg = 0.5
        self._znear = 0.01
        self._zfar = 100.0

    @torch.no_grad()
    def extract_source_gaussians(self, inputs: dict) -> dict:
        """Run frozen backbone on source frame. Returns flat Gaussian dict per batch.

        inputs: Flash3D-format dict with ('color',0,0), ('color_aug',0,0),
                ('K_src',0), relative pose keys, etc.

        Returns dict with keys:
          xyz: (B, N, 3) source-cam coords
          scales: (B, N, 3)
          rotations: (B, N, 4) quaternions
          opacity: (B, N, 1)
          color_rgb: (B, N, 3) linearized from SH DC
          features_dc: (B, N, 1, 3) for rasterizer
          features_rest: (B, N, 3, 3) SH1 coefficients
          source_features: (B, C, Hf, Wf) ResNet last-scale feature map
        """
        from einops import rearrange

        if "target_frame_ids" not in inputs:
            avail = [
                f
                for f in (1, 2, 3, 4, 5)
                if ("T_c2w", f) in inputs or ("T_w2c", f) in inputs
            ]
            inputs["target_frame_ids"] = avail if avail else [1]
        outputs = self.predictor.models["unidepth_extended"](inputs)
        self.predictor.compute_gauss_means(inputs, outputs)
        self.predictor.process_gt_poses(inputs, outputs)

        B = inputs["color", 0, 0].shape[0]
        gpp = self._gpp

        xyz = outputs["gauss_means"]  # (B*gpp, 4, HW) homogeneous
        xyz = rearrange(xyz[:, :3, :], "(b n) c l -> b (n l) c", n=gpp)

        opacity = rearrange(
            outputs["gauss_opacity"], "(b n) c h w -> b (n h w) c", n=gpp
        )
        # Padded per-pixel source resolution; the primitive axis (n h w) enumerates
        # gaussians_per_pixel layers over this (H, W) grid, so a flat primitive
        # index maps to (layer, y, x) via primitive_source_pixels(). Candidates use
        # this to bilinearly sample source_features per-primitive.
        padded_h, padded_w = (
            outputs["gauss_opacity"].shape[-2],
            outputs["gauss_opacity"].shape[-1],
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
        color_rgb = (0.28209479177387814 * fdc + 0.5).clamp(0, 1)

        features_rest = None
        if "gauss_features_rest" in outputs:
            features_rest = rearrange(
                outputs["gauss_features_rest"],
                "(b n) c h w -> b (n h w) c",
                n=gpp,
            )
            features_rest = features_rest.view(B, -1, 3, 3)

        ext = self.predictor.models["unidepth_extended"]
        color_in = inputs["color_aug", 0, 0]
        if ext.cfg.model.backbone.depth_cond:
            depth_outs = ext.unidepth.infer(
                color_in,
                intrinsics=inputs.get(("K_src", 0)),
            )
            x = torch.cat([color_in, depth_outs["depth"] / 20.0], dim=1)
        else:
            x = color_in
        source_features = ext.encoder(x)[-1]

        return {
            "xyz": xyz,
            "scales": scaling,
            "rotations": rotation,
            "opacity": opacity,
            "color_rgb": color_rgb,
            "features_dc": fdc.view(B, -1, 1, 3),
            "features_rest": features_rest,
            "source_features": source_features,
            "pixel_hw": (int(padded_h), int(padded_w)),
            "gaussians_per_pixel": int(gpp),
            "_outputs": outputs,
        }

    @torch.no_grad()
    def predict_depth(self, color: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Predict metric depth for an arbitrary frame via frozen UniDepth.

        color: (B,3,H,W) in [0,1]. K: (B,3,3) intrinsics matched to color resolution.
        Returns: (B,1,H,W) metric depth. Used by S0 teacher builder to get REAL
        per-neighbor depth (not a crude median), so backprojected hidden points are
        geometrically accurate.
        """
        ext = self.predictor.models["unidepth_extended"]
        out = ext.unidepth.infer(color, intrinsics=K)
        return out["depth"]

    def render_gaussians(
        self,
        gauss_flat: dict,
        T_rel: torch.Tensor,
        K_tgt: torch.Tensor,
        H: int,
        W: int,
        batch_idx: int = 0,
    ) -> torch.Tensor:
        """Render source-frame Gaussians at a target pose.

        gauss_flat: output of extract_source_gaussians (batched) or single-item dict
                    with xyz(N,3), scales(N,3), rotations(N,4), opacity(N,1),
                    color_rgb(N,3), features_dc(N,1,3).
        T_rel: (4,4) cam_T_cam source->target (w2c_tgt @ c2w_src).
        K_tgt: (3,3) target intrinsics (unpadded).
        H, W: target resolution.
        batch_idx: which batch element if gauss_flat is batched.

        Returns (3,H,W) rendered image in [0,1].
        """
        device = T_rel.device

        if gauss_flat.get("xyz") is not None and gauss_flat["xyz"].dim() == 3:
            xyz = gauss_flat["xyz"][batch_idx]
            scaling = gauss_flat["scales"][batch_idx]
            rotation = gauss_flat["rotations"][batch_idx]
            opacity = gauss_flat["opacity"][batch_idx]
            color_rgb = gauss_flat["color_rgb"][batch_idx]
        else:
            xyz = gauss_flat["xyz"]
            scaling = gauss_flat["scales"]
            rotation = gauss_flat["rotations"]
            opacity = gauss_flat["opacity"]
            color_rgb = gauss_flat["color_rgb"]

        gauss = {
            "xyz": xyz,
            "scaling": scaling,
            "rotation": rotation,
            "opacity": opacity,
            "rgb_direct": color_rgb,
            "features_dc": color_rgb.reshape(-1, 1, 3),
        }

        _, render_predicted, focal2fov, getProjectionMatrix = _import_flash3d()

        fx, fy = K_tgt[0, 0].item(), K_tgt[1, 1].item()
        fovX = focal2fov(fx, W)
        fovY = focal2fov(fy, H)
        proj_mtrx = getProjectionMatrix(
            self._znear, self._zfar, fovX, fovY, pX=0.0, pY=0.0
        ).to(device)
        proj_mtrx = proj_mtrx.transpose(0, 1).float()

        world_view_transform = T_rel.transpose(0, 1).float()
        camera_center = (
            -world_view_transform[3, :3] @ world_view_transform[:3, :3].transpose(0, 1)
        ).float()
        full_proj = (world_view_transform @ proj_mtrx).float()

        bg_t = torch.full((3,), self._bg, device=device)
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
        return out["render"].clamp(0, 1)

    @staticmethod
    def build_source_input_from_dataloader(inputs: dict) -> SourceInput:
        """Convert Flash3D dataloader batch dict into SourceInput for candidates.

        Uses the unpadded source color and unpadded K_src.
        """
        color = inputs["color", 0, 0]  # (B,3,H,W) unpadded
        K_src = inputs[("K_src", 0)]  # (B,3,3)
        if ("T_c2w_src", 0) in inputs:
            T_c2w_src = inputs[("T_c2w_src", 0)]
        else:
            T_c2w_src = (
                torch.eye(4, device=color.device)
                .unsqueeze(0)
                .expand(color.shape[0], -1, -1)
            )
        return SourceInput(
            source_rgb=color,
            K_src=K_src,
            T_c2w_src=T_c2w_src,
        )

    def parameters(self, recurse=True):
        return iter([])

    def named_parameters(self, prefix="", recurse=True):
        return iter([])
