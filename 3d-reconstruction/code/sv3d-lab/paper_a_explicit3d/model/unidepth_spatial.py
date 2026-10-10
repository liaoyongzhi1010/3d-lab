"""UniDepthSpatial: Flash3D UniDepthExtended + appearance-aware spatial guidance.

Drop-in replacement for models/encoder/unidepth_encoder.py's UniDepthExtended, activated
when cfg.model.name contains "unidepth_spatial". Identical to Flash3D except it enriches
the DEEPEST ResNet feature with a point-cloud cross-attention branch (see
appearance_spatial_branch.py) BEFORE the depth/gaussian decoders run.

At init the fusion gate gamma=0, so this model == Flash3D exactly (clean A/B: any gain is
attributable to the branch, not to re-init).

Deploy: scp this + appearance_spatial_branch.py to
  /root/projects/flash3d/models/encoder/
and register in models/model.py GaussianPredictor.__init__ (see integration note below).

Integration note (model.py):
    if "unidepth_spatial" in cfg.model.name:
        from models.encoder.unidepth_spatial import UniDepthSpatial
        models["unidepth_extended"] = UniDepthSpatial(cfg)
        ...
    elif "unidepth" in cfg.model.name:  # original branch
        ...
The forward() output dict is identical to UniDepthExtended, so the rest of the pipeline
(compute_gauss_means, render) is unchanged.
"""

import torch
import torch.nn as nn
from einops import rearrange

from models.encoder.resnet_encoder import ResnetEncoder
from models.decoder.resnet_decoder import ResnetDecoder, ResnetDepthDecoder
from models.encoder.appearance_spatial_branch import AppearanceSpatialBranch


class UniDepthSpatial(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.unidepth = torch.hub.load(
            "lpiccinelli-eth/UniDepth",
            "UniDepth",
            version=cfg.model.depth.version,
            backbone=cfg.model.depth.backbone,
            pretrained=True,
            trust_repo=True,
            force_reload=False,
        )

        self.parameters_to_train = []
        assert cfg.model.backbone.name == "resnet"
        self.encoder = ResnetEncoder(
            num_layers=cfg.model.backbone.num_layers,
            pretrained=cfg.model.backbone.weights_init == "pretrained",
            bn_order=cfg.model.backbone.resnet_bn_order,
        )
        if cfg.model.backbone.depth_cond:
            self.encoder.encoder.conv1 = nn.Conv2d(
                4,
                self.encoder.encoder.conv1.out_channels,
                kernel_size=self.encoder.encoder.conv1.kernel_size,
                padding=self.encoder.encoder.conv1.padding,
                stride=self.encoder.encoder.conv1.stride,
            )
        self.parameters_to_train += [{"params": self.encoder.parameters()}]

        # ---- appearance-aware spatial branch (the contribution) ----
        deepest_ch = int(self.encoder.num_ch_enc[-1])  # 2048 for resnet50
        d_model = getattr(cfg.model, "spatial_d_model", 256)
        num_points = getattr(cfg.model, "spatial_num_points", 4096)
        num_tokens = getattr(cfg.model, "spatial_num_tokens", 32)
        self.spatial_branch = AppearanceSpatialBranch(
            img_channels=deepest_ch,
            d_model=d_model,
            num_points=num_points,
            num_out_tokens=num_tokens,
        )
        self.parameters_to_train += [{"params": self.spatial_branch.parameters()}]

        models = {}
        if cfg.model.gaussians_per_pixel > 1:
            models["depth"] = ResnetDepthDecoder(
                cfg=cfg, num_ch_enc=self.encoder.num_ch_enc
            )
            self.parameters_to_train += [{"params": models["depth"].parameters()}]
        for i in range(cfg.model.gaussians_per_pixel):
            models["gauss_decoder_" + str(i)] = ResnetDecoder(
                cfg=cfg, num_ch_enc=self.encoder.num_ch_enc
            )
            self.parameters_to_train += [
                {"params": models["gauss_decoder_" + str(i)].parameters()}
            ]
            if cfg.model.one_gauss_decoder:
                break
        self.models = nn.ModuleDict(models)

        # Optional: freeze the loaded Flash3D backbone (encoder + depth/gauss decoders)
        # and train ONLY the appearance-aware spatial branch. This isolates the branch's
        # contribution on a fixed, converged backbone (no drift). Activated by
        # cfg.model.freeze_backbone=true. The spatial branch is always trainable.
        self.freeze_backbone = bool(getattr(cfg.model, "freeze_backbone", False))
        if self.freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            for m in self.models.values():
                for p in m.parameters():
                    p.requires_grad_(False)
            # only the spatial branch remains trainable
            self.parameters_to_train = [{"params": self.spatial_branch.parameters()}]

    def get_parameter_groups(self):
        return self.parameters_to_train

    def forward(self, inputs):
        if ("unidepth", 0, 0) in inputs.keys() and inputs[
            ("unidepth", 0, 0)
        ] is not None:
            depth_outs = dict()
            depth_outs["depth"] = inputs[("unidepth", 0, 0)]
        else:
            with torch.no_grad():
                intrinsics = (
                    inputs[("K_src", 0)] if ("K_src", 0) in inputs.keys() else None
                )
                depth_outs = self.unidepth.infer(
                    inputs["color_aug", 0, 0], intrinsics=intrinsics
                )

        outputs_gauss = {}
        outputs_gauss[("K_src", 0)] = (
            inputs[("K_src", 0)]
            if ("K_src", 0) in inputs.keys()
            else depth_outs["intrinsics"]
        )
        outputs_gauss[("inv_K_src", 0)] = torch.linalg.inv(outputs_gauss[("K_src", 0)])

        if self.cfg.model.backbone.depth_cond:
            enc_input = torch.cat(
                [inputs["color_aug", 0, 0], depth_outs["depth"] / 20.0], dim=1
            )
        else:
            enc_input = inputs["color_aug", 0, 0]
        encoded_features = self.encoder(enc_input)

        # ---- enrich deepest feature with appearance-aware spatial guidance ----
        deep = encoded_features[-1]  # [B, C, hf, wf]
        rgb = inputs["color_aug", 0, 0]
        depth_full = depth_outs["depth"]  # [B,1,H,W]
        # downsample rgb+depth to the deep feature resolution for point sampling
        hf, wf = deep.shape[-2], deep.shape[-1]
        rgb_ds = torch.nn.functional.interpolate(
            rgb, size=(hf, wf), mode="bilinear", align_corners=False
        )
        depth_ds = torch.nn.functional.interpolate(
            depth_full, size=(hf, wf), mode="nearest"
        )
        # inv_K for the downsampled resolution: scale intrinsics by (hf/H, wf/W)
        inv_K = self._inv_K_for_res(
            outputs_gauss[("K_src", 0)],
            depth_full.shape[-2],
            depth_full.shape[-1],
            hf,
            wf,
            deep.device,
            deep.dtype,
        )
        encoded_features[-1] = self.spatial_branch(depth_ds, rgb_ds, inv_K, deep)

        # ---- depth + gaussian decoders (unchanged) ----
        if self.cfg.model.gaussians_per_pixel > 1:
            depth = self.models["depth"](encoded_features)
            depth[("depth", 0)] = rearrange(
                depth[("depth", 0)],
                "(b n) ... -> b n ...",
                n=self.cfg.model.gaussians_per_pixel - 1,
            )
            depth[("depth", 0)] = torch.cumsum(
                torch.cat(
                    (depth_outs["depth"][:, None, ...], depth[("depth", 0)]), dim=1
                ),
                dim=1,
            )
            outputs_gauss[("depth", 0)] = rearrange(
                depth[("depth", 0)],
                "b n c ... -> (b n) c ...",
                n=self.cfg.model.gaussians_per_pixel,
            )
        else:
            outputs_gauss[("depth", 0)] = depth_outs["depth"]

        gauss_outs = dict()
        for i in range(self.cfg.model.gaussians_per_pixel):
            outs = self.models["gauss_decoder_" + str(i)](encoded_features)
            if self.cfg.model.one_gauss_decoder:
                gauss_outs |= outs
                break
            else:
                for key, v in outs.items():
                    gauss_outs[key] = (
                        outs[key]
                        if i == 0
                        else torch.cat([gauss_outs[key], outs[key]], dim=1)
                    )
        for key, v in gauss_outs.items():
            gauss_outs[key] = rearrange(gauss_outs[key], "b n ... -> (b n) ...")
        outputs_gauss |= gauss_outs

        return outputs_gauss

    @staticmethod
    def _inv_K_for_res(K_full, H, W, hf, wf, device, dtype):
        """Scale a full-res intrinsics matrix to the feature resolution and invert."""
        K = K_full.clone().to(device=device, dtype=dtype)
        sx = wf / float(W)
        sy = hf / float(H)
        K[:, 0, 0] = K[:, 0, 0] * sx
        K[:, 0, 2] = K[:, 0, 2] * sx
        K[:, 1, 1] = K[:, 1, 1] * sy
        K[:, 1, 2] = K[:, 1, 2] * sy
        return torch.linalg.inv(K)
