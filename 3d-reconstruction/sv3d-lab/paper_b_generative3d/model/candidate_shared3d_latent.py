"""C4 — shared-3D latent refinement.

A trainable refinement network takes the frozen backbone's per-primitive features
and produces offset/color/opacity corrections. The refinement is applied once to
produce a single scene, which is then rendered to ALL targets — making it
target-order invariant by construction. No target information enters the forward.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .reconstruction_interface import (
    FrozenBackbone,
    GaussianScene,
    ReconstructionModel,
    SourceInput,
    base_scene_from_backbone,
)


class Shared3DLatentRefinement(ReconstructionModel):
    provenance_tag = "c4_shared3d"

    def __init__(self, cfg: dict):
        super().__init__()
        num_points = int(cfg.get("num_points", 24))
        hidden = int(cfg.get("hidden", 8))
        seed = int(cfg.get("seed", 0))
        self.feature_hidden = hidden
        self._build_feature_adapter()
        self.extra_points = int(
            cfg.get("extra_points", min(num_points, self.max_extra_points))
        )
        self.latent_dim = int(cfg.get("latent_dim", 4))
        self.backbone = FrozenBackbone(num_points, hidden=hidden, seed=seed)
        self.scene_encoder = nn.Linear(hidden, self.latent_dim)
        self.latent_to_hidden = nn.Linear(self.latent_dim, hidden)
        self.refine_mean = nn.Linear(hidden, 3)
        self.refine_color = nn.Linear(hidden, 3)
        self.refine_opacity = nn.Linear(hidden, 1)
        with torch.no_grad():
            self.scene_encoder.weight.mul_(0.1)
            self.scene_encoder.bias.zero_()
            self.latent_to_hidden.weight.mul_(0.1)
            self.latent_to_hidden.bias.zero_()
            self.refine_mean.weight.mul_(0.1)
            self.refine_mean.bias.zero_()
            self.refine_color.weight.mul_(0.1)
            self.refine_color.bias.zero_()
            self.refine_opacity.weight.mul_(0.1)
            self.refine_opacity.bias.zero_()

    def refinement_magnitude(self, feat: torch.Tensor) -> float:
        with torch.no_grad():
            return float(
                (self.refine_means_frac * torch.tanh(self.refine_mean(feat)))
                .norm(dim=1)
                .mean()
            )

    def _assemble(
        self, means, scales, rotations, opacity, color, reliability, hidden, scene_id
    ):
        """Refine ALL backbone primitives in place with the ONE shared scene-latent
        code (broadcast to every primitive), plus a SMALL bounded extra set — no
        full duplicate. The single shared latent modulating every refinement is
        C4's per-candidate differentiation."""
        n = means.shape[0]
        refined_means, refined_color, refined_opacity = self.refine_backbone_inplace(
            means,
            color,
            opacity,
            hidden,
            mean_head=self.refine_mean,
            color_head=self.refine_color,
            opacity_head=self.refine_opacity,
        )
        k = self._extra_count(n, self.extra_points)
        sel = self._subsample_indices(n, k, device=means.device)
        extra_means = refined_means[sel] + 0.02 * torch.tanh(
            self.refine_mean(hidden[sel])
        )
        all_means = torch.cat((refined_means, extra_means), dim=0)
        all_color = torch.cat((refined_color, refined_color[sel]), dim=0)
        all_scales = torch.cat((scales, scales[sel]), dim=0)
        all_rot = torch.cat((rotations, rotations[sel]), dim=0)
        all_opacity = torch.cat((refined_opacity, refined_opacity[sel]), dim=0)
        all_rel = torch.cat((reliability, reliability[sel]), dim=0)
        provenance = ["backbone"] * n + [self.provenance_tag] * k
        return GaussianScene(
            means=all_means,
            scales=all_scales,
            rotations=all_rot,
            opacity=all_opacity,
            color=all_color,
            provenance=provenance,
            reliability=all_rel,
            scene_id=scene_id,
        ).validate()

    def build_scene(self, source: SourceInput) -> GaussianScene:
        b = self.backbone.predict(source)
        feat = b["features"]
        return self._assemble(
            b["means"],
            b["scales"],
            b["rotations"],
            b["opacity"],
            b["color"],
            b["reliability"],
            feat,
            scene_id=f"c4_{feat.shape[0]}",
        )

    def build_scene_from_backbone(
        self, backbone_gaussians: dict, source: SourceInput
    ) -> GaussianScene:
        base = base_scene_from_backbone(
            backbone_gaussians, scene_id="c4_real", batch_idx=0
        )
        n = len(base)
        feat = self.per_primitive_features(backbone_gaussians, n)
        scene_latent = self.scene_encoder(feat.mean(dim=0))  # [latent_dim]
        shared = torch.tanh(self.latent_to_hidden(scene_latent))  # [hidden]
        shared = shared.unsqueeze(0).expand(n, -1)  # broadcast one scene latent
        return self._assemble(
            base.means,
            base.scales,
            base.rotations,
            base.opacity,
            base.color,
            base.reliability,
            shared,
            scene_id=f"c4_real_{n}",
        )
