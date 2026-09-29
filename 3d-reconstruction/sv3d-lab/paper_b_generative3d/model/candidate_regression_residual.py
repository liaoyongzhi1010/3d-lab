"""Residual family candidates.

C0 — regression-only control: a single trainable linear residual head on top of
the frozen backbone. NO generative module, NO latent, NO teacher. It owns
trainable parameters (so the funnel can measure whether any generative family
actually beats plain regression).

C1 — confidence-routed residual 3DGS: keeps every backbone primitive and adds a
trainable residual set of Gaussians whose opacity is gated by a reliability
router, so generative capacity is only spent where the backbone is unreliable.

Both keep provenance ``"backbone"`` on backbone primitives and a candidate tag on
added ones, so deletion is causally load-bearing and restores the backbone-only
render (tests/reconstruction/test_deletion.py).
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
from .reliability_router import ReliabilityRouter


class RegressionResidualControl(ReconstructionModel):
    is_control = True
    provenance_tag = "c0_residual"

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
        self.backbone = FrozenBackbone(num_points, hidden=hidden, seed=seed)
        # Load-bearing refinement heads: one per attribute (means/color/opacity),
        # applied to EVERY backbone primitive. Zero-init keeps training stable but
        # the heads are free to grow so the refinement becomes the primary
        # trainable contribution (E-112 fix).
        self.residual = nn.Linear(hidden, 3)
        self.color_residual = nn.Linear(hidden, 3)
        self.opacity_head = nn.Linear(hidden, 1)
        with torch.no_grad():
            for head in (self.residual, self.color_residual, self.opacity_head):
                head.weight.zero_()
                head.bias.zero_()

    def _refine(self, means, color, opacity, feat):
        """Plain per-primitive MLP refinement of ALL backbone primitives (control)."""
        return self.refine_backbone_inplace(
            means,
            color,
            opacity,
            feat,
            mean_head=self.residual,
            color_head=self.color_residual,
            opacity_head=self.opacity_head,
        )

    def _assemble(
        self, means, scales, rotations, opacity, color, reliability, feat, scene_id
    ):
        """Refine ALL backbone primitives in place, then append a SMALL bounded extra set.

        The backbone primitives are refined (means/color/opacity) yet keep
        provenance ``"backbone"`` so they remain the anchor identity — this is now
        the PRIMARY trainable path. Only a bounded extra subset
        (``<= max_extra_points``) is tagged with the candidate provenance, so total
        count stays near ``n + extra`` instead of ``2n``. Deleting the candidate
        tag is causally load-bearing (removes the extra set); resetting the
        refinement heads to zero restores the exact backbone.
        """
        n = means.shape[0]
        refined_means, refined_color, refined_opacity = self._refine(
            means, color, opacity, feat
        )

        k = self._extra_count(n, self.extra_points)
        sel = self._subsample_indices(n, k, device=means.device)
        extra_means = refined_means[sel] + 0.02 * torch.tanh(self.residual(feat[sel]))
        extra_color = refined_color[sel]

        all_means = torch.cat((refined_means, extra_means), dim=0)
        all_color = torch.cat((refined_color, extra_color), dim=0)
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

    def refinement_magnitude(self, feat: torch.Tensor) -> float:
        """Mean per-primitive mean-residual norm; 0 when heads are reset."""
        with torch.no_grad():
            return float(
                (self.refine_means_frac * torch.tanh(self.residual(feat)))
                .norm(dim=1)
                .mean()
            )

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
            scene_id=f"c0_{feat.shape[0]}",
        )

    def build_scene_from_backbone(
        self, backbone_gaussians: dict, source: SourceInput
    ) -> GaussianScene:
        base = base_scene_from_backbone(
            backbone_gaussians, scene_id="c0_real", batch_idx=0
        )
        n = len(base)
        feat = self.per_primitive_features(backbone_gaussians, n)
        return self._assemble(
            base.means,
            base.scales,
            base.rotations,
            base.opacity,
            base.color,
            base.reliability,
            feat,
            scene_id=f"c0_real_{n}",
        )


class ConfidenceRoutedResidual(ReconstructionModel):
    provenance_tag = "c1_residual"

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
        self.backbone = FrozenBackbone(num_points, hidden=hidden, seed=seed)
        self.router = ReliabilityRouter(hidden, seed=seed)
        self.mean_head = nn.Linear(hidden, 3)
        self.color_head = nn.Linear(hidden, 3)
        self.opacity_head = nn.Linear(hidden, 1)
        with torch.no_grad():
            for head in (self.mean_head, self.color_head, self.opacity_head):
                head.weight.zero_()
                head.bias.zero_()

    def refinement_magnitude(self, feat: torch.Tensor) -> float:
        with torch.no_grad():
            return float(
                (self.refine_means_frac * torch.tanh(self.mean_head(feat)))
                .norm(dim=1)
                .mean()
            )

    def _assemble(self, means, scales, rotations, opacity, color, feat, scene_id):
        n = means.shape[0]
        reliability = self.router(feat)  # [n,1] trainable, gated capacity
        gate = (1.0 - reliability).clamp(0, 1)  # spend capacity where unreliable

        # Load-bearing in-place refinement of ALL backbone primitives (provenance
        # stays "backbone"). The reliability router GATES the color/opacity/means
        # residuals by ``1 - reliability`` so refinement is stronger where the
        # backbone is unreliable — this is C1's per-candidate differentiation.
        refined_means, refined_color, refined_opacity = self.refine_backbone_inplace(
            means,
            color,
            opacity,
            feat,
            mean_head=self.mean_head,
            color_head=self.color_head,
            opacity_head=self.opacity_head,
            gate=gate,
        )

        # Extra set: subsample the LEAST reliable primitives (highest gate), so the
        # generative capacity is only spent where the backbone is unreliable.
        k = self._extra_count(n, self.extra_points)
        order = torch.argsort(gate.squeeze(1), descending=True)
        sel = order[:k]
        extra_means = refined_means[sel] + 0.05 * torch.tanh(self.mean_head(feat[sel]))
        extra_color = refined_color[sel]
        extra_opacity = (gate[sel] * refined_opacity[sel]).clamp(0, 1)

        all_means = torch.cat((refined_means, extra_means), dim=0)
        all_color = torch.cat((refined_color, extra_color), dim=0)
        all_scales = torch.cat((scales, scales[sel]), dim=0)
        all_rot = torch.cat((rotations, rotations[sel]), dim=0)
        all_opacity = torch.cat((refined_opacity, extra_opacity), dim=0)
        all_rel = torch.cat((reliability, reliability[sel]), dim=0).clamp(0, 1)
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
            feat,
            scene_id=f"c1_{feat.shape[0]}",
        )

    def build_scene_from_backbone(
        self, backbone_gaussians: dict, source: SourceInput
    ) -> GaussianScene:
        base = base_scene_from_backbone(
            backbone_gaussians, scene_id="c1_real", batch_idx=0
        )
        n = len(base)
        feat = self.per_primitive_features(backbone_gaussians, n)
        return self._assemble(
            base.means,
            base.scales,
            base.rotations,
            base.opacity,
            base.color,
            feat,
            scene_id=f"c1_real_{n}",
        )
