"""C5 — multi-hypothesis refinement + source-only selector.

Generates K diverse REFINEMENT hypotheses of the frozen backbone (each hypothesis
is its own set of load-bearing per-primitive refinement heads acting on ALL
backbone primitives) and selects the best one using ONLY source-derived
information. The selector cannot inspect ground-truth target images — it uses a
source-driven scoring net over the source features as a proxy for quality.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .reconstruction_interface import (
    FrozenBackbone,
    GaussianScene,
    ReconstructionModel,
    SourceInput,
    base_scene_from_backbone,
)


class MultiHypothesisSelector(ReconstructionModel):
    provenance_tag = "c5_hypothesis"

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
        self.num_hypotheses = int(cfg.get("num_hypotheses", 3))
        self.backbone = FrozenBackbone(num_points, hidden=hidden, seed=seed)
        # K load-bearing refinement hypotheses: each has its own mean/color/opacity
        # head applied to ALL backbone primitives. The source-only selector picks
        # one. Heads are zero-init so an untrained model restores the exact
        # backbone (deletion reset-test), yet are free to grow during training.
        self.hyp_mean_heads = nn.ModuleList(
            [nn.Linear(hidden, 3) for _ in range(self.num_hypotheses)]
        )
        self.hyp_color_heads = nn.ModuleList(
            [nn.Linear(hidden, 3) for _ in range(self.num_hypotheses)]
        )
        self.hyp_opacity_heads = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in range(self.num_hypotheses)]
        )
        self.selector_net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.num_hypotheses),
        )
        with torch.no_grad():
            for heads in (
                self.hyp_mean_heads,
                self.hyp_color_heads,
                self.hyp_opacity_heads,
            ):
                for head in heads:
                    head.weight.zero_()
                    head.bias.zero_()

    def refinement_magnitude(self, feat: torch.Tensor) -> float:
        with torch.no_grad():
            return float(
                (self.refine_means_frac * torch.tanh(self.hyp_mean_heads[0](feat)))
                .norm(dim=1)
                .mean()
            )

    def _hypothesis(
        self,
        means,
        scales,
        rotations,
        opacity,
        color,
        reliability,
        feat,
        i,
        scene_id,
    ):
        """One hypothesis: refine ALL backbone primitives with hypothesis ``i``'s
        heads (provenance "backbone"), plus a SMALL bounded generated extra set."""
        n = means.shape[0]
        refined_means, refined_color, refined_opacity = self.refine_backbone_inplace(
            means,
            color,
            opacity,
            feat,
            mean_head=self.hyp_mean_heads[i],
            color_head=self.hyp_color_heads[i],
            opacity_head=self.hyp_opacity_heads[i],
        )
        k = self._extra_count(n, self.extra_points)
        sel = self._subsample_indices(n, k, device=means.device)
        extra_means = refined_means[sel] + 0.05 * torch.tanh(
            self.hyp_mean_heads[i](feat[sel])
        )
        extra_color = refined_color[sel]
        all_means = torch.cat((refined_means, extra_means), dim=0)
        all_color = torch.cat((refined_color, extra_color), dim=0)
        all_scales = torch.cat((scales, scales[sel]), dim=0)
        all_rot = torch.cat((rotations, rotations[sel]), dim=0)
        all_opacity = torch.cat((refined_opacity, refined_opacity[sel]), dim=0)
        all_rel = torch.cat((reliability, reliability[sel]), dim=0)
        provenance = ["backbone"] * n + [f"{self.provenance_tag}_{i}"] * k
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

    def build_hypotheses(self, source: SourceInput) -> List[GaussianScene]:
        b = self.backbone.predict(source)
        feat = b["features"]
        n = feat.shape[0]
        scenes = []
        for i in range(self.num_hypotheses):
            scenes.append(
                self._hypothesis(
                    b["means"],
                    b["scales"],
                    b["rotations"],
                    b["opacity"],
                    b["color"],
                    b["reliability"],
                    feat,
                    i,
                    scene_id=f"c5_{n}_h{i}",
                )
            )
        return scenes

    def select_hypothesis(
        self, source: SourceInput, hypotheses: List[GaussianScene]
    ) -> int:
        b = self.backbone.predict(source)
        feat = b["features"]
        scores = self.selector_net(feat.mean(dim=0, keepdim=True))
        return int(scores[0, : len(hypotheses)].argmax().item())

    def build_scene(self, source: SourceInput) -> GaussianScene:
        hypotheses = self.build_hypotheses(source)
        idx = self.select_hypothesis(source, hypotheses)
        return hypotheses[idx]

    def build_hypotheses_from_backbone(
        self, backbone_gaussians: dict
    ) -> List[GaussianScene]:
        base = base_scene_from_backbone(
            backbone_gaussians, scene_id="c5_real", batch_idx=0
        )
        n = len(base)
        feat = self.per_primitive_features(backbone_gaussians, n)
        scenes = []
        for i in range(self.num_hypotheses):
            scenes.append(
                self._hypothesis(
                    base.means,
                    base.scales,
                    base.rotations,
                    base.opacity,
                    base.color,
                    base.reliability,
                    feat,
                    i,
                    scene_id=f"c5_real_{n}_h{i}",
                )
            )
        return scenes

    def select_hypothesis_from_backbone(
        self, backbone_gaussians: dict, hypotheses: List[GaussianScene]
    ) -> int:
        base = base_scene_from_backbone(
            backbone_gaussians, scene_id="c5_sel", batch_idx=0
        )
        feat = self.per_primitive_features(backbone_gaussians, len(base))
        scores = self.selector_net(feat.mean(dim=0, keepdim=True))
        return int(scores[0, : len(hypotheses)].argmax().item())

    def build_scene_from_backbone(
        self, backbone_gaussians: dict, source: SourceInput
    ) -> GaussianScene:
        hypotheses = self.build_hypotheses_from_backbone(backbone_gaussians)
        idx = self.select_hypothesis_from_backbone(backbone_gaussians, hypotheses)
        return hypotheses[idx]
