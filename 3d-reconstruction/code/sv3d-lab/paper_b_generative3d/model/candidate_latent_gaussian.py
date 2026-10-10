"""C2 — latent Gaussian generation.

A small trainable generator maps a scene latent to a spatially-coherent field of
Gaussian offsets over the backbone primitives. The latent is decoded through a
low-frequency spatial basis so that perturbing it moves neighbouring primitives
together (coherent), unlike an i.i.d. per-primitive perturbation (the incoherence
baseline). This is what makes the generation 3D-native rather than per-pixel noise.
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


class LatentGaussianGeneration(ReconstructionModel):
    provenance_tag = "c2_latent"

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
        self.n_basis = int(cfg.get("n_basis", 3))
        self.backbone = FrozenBackbone(num_points, hidden=hidden, seed=seed)
        self.decode = nn.Linear(self.latent_dim, self.n_basis * 3)
        self.latent_encoder = nn.Linear(hidden, self.latent_dim)
        # Load-bearing latent-modulated refinement of ALL backbone primitives.
        # The scene latent is broadcast and concatenated onto each primitive's
        # feature, so the latent CONDITIONS the refinement field (C2's
        # per-candidate differentiation). Heads are zero-init and named so the
        # deletion reset-test can restore the exact backbone.
        self.mean_head = nn.Linear(hidden + self.latent_dim, 3)
        self.color_head = nn.Linear(hidden + self.latent_dim, 3)
        self.opacity_head = nn.Linear(hidden + self.latent_dim, 1)
        with torch.no_grad():
            self.decode.weight.mul_(0.5)
            self.latent_encoder.weight.mul_(0.1)
            self.latent_encoder.bias.zero_()
            for head in (self.mean_head, self.color_head, self.opacity_head):
                head.weight.zero_()
                head.bias.zero_()

    def _latent_modulated_feat(self, feat: torch.Tensor, latent: torch.Tensor):
        """Condition each primitive's feature on the shared scene latent."""
        lat = latent.reshape(1, -1).expand(feat.shape[0], -1)
        return torch.cat((feat, lat), dim=1)

    def sample_latent(self, seed: int = 0) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(self.latent_dim, generator=gen)

    def _spatial_basis(self, means: torch.Tensor) -> torch.Tensor:
        # Low-frequency cosine basis over the primitive x-coordinate: nearby
        # primitives get near-identical basis values -> coherent motion.
        x = means[:, 0]
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        freqs = torch.arange(
            1, self.n_basis + 1, dtype=means.dtype, device=means.device
        )
        return torch.cos(torch.pi * x.unsqueeze(1) * freqs.unsqueeze(0))  # [n,B]

    def _decode_offsets(self, latent: torch.Tensor, means: torch.Tensor):
        basis = self._spatial_basis(means)  # [n,B]
        coeff = self.decode(latent).reshape(self.n_basis, 3)  # [B,3]
        return basis @ coeff  # [n,3] coherent offset field

    def build_scene(self, source: SourceInput) -> GaussianScene:
        latent = torch.zeros(self.latent_dim)
        return self.build_scene_with_latent(source, latent)

    def build_scene_with_latent(
        self, source: SourceInput, latent: torch.Tensor
    ) -> GaussianScene:
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
            latent,
            scene_id=f"c2_{feat.shape[0]}",
        )

    def _assemble(
        self,
        means,
        scales,
        rotations,
        opacity,
        color,
        reliability,
        feat,
        latent,
        scene_id,
    ):
        """Refine ALL backbone anchors in place with a LATENT-MODULATED field
        (provenance "backbone", the primary trainable path), plus a SMALL bounded
        extra set of latent-generated Gaussians (secondary, deletable). The extra
        count is bounded (not a full duplicate), keeping the total near
        ``n + extra``."""
        n = means.shape[0]
        mod_feat = self._latent_modulated_feat(feat, latent)
        refined_means, refined_color, refined_opacity = self.refine_backbone_inplace(
            means,
            color,
            opacity,
            mod_feat,
            mean_head=self.mean_head,
            color_head=self.color_head,
            opacity_head=self.opacity_head,
        )

        k = self._extra_count(n, self.extra_points)
        sel = self._subsample_indices(n, k, device=means.device)
        sub_means = refined_means[sel]
        offsets = self._decode_offsets(latent, sub_means)  # coherent over subset
        extra_means = sub_means + 0.3 * offsets
        extra_color = torch.sigmoid(
            self.color_head(mod_feat[sel])
        )  # latent-conditioned color
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

    def build_scene_from_backbone(
        self, backbone_gaussians: dict, source: SourceInput
    ) -> GaussianScene:
        base = base_scene_from_backbone(
            backbone_gaussians, scene_id="c2_real", batch_idx=0
        )
        n = len(base)
        feat = self.per_primitive_features(backbone_gaussians, n)
        latent = self.latent_encoder(feat.mean(dim=0))  # [latent_dim]
        return self._assemble(
            base.means,
            base.scales,
            base.rotations,
            base.opacity,
            base.color,
            base.reliability,
            feat,
            latent,
            scene_id=f"c2_real_{n}",
        )

    @staticmethod
    def _added(scene: GaussianScene) -> torch.Tensor:
        idx = [i for i, p in enumerate(scene.provenance) if p != "backbone"]
        return scene.means[torch.tensor(idx, dtype=torch.long)]

    def latent_spatial_coherence(self, base: GaussianScene, perturbed: GaussianScene):
        """Higher = neighbouring primitives move more similarly (coherent)."""
        d = self._added(perturbed) - self._added(base)  # [n,3]
        order = torch.argsort(self._added(base)[:, 0])
        d = d[order]
        neighbour_diff = (d[1:] - d[:-1]).norm(dim=1).mean()
        return float(-neighbour_diff)

    def latent_incoherence_baseline(
        self, base: GaussianScene, perturbed: GaussianScene
    ):
        """Coherence of a shuffled (i.i.d.) version of the same displacement."""
        d = self._added(perturbed) - self._added(base)
        gen = torch.Generator().manual_seed(0)
        perm = torch.randperm(d.shape[0], generator=gen)
        d = d[perm]
        neighbour_diff = (d[1:] - d[:-1]).norm(dim=1).mean()
        return float(-neighbour_diff)
