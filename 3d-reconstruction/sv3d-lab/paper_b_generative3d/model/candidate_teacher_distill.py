"""C3 — generative-teacher distillation with a strictly source-only student.

A powerful generative teacher (e.g. Gen3R) produces target-consistent point sets
OFFLINE for TRAIN scenes only; they are cached and used purely as training
supervision. The student's ``build_scene`` is source-only — it never receives the
teacher output or any target signal at inference. The teacher cache refuses to
serve any scene registered under a non-train split (charter: teacher/oracle are
upper bounds and must not leak into the student forward or into test).
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


class TeacherDistillation(ReconstructionModel):
    provenance_tag = "c3_distilled"

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
        # Student refinement heads (source-only). The teacher supervises these
        # OFFLINE on train scenes; at inference only these source-driven heads run,
        # refining ALL backbone primitives in place (the primary trainable path).
        self.student_mean = nn.Linear(hidden, 3)
        self.student_color = nn.Linear(hidden, 3)
        self.opacity_head = nn.Linear(hidden, 1)
        with torch.no_grad():
            for head in (self.student_mean, self.student_color, self.opacity_head):
                head.weight.zero_()
                head.bias.zero_()
        self._teacher_cache = {}
        self._teacher_split = {}

    def register_teacher_cache(self, cache: dict, *, split: str) -> None:
        for scene_id, points in cache.items():
            self._teacher_cache[scene_id] = points
            self._teacher_split[scene_id] = split

    def lookup_teacher(self, scene_id: str, *, requesting_split: str = "train"):
        cached_split = self._teacher_split.get(scene_id)
        if cached_split is None:
            return None
        if requesting_split != "train" or cached_split != "train":
            raise ValueError(
                "teacher cache is train-only; refusing test/dev scene distillation"
            )
        return self._teacher_cache[scene_id]

    def refinement_magnitude(self, feat: torch.Tensor) -> float:
        with torch.no_grad():
            return float(
                (self.refine_means_frac * torch.tanh(self.student_mean(feat)))
                .norm(dim=1)
                .mean()
            )

    def _assemble(
        self, means, scales, rotations, opacity, color, reliability, feat, scene_id
    ):
        n = means.shape[0]
        # Load-bearing in-place refinement of ALL backbone primitives with the
        # source-only student head (teacher-distilled, teacher used train-only).
        refined_means, refined_color, refined_opacity = self.refine_backbone_inplace(
            means,
            color,
            opacity,
            feat,
            mean_head=self.student_mean,
            color_head=self.student_color,
            opacity_head=self.opacity_head,
        )
        k = self._extra_count(n, self.extra_points)
        sel = self._subsample_indices(n, k, device=means.device)
        extra_means = refined_means[sel] + 0.03 * torch.tanh(
            self.student_mean(feat[sel])
        )
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
            scene_id=f"c3_{feat.shape[0]}",
        )

    def build_scene_from_backbone(
        self, backbone_gaussians: dict, source: SourceInput
    ) -> GaussianScene:
        base = base_scene_from_backbone(
            backbone_gaussians, scene_id="c3_real", batch_idx=0
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
            scene_id=f"c3_real_{n}",
        )
