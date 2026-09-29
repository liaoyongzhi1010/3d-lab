"""Source-only reconstruction interface and the renderable GaussianScene contract.

Charter contract (research_charter.md sections 3, and the three-route spec):
  * Inference input is a single source RGB, source intrinsics, and canonical source
    camera pose only. NO target rgb/depth/feature ever reaches ``build_scene``.
  * The output is ONE unified, renderable explicit 3D representation (Gaussians).
    Every requested target view is rendered from this single scene.
  * Deleting provenance-tagged primitives must produce an attributable change and,
    for a residual-style candidate, restore the frozen-backbone-only render.

Everything here is pure-torch and CPU friendly. The renderer is a small,
deterministic, differentiable soft-splat that does not require CUDA, gsplat, or
Flash3D weights, so the whole contract is unit-testable on a laptop while the
real remote runs swap in the Flash3D renderer behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

SOURCE_CAMERA = "source_camera"

#: Channel count of the Flash3D backbone ``source_features`` map — the ResNet50
#: last-scale feature map (``ext.encoder(x)[-1]``) has 2048 channels. This is a
#: FIXED architectural constant of the real backbone, so the per-primitive feature
#: adapter can be built EAGERLY in ``__init__`` at a deterministic state-dict key
#: instead of lazily on the first forward (which left it out of the optimizer and
#: silently dropped on ``load_state_dict``). CPU unit tests feed a different (tiny)
#: channel count on purpose; ``per_primitive_features`` detects the mismatch and
#: falls back to a per-dim adapter for those.
SOURCE_FEATURE_DIM = 2048


@dataclass
class SourceInput:
    """The ONLY information a source-only model may condition on."""

    source_rgb: torch.Tensor  # [B,3,H,W] in [0,1]
    K_src: torch.Tensor  # [B,3,3]
    T_c2w_src: torch.Tensor  # [B,4,4]

    def __post_init__(self):
        if self.source_rgb.dim() != 4 or self.source_rgb.shape[1] != 3:
            raise ValueError("source_rgb must be [B,3,H,W]")
        if self.K_src.shape[-2:] != (3, 3):
            raise ValueError("K_src must be [B,3,3]")
        if self.T_c2w_src.shape[-2:] != (4, 4):
            raise ValueError("T_c2w_src must be [B,4,4]")

    @property
    def batch(self) -> int:
        return self.source_rgb.shape[0]


@dataclass
class TargetCameras:
    """Target *positions* only — never any target image content."""

    K: torch.Tensor  # [B,V,3,3]
    T_c2w: torch.Tensor  # [B,V,4,4]

    def __post_init__(self):
        if self.K.dim() != 4 or self.K.shape[-2:] != (3, 3):
            raise ValueError("K must be [B,V,3,3]")
        if self.T_c2w.dim() != 4 or self.T_c2w.shape[-2:] != (4, 4):
            raise ValueError("T_c2w must be [B,V,4,4]")

    @property
    def num_views(self) -> int:
        return self.K.shape[1]


@dataclass
class RenderOutput:
    images: torch.Tensor  # [B,V,3,H,W]
    scene_id: str


_FIELD_SHAPES = {
    "means": 3,
    "scales": 3,
    "rotations": 4,
    "opacity": 1,
    "color": 3,
    "reliability": 1,
}


@dataclass
class GaussianScene:
    """One unified renderable explicit-3D representation in the source camera frame."""

    means: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacity: torch.Tensor
    color: torch.Tensor
    provenance: List[str]
    reliability: torch.Tensor
    scene_id: str
    camera_convention: str = SOURCE_CAMERA

    def __len__(self) -> int:
        return self.means.shape[0]

    def validate(self) -> "GaussianScene":
        n = self.means.shape[0]
        for name, dim in _FIELD_SHAPES.items():
            tensor = getattr(self, name)
            if not torch.is_tensor(tensor):
                raise ValueError(f"{name} must be a tensor")
            if tensor.dim() != 2 or tensor.shape[1] != dim:
                raise ValueError(f"{name} must be [N,{dim}]")
            if tensor.shape[0] != n:
                raise ValueError(f"{name} count {tensor.shape[0]} != means count {n}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite")
        if len(self.provenance) != n:
            raise ValueError(f"provenance count {len(self.provenance)} != count {n}")
        if (self.scales <= 0).any():
            raise ValueError("scales must be positive")
        if torch.linalg.norm(self.rotations, dim=1).min() <= 1e-8:
            raise ValueError("rotations must be non-zero quaternions")
        if (self.opacity < 0).any() or (self.opacity > 1).any():
            raise ValueError("opacity must be in [0,1]")
        if (self.color < 0).any() or (self.color > 1).any():
            raise ValueError("color must be in [0,1]")
        if (self.reliability < 0).any() or (self.reliability > 1).any():
            raise ValueError("reliability must be in [0,1]")
        if self.camera_convention != SOURCE_CAMERA:
            raise ValueError("scene must use the source camera convention")
        return self

    def to_dict(self) -> dict:
        return {
            "means": self.means.detach().cpu().tolist(),
            "scales": self.scales.detach().cpu().tolist(),
            "rotations": self.rotations.detach().cpu().tolist(),
            "opacity": self.opacity.detach().cpu().tolist(),
            "color": self.color.detach().cpu().tolist(),
            "reliability": self.reliability.detach().cpu().tolist(),
            "provenance": list(self.provenance),
            "scene_id": self.scene_id,
            "camera_convention": self.camera_convention,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "GaussianScene":
        def t(key):
            return torch.tensor(payload[key], dtype=torch.float32)

        return cls(
            means=t("means"),
            scales=t("scales"),
            rotations=t("rotations"),
            opacity=t("opacity"),
            color=t("color"),
            provenance=list(payload["provenance"]),
            reliability=t("reliability"),
            scene_id=payload["scene_id"],
            camera_convention=payload.get("camera_convention", SOURCE_CAMERA),
        )

    def delete_by_provenance(self, *tags: str) -> "GaussianScene":
        drop = set(tags)
        keep = [i for i, p in enumerate(self.provenance) if p not in drop]
        index = torch.tensor(keep, dtype=torch.long)
        return GaussianScene(
            means=self.means[index],
            scales=self.scales[index],
            rotations=self.rotations[index],
            opacity=self.opacity[index],
            color=self.color[index],
            provenance=[self.provenance[i] for i in keep],
            reliability=self.reliability[index],
            scene_id=self.scene_id,
            camera_convention=self.camera_convention,
        )


def render_scene(scene: GaussianScene, cameras: TargetCameras, *, height=8, width=12):
    """Deterministic soft-splat renderer.

    Projects Gaussian means through each target camera relative to the source
    camera and accumulates opacity-weighted colour. Order over views is decided
    entirely by the camera list, and the SAME scene tensors feed every view, so
    the result is target-order invariant by construction.
    """

    means = scene.means
    device = means.device
    B = cameras.T_c2w.shape[0]
    V = cameras.num_views
    images = []
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=means.dtype),
        torch.arange(width, device=device, dtype=means.dtype),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=-1).reshape(-1, 2)  # [P,2]
    for b in range(B):
        view_imgs = []
        for v in range(V):
            K = cameras.K[b, v]
            pose = cameras.T_c2w[b, v]
            R = pose[:3, :3]
            t = pose[:3, 3]
            cam_pts = means @ R.T + t  # world==source frame -> target cam
            z = cam_pts[:, 2].clamp(min=1e-3)
            uv = (cam_pts @ K.T)[:, :2] / z.unsqueeze(1)
            sigma = scene.scales.mean(dim=1).clamp(min=1e-2) * 4.0
            diff = grid.unsqueeze(1) - uv.unsqueeze(0)  # [P,N,2]
            weight = torch.exp(-(diff.pow(2).sum(-1)) / (2.0 * sigma.unsqueeze(0) ** 2))
            weight = weight * scene.opacity.squeeze(1).unsqueeze(0) / z.unsqueeze(0)
            wsum = weight.sum(1, keepdim=True).clamp(min=1e-6)
            color = (weight @ scene.color) / wsum  # [P,3]
            view_imgs.append(color.reshape(height, width, 3).permute(2, 0, 1))
        images.append(torch.stack(view_imgs))
    return RenderOutput(images=torch.stack(images), scene_id=scene.scene_id)


BACKBONE_TAG = "backbone"


def _pick_batch_item(tensor: torch.Tensor, batch_idx: int) -> torch.Tensor:
    """Return a per-item ``[N, C]`` view whether ``tensor`` is batched or flat."""

    if tensor is None:
        return None
    if tensor.dim() == 3:
        return tensor[batch_idx]
    return tensor


def base_scene_from_backbone(
    backbone_gaussians: dict,
    *,
    provenance_tag: str = BACKBONE_TAG,
    scene_id: str,
    batch_idx: int = 0,
) -> GaussianScene:
    """Build the frozen-anchor ``GaussianScene`` from a real backbone dict.

    ``backbone_gaussians`` is the exact dict returned by
    ``Flash3DBackbone.extract_source_gaussians`` (batched ``[B,N,...]``) or an
    already single-item dict. These primitives are the FROZEN source-camera
    anchors: their tensors are ``.detach()``-ed so no gradient flows to the
    backbone, and they are tagged ``"backbone"`` so deletion restores the
    backbone-only render.
    """

    xyz = _pick_batch_item(backbone_gaussians["xyz"], batch_idx).detach()
    scales = _pick_batch_item(backbone_gaussians["scales"], batch_idx).detach()
    rotations = _pick_batch_item(backbone_gaussians["rotations"], batch_idx).detach()
    opacity = _pick_batch_item(backbone_gaussians["opacity"], batch_idx).detach()
    color = _pick_batch_item(backbone_gaussians["color_rgb"], batch_idx).detach()

    scales = scales.clamp(min=1e-4)
    opacity = opacity.clamp(0.0, 1.0)
    color = color.clamp(0.0, 1.0)
    rnorm = torch.linalg.norm(rotations, dim=1, keepdim=True).clamp(min=1e-8)
    rotations = rotations / rnorm

    n = xyz.shape[0]
    reliability = torch.full((n, 1), 0.5, dtype=xyz.dtype, device=xyz.device)
    return GaussianScene(
        means=xyz,
        scales=scales,
        rotations=rotations,
        opacity=opacity,
        color=color,
        provenance=[provenance_tag] * n,
        reliability=reliability,
        scene_id=scene_id,
    )


class ReconstructionModel(nn.Module):
    """Abstract single-view reconstruction candidate.

    Subclasses implement ``build_scene(self, source)`` — note the signature
    accepts ONLY a ``SourceInput``. No target rgb/depth/feature parameter may be
    added; ``tests/reconstruction/test_source_only_contract.py`` enforces this.

    Subclasses ALSO implement ``build_scene_from_backbone(backbone_gaussians,
    source)`` which consumes the REAL Flash3D backbone output (see
    ``Flash3DBackbone.extract_source_gaussians``) as the frozen anchor and adds
    only trainable, provenance-tagged residual capacity. Same source-only rule:
    no target rgb/depth/feature parameter is allowed.
    """

    render_height = 8
    render_width = 12
    feature_hidden = 8

    def build_scene(self, source: SourceInput) -> GaussianScene:  # pragma: no cover
        raise NotImplementedError

    def build_scene_from_backbone(
        self, backbone_gaussians: dict, source: SourceInput
    ) -> GaussianScene:  # pragma: no cover
        raise NotImplementedError

    def _build_feature_adapter(self) -> None:
        """EAGERLY create the trainable Cin->hidden feature adapter.

        Called from every candidate ``__init__`` (after ``feature_hidden`` is set)
        so ``self._feat_adapter`` is ALWAYS present in ``.parameters()`` and in the
        ``state_dict`` at a FIXED key (``_feat_adapter.weight`` /
        ``_feat_adapter.bias``) — before any forward. This fixes two bugs:

          1. ``train_real.py`` builds the optimizer from ``candidate.parameters()``
             before the first forward; a lazily-created adapter was silently absent
             and never trained.
          2. ``eval_candidates_real.py`` runs ``load_state_dict`` before any
             forward; a lazily-created adapter did not exist yet, so the trained
             adapter weights were silently dropped and the candidate render
             collapsed to the backbone render (delta PSNR ~0).

        The real Flash3D ``source_features`` channel count is the fixed
        ``SOURCE_FEATURE_DIM`` constant, so the eager adapter is deterministic. CPU
        unit tests feed a different (tiny) channel count; those hit the per-dim
        fallback in ``per_primitive_features``.
        """

        adapter = nn.Linear(SOURCE_FEATURE_DIM, self.feature_hidden)
        with torch.no_grad():
            adapter.weight.mul_(0.1)
            adapter.bias.zero_()
        self._feat_adapter = adapter

    def _feature_adapter(self, in_dim: int) -> nn.Linear:
        """Return a trainable ``in_dim -> hidden`` adapter for ``source_features``.

        When ``in_dim`` matches the eager ``SOURCE_FEATURE_DIM`` adapter (the real
        Flash3D path) we return the eager module so its weights are trained and
        round-trip through the checkpoint. When it does NOT match (the CPU
        fake-backbone tests, which use a tiny channel count) we lazily create and
        cache a per-dim adapter in ``_feat_adapters`` — this preserves the existing
        CPU-test behaviour without weakening the real-path guarantee.
        """

        if getattr(self, "_feat_adapter", None) is not None and (
            in_dim == self._feat_adapter.in_features
        ):
            return self._feat_adapter

        if not hasattr(self, "_feat_adapters"):
            self._feat_adapters = nn.ModuleDict()
        key = str(in_dim)
        if key not in self._feat_adapters:
            adapter = nn.Linear(in_dim, self.feature_hidden)
            with torch.no_grad():
                adapter.weight.mul_(0.1)
                adapter.bias.zero_()
            self._feat_adapters[key] = adapter.to(
                next(iter(self.parameters())).device
                if any(True for _ in self.parameters())
                else "cpu"
            )
        return self._feat_adapters[key]

    @staticmethod
    def primitive_source_pixels(
        n: int,
        pixel_hw: Sequence[int],
        gaussians_per_pixel: int,
        *,
        device=None,
        dtype=torch.float32,
    ) -> torch.Tensor:
        """Map flat primitive indices ``0..n-1`` to their source-pixel ``(x, y)``.

        The Flash3D backbone primitives are per-pixel with the fixed layout coming
        from ``extract_source_gaussians``' ``rearrange("(b n) c l -> b (n l) c",
        n=gpp)`` where ``l = padded_H * padded_W`` is the flattened source-pixel
        grid. Hence for a primitive index ``i``::

            layer = i // (padded_H * padded_W)
            pix   = i %  (padded_H * padded_W)
            y     = pix // padded_W
            x     = pix %  padded_W

        Every one of the ``gaussians_per_pixel`` layers over the same pixel shares
        the same ``(x, y)``. Returns a ``[n, 2]`` tensor of integer pixel
        coordinates as floats.
        """

        padded_h, padded_w = int(pixel_hw[0]), int(pixel_hw[1])
        area = padded_h * padded_w
        idx = torch.arange(n, device=device)
        pix = idx % area
        y = (pix // padded_w).to(dtype)
        x = (pix % padded_w).to(dtype)
        return torch.stack((x, y), dim=1)  # [n,2]

    def per_primitive_features(
        self,
        backbone_gaussians: dict,
        n: int,
        batch_idx: int = 0,
        *,
        pixel_hw: Optional[Sequence[int]] = None,
        gaussians_per_pixel: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample ``source_features [B,C,hf,wf]`` per-primitive at its source pixel.

        The Flash3D primitives are per-pixel: primitive index ``i`` corresponds to
        source pixel ``(x, y)`` over the padded ``(padded_H, padded_W)`` grid with
        ``gaussians_per_pixel`` layers (see ``primitive_source_pixels`` and
        ``backbone_flash3d.extract_source_gaussians``). ``source_features`` lives at
        a DIFFERENT encoder resolution ``(hf, wf)``, so we bilinearly sample it at
        each primitive's normalized ``(x, y)`` source-pixel coordinate, producing a
        ``[n, C]`` tensor, then push it through a trainable ``Cin -> hidden`` adapter
        with ``tanh`` to ``[n, hidden]``.

        ``pixel_hw`` / ``gaussians_per_pixel`` are read from ``backbone_gaussians``
        (keys ``"pixel_hw"`` and ``"gaussians_per_pixel"`` added by the real
        backbone) unless passed explicitly. When no spatial layout information is
        available (e.g. the CPU fake-backbone dict) we fall back to the pooled
        broadcast so existing CPU tests keep passing.
        """

        feats = backbone_gaussians.get("source_features")
        if feats is None:
            raise ValueError("backbone_gaussians must include 'source_features'")
        if feats.dim() == 4:
            feats = feats[batch_idx]  # [C,hf,wf]
        adapter = self._feature_adapter(feats.shape[0])
        feats = feats.to(adapter.weight.dtype)

        if pixel_hw is None:
            pixel_hw = backbone_gaussians.get("pixel_hw")
        if gaussians_per_pixel is None:
            gaussians_per_pixel = backbone_gaussians.get("gaussians_per_pixel")

        if pixel_hw is None or feats.dim() != 3:
            # CPU-safe fallback: no spatial layout -> pooled broadcast.
            pooled = feats.reshape(feats.shape[0], -1).mean(dim=1)  # [C]
            hidden = torch.tanh(adapter(pooled))  # [hidden]
            return hidden.unsqueeze(0).expand(n, -1)  # [n,hidden]

        padded_h, padded_w = int(pixel_hw[0]), int(pixel_hw[1])
        gpp = int(gaussians_per_pixel) if gaussians_per_pixel else 1
        xy = self.primitive_source_pixels(
            n, pixel_hw, gpp, device=feats.device, dtype=feats.dtype
        )  # [n,2] pixel coords in padded (x,y)
        # Normalize pixel-center coords to grid_sample's [-1,1] range. The pixel
        # grid spans [0, padded_w-1] x [0, padded_h-1]; align_corners=True maps the
        # first/last sample centers to -1/+1, matching integer pixel indices.
        gx = xy[:, 0] / max(padded_w - 1, 1) * 2.0 - 1.0
        gy = xy[:, 1] / max(padded_h - 1, 1) * 2.0 - 1.0
        grid = torch.stack((gx, gy), dim=1).view(1, n, 1, 2)  # [1,n,1,2]
        sampled = torch.nn.functional.grid_sample(
            feats.unsqueeze(0),  # [1,C,hf,wf]
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # [1,C,n,1]
        per_prim = sampled.squeeze(0).squeeze(-1).transpose(0, 1)  # [n,C]
        return torch.tanh(adapter(per_prim))  # [n,hidden]

    #: Upper bound on the number of *extra* (deletable, candidate-tagged) primitives
    #: a refinement-family candidate may append on top of the refined backbone set.
    #: The point count therefore stays near ``backbone + max_extra_points`` rather
    #: than doubling to ``2 * backbone`` as the old duplicate-full-set design did.
    max_extra_points = 4096

    def _extra_count(self, n: int, requested: Optional[int] = None) -> int:
        """Bounded number of extra primitives: ``min(requested or default, n, cap)``."""

        want = (
            int(requested) if requested is not None else min(n, self.max_extra_points)
        )
        return max(1, min(want, n, self.max_extra_points))

    @staticmethod
    def _subsample_indices(n: int, k: int, device=None) -> torch.Tensor:
        """Deterministic evenly-spaced indices selecting ``k`` of ``n`` primitives."""

        k = max(1, min(k, n))
        if k == n:
            return torch.arange(n, device=device)
        return torch.linspace(0, n - 1, steps=k, device=device).round().long()

    #: Bounded residual scales for the load-bearing in-place refinement of the
    #: backbone primitives. These are small-but-non-negligible: they must be able
    #: to actually change the render (the E-112 fix), yet stay bounded so the
    #: refinement can never blow up the frozen anchors. ``means`` is scaled by the
    #: scene extent so the offset is a fixed *fraction* of the scene regardless of
    #: units; ``color``/``opacity`` are additive logit deltas bounded by ``tanh``.
    refine_means_frac = 0.05  # fraction of scene scale
    refine_color_logit = 2.0  # max +/- color-logit shift
    refine_opacity_logit = 2.0  # max +/- opacity-logit shift

    @staticmethod
    def _scene_scale(means: torch.Tensor) -> torch.Tensor:
        """Robust scalar extent of the scene, used to bound the means offset."""

        if means.shape[0] == 0:
            return means.new_tensor(1.0)
        center = means.mean(dim=0, keepdim=True)
        return (means - center).norm(dim=1).mean().clamp(min=1e-3)

    def refine_backbone_inplace(
        self,
        means: torch.Tensor,
        color: torch.Tensor,
        opacity: torch.Tensor,
        feat: torch.Tensor,
        *,
        mean_head: nn.Module,
        color_head: nn.Module,
        opacity_head: nn.Module,
        gate: Optional[torch.Tensor] = None,
    ):
        """Load-bearing in-place refinement of ALL backbone primitives.

        The refinement is a bounded residual driven by the per-primitive source
        features ``feat`` (source-only, no target signal). Each head is applied to
        every primitive so the trainable capacity acts on the full backbone set,
        not just a handful of extra points (the E-112 fix). All three residuals are
        zero when the heads' ``weight`` and ``bias`` are zero, so resetting the
        heads restores the exact backbone anchors (deletion contract).

        ``gate`` (``[n,1]`` in ``[0,1]``) optionally scales the opacity/color
        residual so capacity is only spent where requested (C1 reliability gating).
        """

        scene_scale = self._scene_scale(means)
        gate = 1.0 if gate is None else gate

        mean_delta = self.refine_means_frac * scene_scale * torch.tanh(mean_head(feat))
        refined_means = means + mean_delta

        color_logit = torch.logit(color.clamp(1e-4, 1 - 1e-4))
        color_delta = self.refine_color_logit * torch.tanh(color_head(feat))
        refined_color = torch.sigmoid(color_logit + gate * color_delta)

        opacity_logit = torch.logit(opacity.clamp(1e-4, 1 - 1e-4))
        opacity_delta = self.refine_opacity_logit * torch.tanh(opacity_head(feat))
        refined_opacity = torch.sigmoid(opacity_logit + gate * opacity_delta).clamp(
            0.0, 1.0
        )
        return refined_means, refined_color, refined_opacity

    def render_targets(
        self, scene: GaussianScene, target_cameras: TargetCameras
    ) -> RenderOutput:
        return render_scene(
            scene,
            target_cameras,
            height=self.render_height,
            width=self.render_width,
        )

    def frozen_backbone_state(self):
        return {
            name: p.detach()
            for name, p in self.named_parameters()
            if not p.requires_grad
        }


class FrozenBackbone(nn.Module):
    """A tiny stand-in for a frozen Flash3D-style per-pixel Gaussian predictor.

    On the remote flash3d venv this is swapped for the real GaussianPredictor
    (loaded from ``model_re10k_v2.pth`` and frozen). Locally it is a small fixed
    network with ``requires_grad=False`` on every parameter, so tests can prove
    that (a) gradients never reach it and (b) training does not change it.
    Deterministic given the source input.
    """

    def __init__(self, num_points: int, hidden: int = 8, seed: int = 0):
        super().__init__()
        self.num_points = num_points
        gen = torch.Generator().manual_seed(seed)
        self.proj = nn.Linear(3, hidden)
        self.to_mean = nn.Linear(hidden, 3)
        self.to_color = nn.Linear(hidden, 3)
        with torch.no_grad():
            for layer in (self.proj, self.to_mean, self.to_color):
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=gen) * 0.3)
                layer.bias.copy_(torch.randn(layer.bias.shape, generator=gen) * 0.1)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def predict(self, source: "SourceInput"):
        rgb = source.source_rgb[0]  # [3,H,W]
        _, h, w = rgb.shape
        n = self.num_points
        gen = torch.Generator().manual_seed(1234)
        ys = torch.randint(0, h, (n,), generator=gen)
        xs = torch.randint(0, w, (n,), generator=gen)
        pix = rgb[:, ys, xs].T  # [n,3]
        feat = torch.tanh(self.proj(pix))
        offsets = self.to_mean(feat)
        base = torch.stack(
            (
                xs.float() / w - 0.5,
                ys.float() / h - 0.5,
                torch.full((n,), 1.5),
            ),
            dim=1,
        )
        means = base + 0.1 * offsets
        color = torch.sigmoid(self.to_color(feat))
        scales = torch.full((n, 3), 0.1)
        rotations = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(n, 1)
        opacity = torch.full((n, 1), 0.6)
        reliability = torch.full((n, 1), 0.5)
        return {
            "means": means,
            "scales": scales,
            "rotations": rotations,
            "opacity": opacity,
            "color": color,
            "reliability": reliability,
            "features": feat,
        }
