"""Geometry-DPC: a source-camouflaged white-box attack on explicit-3D structure.

Where the render-level DPC (attacks/dpc_attack.py) diverges *rendered novel views*, geometry-
DPC diverges the *explicit 3D primitives* (Gaussian means) that a feed-forward model predicts,
while keeping the source appearance camouflaged. It reuses DPC's auxiliary-pose machinery and
the shared source-camouflage philosophy, but its divergence term is computed in common target
coordinates on the primitive point cloud, not on rendered RGB.

Design (matches the spec):
  * The clean scene / GeometryState is cached once (no grad); the adversarial GeometryState is
    rebuilt every step from ``image + delta`` through a differentiable ``build_state`` callable.
  * SOFT differentiable objectives drive optimization:
      - ``primitive``: aligned 3D point displacement (scale-removed) of matched primitives.
      - ``depth``: divergence of per-primitive target depth after global-scale alignment.
      - ``ordering``: a smooth surrogate that encourages near/far order reversals of well-
        separated clean pairs (hinge on pairwise depth-difference sign).
      - ``hybrid``: weighted sum of the three.
  * The HARD semantic predicates (valid order reversal, mutual visibility) stay in
    geometry_metrics.py and are evaluation-only; they are never used as gradients.

CPU-testable through the injected ``build_state`` callable; the Flash3D adapter provides the
real one on the server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from attacks.dpc_attack import DPCConfig, sample_auxiliary_poses
from attacks.geometry_metrics import (
    GeometryState,
    align_scale_least_squares,
    transform_points,
)

BuildStateFn = Callable[[torch.Tensor], GeometryState]
"""build_state(image[1,3,H,W]) -> GeometryState with primitive means in source-camera coords.

Must be differentiable wrt the input image so gradients flow into the perturbation.
"""


@dataclass
class GeometryAttackConfig:
    epsilon: float = 8.0 / 255.0
    steps: int = 40
    step_size: float = 1.0 / 255.0
    lambda_src: float = 10.0
    mode: str = "hybrid"  # one of primitive|depth|ordering|hybrid
    w_primitive: float = 1.0
    w_depth: float = 1.0
    w_ordering: float = 1.0
    n_aux_poses: int = 4
    seed: int = 0
    momentum: float = 0.9
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    order_margin_rel: float = 0.05


def _aligned_target_depths(
    adv_state: GeometryState,
    clean_state: GeometryState,
    T: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Matched, global-scale-aligned target-frame primitive positions (adv, clean).

    Matching is by shared primitive id (in clean order). Differentiable wrt adv means.
    """
    adv_index = {pid: i for i, pid in enumerate(adv_state.primitive_ids)}
    rows_adv = []
    rows_clean = []
    for j, pid in enumerate(clean_state.primitive_ids):
        if pid in adv_index:
            rows_adv.append(adv_index[pid])
            rows_clean.append(j)
    if not rows_clean:
        raise ValueError("no shared primitive ids between clean and adv states")
    idx_adv = torch.tensor(rows_adv, dtype=torch.long)
    idx_clean = torch.tensor(rows_clean, dtype=torch.long)

    adv_t = transform_points(adv_state.means, T)[idx_adv]
    clean_t = transform_points(clean_state.means, T)[idx_clean]
    # global scale from depth (z); detach clean, keep grad through adv.
    s = align_scale_least_squares(adv_t[:, 2].detach(), clean_t[:, 2])
    adv_aligned = adv_t / max(s, 1e-8)
    return adv_aligned, clean_t


def geometry_soft_objective(
    adv_state: GeometryState,
    clean_state: GeometryState,
    source_adv: torch.Tensor,
    source_clean: torch.Tensor,
    target_poses: list,
    lambda_src: float,
    mode: str = "hybrid",
    w_primitive: float = 1.0,
    w_depth: float = 1.0,
    w_ordering: float = 1.0,
    order_margin_rel: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """Source-camouflaged soft geometry objective (to MINIMIZE).

    ``loss = lambda_src * source_cost - divergence`` where divergence is the selected
    combination of primitive / depth / ordering surrogates averaged over target poses.
    """
    if mode not in ("primitive", "depth", "ordering", "hybrid"):
        raise ValueError(f"unknown mode {mode!r}")
    if not target_poses:
        raise ValueError("geometry objective requires at least one target pose")

    source_cost = (source_adv - source_clean).pow(2).mean()

    prim_terms = []
    depth_terms = []
    order_terms = []
    for T in target_poses:
        adv_aligned, clean_t = _aligned_target_depths(adv_state, clean_state, T)
        # primitive: mean 3D displacement (scale-removed).
        prim_terms.append((adv_aligned - clean_t).norm(dim=1).mean())
        # depth: divergence of aligned target depth.
        depth_terms.append((adv_aligned[:, 2] - clean_t[:, 2]).pow(2).mean())
        # ordering: reward flipping the sign of pairwise depth gaps for well-separated pairs.
        order_terms.append(
            _ordering_surrogate(adv_aligned[:, 2], clean_t[:, 2], order_margin_rel)
        )

    primitive = torch.stack(prim_terms).mean()
    depth = torch.stack(depth_terms).mean()
    ordering = torch.stack(order_terms).mean()

    if mode == "primitive":
        divergence = primitive
    elif mode == "depth":
        divergence = depth
    elif mode == "ordering":
        divergence = ordering
    else:  # hybrid
        divergence = w_primitive * primitive + w_depth * depth + w_ordering * ordering

    loss = lambda_src * source_cost - divergence
    comps = {
        "source_cost": source_cost,
        "primitive": primitive,
        "depth": depth,
        "ordering": ordering,
        "divergence": divergence,
    }
    return loss, comps


def _ordering_surrogate(
    adv_depth: torch.Tensor,
    clean_depth: torch.Tensor,
    margin_rel: float,
    max_pairs: int = 4096,
    seed: int = 42,
) -> torch.Tensor:
    """Smooth surrogate that grows when clean-separated near/far pairs flip order.

    Uses pre-sampled eligible pairs to avoid N^2 memory.
    """
    n = clean_depth.shape[0]
    if n < 2:
        return adv_depth.sum() * 0.0
    device = clean_depth.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_sample = min(max_pairs * 10, n * (n - 1) // 2, 500000)
    idx_i = torch.randint(n, (n_sample,), generator=g).to(device)
    idx_j = torch.randint(n, (n_sample,), generator=g).to(device)
    valid = idx_i != idx_j
    idx_i, idx_j = idx_i[valid], idx_j[valid]
    ci = clean_depth[idx_i]
    cj = clean_depth[idx_j]
    clean_gap = ci - cj
    denom = torch.maximum(ci.abs(), cj.abs()).clamp_min(1e-8)
    eligible = (clean_gap.abs() / denom) >= margin_rel
    idx_i, idx_j = idx_i[eligible], idx_j[eligible]
    clean_gap = clean_gap[eligible]
    if idx_i.shape[0] == 0:
        return adv_depth.sum() * 0.0
    if idx_i.shape[0] > max_pairs:
        perm = torch.randperm(idx_i.shape[0], generator=g)[:max_pairs]
        idx_i, idx_j, clean_gap = idx_i[perm], idx_j[perm], clean_gap[perm]
    ai = adv_depth[idx_i]
    aj = adv_depth[idx_j]
    adv_gap = ai - aj
    flip = torch.relu(-torch.sign(clean_gap) * adv_gap)
    return flip.mean()


def geometry_attack(
    image: torch.Tensor,
    build_state: BuildStateFn,
    cfg: GeometryAttackConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Run geometry-DPC. Returns ``(attacked_image, delta, info)``, delta L-inf <= epsilon."""
    if image.ndim != 4:
        raise ValueError(f"Expected BCHW image, got {tuple(image.shape)}")
    device = image.device

    dpc_cfg = DPCConfig(n_aux_poses=cfg.n_aux_poses, seed=cfg.seed)
    target_poses = sample_auxiliary_poses(dpc_cfg, device)

    with torch.no_grad():
        clean_state = build_state(image.detach())
        # snapshot clean means/ids/K so the cached scene never changes.
        clean_state = GeometryState(
            means=clean_state.means.detach(),
            primitive_ids=list(clean_state.primitive_ids),
            K_src=clean_state.K_src,
        )

    g = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)
    if cfg.epsilon > 0:
        init = (torch.rand(image.shape, generator=g).to(device) * 2 - 1) * cfg.epsilon
        delta = ((image + init).clamp(cfg.clamp_min, cfg.clamp_max) - image).detach()
    else:
        delta = torch.zeros_like(image)
    delta.requires_grad_(True)
    grad_ema = torch.zeros_like(image)
    history: list[dict] = []

    for step in range(cfg.steps):
        adv = (image + delta).clamp(cfg.clamp_min, cfg.clamp_max)
        adv_state = build_state(adv)
        loss, comps = geometry_soft_objective(
            adv_state,
            clean_state,
            source_adv=adv,
            source_clean=image.detach(),
            target_poses=target_poses,
            lambda_src=cfg.lambda_src,
            mode=cfg.mode,
            w_primitive=cfg.w_primitive,
            w_depth=cfg.w_depth,
            w_ordering=cfg.w_ordering,
            order_margin_rel=cfg.order_margin_rel,
        )
        grad = torch.autograd.grad(loss, delta)[0]
        grad_ema = cfg.momentum * grad_ema + grad / (grad.abs().mean() + 1e-12)
        delta = delta.detach() - cfg.step_size * grad_ema.sign()
        delta = delta.clamp(-cfg.epsilon, cfg.epsilon)
        delta = ((image + delta).clamp(cfg.clamp_min, cfg.clamp_max) - image).detach()
        delta.requires_grad_(True)

        history.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "source_cost": float(comps["source_cost"].detach()),
                "primitive": float(comps["primitive"].detach()),
                "depth": float(comps["depth"].detach()),
                "ordering": float(comps["ordering"].detach()),
            }
        )

    attacked = (image + delta).clamp(cfg.clamp_min, cfg.clamp_max).detach()
    info = {
        "history": history,
        "linf": float((attacked - image).abs().max()),
        "mode": cfg.mode,
        "n_target_poses": len(target_poses),
    }
    return attacked, (attacked - image).detach(), info
