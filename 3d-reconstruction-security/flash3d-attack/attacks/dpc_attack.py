"""Depth-Parallax Confusion (DPC) attack for feed-forward single-image 3DGS.

Key insight
-----------
Feed-forward single-image 3DGS (Flash3D, pixelSplat, Splatter Image, ...) is trained
with novel-view photometric loss. For one source image there exist many (depth, appearance)
explanations that render identically in the SOURCE view; they only diverge under parallax
in NOVEL views. This depth-appearance ambiguity is a structural blind spot.

DPC exploits it with a source-camouflaged, self-referential objective:

    min_delta   lambda_src * || R_src(x+delta) - R_src(x) ||^2      (camouflage source view)
              -  sum_j       || R_{P_j}(x+delta) - R_{P_j}(x) ||^2   (diverge novel views)

subject to || delta ||_inf <= eps.

Differences vs PGD / AdvSplat:
  * Opposite-sign dual objective (preserve source, destroy novel) rather than maximizing a
    single task loss.
  * Self-referential target (attacked-vs-clean self-renders); needs no ground-truth novel view.
  * Pose-agnostic: optimized on attacker-sampled auxiliary poses, evaluated on unseen poses.

This module is model-agnostic through a small `RenderFn` protocol so the same attack transfers
to any feed-forward 3DGS that can render a source view and a posed novel view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass
class DPCConfig:
    epsilon: float = 8.0 / 255.0
    steps: int = 40
    step_size: float = 1.0 / 255.0
    lambda_src: float = 10.0
    n_aux_poses: int = 4
    aux_yaw_range_deg: float = 8.0
    aux_pitch_range_deg: float = 4.0
    aux_trans: float = 0.10
    seed: int = 0
    momentum: float = 0.9
    clamp_min: float = 0.0
    clamp_max: float = 1.0


AuxPose = torch.Tensor  # (4,4) cam_T_cam relative pose
RenderFn = Callable[[torch.Tensor, AuxPose], torch.Tensor]
"""RenderFn(image, relative_pose) -> rendered RGB (B,3,H,W).

`relative_pose` is a 4x4 target-from-source camera transform. The identity pose must
render the source view. Implementations wrap a frozen 3DGS model.
"""


def sample_auxiliary_poses(cfg: DPCConfig, device: torch.device) -> list[AuxPose]:
    """Deterministically sample small rigid novel poses (attacker-chosen, not eval poses)."""
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    poses: list[AuxPose] = []
    for _ in range(cfg.n_aux_poses):
        yaw = (torch.rand(1, generator=g).item() * 2 - 1) * cfg.aux_yaw_range_deg
        pitch = (torch.rand(1, generator=g).item() * 2 - 1) * cfg.aux_pitch_range_deg
        tx = (torch.rand(1, generator=g).item() * 2 - 1) * cfg.aux_trans
        tz = (torch.rand(1, generator=g).item() * 2 - 1) * cfg.aux_trans
        ry = torch.deg2rad(torch.tensor(yaw))
        rp = torch.deg2rad(torch.tensor(pitch))
        cy, sy = torch.cos(ry), torch.sin(ry)
        cp, sp = torch.cos(rp), torch.sin(rp)
        R_yaw = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        R_pitch = torch.tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        R = R_pitch @ R_yaw
        T = torch.eye(4)
        T[:3, :3] = R
        T[0, 3] = tx
        T[2, 3] = tz
        poses.append(T.to(device))
    return poses


def dpc_objective(
    attacked_src: torch.Tensor,
    clean_src: torch.Tensor,
    attacked_novel: list[torch.Tensor],
    clean_novel: list[torch.Tensor],
    lambda_src: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the source-camouflaged DPC objective shared by all attacks."""
    if not attacked_novel:
        raise ValueError("DPC objective requires at least one novel view")
    if len(attacked_novel) != len(clean_novel):
        raise ValueError("Attacked and clean novel views must have the same length")

    source_cost = (attacked_src - clean_src).pow(2).mean()
    novel_divergence = torch.stack(
        [
            (attacked - clean).pow(2).mean()
            for attacked, clean in zip(attacked_novel, clean_novel)
        ]
    ).mean()
    loss = lambda_src * source_cost - novel_divergence
    return loss, source_cost, novel_divergence


def dpc_attack(
    image: torch.Tensor,
    render_fn: RenderFn,
    cfg: DPCConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Run the DPC attack.

    Args:
        image: clean source image (B,3,H,W) in [0,1].
        render_fn: frozen-model renderer, render_fn(image, relative_pose)->(B,3,H,W).
        cfg: attack configuration.

    Returns:
        (attacked_image, delta, info) where delta is L-inf bounded by cfg.epsilon.
    """
    if image.ndim != 4:
        raise ValueError(f"Expected BCHW image, got {tuple(image.shape)}")
    device = image.device
    identity = torch.eye(4, device=device)
    aux_poses = sample_auxiliary_poses(cfg, device)

    with torch.no_grad():
        clean_src = render_fn(image, identity).detach()
        clean_novel = [render_fn(image, P).detach() for P in aux_poses]

    # Random start inside the L-inf ball. This is important: the novel-divergence term is a
    # squared residual whose gradient vanishes exactly at delta=0, so a nonzero start is
    # required to escape that saddle (standard for PGD-style attacks too).
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
        src_adv = render_fn(adv, identity)
        novel_adv = [render_fn(adv, P) for P in aux_poses]
        loss, src_cost, novel_div = dpc_objective(
            src_adv,
            clean_src,
            novel_adv,
            clean_novel,
            lambda_src=cfg.lambda_src,
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
                "src_cost": float(src_cost.detach()),
                "novel_div": float(novel_div.detach()),
                "loss": float(loss.detach()),
            }
        )

    attacked = (image + delta).clamp(cfg.clamp_min, cfg.clamp_max).detach()
    info = {
        "history": history,
        "final_src_cost": history[-1]["src_cost"] if history else None,
        "final_novel_div": history[-1]["novel_div"] if history else None,
        "linf": float(delta.detach().abs().max()),
    }
    return attacked, (attacked - image).detach(), info


def random_delta_baseline(
    image: torch.Tensor, epsilon: float, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Non-optimized random uniform noise baseline at the same L-inf budget."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    delta = (torch.rand(image.shape, generator=g).to(image.device) * 2 - 1) * epsilon
    attacked = (image + delta).clamp(0.0, 1.0)
    delta_actual = attacked - image
    return attacked, delta_actual, {"linf": float(delta_actual.abs().max())}


def naive_pgd_attack(
    image: torch.Tensor,
    render_fn: RenderFn,
    cfg: DPCConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Standard PGD that maximizes novel-view divergence WITHOUT source camouflage.

    This is the natural ablation: same gradient-based optimization budget, same epsilon,
    but no source-preservation term (lambda_src=0). Proves that DPC's camouflage is a
    genuine contribution, not just a weaker version of unconstrained PGD.
    """
    ablation_cfg = DPCConfig(
        epsilon=cfg.epsilon,
        steps=cfg.steps,
        step_size=cfg.step_size,
        lambda_src=0.0,
        n_aux_poses=cfg.n_aux_poses,
        aux_yaw_range_deg=cfg.aux_yaw_range_deg,
        aux_pitch_range_deg=cfg.aux_pitch_range_deg,
        aux_trans=cfg.aux_trans,
        seed=cfg.seed,
        momentum=cfg.momentum,
        clamp_min=cfg.clamp_min,
        clamp_max=cfg.clamp_max,
    )
    return dpc_attack(image, render_fn, ablation_cfg)
