"""Budget-aware query-only DPC using NES over low-frequency DCT coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from attacks.blackbox_api import CountingRenderAPI
from attacks.dpc_attack import (
    DPCConfig,
    RenderFn,
    dpc_objective,
    sample_auxiliary_poses,
)


@dataclass
class BlackBoxDPCConfig:
    epsilon: float = 8.0 / 255.0
    iters: int = 60
    n_freq: int = 8
    nes_samples: int = 20
    nes_sigma: float = 0.01
    lr: float = 0.05
    lambda_src: float = 3.0
    n_aux_poses: int = 4
    aux_yaw_range_deg: float = 8.0
    aux_pitch_range_deg: float = 4.0
    aux_trans: float = 0.10
    seed: int = 0
    checkpoints: tuple[int, ...] = (1000, 5000, 12300)


def _dct_basis(n_pix: int, n_freq: int, device, dtype) -> torch.Tensor:
    """1D DCT-II basis matrix (n_freq x n_pix)."""
    x = torch.arange(n_pix, device=device, dtype=dtype)
    k = torch.arange(n_freq, device=device, dtype=dtype).unsqueeze(1)
    return torch.cos((torch.pi / n_pix) * (x + 0.5) * k)


def coeffs_to_delta(
    z: torch.Tensor,
    h: int,
    w: int,
    epsilon: float,
    basis_h: torch.Tensor,
    basis_w: torch.Tensor,
) -> torch.Tensor:
    """Map DCT coefficients to an L-infinity-bounded pixel perturbation."""
    spatial = torch.einsum("fh,cfg,gw->chw", basis_h, z, basis_w)
    return (torch.tanh(spatial) * epsilon).unsqueeze(0)


def prepare_clean_references(
    api: CountingRenderAPI,
    image: torch.Tensor,
    aux_poses: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    return api.clean_references(image, aux_poses)


def score_candidate(
    api: CountingRenderAPI,
    candidate: torch.Tensor,
    aux_poses: list[torch.Tensor],
    clean_src: torch.Tensor,
    clean_novel: list[torch.Tensor],
    *,
    lambda_src: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score one candidate; cost is exactly source plus every novel pose."""
    attacked_src = api.render(
        candidate,
        torch.eye(4, device=candidate.device),
        phase="attack",
        purpose="candidate_source",
    )
    attacked_novel = [
        api.render(candidate, pose, phase="attack", purpose="candidate_novel")
        for pose in aux_poses
    ]
    return dpc_objective(
        attacked_src,
        clean_src,
        attacked_novel,
        clean_novel,
        lambda_src=lambda_src,
    )


def blackbox_dpc_attack(
    image: torch.Tensor,
    render_fn: RenderFn | CountingRenderAPI,
    cfg: BlackBoxDPCConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Run DCT-NES without ever exceeding the render API's query budget."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected BCHW image with B=1, got {tuple(image.shape)}")
    api = (
        render_fn
        if isinstance(render_fn, CountingRenderAPI)
        else CountingRenderAPI(render_fn)
    )
    _, c, h, w = image.shape
    device, dtype = image.device, image.dtype
    pose_cfg = DPCConfig(
        n_aux_poses=cfg.n_aux_poses,
        aux_yaw_range_deg=cfg.aux_yaw_range_deg,
        aux_pitch_range_deg=cfg.aux_pitch_range_deg,
        aux_trans=cfg.aux_trans,
        seed=cfg.seed,
    )
    aux_poses = sample_auxiliary_poses(pose_cfg, device)
    score_cost = 1 + len(aux_poses)
    if not api.can_afford(2 * score_cost):
        raise ValueError("Budget must cover clean references and one candidate score")

    basis_h = _dct_basis(h, cfg.n_freq, device, dtype)
    basis_w = _dct_basis(w, cfg.n_freq, device, dtype)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 5)

    with torch.no_grad():
        clean_src, clean_novel = prepare_clean_references(api, image, aux_poses)
        z = torch.zeros(c, cfg.n_freq, cfg.n_freq, device=device, dtype=dtype)

        def candidate(current_z: torch.Tensor) -> torch.Tensor:
            delta = coeffs_to_delta(current_z, h, w, cfg.epsilon, basis_h, basis_w)
            return (image + delta).clamp(0.0, 1.0)

        best_image = candidate(z)
        best_loss, best_source, best_novel = score_candidate(
            api,
            best_image,
            aux_poses,
            clean_src,
            clean_novel,
            lambda_src=cfg.lambda_src,
        )
        history = [
            {
                "iter": -1,
                "loss": float(best_loss),
                "src_cost": float(best_source),
                "novel_div": float(best_novel),
                "queries": api.queries,
            }
        ]
        checkpoints = {
            budget: None
            for budget in cfg.checkpoints
            if api.budget is None or budget <= api.budget
        }
        stop_reason = "iterations_complete"

        for iteration in range(cfg.iters):
            iteration_cost = (2 * cfg.nes_samples + 1) * score_cost
            if not api.can_afford(iteration_cost):
                stop_reason = "budget_exhausted"
                break
            noise = torch.randn(
                cfg.nes_samples, c, cfg.n_freq, cfg.n_freq, generator=generator
            ).to(device=device, dtype=dtype)
            grad_est = torch.zeros_like(z)
            for sample in noise:
                plus, _, _ = score_candidate(
                    api,
                    candidate(z + cfg.nes_sigma * sample),
                    aux_poses,
                    clean_src,
                    clean_novel,
                    lambda_src=cfg.lambda_src,
                )
                minus, _, _ = score_candidate(
                    api,
                    candidate(z - cfg.nes_sigma * sample),
                    aux_poses,
                    clean_src,
                    clean_novel,
                    lambda_src=cfg.lambda_src,
                )
                grad_est += (plus - minus) * sample
            grad_est /= 2 * cfg.nes_samples * cfg.nes_sigma
            proposal_z = z - cfg.lr * grad_est
            proposal_image = candidate(proposal_z)
            loss, source_cost, novel_divergence = score_candidate(
                api,
                proposal_image,
                aux_poses,
                clean_src,
                clean_novel,
                lambda_src=cfg.lambda_src,
            )
            if loss <= best_loss:
                z = proposal_z
                best_image = proposal_image
                best_loss = loss
                best_source = source_cost
                best_novel = novel_divergence
            history.append(
                {
                    "iter": iteration,
                    "loss": float(best_loss),
                    "src_cost": float(best_source),
                    "novel_div": float(best_novel),
                    "queries": api.queries,
                }
            )
            for budget in checkpoints:
                if checkpoints[budget] is None and api.queries == budget:
                    checkpoints[budget] = dict(history[-1])

    delta = (best_image - image).detach()
    return (
        best_image.detach(),
        delta,
        {
            "history": history,
            "linf": float(delta.abs().max()),
            "queries": api.queries,
            "stop_reason": stop_reason,
            "checkpoints": checkpoints,
        },
    )
