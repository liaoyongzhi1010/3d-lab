"""Query-budgeted random and Square-style score baselines."""

from __future__ import annotations

from typing import Callable

import torch

from attacks.blackbox_api import CountingRenderAPI
from attacks.dpc_blackbox import prepare_clean_references, score_candidate


def _run_search(
    image: torch.Tensor,
    api: CountingRenderAPI,
    poses: list[torch.Tensor],
    epsilon: float,
    lambda_src: float,
    seed: int,
    propose: Callable[[torch.Tensor, int, torch.Generator], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    score_cost = 1 + len(poses)
    if not api.can_afford(2 * score_cost):
        raise ValueError("Budget must cover references and one candidate")
    clean_src, clean_novel = prepare_clean_references(api, image, poses)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best = image.clone()
    best_loss, _, _ = score_candidate(
        api, best, poses, clean_src, clean_novel, lambda_src=lambda_src
    )
    candidate_queries = [api.queries]
    best_query = api.queries
    iteration = 0
    while api.can_afford(score_cost):
        candidate = propose(best, iteration, generator)
        candidate = torch.max(
            torch.min(candidate, image + epsilon), image - epsilon
        ).clamp(0.0, 1.0)
        loss, _, _ = score_candidate(
            api, candidate, poses, clean_src, clean_novel, lambda_src=lambda_src
        )
        candidate_queries.append(api.queries)
        if loss < best_loss:
            best, best_loss, best_query = candidate, loss, api.queries
        iteration += 1
    delta = (best - image).detach()
    return (
        best.detach(),
        delta,
        {
            "queries": api.queries,
            "candidate_queries": candidate_queries,
            "best_query": best_query,
            "loss": float(best_loss),
        },
    )


def random_search_attack(
    image: torch.Tensor,
    api: CountingRenderAPI,
    poses: list[torch.Tensor],
    *,
    epsilon: float,
    lambda_src: float,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    def propose(_best, _iteration, generator):
        noise = torch.rand(image.shape, generator=generator).to(
            image.device, image.dtype
        )
        return image + (2 * noise - 1) * epsilon

    return _run_search(image, api, poses, epsilon, lambda_src, seed, propose)


def square_attack(
    image: torch.Tensor,
    api: CountingRenderAPI,
    poses: list[torch.Tensor],
    *,
    epsilon: float,
    lambda_src: float,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    _, _, height, width = image.shape

    def propose(best, iteration, generator):
        side = max(1, round(min(height, width) * (0.5 / (iteration + 1) ** 0.5)))
        top = int(torch.randint(height - side + 1, (1,), generator=generator).item())
        left = int(torch.randint(width - side + 1, (1,), generator=generator).item())
        sign = 1.0 if torch.rand((), generator=generator).item() >= 0.5 else -1.0
        candidate = best.clone()
        candidate[:, :, top : top + side, left : left + side] = (
            image[:, :, top : top + side, left : left + side] + sign * epsilon
        )
        return candidate

    return _run_search(image, api, poses, epsilon, lambda_src, seed, propose)
