"""Target-isolated transfer attack crafting and evaluation foundations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from attacks.blackbox_api import CountingRenderAPI
from attacks.dpc_attack import DPCConfig, dpc_attack, random_delta_baseline
from attacks.dpc_blackbox import BlackBoxDPCConfig, blackbox_dpc_attack


@dataclass(frozen=True)
class TransferArtifact:
    clean: torch.Tensor
    attacked: torch.Tensor
    method: str
    epsilon: float
    metadata: dict


def craft_transfer_attack(
    image: torch.Tensor,
    surrogate_render_fn,
    *,
    method: str,
    epsilon: float,
    seed: int = 0,
    config=None,
    second_surrogate_render_fn=None,
) -> TransferArtifact:
    """Craft using surrogates only; target models cannot enter this interface."""
    if method == "random":
        attacked, _, info = random_delta_baseline(image, epsilon, seed)
    elif method == "dpc":
        cfg = config or DPCConfig(epsilon=epsilon, seed=seed)
        attacked, _, info = dpc_attack(image, surrogate_render_fn, cfg)
    elif method == "dct_dpc":
        cfg = config or BlackBoxDPCConfig(epsilon=epsilon, seed=seed)
        attacked, _, info = blackbox_dpc_attack(image, surrogate_render_fn, cfg)
    elif method == "ensemble":
        if second_surrogate_render_fn is None:
            raise ValueError("ensemble requires two valid surrogates")
        first, _, first_info = dpc_attack(
            image, surrogate_render_fn, config or DPCConfig(epsilon=epsilon, seed=seed)
        )
        second, _, second_info = dpc_attack(
            image,
            second_surrogate_render_fn,
            config or DPCConfig(epsilon=epsilon, seed=seed + 1),
        )
        delta = ((first - image) + (second - image)) / 2
        attacked = (image + delta.clamp(-epsilon, epsilon)).clamp(0.0, 1.0)
        info = {"surrogates": [first_info, second_info]}
    else:
        raise ValueError(f"Unknown transfer method: {method}")
    return TransferArtifact(image.detach(), attacked.detach(), method, epsilon, info)


def evaluate_transfer_artifact(
    artifact: TransferArtifact,
    target_api: CountingRenderAPI,
    target_poses: list[torch.Tensor],
) -> dict:
    """Evaluate a fixed artifact; all target calls occur only in this function."""
    clean_renders = [
        target_api.render(
            artifact.clean, pose, phase="evaluation", purpose="target_clean"
        )
        for pose in target_poses
    ]
    attacked_renders = [
        target_api.render(
            artifact.attacked, pose, phase="evaluation", purpose="target_attacked"
        )
        for pose in target_poses
    ]
    divergences = [
        float((attacked - clean).pow(2).mean())
        for attacked, clean in zip(attacked_renders, clean_renders)
    ]
    return {
        "schema": "transfer",
        "method": artifact.method,
        "queries": target_api.queries,
        "per_pose_divergence": divergences,
        "mean_divergence": sum(divergences) / len(divergences) if divergences else None,
    }
