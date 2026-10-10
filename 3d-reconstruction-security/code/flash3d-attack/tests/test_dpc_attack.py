"""Unit tests for the DPC attack core (model-agnostic, synthetic renderer).

Run on server (Flash3D venv) or any torch env:
  cd /root/flash3d-attack && python -m pytest tests/test_dpc_attack.py -q
"""

import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attacks.dpc_attack import (
    DPCConfig,
    dpc_attack,
    dpc_objective,
    naive_pgd_attack,
    random_delta_baseline,
    sample_auxiliary_poses,
)


def _synthetic_render_fn(depth_sensitivity=5.0):
    """A differentiable toy renderer with a depth-appearance ambiguity.

    Source view (identity pose) depends only on a low-frequency (blurred) version of the
    image, so high-frequency perturbations are invisible in the source view. Novel views
    (non-identity pose) add a pose-scaled high-frequency term, so the SAME high-frequency
    perturbation is amplified under parallax. This mirrors the real blind spot: source-view
    photometric consistency does not constrain parallax-only geometry.
    """

    def _low_freq(image):
        return image.mean(dim=(2, 3), keepdim=True).expand_as(image)

    def render_fn(image, pose):
        low = _low_freq(image)
        hf = image - low
        parallax = (
            float(pose[0, 3].item())
            + float(pose[2, 3].item())
            + float(pose[0, 2].item())
        )
        return low + depth_sensitivity * parallax * hf

    return render_fn


def test_sample_auxiliary_poses_deterministic_and_shaped():
    device = torch.device("cpu")
    cfg = DPCConfig(n_aux_poses=4, seed=123)
    a = sample_auxiliary_poses(cfg, device)
    b = sample_auxiliary_poses(cfg, device)
    assert len(a) == 4
    for pa, pb in zip(a, b):
        assert pa.shape == (4, 4)
        assert torch.equal(pa, pb)


def test_dpc_objective_matches_formula_and_preserves_gradients():
    attacked_src = torch.tensor([[[[2.0]]]], requires_grad=True)
    clean_src = torch.zeros_like(attacked_src)
    attacked_novel = [torch.tensor([[[[3.0]]]], requires_grad=True)]
    clean_novel = [torch.tensor([[[[1.0]]]])]

    loss, source_cost, novel_divergence = dpc_objective(
        attacked_src, clean_src, attacked_novel, clean_novel, lambda_src=2.0
    )

    assert source_cost.item() == pytest.approx(4.0)
    assert novel_divergence.item() == pytest.approx(4.0)
    assert loss.item() == pytest.approx(4.0)
    loss.backward()
    assert attacked_src.grad.item() == pytest.approx(8.0)
    assert attacked_novel[0].grad.item() == pytest.approx(-4.0)


def test_dpc_objective_requires_matching_nonempty_novel_views():
    value = torch.zeros(1, 3, 2, 2)
    with pytest.raises(ValueError, match="at least one"):
        dpc_objective(value, value, [], [], lambda_src=1.0)
    with pytest.raises(ValueError, match="same length"):
        dpc_objective(value, value, [value], [value, value], lambda_src=1.0)


def test_dpc_respects_linf_bound():
    torch.manual_seed(0)
    image = torch.rand(1, 3, 16, 24)
    cfg = DPCConfig(epsilon=6.0 / 255.0, steps=10, step_size=1.0 / 255.0)
    attacked, delta, info = dpc_attack(image, _synthetic_render_fn(), cfg)
    assert delta.abs().max().item() <= cfg.epsilon + 1e-6
    assert attacked.min().item() >= 0.0
    assert attacked.max().item() <= 1.0
    assert info["linf"] <= cfg.epsilon + 1e-6


def test_dpc_increases_novel_divergence_more_than_source():
    torch.manual_seed(0)
    image = torch.rand(1, 3, 16, 24)
    cfg = DPCConfig(
        epsilon=10.0 / 255.0, steps=60, step_size=1.0 / 255.0, lambda_src=20.0
    )
    render_fn = _synthetic_render_fn(depth_sensitivity=6.0)
    attacked, delta, info = dpc_attack(image, render_fn, cfg)

    identity = torch.eye(4)
    poses = sample_auxiliary_poses(cfg, torch.device("cpu"))

    src_change = (
        (render_fn(attacked, identity) - render_fn(image, identity))
        .pow(2)
        .mean()
        .item()
    )
    novel_change = sum(
        (render_fn(attacked, P) - render_fn(image, P)).pow(2).mean().item()
        for P in poses
    ) / len(poses)

    # The whole point: novel-view corruption dominates source-view change.
    assert novel_change > src_change
    assert novel_change > 1e-6


def test_dpc_zero_epsilon_is_noop():
    image = torch.rand(1, 3, 8, 8)
    cfg = DPCConfig(epsilon=0.0, steps=5)
    attacked, delta, _ = dpc_attack(image, _synthetic_render_fn(), cfg)
    assert torch.allclose(attacked, image, atol=1e-6)
    assert delta.abs().max().item() <= 1e-6


def test_random_delta_baseline_respects_budget_and_deterministic():
    image = torch.rand(1, 3, 12, 12)
    eps = 6.0 / 255.0
    a1, d1, info1 = random_delta_baseline(image, eps, seed=7)
    a2, d2, _ = random_delta_baseline(image, eps, seed=7)
    assert d1.abs().max().item() <= eps + 1e-6
    assert a1.min().item() >= 0.0 and a1.max().item() <= 1.0
    assert torch.equal(a1, a2)
    assert info1["linf"] <= eps + 1e-6


def test_naive_pgd_hurts_source_more_than_dpc():
    """Without camouflage, naive PGD should damage the SOURCE view more than DPC does."""
    torch.manual_seed(0)
    image = torch.rand(1, 3, 16, 24)
    render_fn = _synthetic_render_fn(depth_sensitivity=6.0)
    cfg = DPCConfig(
        epsilon=10.0 / 255.0, steps=60, step_size=1.0 / 255.0, lambda_src=20.0
    )

    dpc_att, _, _ = dpc_attack(image, render_fn, cfg)
    pgd_att, _, _ = naive_pgd_attack(image, render_fn, cfg)

    identity = torch.eye(4)
    dpc_src = (
        (render_fn(dpc_att, identity) - render_fn(image, identity)).pow(2).mean().item()
    )
    pgd_src = (
        (render_fn(pgd_att, identity) - render_fn(image, identity)).pow(2).mean().item()
    )

    assert pgd_src >= dpc_src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
