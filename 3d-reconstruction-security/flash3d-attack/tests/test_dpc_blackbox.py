"""Unit tests for the black-box DPC attack (synthetic renderer, no model)."""

import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attacks.blackbox_api import CountingRenderAPI
from attacks.dpc_attack import dpc_objective, sample_auxiliary_poses, DPCConfig
from attacks.dpc_blackbox import (
    BlackBoxDPCConfig,
    blackbox_dpc_attack,
    coeffs_to_delta,
    prepare_clean_references,
    score_candidate,
    _dct_basis,
)


def _synthetic_render_fn(depth_sensitivity=5.0):
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


def test_coeffs_to_delta_respects_linf_bound():
    c, h, w, nf = 3, 16, 24, 6
    basis_h = _dct_basis(h, nf, torch.device("cpu"), torch.float32)
    basis_w = _dct_basis(w, nf, torch.device("cpu"), torch.float32)
    z = torch.randn(c, nf, nf) * 100.0  # large -> saturates tanh
    eps = 8.0 / 255.0
    delta = coeffs_to_delta(z, h, w, eps, basis_h, basis_w)
    assert delta.shape == (1, c, h, w)
    assert delta.abs().max().item() <= eps + 1e-6


def test_clean_references_and_candidate_score_have_exact_cost_and_objective():
    image = torch.rand(1, 3, 8, 8)
    render_fn = _synthetic_render_fn(6.0)
    cfg = BlackBoxDPCConfig(n_aux_poses=4, iters=0)
    poses = sample_auxiliary_poses(
        DPCConfig(n_aux_poses=4, seed=cfg.seed), torch.device("cpu")
    )
    api = CountingRenderAPI(render_fn, budget=10)
    clean_src, clean_novel = prepare_clean_references(api, image, poses)
    assert api.queries == 5

    score = score_candidate(
        api, image, poses, clean_src, clean_novel, lambda_src=cfg.lambda_src
    )
    assert api.queries == 10
    attacked_src = render_fn(image, torch.eye(4))
    attacked_novel = [render_fn(image, pose) for pose in poses]
    expected = dpc_objective(
        attacked_src,
        clean_src,
        attacked_novel,
        clean_novel,
        lambda_src=cfg.lambda_src,
    )
    assert all(
        torch.allclose(actual, wanted) for actual, wanted in zip(score, expected)
    )


def test_blackbox_dpc_never_overshoots_partial_iteration_budget():
    image = torch.rand(1, 3, 8, 8)
    api = CountingRenderAPI(_synthetic_render_fn(6.0), budget=19)
    cfg = BlackBoxDPCConfig(
        epsilon=0.05, iters=10, n_freq=3, nes_samples=2, n_aux_poses=1
    )
    _, _, info = blackbox_dpc_attack(image, api, cfg)
    assert api.queries == info["queries"] == 14
    assert api.queries <= api.budget
    assert info["stop_reason"] == "budget_exhausted"
    assert all(item["queries"] <= api.budget for item in info["history"])


def test_blackbox_dpc_respects_budget_and_reduces_objective():
    torch.manual_seed(0)
    image = torch.rand(1, 3, 16, 24)
    cfg = BlackBoxDPCConfig(
        epsilon=10.0 / 255.0, iters=15, n_freq=5, nes_samples=8, nes_sigma=0.02, lr=0.1
    )
    attacked, delta, info = blackbox_dpc_attack(image, _synthetic_render_fn(6.0), cfg)
    assert delta.abs().max().item() <= cfg.epsilon + 1e-6
    assert attacked.min().item() >= 0.0 and attacked.max().item() <= 1.0
    # NES should reduce the objective from first to last iteration.
    assert info["history"][-1]["loss"] <= info["history"][0]["loss"] + 1e-6
    assert info["queries"] > 0


def test_blackbox_dpc_increases_novel_divergence():
    torch.manual_seed(0)
    image = torch.rand(1, 3, 16, 24)
    render_fn = _synthetic_render_fn(6.0)
    cfg = BlackBoxDPCConfig(
        epsilon=12.0 / 255.0, iters=25, n_freq=6, nes_samples=10, nes_sigma=0.02, lr=0.1
    )
    attacked, _, info = blackbox_dpc_attack(image, render_fn, cfg)
    assert info["history"][-1]["novel_div"] > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
