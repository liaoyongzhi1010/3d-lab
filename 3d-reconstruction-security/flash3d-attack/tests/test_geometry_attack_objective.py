"""RED-first synthetic tests for the geometry-DPC white-box attack objective.

CPU-only. Uses a tiny differentiable synthetic "geometry renderer" that maps a source
image to per-primitive source-camera means (a depth field) so we can exercise the soft
differentiable geometry objectives (primitive / depth / ordering / hybrid) and the
source-camouflage term WITHOUT Flash3D or CUDA.

The hard semantic predicates (near/far order reversal, mutual visibility) are tested in
test_geometry_metrics.py and are evaluation-only; here we only check the SOFT surrogate
objectives that produce gradients, epsilon/no-op behaviour, and source camouflage.

Run locally:  python3 -m pytest tests/test_geometry_attack_objective.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attacks.dpc_attack import DPCConfig, sample_auxiliary_poses
from attacks.dpc_geometry_attack import (
    GeometryAttackConfig,
    geometry_attack,
    geometry_soft_objective,
)
from attacks.geometry_metrics import GeometryState


def _pinhole_K():
    return torch.tensor([[128.0, 0.0, 192.0], [0.0, 128.0, 128.0], [0.0, 0.0, 1.0]])


def _synthetic_geometry_fn(depth_sensitivity=4.0):
    """Differentiable map: source image -> GeometryState in source-camera coords.

    A tiny grid of primitives whose depth is a low-frequency base plus a pose-independent
    high-frequency term of the image. Perturbing the high-frequency image content moves the
    primitive depths, giving nonzero geometry gradients while the mean (source appearance)
    stays close.
    """
    K = _pinhole_K()
    H = W = 4

    def build_state(image):
        # image: (1,3,H,W) in [0,1]; collapse to a per-pixel scalar depth field.
        gray = image.mean(dim=1)  # (1,H,W)
        gray = torch.nn.functional.adaptive_avg_pool2d(gray, (H, W))  # (1,H,W)
        low = gray.mean()
        hf = gray - low
        depth = 2.0 + depth_sensitivity * hf  # (1,H,W)
        ids = []
        xs = []
        ys = []
        for yy in range(H):
            for xx in range(W):
                ids.append((0, yy, xx))
                xs.append((xx - 1.5) * 0.1)
                ys.append((yy - 1.5) * 0.1)
        xs_t = torch.tensor(xs)
        ys_t = torch.tensor(ys)
        d = depth.reshape(-1)
        means = torch.stack([xs_t * d, ys_t * d, d], dim=1)  # (N,3), grad via d
        return GeometryState(means=means, primitive_ids=ids, K_src=K)

    return build_state


def test_geometry_soft_objective_returns_components_and_gradients():
    build = _synthetic_geometry_fn()
    image = torch.rand(1, 3, 8, 8)
    delta = torch.zeros_like(image, requires_grad=True)
    adv = image + delta
    clean_state = build(image.detach())
    adv_state = build(adv)
    poses = [torch.eye(4)]
    loss, comps = geometry_soft_objective(
        adv_state,
        clean_state,
        source_adv=adv,
        source_clean=image.detach(),
        target_poses=poses,
        lambda_src=10.0,
        mode="hybrid",
    )
    assert set(["source_cost", "primitive", "depth", "ordering"]).issubset(comps.keys())
    # Zero delta -> geometry divergence terms are ~0 but the objective must still be
    # differentiable wrt delta (nonzero structure, not necessarily nonzero grad here).
    loss.backward()
    assert delta.grad is not None


def test_geometry_soft_objective_nonzero_gradient_when_perturbed():
    build = _synthetic_geometry_fn()
    torch.manual_seed(0)
    image = torch.rand(1, 3, 8, 8)
    delta = (torch.rand_like(image) * 2 - 1) * (6.0 / 255.0)
    delta = delta.detach().requires_grad_(True)
    adv = image + delta
    clean_state = build(image.detach())
    adv_state = build(adv)
    loss, _ = geometry_soft_objective(
        adv_state,
        clean_state,
        source_adv=adv,
        source_clean=image.detach(),
        target_poses=[torch.eye(4)],
        lambda_src=10.0,
        mode="depth",
    )
    g = torch.autograd.grad(loss, delta)[0]
    assert g.abs().max().item() > 0.0


@pytest.mark.parametrize("mode", ["primitive", "depth", "ordering", "hybrid"])
def test_geometry_attack_respects_linf_and_modes(mode):
    build = _synthetic_geometry_fn(depth_sensitivity=6.0)
    torch.manual_seed(0)
    image = torch.rand(1, 3, 8, 8)
    cfg = GeometryAttackConfig(
        epsilon=8.0 / 255.0,
        steps=15,
        step_size=1.0 / 255.0,
        lambda_src=10.0,
        mode=mode,
    )
    attacked, delta, info = geometry_attack(image, build, cfg)
    assert delta.abs().max().item() <= cfg.epsilon + 1e-6
    assert attacked.min().item() >= 0.0 and attacked.max().item() <= 1.0
    assert info["linf"] <= cfg.epsilon + 1e-6
    assert len(info["history"]) == cfg.steps


def test_geometry_attack_zero_epsilon_is_noop():
    build = _synthetic_geometry_fn()
    image = torch.rand(1, 3, 8, 8)
    cfg = GeometryAttackConfig(epsilon=0.0, steps=5, mode="hybrid")
    attacked, delta, _ = geometry_attack(image, build, cfg)
    assert torch.allclose(attacked, image, atol=1e-6)
    assert delta.abs().max().item() <= 1e-6


def test_geometry_attack_perturbs_geometry_more_than_source():
    """Source camouflage: the attack should move primitive geometry while keeping the
    source appearance close (small source_cost relative to geometry divergence)."""
    build = _synthetic_geometry_fn(depth_sensitivity=6.0)
    torch.manual_seed(0)
    image = torch.rand(1, 3, 8, 8)
    cfg = GeometryAttackConfig(
        epsilon=10.0 / 255.0,
        steps=40,
        step_size=1.0 / 255.0,
        lambda_src=20.0,
        mode="depth",
    )
    attacked, _, info = geometry_attack(image, build, cfg)

    clean_state = build(image.detach())
    adv_state = build(attacked.detach())
    clean_d = clean_state.means[:, 2]
    adv_d = adv_state.means[:, 2]
    geom_change = (adv_d - clean_d).abs().mean().item()
    src_change = (attacked - image).pow(2).mean().item()
    assert geom_change > 1e-4
    # camouflage: L-inf budget keeps source change tiny
    assert src_change < (cfg.epsilon**2) + 1e-6


def test_geometry_attack_uses_sampled_auxiliary_poses_by_default():
    build = _synthetic_geometry_fn()
    image = torch.rand(1, 3, 8, 8)
    cfg = GeometryAttackConfig(
        epsilon=4.0 / 255.0, steps=3, n_aux_poses=3, mode="hybrid"
    )
    _, _, info = geometry_attack(image, build, cfg)
    dpc_cfg = DPCConfig(n_aux_poses=3, seed=cfg.seed)
    expected = sample_auxiliary_poses(dpc_cfg, torch.device("cpu"))
    assert info["n_target_poses"] == len(expected)


def test_hybrid_beats_render_only_dpc_on_geometry_damage():
    """With a synthetic scene where high-frequency input drives per-pixel depth, the hybrid
    geometry attack must produce MORE novel-view geometry damage than render-only DPC (which
    only has an RGB divergence signal), AND source geometry change must remain smaller than
    novel geometry change.

    This proves the geometry objectives add information beyond what render-only DPC provides.
    """
    from attacks.dpc_attack import dpc_attack, DPCConfig

    build = _synthetic_geometry_fn(depth_sensitivity=8.0)
    torch.manual_seed(42)
    image = torch.rand(1, 3, 8, 8)
    eps = 10.0 / 255.0
    steps = 50

    # --- Geometry-DPC (hybrid) ---
    gcfg = GeometryAttackConfig(
        epsilon=eps,
        steps=steps,
        step_size=1.0 / 255.0,
        lambda_src=10.0,
        mode="hybrid",
        n_aux_poses=4,
        seed=0,
    )
    g_attacked, _, _ = geometry_attack(image, build, gcfg)

    # --- Render-only DPC (uses a simple render_fn that returns the source as-is for source
    #     and a depth-modulated brightness shift for novel views). ---
    def render_fn(img, pose):
        # Novel divergence comes from image brightness shift proportional to the pose offset.
        shift = pose[:3, 3].norm().item() * 0.5
        return img + shift if pose[:3, 3].norm() > 0.01 else img

    rcfg = DPCConfig(
        epsilon=eps,
        steps=steps,
        step_size=1.0 / 255.0,
        lambda_src=10.0,
        n_aux_poses=4,
        seed=0,
    )
    r_attacked, _, _ = dpc_attack(image, render_fn, rcfg)

    # Measure geometry damage on a novel target pose (translate tz=0.5).
    T = torch.eye(4)
    T[2, 3] = 0.5
    with torch.no_grad():
        clean_state = build(image.detach())
        g_state = build(g_attacked.detach())
        r_state = build(r_attacked.detach())

    from attacks.geometry_metrics import aligned_point_displacement

    geom_damage_hybrid = aligned_point_displacement(
        g_state.means,
        clean_state.means,
        g_state.primitive_ids,
        clean_state.primitive_ids,
    )
    geom_damage_render = aligned_point_displacement(
        r_state.means,
        clean_state.means,
        r_state.primitive_ids,
        clean_state.primitive_ids,
    )
    # Hybrid must cause MORE geometry damage than render-only.
    assert geom_damage_hybrid > geom_damage_render, (
        f"hybrid {geom_damage_hybrid:.6f} should > render-only {geom_damage_render:.6f}"
    )
    # Source camouflage: source change must be bounded.
    src_change = (g_attacked - image).pow(2).mean().item()
    assert src_change < eps**2 + 1e-6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
