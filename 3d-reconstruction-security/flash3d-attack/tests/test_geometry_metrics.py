"""RED-first synthetic tests for correspondence-aware explicit-3D geometry metrics.

These tests run on CPU with tiny synthetic scenes (no Flash3D / no CUDA). They pin the
exact geometric conventions the white-box geometry attack depends on:

  * primitives live in SOURCE-camera coordinates and carry stable ids (layer, y, x);
  * projection to a target view uses a target-from-source rigid transform;
  * global depth scale is removed by median / least-squares alignment before measuring
    deformation (a pure global 2x scale must align to ~0 damage);
  * a raster depth buffer decides which primitive is visible at each target pixel;
  * correspondences are MUTUAL: a primitive counts only if visible in BOTH the clean and
    adversarial rasterizations at the same target pixel;
  * near/far order reversals require a clean separation margin and an adversarial reversal
    margin, same-direction motion is NOT a flip, disappeared pairs are excluded, and the
    denominator is the frozen clean-eligible pair count.

Run locally:  python3 -m pytest tests/test_geometry_metrics.py -q
Run on server: /root/projects/flash3d/.venv/bin/python -m pytest tests/test_geometry_metrics.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attacks.geometry_metrics import (
    GeometryState,
    OrderReversalConfig,
    align_scale_least_squares,
    align_scale_median,
    aligned_point_displacement,
    mutual_visibility_correspondences,
    order_reversal_rate,
    project_points,
    raster_depth_visibility,
    reprojection_displacement,
    transform_points,
)


def _pinhole_K(fx=128.0, fy=128.0, cx=192.0, cy=128.0):
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _identity_T():
    return torch.eye(4)


def _translate_T(tx=0.0, ty=0.0, tz=0.0):
    T = torch.eye(4)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


def test_project_points_maps_principal_axis_to_principal_point():
    K = _pinhole_K()
    # A point straight ahead on the optical axis projects to the principal point.
    pts = torch.tensor([[0.0, 0.0, 2.0]])
    pix, depth = project_points(pts, K)
    assert pix.shape == (1, 2)
    assert depth.shape == (1,)
    assert pix[0, 0].item() == pytest.approx(192.0, abs=1e-4)
    assert pix[0, 1].item() == pytest.approx(128.0, abs=1e-4)
    assert depth[0].item() == pytest.approx(2.0, abs=1e-6)


def test_project_points_offaxis_uses_fx_fy():
    K = _pinhole_K()
    pts = torch.tensor([[1.0, 0.5, 2.0]])
    pix, depth = project_points(pts, K)
    # u = fx * X/Z + cx = 128 * 0.5 + 192 = 256 ; v = fy * Y/Z + cy = 128 * 0.25 + 128 = 160
    assert pix[0, 0].item() == pytest.approx(256.0, abs=1e-4)
    assert pix[0, 1].item() == pytest.approx(160.0, abs=1e-4)


def test_transform_points_applies_target_from_source():
    pts = torch.tensor([[0.0, 0.0, 2.0], [1.0, -1.0, 3.0]])
    T = _translate_T(tx=0.5, tz=1.0)
    out = transform_points(pts, T)
    assert out.shape == (2, 3)
    assert torch.allclose(out[0], torch.tensor([0.5, 0.0, 3.0]), atol=1e-6)
    assert torch.allclose(out[1], torch.tensor([1.5, -1.0, 4.0]), atol=1e-6)


def test_align_scale_median_removes_global_scale():
    clean = torch.tensor([1.0, 2.0, 4.0, 8.0])
    adv = clean * 2.0
    s = align_scale_median(adv, clean)
    assert s == pytest.approx(2.0, rel=1e-6)
    # After dividing adv by s, aligned depths match clean.
    assert torch.allclose(adv / s, clean, atol=1e-6)


def test_align_scale_least_squares_removes_global_scale():
    clean = torch.tensor([1.0, 3.0, 5.0, 7.0])
    adv = clean * 0.5
    s = align_scale_least_squares(adv, clean)
    assert s == pytest.approx(0.5, rel=1e-6)


def test_aligned_point_displacement_zero_under_pure_global_scale():
    # A 2x global depth scale is not geometric damage: it must align to ~0.
    ids = [(0, 0, 0), (0, 0, 1), (0, 1, 0)]
    clean_pts = torch.tensor([[0.0, 0.0, 1.0], [0.5, 0.0, 2.0], [0.0, 0.5, 4.0]])
    adv_pts = clean_pts * 2.0
    disp = aligned_point_displacement(adv_pts, clean_pts, ids, ids)
    assert disp == pytest.approx(0.0, abs=1e-5)


def test_aligned_point_displacement_detects_local_deformation():
    ids = [(0, 0, 0), (0, 0, 1), (0, 1, 0)]
    clean_pts = torch.tensor([[0.0, 0.0, 1.0], [0.5, 0.0, 2.0], [0.0, 0.5, 4.0]])
    adv_pts = clean_pts.clone()
    adv_pts[1, 2] = 5.0  # bend one primitive only -> not removable by global scale
    disp = aligned_point_displacement(adv_pts, clean_pts, ids, ids)
    assert disp > 1e-3


def test_raster_depth_visibility_keeps_nearest_per_pixel():
    K = _pinhole_K()
    # Two primitives fall on the SAME pixel; only the nearer one is visible.
    near = torch.tensor([0.0, 0.0, 1.0])
    far = torch.tensor([0.0, 0.0, 3.0])
    pts = torch.stack([far, near])  # order shouldn't matter
    ids = [(0, 0, 0), (0, 0, 1)]
    vis = raster_depth_visibility(pts, K, (256, 384), ids)
    # visible set maps pixel -> id; the near primitive (id index 1) wins.
    assert (0, 0, 1) in vis.visible_ids
    assert (0, 0, 0) not in vis.visible_ids


def test_raster_depth_visibility_drops_out_of_frame_and_behind():
    K = _pinhole_K()
    behind = torch.tensor([0.0, 0.0, -1.0])
    out_of_frame = torch.tensor([100.0, 0.0, 1.0])
    inframe = torch.tensor([0.0, 0.0, 2.0])
    pts = torch.stack([behind, out_of_frame, inframe])
    ids = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
    vis = raster_depth_visibility(pts, K, (256, 384), ids)
    assert (0, 0, 0) not in vis.visible_ids
    assert (0, 0, 1) not in vis.visible_ids
    assert (0, 0, 2) in vis.visible_ids


def test_mutual_visibility_requires_both_clean_and_adv():
    K = _pinhole_K()
    ids = [(0, 0, 0), (0, 0, 1)]
    clean_pts = torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
    # In adv, primitive 1 goes behind the camera -> disappears; only 0 is mutual.
    adv_pts = torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, -2.0]])
    clean_vis = raster_depth_visibility(clean_pts, K, (256, 384), ids)
    adv_vis = raster_depth_visibility(adv_pts, K, (256, 384), ids)
    corr = mutual_visibility_correspondences(clean_vis, adv_vis)
    assert (0, 0, 0) in corr
    assert (0, 0, 1) not in corr


def test_reprojection_displacement_zero_when_static():
    K = _pinhole_K()
    ids = [(0, 0, 0), (0, 0, 1)]
    pts = torch.tensor([[0.3, 0.1, 2.0], [-0.4, 0.2, 3.0]])
    disp = reprojection_displacement(pts, pts, K, ids, ids)
    assert disp == pytest.approx(0.0, abs=1e-5)


def test_reprojection_displacement_positive_when_moved():
    K = _pinhole_K()
    ids = [(0, 0, 0)]
    clean = torch.tensor([[0.0, 0.0, 2.0]])
    adv = torch.tensor([[0.5, 0.0, 2.0]])
    disp = reprojection_displacement(adv, clean, K, ids, ids)
    assert disp > 1.0  # pixels


def _make_state(pts, ids, K):
    return GeometryState(
        means=torch.as_tensor(pts, dtype=torch.float32),
        primitive_ids=list(ids),
        K_src=K,
    )


def test_order_reversal_counts_true_near_far_flip():
    # Clean: primitive A is clearly nearer than B (margin >= 5% rel).
    # Adv:   B becomes clearly nearer than A (reversal >= 2% rel).
    K = _pinhole_K()
    ids = [(0, 0, 0), (0, 0, 1)]
    # Project both to the SAME target pixel region but keep them mutually visible by
    # placing them at distinct pixels (the pair is defined by ids, not shared pixel).
    clean_pts = torch.tensor([[0.0, 0.0, 1.0], [0.2, 0.0, 2.0]])
    adv_pts = torch.tensor([[0.0, 0.0, 2.5], [0.2, 0.0, 1.5]])
    clean_state = _make_state(clean_pts, ids, K)
    adv_state = _make_state(adv_pts, ids, K)
    T = _identity_T()
    cfg = OrderReversalConfig(clean_margin_rel=0.05, adv_reverse_rel=0.02)
    result = order_reversal_rate(clean_state, adv_state, T, cfg)
    assert result.n_eligible_pairs == 1
    assert result.n_reversed == 1
    assert result.rate == pytest.approx(1.0, abs=1e-6)


def test_order_reversal_same_direction_is_not_a_flip():
    # Both primitives move farther but keep their relative order -> NOT a flip.
    K = _pinhole_K()
    ids = [(0, 0, 0), (0, 0, 1)]
    clean_pts = torch.tensor([[0.0, 0.0, 1.0], [0.2, 0.0, 2.0]])
    adv_pts = torch.tensor([[0.0, 0.0, 1.5], [0.2, 0.0, 3.0]])
    clean_state = _make_state(clean_pts, ids, K)
    adv_state = _make_state(adv_pts, ids, K)
    result = order_reversal_rate(
        clean_state, adv_state, _identity_T(), OrderReversalConfig()
    )
    assert result.n_eligible_pairs == 1
    assert result.n_reversed == 0
    assert result.rate == pytest.approx(0.0, abs=1e-6)


def test_order_reversal_excludes_pairs_without_clean_margin():
    # Nearly-equal clean depths -> pair is NOT eligible (no clean margin).
    K = _pinhole_K()
    ids = [(0, 0, 0), (0, 0, 1)]
    clean_pts = torch.tensor([[0.0, 0.0, 2.00], [0.2, 0.0, 2.001]])
    adv_pts = torch.tensor([[0.0, 0.0, 3.0], [0.2, 0.0, 1.0]])
    clean_state = _make_state(clean_pts, ids, K)
    adv_state = _make_state(adv_pts, ids, K)
    result = order_reversal_rate(
        clean_state, adv_state, _identity_T(), OrderReversalConfig()
    )
    assert result.n_eligible_pairs == 0
    # Empty eligible set is an explicit failure, not zero damage.
    assert result.rate is None


def test_order_reversal_excludes_disappeared_pairs_but_keeps_denominator():
    # Two eligible clean pairs; in adv, one primitive disappears (behind camera).
    # The pair involving it is excluded from the numerator but the ELIGIBLE denominator
    # stays frozen from the clean scene.
    K = _pinhole_K()
    ids = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
    clean_pts = torch.tensor([[0.0, 0.0, 1.0], [0.2, 0.0, 2.0], [-0.2, 0.0, 3.0]])
    # adv: reverse 0 vs 1 (a real flip); primitive 2 goes behind camera.
    adv_pts = torch.tensor([[0.0, 0.0, 2.5], [0.2, 0.0, 1.5], [-0.2, 0.0, -1.0]])
    clean_state = _make_state(clean_pts, ids, K)
    adv_state = _make_state(adv_pts, ids, K)
    result = order_reversal_rate(
        clean_state, adv_state, _identity_T(), OrderReversalConfig()
    )
    # Clean eligible pairs among {0,1,2}: all three C(3,2)=3 have clean margin >=5%.
    assert result.n_eligible_pairs == 3
    assert result.n_reversed == 1  # only 0-1 flips; pairs with 2 are disappeared
    assert result.rate == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_local_explosion_detects_radial_ratio_outside_band():
    """After global scale alignment, a primitive whose radial distance from the optical axis
    explodes (ratio > 2.0 or < 0.5 relative to clean) should count as a local explosion."""
    from attacks.geometry_metrics import local_explosion_rate

    K = _pinhole_K()
    ids = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)]
    # Clean: well-behaved radial distances. Primitive 0 is on-axis (excluded from denominator).
    clean_pts = torch.tensor(
        [
            [0.0, 0.0, 2.0],
            [0.5, 0.0, 2.0],
            [-0.3, 0.2, 3.0],
            [0.1, 0.4, 4.0],
        ]
    )
    # Adv: same global scale but primitive 1 has exploded radially (ratio > 2.0).
    adv_pts = clean_pts.clone()
    adv_pts[1, 0] = 1.5  # radial ratio = 1.5/0.5 = 3.0 -> outside [0.5, 2.0]
    result = local_explosion_rate(adv_pts, clean_pts, ids, ids)
    assert result["n_valid"] == 3  # primitive 0 on-axis excluded
    assert result["n_exploded"] == 1
    assert result["rate"] == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_local_explosion_pure_global_scale_no_explosion():
    """A pure 2x global scale should produce zero local explosions after alignment."""
    from attacks.geometry_metrics import local_explosion_rate

    ids = [(0, 0, 0), (0, 0, 1), (0, 1, 0)]
    clean_pts = torch.tensor([[0.3, 0.1, 2.0], [0.5, 0.0, 3.0], [-0.2, 0.4, 4.0]])
    adv_pts = clean_pts * 2.0
    result = local_explosion_rate(adv_pts, clean_pts, ids, ids)
    assert result["n_exploded"] == 0
    assert result["rate"] == pytest.approx(0.0, abs=1e-6)


def test_local_explosion_excludes_on_axis_primitives():
    """Primitives at the optical axis (radial ~0 in clean) should be excluded from the
    explosion denominator since ratios are ill-defined."""
    from attacks.geometry_metrics import local_explosion_rate

    ids = [(0, 0, 0), (0, 0, 1)]
    clean_pts = torch.tensor([[0.0, 0.0, 2.0], [0.5, 0.0, 2.0]])
    adv_pts = torch.tensor([[0.3, 0.0, 2.0], [0.5, 0.0, 2.0]])
    result = local_explosion_rate(adv_pts, clean_pts, ids, ids)
    # Only primitive 1 is valid (primitive 0 is on-axis); primitive 1 is not exploded.
    assert result["n_valid"] == 1
    assert result["n_exploded"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
