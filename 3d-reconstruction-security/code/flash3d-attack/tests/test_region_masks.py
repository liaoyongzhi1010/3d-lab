"""Unit tests for the method-agnostic region partition (attacks/region_masks.py).

Run on server (Flash3D venv) or any torch env:
  cd /root/flash3d-attack && python -m pytest tests/test_region_masks.py -q
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attacks.region_masks import region_partition, masked_psnr, forward_warp_coverage


def _K(H, W, f=1.0):
    k = torch.eye(3)
    k[0, 0] = f * W
    k[1, 1] = f * H
    k[0, 2] = W / 2.0
    k[1, 2] = H / 2.0
    return k


def test_partition_is_exclusive_and_exhaustive():
    H = W = 32
    device = torch.device("cpu")
    depth = torch.ones(H, W) * 2.0
    K = _K(H, W)
    T = torch.eye(4)
    T[0, 3] = 0.3  # sideways translation -> parallax
    vis, occ, beyond = region_partition(depth, K, T, H, W, device)
    # mutually exclusive
    assert not (vis & occ).any()
    assert not (vis & beyond).any()
    assert not (occ & beyond).any()
    # exhaustive
    assert bool((vis | occ | beyond).all())


def test_identity_pose_mostly_visible():
    H = W = 32
    device = torch.device("cpu")
    depth = torch.ones(H, W) * 2.0
    K = _K(H, W)
    T = torch.eye(4)  # no motion -> every pixel maps to itself
    vis, occ, beyond = region_partition(depth, K, T, H, W, device)
    assert float(vis.float().mean()) > 0.9


def test_translation_creates_beyond_region():
    H = W = 48
    device = torch.device("cpu")
    depth = torch.ones(H, W) * 1.5
    K = _K(H, W)
    T = torch.eye(4)
    T[0, 3] = 0.6  # large sideways shift pushes content off one side
    vis, occ, beyond = region_partition(depth, K, T, H, W, device)
    # a strong translation must leave some non-visible (occluded or beyond) area
    assert float((occ | beyond).float().mean()) > 0.05


def test_masked_psnr_small_region_returns_none():
    pred = torch.rand(3, 16, 16)
    gt = torch.rand(3, 16, 16)
    mask = torch.zeros(16, 16, dtype=torch.bool)
    mask[0, 0] = True  # only 1 pixel -> below min_px
    assert masked_psnr(pred, gt, mask) is None


def test_masked_psnr_identity_is_high():
    img = torch.rand(3, 16, 16)
    mask = torch.ones(16, 16, dtype=torch.bool)
    val = masked_psnr(img, img, mask)
    assert val is not None and val > 60.0


def test_coverage_shrinks_with_larger_translation():
    H = W = 40
    device = torch.device("cpu")
    depth = torch.ones(H, W) * 1.2
    K = _K(H, W)
    T_small = torch.eye(4)
    T_small[0, 3] = 0.1
    T_big = torch.eye(4)
    T_big[0, 3] = 0.7
    cov_small = forward_warp_coverage(depth, K, T_small, H, W, device).float().mean()
    cov_big = forward_warp_coverage(depth, K, T_big, H, W, device).float().mean()
    assert float(cov_big) <= float(cov_small)
