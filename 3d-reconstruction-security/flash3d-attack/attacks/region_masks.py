"""Method-agnostic visibility partition for region-separated attack analysis.

Given only the SOURCE depth and the source->target camera transform (no method-specific
Gaussians), forward-warp the source pixels into a target view and split the target image into:

    visible  : target pixel receives a front-most warped source pixel,
    occluded : not visible, but inside the source-coverage hull (disocclusion hole),
    beyond   : not visible and outside the coverage hull (beyond the source frustum, OOF).

Because the partition depends only on source geometry + poses, it is identical for any
reconstruction method and is a fair way to ask *where* DPC concentrates its damage. We expect a
geometry-level attack to hurt occluded / beyond-frustum regions (unobservable from the source)
more than the visible region.

Self-contained: depends only on torch. Not imported from any other project.
"""

from __future__ import annotations

import torch


def forward_warp_coverage(depth_src, K, T_src2tgt, H, W, device):
    """Forward-project source pixels into the target; return a z-buffered coverage mask.

    depth_src : (H,W) metric depth in the source camera.
    K         : (3,3) pixel intrinsics (shared source/target).
    T_src2tgt : (4,4) source-cam -> target-cam rigid transform.
    Returns covered (H,W) bool: target pixels that received >=1 front-most source pixel.
    """
    inv_K = torch.linalg.inv(K)
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], 0).reshape(3, -1)
    rays = inv_K @ pix
    pts_src = rays * depth_src.reshape(1, -1)
    n = pts_src.shape[1]
    homo = torch.cat([pts_src, torch.ones(1, n, device=device)], 0)
    pts_tgt = (T_src2tgt @ homo)[:3]
    z = pts_tgt[2].clamp(min=1e-6)
    u = (pts_tgt[0] / z * K[0, 0] + K[0, 2]).round().long()
    v = (pts_tgt[1] / z * K[1, 1] + K[1, 2]).round().long()
    valid = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    covered = torch.zeros(H * W, dtype=torch.bool, device=device)
    lin = v[valid] * W + u[valid]
    covered[lin] = True
    return covered.reshape(H, W)


def _dilate(mask, r):
    if r <= 0:
        return mask
    m = mask.float().unsqueeze(0).unsqueeze(0)
    k = 2 * r + 1
    m = torch.nn.functional.max_pool2d(m, k, stride=1, padding=r)
    return m.squeeze() > 0.5


def _coverage_hull(covered, H, W, device):
    """A pixel is inside the hull if covered pixels exist both left & right in its row AND
    above & below in its column (a cheap convex-ish extent)."""
    cov = covered
    col_idx = torch.arange(W, device=device)
    row_idx = torch.arange(H, device=device)
    row_has = cov.any(dim=1)
    left = (
        torch.where(
            cov, col_idx.unsqueeze(0), torch.full_like(cov, W, dtype=torch.long)
        )
        .min(dim=1)
        .values
    )
    right = (
        torch.where(
            cov, col_idx.unsqueeze(0), torch.full_like(cov, -1, dtype=torch.long)
        )
        .max(dim=1)
        .values
    )
    top = (
        torch.where(
            cov, row_idx.unsqueeze(1), torch.full_like(cov, H, dtype=torch.long)
        )
        .min(dim=0)
        .values
    )
    bot = (
        torch.where(
            cov, row_idx.unsqueeze(1), torch.full_like(cov, -1, dtype=torch.long)
        )
        .max(dim=0)
        .values
    )
    uu, vv = torch.meshgrid(col_idx, row_idx, indexing="xy")
    in_row = (
        (uu >= left.unsqueeze(1)) & (uu <= right.unsqueeze(1)) & row_has.unsqueeze(1)
    )
    in_col = (vv >= top.unsqueeze(0)) & (vv <= bot.unsqueeze(0))
    return in_row & in_col


def region_partition(depth_src, K, T_src2tgt, H, W, device, dilate=2):
    """Return (visible, occluded, beyond) boolean (H,W) masks: mutually exclusive & exhaustive."""
    covered = forward_warp_coverage(depth_src, K, T_src2tgt, H, W, device)
    visible = _dilate(covered, dilate)
    hull = _coverage_hull(covered, H, W, device)
    occluded = (~visible) & hull
    beyond = (~visible) & (~hull)
    return visible, occluded, beyond


def masked_psnr(pred, gt, mask, eps=1e-10, min_px=25):
    """PSNR over masked pixels. pred/gt: (3,H,W) or (1,3,H,W). mask: (H,W) bool.
    Returns None if the region is too small to be meaningful."""
    if pred.dim() == 4:
        pred = pred[0]
    if gt.dim() == 4:
        gt = gt[0]
    if mask.sum() < min_px:
        return None
    p = pred[:, mask]
    g = gt[:, mask]
    mse = (p.clamp(0, 1) - g.clamp(0, 1)).pow(2).mean()
    return float(-10.0 * torch.log10(mse + eps))
