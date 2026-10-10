"""
Method-agnostic visibility masks via forward-warp disocclusion.
Reused by the Representation Oracle and Paper A region-separated evaluation.

Definitions (charter §8: masks must be method-agnostic):
- Forward-warp the SOURCE depth+grid into the TARGET view (splat with z-buffer).
- A target pixel is:
    VISIBLE   if it receives a valid warped source pixel (front-most),
    OCCLUDED  if it lies inside the warped-source frustum coverage region but has NO source pixel
              (a disocclusion hole behind foreground) — i.e. in-frame gap,
    OOF       if it lies outside the region any source pixel could cover (beyond source frustum).
These depend ONLY on source geometry + camera poses, NOT on any method's Gaussians. So they are
identical for Flash3D, the oracle, and Paper A — a fair, method-agnostic partition.
"""

import torch


def forward_warp_source_to_target(depth_src, K, T_src2tgt, H, W, device):
    """Forward-project source pixels into the target image; return a z-buffered coverage map.

    depth_src: (H,W) metric depth in source camera.
    K: (3,3) pixel intrinsics (shared src/tgt here).
    T_src2tgt: (4,4) source-cam -> target-cam transform (cam_T_cam(src->tgt)).
    Returns:
      covered: (H,W) bool — target pixels that received >=1 source pixel (front-most).
      warped_z: (H,W) float — z-buffer depth at each covered target pixel (inf elsewhere).
    """
    inv_K = torch.linalg.inv(K)
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], 0).reshape(3, -1)  # (3,HW)
    rays = inv_K @ pix
    pts_src = rays * depth_src.reshape(1, -1)  # (3,HW) source cam
    N = pts_src.shape[1]
    homo = torch.cat([pts_src, torch.ones(1, N, device=device)], 0)  # (4,HW)
    pts_tgt = (T_src2tgt @ homo)[:3]  # (3,HW) target cam
    z = pts_tgt[2].clamp(min=1e-6)
    u = (pts_tgt[0] / z * K[0, 0] + K[0, 2]).round().long()
    v = (pts_tgt[1] / z * K[1, 1] + K[1, 2]).round().long()
    valid = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    warped_z = torch.full((H * W,), float("inf"), device=device)
    covered = torch.zeros(H * W, dtype=torch.bool, device=device)
    lin = v[valid] * W + u[valid]
    zv = z[valid]
    # z-buffer: scatter-min the depth per target pixel
    order = torch.argsort(zv, descending=True)  # far-to-near so nearest overwrites
    lin_o = lin[order]
    zv_o = zv[order]
    warped_z[lin_o] = zv_o
    covered[lin_o] = True
    return covered.reshape(H, W), warped_z.reshape(H, W)


def source_frustum_coverage_mask(depth_src, K, T_src2tgt, H, W, device, dilate=2):
    """Region of the target image that ANY source pixel could plausibly cover (in-frustum),
    approximated by the convex extent of warped source pixels, dilated slightly.
    Target pixels outside this region are OOF (beyond source frustum)."""
    covered, _ = forward_warp_source_to_target(depth_src, K, T_src2tgt, H, W, device)
    # morphological close: dilate the covered region to fill thin disocclusion cracks,
    # then anything still uncovered but inside the covered bounding hull is OCCLUDED,
    # outside is OOF. We approximate the hull by row/col min-max of covered pixels.
    cov = covered.clone()
    if dilate > 0:
        c = cov.float().unsqueeze(0).unsqueeze(0)
        k = 2 * dilate + 1
        c = torch.nn.functional.max_pool2d(c, k, stride=1, padding=dilate)
        cov_dil = c.squeeze() > 0.5
    else:
        cov_dil = cov
    return covered, cov_dil


def visibility_partition(depth_src, K, T_src2tgt, H, W, device, dilate=2):
    """Return (visible, occluded, oof) boolean (H,W) masks, mutually exclusive & exhaustive.
    visible : target pixel received a front-most source pixel (after small dilation to close cracks).
    occluded: not visible, but inside the source-coverage hull (disocclusion hole).
    oof     : not visible, outside the source-coverage hull (beyond source frustum)."""
    covered, cov_dil = source_frustum_coverage_mask(
        depth_src, K, T_src2tgt, H, W, device, dilate
    )
    visible = cov_dil
    # coverage hull: bounding region of covered pixels (per-row and per-col spans)
    hull = _coverage_hull(covered, H, W, device)
    occluded = (~visible) & hull
    oof = (~visible) & (~hull)
    return visible, occluded, oof


def _coverage_hull(covered, H, W, device):
    """Approximate convex-ish hull: a pixel is inside the hull if there are covered pixels
    both to its left AND right in its row, AND above AND below in its column."""
    cov = covered
    # row spans
    col_idx = torch.arange(W, device=device)
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
    row_idx = torch.arange(H, device=device)
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
    uu, vv = torch.meshgrid(col_idx, row_idx, indexing="xy")  # (H,W)
    in_row = (
        (uu >= left.unsqueeze(1)) & (uu <= right.unsqueeze(1)) & row_has.unsqueeze(1)
    )
    in_col = (vv >= top.unsqueeze(0)) & (vv <= bot.unsqueeze(0))
    return in_row & in_col
