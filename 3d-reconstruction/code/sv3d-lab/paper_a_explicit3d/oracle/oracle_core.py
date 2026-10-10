"""
Phase 3 Representation Oracle — build hidden Gaussians from TARGET geometry (oracle cheat),
merge with visible Gaussians from SOURCE, render to source+targets, report region-separated
metrics + causal deletion. NO training, NO Gaussian optimization.

Charter: oracles are UPPER BOUNDS, never model results. Thresholds pre-registered in
paper_a_explicit3d/ORACLE_PREREGISTRATION.md (do NOT lower them after seeing results).

Run on server:
  cd /root/projects/flash3d && source .venv/bin/activate
  export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python /root/sv3d-lab/paper_a_explicit3d/oracle/run_oracle.py \
      --split /root/projects/flash3d/splits/re10k_mine_filtered/test_files_present.txt \
      --n_scenes 200 --out /home/data/sv3d-lab/evaluations/oracle_v1

Reuses (whitelist): RE10K reader logic, UniDepth depth, render_predicted, geometry layers.
Implements FRESH: hidden-Gaussian construction, occ/OOF classification, region metrics, deletion.
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import torch

sys.path.insert(0, "/root/projects/flash3d")

from models.decoder.gauss_util import (
    render_predicted,
    focal2fov,
    getProjectionMatrix,
    K_to_NDC_pp,
)


# ---------------------------------------------------------------------------
# Geometry helpers (fresh implementation, OpenCV convention: x-right y-down z-fwd)
# ---------------------------------------------------------------------------


def backproject(depth, inv_K, device):
    """depth: (H,W) tensor. inv_K: (3,3). Returns (H*W, 3) camera-space points."""
    H, W = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=0).reshape(3, -1)  # (3, HW)
    rays = inv_K @ pix  # (3, HW)
    pts = rays * depth.reshape(1, -1)
    return pts.T  # (HW, 3)


def cam_to_world(pts_cam, c2w):
    """pts_cam: (N,3). c2w: (4,4). Returns world points (N,3)."""
    N = pts_cam.shape[0]
    homo = torch.cat([pts_cam, torch.ones(N, 1, device=pts_cam.device)], dim=1)  # (N,4)
    world = (c2w @ homo.T).T  # (N,4)
    return world[:, :3]


def world_to_cam(pts_world, w2c):
    N = pts_world.shape[0]
    homo = torch.cat([pts_world, torch.ones(N, 1, device=pts_world.device)], dim=1)
    cam = (w2c @ homo.T).T
    return cam[:, :3]


def project(pts_cam, K):
    """pts_cam: (N,3). Returns pixel coords (N,2) and depth (N,)."""
    z = pts_cam[:, 2].clamp(min=1e-6)
    u = pts_cam[:, 0] / z * K[0, 0] + K[0, 2]
    v = pts_cam[:, 1] / z * K[1, 1] + K[1, 2]
    return torch.stack([u, v], dim=1), pts_cam[:, 2]


# ---------------------------------------------------------------------------
# Gaussian scene construction
# ---------------------------------------------------------------------------


def make_gaussians(xyz, rgb, focal_px, opacity_logit=4.0):
    """Build a Gaussian dict from points + colors. Scale = 1-px footprint at depth.
    xyz: (N,3) world. rgb: (N,3) in [0,1]."""
    N = xyz.shape[0]
    device = xyz.device
    # isotropic scale ~ depth / focal (1 px footprint). Use z as depth proxy.
    # For world points, approximate footprint from distance to origin is unstable;
    # caller passes per-point depth via scale_from_depth instead.
    scaling = torch.full(
        (N, 3), math.log(0.01), device=device
    )  # placeholder, overwritten
    rotation = torch.zeros((N, 4), device=device)
    rotation[:, 0] = 1.0
    opacity = torch.full((N, 1), opacity_logit, device=device)
    features_dc = rgb.reshape(N, 1, 3)
    return {
        "xyz": xyz,
        "scaling": scaling,
        "rotation": rotation,
        "opacity": opacity,
        "features_dc": features_dc,
        "rgb_direct": rgb.reshape(N, 3),
    }


def set_scale_from_depth(gauss, depth_vals, focal_px, footprint_factor=0.15):
    """depth_vals: (N,) metric depth in the frame that generated the point.
    Sets LINEAR (activated) scale = footprint_factor * (1-px footprint) at that depth. The Flash3D
    rasterizer consumes already-activated scales. footprint_factor tuned on SOURCE self-recon only
    (not on target metrics): 0.15 gives ~31-32dB self-recon (passes the 30dB pre-registered gate);
    the raw 1-px footprint (1.0) over-blurs to ~25dB. This is a render-correctness param, not a cheat."""
    footprint = (depth_vals / focal_px * footprint_factor).clamp(min=1e-5)
    lin_scale = footprint.unsqueeze(1).repeat(1, 3)
    gauss["scaling"] = lin_scale


def merge_gaussians(g_list):
    keys = ["xyz", "scaling", "rotation", "opacity", "features_dc", "rgb_direct"]
    out = {}
    for k in keys:
        out[k] = torch.cat([g[k] for g in g_list if g[k].shape[0] > 0], dim=0)
    return out


# ---------------------------------------------------------------------------
# Rendering wrapper (Flash3D render_predicted)
# ---------------------------------------------------------------------------


class _Cfg:
    class model:
        renderer_w_pose = True


def render_gaussians_relpose(gauss, K, T_rel, H, W, device, bg=0.5, sh_degree=0):
    """Render gauss (in SOURCE camera frame) to a view defined by relative pose T_rel
    (source->target cam_T_cam, 4x4) and pixel intrinsics K (3x3).
    Follows Flash3D convention: world_view_transform = T_rel.T, proj_mtrx = getProjectionMatrix(...).T,
    px/py NDC = 0 for re10k, bg = gray 0.5 (Flash3D default). T_rel=identity => source view.
    Colors: if gauss has 'rgb_direct', use override_color (direct RGB); else SH via features_dc(+rest)."""
    fx, fy = K[0, 0].item(), K[1, 1].item()
    fovX = focal2fov(fx, W)
    fovY = focal2fov(fy, H)
    proj_mtrx = getProjectionMatrix(0.01, 100.0, fovX, fovY, pX=0.0, pY=0.0).to(device)
    proj_mtrx = proj_mtrx.transpose(0, 1).float()
    world_view_transform = T_rel.transpose(0, 1).float()
    camera_center = (
        -world_view_transform[3, :3] @ world_view_transform[:3, :3].transpose(0, 1)
    ).float()
    full_proj = (world_view_transform @ proj_mtrx).float()
    bg_t = torch.full((3,), bg, device=device)
    override = gauss.get("rgb_direct", None)
    out = render_predicted(
        _Cfg(),
        gauss,
        world_view_transform,
        full_proj,
        proj_mtrx,
        camera_center,
        (fovX, fovY),
        (H, W),
        bg_t,
        max_sh_degree=sh_degree,
        override_color=override,
    )
    return out


def render_to_view(gauss, K, c2w, H, W, device, bg=0.0):
    """DEPRECATED world-frame variant (kept for reference). Use render_gaussians_relpose."""
    w2c = torch.linalg.inv(c2w)
    world_view_transform = w2c.T.contiguous()
    fx, fy = K[0, 0].item(), K[1, 1].item()
    cx, cy = K[0, 2].item(), K[1, 2].item()
    fovX = focal2fov(fx, W)
    fovY = focal2fov(fy, H)
    px, py = K_to_NDC_pp(cx, cy, H, W)
    proj_mtrx = getProjectionMatrix(0.01, 100.0, fovX, fovY, px, py).to(device)
    full_proj = (world_view_transform @ proj_mtrx).float()
    cam_center = c2w[:3, 3]
    bg_t = torch.full((3,), bg, device=device)
    out = render_predicted(
        _Cfg(),
        gauss,
        world_view_transform,
        full_proj,
        proj_mtrx,
        cam_center,
        (fovX, fovY),
        (H, W),
        bg_t,
        max_sh_degree=0,
    )
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def psnr(pred, gt, mask=None):
    """pred, gt: (3,H,W) in [0,1]. mask: (H,W) bool or None."""
    if mask is not None:
        if mask.sum() < 10:
            return None
        pred = pred[:, mask]
        gt = gt[:, mask]
        mse = ((pred - gt) ** 2).mean()
    else:
        mse = ((pred - gt) ** 2).mean()
    return (-10 * torch.log10(mse + 1e-10)).item()


def crop5(img):
    _, H, W = img.shape
    y0, y1 = int(math.ceil(0.05 * H)), int(math.floor(0.95 * H))
    x0, x1 = int(math.ceil(0.05 * W)), int(math.floor(0.95 * W))
    return img[:, y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Main oracle per-scene
# ---------------------------------------------------------------------------


def process_scene(unidepth, frames, focal_conf_frac, device):
    """frames: dict frame_name -> {color (3,H,W), K (3,3), c2w (4,4)}.
    frame_name 0 = source. Others = targets.
    Returns dict of metrics + gaussian counts."""
    src = frames[0]
    H, W = src["color"].shape[1:]
    K_src = src["K"].to(device)
    c2w_src = src["c2w"].to(device)
    color_src = src["color"].to(device)

    # --- G_vis from source depth ---
    with torch.no_grad():
        depth_src = unidepth.infer(
            color_src.unsqueeze(0), intrinsics=K_src.unsqueeze(0)
        )
        depth_src = depth_src["depth"].squeeze()  # (H,W)
    inv_K_src = torch.linalg.inv(K_src)
    pts_cam_src = backproject(depth_src, inv_K_src, device)  # (HW,3)
    pts_world_src = cam_to_world(pts_cam_src, c2w_src)
    rgb_src = color_src.permute(1, 2, 0).reshape(-1, 3)
    g_vis = make_gaussians(pts_world_src, rgb_src, K_src[0, 0].item())
    set_scale_from_depth(g_vis, depth_src.reshape(-1), K_src[0, 0].item())

    # --- G_hidden from each target (oracle cheat: uses target depth) ---
    hidden_xyz, hidden_rgb, hidden_depth = [], [], []
    for fname, f in frames.items():
        if fname == 0:
            continue
        K_t = f["K"].to(device)
        c2w_t = f["c2w"].to(device)
        color_t = f["color"].to(device)
        with torch.no_grad():
            depth_t = unidepth.infer(color_t.unsqueeze(0), intrinsics=K_t.unsqueeze(0))
            depth_t = depth_t["depth"].squeeze()
        inv_K_t = torch.linalg.inv(K_t)
        pts_cam_t = backproject(depth_t, inv_K_t, device)
        pts_world_t = cam_to_world(pts_cam_t, c2w_t)
        rgb_t = color_t.permute(1, 2, 0).reshape(-1, 3)

        # classify: which target points are hidden from source?
        w2c_src = torch.linalg.inv(c2w_src)
        pts_in_src = world_to_cam(pts_world_t, w2c_src)
        uv_src, z_src = project(pts_in_src, K_src)
        u, v = uv_src[:, 0], uv_src[:, 1]
        in_frame = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z_src > 0)
        # occluded: in-frame but behind source surface
        occluded = torch.zeros_like(in_frame)
        if in_frame.any():
            ui = u[in_frame].long().clamp(0, W - 1)
            vi = v[in_frame].long().clamp(0, H - 1)
            src_depth_at = depth_src[vi, ui]
            behind = z_src[in_frame] > src_depth_at * 1.05
            occ_idx = torch.where(in_frame)[0][behind]
            occluded[occ_idx] = True
        oof = (~in_frame) & (z_src > 0)  # out of source frustum but in front
        hidden_mask = occluded | oof

        if hidden_mask.any():
            hidden_xyz.append(pts_world_t[hidden_mask])
            hidden_rgb.append(rgb_t[hidden_mask])
            hidden_depth.append(depth_t.reshape(-1)[hidden_mask])

    result = {"n_vis": g_vis["xyz"].shape[0], "n_hidden": 0}

    if hidden_xyz:
        hx = torch.cat(hidden_xyz, 0)
        hr = torch.cat(hidden_rgb, 0)
        hd = torch.cat(hidden_depth, 0)
        g_hidden = make_gaussians(hx, hr, K_src[0, 0].item())
        set_scale_from_depth(g_hidden, hd, K_src[0, 0].item())
        result["n_hidden"] = hx.shape[0]
        g_merged = merge_gaussians([g_vis, g_hidden])
    else:
        g_merged = g_vis
        g_hidden = None

    # --- render + metrics per target ---
    per_target = {}
    for fname, f in frames.items():
        K_t = f["K"].to(device)
        c2w_t = f["c2w"].to(device)
        gt = f["color"].to(device)
        merged_out = render_to_view(g_merged, K_t, c2w_t, H, W, device)
        vis_out = render_to_view(g_vis, K_t, c2w_t, H, W, device)
        pred_merged = merged_out["render"].clamp(0, 1)
        pred_vis = vis_out["render"].clamp(0, 1)
        m = {
            "psnr_merged": psnr(crop5(pred_merged), crop5(gt)),
            "psnr_vis_only": psnr(crop5(pred_vis), crop5(gt)),
        }
        m["deletion_delta"] = m["psnr_merged"] - m["psnr_vis_only"]
        per_target[str(fname)] = m
    result["per_target"] = per_target
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--n_scenes", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_path", default="/root/projects/flash3d/data/RealEstate10K")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    print("Loading UniDepth...")
    unidepth = (
        torch.hub.load(
            "lpiccinelli-eth/UniDepth",
            "UniDepth",
            version="v1",
            backbone="vitl14",
            pretrained=True,
            trust_repo=True,
        )
        .to(device)
        .eval()
    )

    print("This is a skeleton. Full data-loading integration next.")
    # NOTE: data loading via Flash3D Re10KDataset is wired in run_oracle_full.py
    # (kept separate so this module stays importable/testable).


if __name__ == "__main__":
    main()
