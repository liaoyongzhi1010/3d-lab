"""Step S0 v2: Build HIGH-QUALITY canonical hidden-GS teacher pseudo-GT.

Fixes the v1 ceiling problem (median-depth heuristic gave teacher only +0.11dB in
disocc region -> useless as supervision). v2 predicts REAL per-neighbor depth via the
frozen UniDepth predictor and backprojects actual per-pixel geometry, then filters to
points hidden from the source view.

Pipeline per scene:
  1. Source depth (UniDepth on source frame)
  2. For each neighbor frame: predict its REAL depth (UniDepth), backproject its pixels
     to WORLD using known T_c2w_neighbor -> accurate 3D points
  3. Visibility test from source: keep points NOT visible from source (outside frustum
     or behind source surface) = hidden geometry
  4. Subsample + per-point color from neighbor image

Output per scene: {xyz:[M,3] world, color:[M,3], n_points}. S1 transforms world->source-cam.

Run (Flash3D venv):
  python -m paper_b_generative3d.build_teacher_hidden_gs_v2 \
    --max_scenes 620 --k_neighbors 3 \
    --out /home/data/sv3d-lab/reconstruction/teacher_hidden_gs_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from paper_b_generative3d.backbone_flash3d import Flash3DBackbone
from paper_b_generative3d.data_loader_re10k import build_flash3d_cfg, FLASH3D_CKPT


def backproject(depth, K, T_c2w, H, W):
    """depth:[H,W] -> world pts [HW,3], valid [HW]."""
    device = depth.device
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    xs = xs.float().flatten()
    ys = ys.float().flatten()
    zs = depth.flatten()
    valid = zs > 0.01
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs - cx) / fx * zs
    y_cam = (ys - cy) / fy * zs
    pts_cam = torch.stack([x_cam, y_cam, zs, torch.ones_like(zs)], -1)
    pts_world = (T_c2w @ pts_cam.T).T[:, :3]
    return pts_world, valid


def hidden_from_source(pts_world, depth_src, K_src, T_w2c_src, H, W, margin=0.15):
    """True = point not visible from source (outside frustum or behind source surface)."""
    device = pts_world.device
    N = pts_world.shape[0]
    ones = torch.ones(N, 1, device=device)
    pc = (T_w2c_src @ torch.cat([pts_world, ones], 1).T).T[:, :3]
    z = pc[:, 2]
    x = pc[:, 0] / (z + 1e-8)
    y = pc[:, 1] / (z + 1e-8)
    fx, fy = K_src[0, 0], K_src[1, 1]
    cx, cy = K_src[0, 2], K_src[1, 2]
    px = (x * fx + cx).long()
    py = (y * fy + cy).long()
    outside = (px < 0) | (px >= W) | (py < 0) | (py >= H) | (z < 0.01)
    pxc = px.clamp(0, W - 1)
    pyc = py.clamp(0, H - 1)
    src_d = depth_src[pyc, pxc]
    behind = z > (src_d + margin)
    return outside | behind


def resize_K(K, H_old, W_old, H_new, W_new):
    K = K.clone()
    K[0, 0] *= W_new / W_old
    K[0, 2] *= W_new / W_old
    K[1, 1] *= H_new / H_old
    K[1, 2] *= H_new / H_old
    return K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_scenes", type=int, default=620)
    ap.add_argument("--k_neighbors", type=int, default=3)
    ap.add_argument("--max_hidden_pts", type=int, default=8192)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    cfg = build_flash3d_cfg(batch_size=1, num_workers=0, stage="dev")
    backbone = Flash3DBackbone(cfg, FLASH3D_CKPT, device=str(device))
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    from paper_b_generative3d.viz_compare import _create_loader_from_cfg

    _, loader = _create_loader_from_cfg(cfg)
    H, W = 256, 384
    stats = {"n_scenes": 0, "total_hidden": 0}
    n_done = 0
    for inputs in loader:
        if n_done >= args.max_scenes:
            break
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)

        T_w2c_src_t = inputs.get(("T_w2c", 0))
        T_c2w_src_t = inputs.get(("T_c2w", 0))
        if T_w2c_src_t is None or T_c2w_src_t is None:
            n_done += 1
            continue
        T_w2c_src = T_w2c_src_t[0]

        # Source depth (real UniDepth on source color at H,W)
        src_color = inputs[("color", 0, 0)]  # [1,3,H,W]
        K_src_full = inputs[("K_src", 0)][0]
        # K_src is for unpadded color at (H,W)? assume matches ("color",0,0) size.
        sc_h, sc_w = src_color.shape[-2], src_color.shape[-1]
        K_src_d = (
            resize_K(K_src_full, sc_h, sc_w, H, W)
            if (sc_h != H or sc_w != W)
            else K_src_full
        )
        src_rs = F.interpolate(src_color, (H, W), mode="bilinear", align_corners=False)
        depth_src = backbone.predict_depth(src_rs, K_src_d.unsqueeze(0))[0, 0]  # [H,W]

        neighbor_fids = [
            f for f in [1, 2, 3] if ("T_c2w", f) in inputs and ("color", f, 0) in inputs
        ][: args.k_neighbors]

        all_pts, all_col = [], []
        for fid in neighbor_fids:
            T_c2w_n = inputs[("T_c2w", fid)][0]
            n_img = inputs[("color", fid, 0)]  # [1,3,H,W]
            nh, nw = n_img.shape[-2], n_img.shape[-1]
            K_n_full = inputs.get(
                ("K_tgt", fid), inputs.get(("K_src", fid), inputs[("K_src", 0)])
            )
            K_n_full = K_n_full[0] if K_n_full.dim() == 3 else K_n_full
            K_n = resize_K(K_n_full, nh, nw, H, W) if (nh != H or nw != W) else K_n_full
            n_rs = F.interpolate(n_img, (H, W), mode="bilinear", align_corners=False)
            depth_n = backbone.predict_depth(n_rs, K_n.unsqueeze(0))[0, 0]  # [H,W] REAL

            pts_world, valid = backproject(depth_n, K_n, T_c2w_n, H, W)
            hidden = hidden_from_source(pts_world, depth_src, K_src_d, T_w2c_src, H, W)
            keep = hidden & valid
            # subsample by stride grid to limit count
            grid_mask = torch.zeros(H * W, dtype=torch.bool, device=device)
            grid_mask.view(H, W)[:: args.stride, :: args.stride] = True
            keep = keep & grid_mask
            if keep.sum() < 5:
                continue
            pts = pts_world[keep]
            n_flat = n_rs[0].permute(1, 2, 0).reshape(-1, 3)  # [HW,3]
            col = n_flat[keep]
            all_pts.append(pts)
            all_col.append(col)

        if not all_pts:
            n_done += 1
            continue
        pts = torch.cat(all_pts, 0)
        col = torch.cat(all_col, 0)
        if pts.shape[0] > args.max_hidden_pts:
            idx = torch.randperm(pts.shape[0], device=device)[: args.max_hidden_pts]
            pts = pts[idx]
            col = col[idx]

        torch.save(
            {
                "xyz": pts.cpu().half(),
                "color": col.cpu().half(),
                "n_points": pts.shape[0],
            },
            out / f"{n_done:06d}.pt",
        )
        stats["total_hidden"] += pts.shape[0]
        stats["n_scenes"] += 1
        n_done += 1
        if n_done % args.log_every == 0:
            mean_pts = stats["total_hidden"] / max(stats["n_scenes"], 1)
            print(
                f"[{n_done}/{args.max_scenes}] mean_hidden={mean_pts:.0f} pts/scene",
                flush=True,
            )

    mean_pts = stats["total_hidden"] / max(stats["n_scenes"], 1)
    stats["mean_hidden_pts"] = mean_pts
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(
        f"\n=== S0 v2 done: {stats['n_scenes']} scenes, mean {mean_pts:.0f} hidden pts ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
