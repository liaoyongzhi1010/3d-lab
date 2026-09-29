"""
Phase 3 FULL Representation Oracle runner.
Builds G_vis (source) + G_hidden (target UniDepth, oracle cheat), classifies occ/OOF, merges in
SOURCE camera frame, renders to source+targets via validated render_gaussians_relpose, reports
region-separated PSNR + causal deletion. Pre-registration: ORACLE_PREREGISTRATION.md.

Gaussians live in SOURCE camera frame (Flash3D convention). Target points are transformed into
source frame via cam_T_cam(tgt->src) = w2c_src @ c2w_tgt.

Run on server:
  cd /root/projects/flash3d && source .venv/bin/activate
  export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python /root/sv3d-lab/paper_a_explicit3d/oracle/run_oracle.py --n_scenes 100 \
      --out /home/data/sv3d-lab/evaluations/oracle_v1
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, "/root/sv3d-lab/paper_a_explicit3d/oracle")
sys.path.insert(0, "/root/sv3d-lab")

from oracle_core import (
    backproject,
    make_gaussians,
    set_scale_from_depth,
    merge_gaussians,
    render_gaussians_relpose,
    project,
    psnr,
    crop5,
)
from common.geometry.visibility import visibility_partition


def build_cfg(split_path="splits/re10k_mine_filtered/test_files_present.txt"):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir="/root/projects/flash3d/configs", version_base=None
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=layered_re10k",
                "+dataset.crop_border=true",
                f"dataset.test_split_path={split_path}",
                "model.depth.version=v1",
                "data_loader.batch_size=1",
            ],
        )
    return cfg


def region_masks(pts_src_cam, K, H, W, src_depth_map, occ_margin=1.05):
    """Classify hidden target points (already in source-cam frame) as:
    OOF (project outside source frame) or OCCLUDED (in-frame but behind source surface).
    Returns (occluded_mask, oof_mask) over points, plus in_front mask."""
    uv, z = project(pts_src_cam, K)
    u, v = uv[:, 0], uv[:, 1]
    in_front = z > 0.05
    in_frame = (u >= 0) & (u < W) & (v >= 0) & (v < H) & in_front
    occluded = torch.zeros_like(in_frame)
    if in_frame.any():
        ui = u[in_frame].long().clamp(0, W - 1)
        vi = v[in_frame].long().clamp(0, H - 1)
        src_d = src_depth_map[vi, ui]
        behind = z[in_frame] > src_d * occ_margin
        idx = torch.where(in_frame)[0][behind]
        occluded[idx] = True
    oof = (~in_frame) & in_front
    return occluded, oof


def target_pixel_region_mask(g_vis, K, T_rel, H, W, device, gt_depth_tgt):
    """Build method-agnostic per-pixel region mask AT the target view:
    render G_vis alpha to target; pixels with low alpha = not-covered-by-visible (occluded or OOF
    from source). Returns bool masks (visible, hidden) over target pixels."""
    out = render_gaussians_relpose(g_vis, K, T_rel, H, W, device)
    alpha = out.get("rendered_alpha", None)
    if alpha is None:
        return None, None
    alpha = alpha.squeeze()
    visible = alpha > 0.5
    hidden = ~visible
    return visible, hidden


def process_scene(unidepth, inputs, device, footprint_factor=0.15):
    color_src = inputs[("color", 0, 0)].to(device)
    K = inputs[("K_tgt", 0)].to(device)
    c2w_src = inputs[("T_c2w", 0)].to(device)
    w2c_src = inputs[("T_w2c", 0)].to(device)
    H, W = color_src.shape[1:]

    with torch.no_grad():
        depth_src = unidepth.infer(color_src.unsqueeze(0), intrinsics=K.unsqueeze(0))[
            "depth"
        ].squeeze()
    inv_K = torch.linalg.inv(K)
    pts_src = backproject(depth_src, inv_K, device)
    rgb_src = color_src.permute(1, 2, 0).reshape(-1, 3)
    g_vis = make_gaussians(pts_src, rgb_src, K[0, 0].item())
    set_scale_from_depth(g_vis, depth_src.reshape(-1), K[0, 0].item(), footprint_factor)

    # Build hidden gaussians from each available target.
    # A target pixel is "hidden" if it falls in the forward-warp disocclusion/OOF mask
    # (method-agnostic, source-geometry-only). Its backprojected 3D point (in source frame)
    # becomes an oracle hidden Gaussian with the target pixel's true color.
    hidden_xyz, hidden_rgb, hidden_depth = [], [], []
    n_occ = n_oof = 0
    target_frames = [f for f in [1, 2, 3] if ("color", f, 0) in inputs]
    for f in target_frames:
        color_t = inputs[("color", f, 0)].to(device)
        c2w_t = inputs[("T_c2w", f)].to(device)
        with torch.no_grad():
            depth_t = unidepth.infer(color_t.unsqueeze(0), intrinsics=K.unsqueeze(0))[
                "depth"
            ].squeeze()
        pts_t_cam = backproject(depth_t, inv_K, device)  # target-cam frame
        T_ts = w2c_src @ c2w_t
        N = pts_t_cam.shape[0]
        homo = torch.cat([pts_t_cam, torch.ones(N, 1, device=device)], 1)
        pts_in_src = (T_ts @ homo.T).T[:, :3]
        rgb_t = color_t.permute(1, 2, 0).reshape(-1, 3)

        # method-agnostic mask at THIS target (source geometry warped to target)
        T_st = torch.linalg.inv(c2w_t) @ c2w_src  # src->tgt
        _, occ_mask, oof_mask = visibility_partition(
            depth_src, K, T_st, H, W, device, dilate=2
        )
        hidden_pix = (occ_mask | oof_mask).reshape(-1)  # (HW,) over target pixels
        n_occ += int(occ_mask.sum())
        n_oof += int(oof_mask.sum())
        if hidden_pix.any():
            hidden_xyz.append(pts_in_src[hidden_pix])
            hidden_rgb.append(rgb_t[hidden_pix])
            hidden_depth.append(pts_in_src[hidden_pix][:, 2].clamp(min=0.1))

    res = {
        "n_vis": g_vis["xyz"].shape[0],
        "n_hidden": 0,
        "n_occ": n_occ,
        "n_oof": n_oof,
    }
    if hidden_xyz:
        hx = torch.cat(hidden_xyz, 0)
        hr = torch.cat(hidden_rgb, 0)
        hd = torch.cat(hidden_depth, 0)
        g_hidden = make_gaussians(hx, hr, K[0, 0].item())
        set_scale_from_depth(g_hidden, hd, K[0, 0].item(), footprint_factor)
        g_merged = merge_gaussians([g_vis, g_hidden])
        res["n_hidden"] = hx.shape[0]
    else:
        g_merged = g_vis

    # metrics per target: merged vs vis-only, region-separated
    per_target = {}
    for f in [0] + target_frames:
        gt = inputs[("color", f, 0)].to(device)
        if f == 0:
            T_rel = torch.eye(4, device=device)
        else:
            c2w_t = inputs[("T_c2w", f)].to(device)
            w2c_t = inputs[("T_w2c", f)].to(device)
            T_rel = w2c_t @ c2w_src  # src-cam -> tgt-cam (matches sanity gate)
        merged = render_gaussians_relpose(g_merged, K, T_rel, H, W, device)[
            "render"
        ].clamp(0, 1)
        visonly = render_gaussians_relpose(g_vis, K, T_rel, H, W, device)[
            "render"
        ].clamp(0, 1)
        m = {
            "psnr_merged": psnr(crop5(merged), crop5(gt)),
            "psnr_vis_only": psnr(crop5(visonly), crop5(gt)),
        }
        m["deletion_delta"] = m["psnr_merged"] - m["psnr_vis_only"]
        if f != 0:
            # method-agnostic forward-warp visibility partition (source geometry only)
            visible, occluded, oof = visibility_partition(
                depth_src, K, T_rel, H, W, device, dilate=2
            )
            hid_mask = occluded | oof
            m["psnr_merged_hidden"] = psnr(merged, gt, mask=hid_mask)
            m["psnr_vis_hidden"] = psnr(visonly, gt, mask=hid_mask)
            m["psnr_merged_occ"] = psnr(merged, gt, mask=occluded)
            m["psnr_vis_occ"] = psnr(visonly, gt, mask=occluded)
            m["psnr_merged_oof"] = psnr(merged, gt, mask=oof)
            m["psnr_vis_oof"] = psnr(visonly, gt, mask=oof)
            m["hidden_frac"] = float(hid_mask.float().mean())
            m["occ_frac"] = float(occluded.float().mean())
            m["oof_frac"] = float(oof.float().mean())
        per_target[str(f)] = m
    res["per_target"] = per_target
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_scenes", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--footprint", type=float, default=0.15)
    ap.add_argument(
        "--split", default="splits/re10k_mine_filtered/test_files_present.txt"
    )
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
    cfg = build_cfg(args.split)
    from datasets.re10k import Re10KDataset

    ds = Re10KDataset(cfg, split="test")
    print(f"Dataset length: {len(ds)}")

    all_res = []
    agg = {
        "deletion_delta": [],
        "psnr_merged": [],
        "psnr_vis_only": [],
        "psnr_merged_hidden": [],
        "psnr_vis_hidden": [],
        "hidden_frac": [],
        "n_hidden": [],
        "n_occ": [],
        "n_oof": [],
    }
    for i in range(min(args.n_scenes, len(ds))):
        try:
            r = process_scene(unidepth, ds[i], device, args.footprint)
        except Exception as e:
            print(f"scene {i} FAILED: {e}")
            continue
        all_res.append(r)
        agg["n_hidden"].append(r["n_hidden"])
        agg["n_occ"].append(r["n_occ"])
        agg["n_oof"].append(r["n_oof"])
        # average over target frames (exclude source frame 0 for delta)
        for f, m in r["per_target"].items():
            if f == "0":
                continue
            agg["deletion_delta"].append(m["deletion_delta"])
            agg["psnr_merged"].append(m["psnr_merged"])
            agg["psnr_vis_only"].append(m["psnr_vis_only"])
            if "psnr_merged_hidden" in m and m["psnr_merged_hidden"] is not None:
                agg["psnr_merged_hidden"].append(m["psnr_merged_hidden"])
            if "psnr_vis_hidden" in m and m["psnr_vis_hidden"] is not None:
                agg["psnr_vis_hidden"].append(m["psnr_vis_hidden"])
            if "hidden_frac" in m:
                agg["hidden_frac"].append(m["hidden_frac"])
        if (i + 1) % 10 == 0:
            print(
                f"[{i + 1}] merged={np.mean(agg['psnr_merged']):.2f} "
                f"vis_only={np.mean(agg['psnr_vis_only']):.2f} "
                f"delta={np.mean(agg['deletion_delta']):.3f} "
                f"hidden_frac={np.mean(agg['hidden_frac']):.3f}"
            )

    summary = {k: (float(np.mean(v)) if v else None) for k, v in agg.items()}
    summary["n_scenes"] = len(all_res)
    summary["footprint_factor"] = args.footprint
    with open(os.path.join(args.out, "oracle_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)
    with open(os.path.join(args.out, "oracle_per_scene.json"), "w") as fp:
        json.dump(all_res, fp, indent=2)

    print("\n=== ORACLE SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\n=== PRE-REGISTERED THRESHOLDS (ORACLE_PREREGISTRATION.md) ===")
    mh = summary.get("psnr_merged_hidden")
    vh = summary.get("psnr_vis_hidden")
    hidden_delta = (mh - vh) if (mh is not None and vh is not None) else None
    print(
        f"P1 merged overall PSNR >= 24: {summary['psnr_merged']:.2f} "
        f"{'PASS' if summary['psnr_merged'] and summary['psnr_merged'] >= 24 else 'FAIL'}"
    )
    if mh is not None:
        print(f"P2/P3 hidden-region merged PSNR: {mh:.2f} (vis-only {vh:.2f})")
    if hidden_delta is not None:
        print(
            f"P4 HIDDEN-REGION deletion delta >= 3.0dB: {hidden_delta:.3f} "
            f"{'PASS' if hidden_delta >= 3.0 else 'FAIL'}  "
            f"(overall-image delta {summary['deletion_delta']:.3f})"
        )
    summary["hidden_region_deletion_delta"] = hidden_delta
    with open(os.path.join(args.out, "oracle_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)


if __name__ == "__main__":
    main()
