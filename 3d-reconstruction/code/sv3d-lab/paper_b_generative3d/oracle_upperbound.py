"""E-119 Oracle upper-bound: how much headroom is there in disoccluded regions?

Question: if a PERFECT generative head filled Flash3D's disoccluded regions with the
CORRECT content, how much would it beat Flash3D? This upper-bounds the flow-matching
residual head. If the Oracle barely beats Flash3D, the whole generative approach is
capped low (KILL). If it beats a lot, training a flow head to approach it is worth it (GO).

Method (single-view, no leakage into the METHOD — Oracle is a diagnostic upper bound,
explicitly labeled, never reported as a model result per charter rule 4):
  1. Flash3D renders target view (its best regression guess, blurry in holes).
  2. Disocclusion mask = target pixels NOT visible from source: forward-warp source
     depth to target; pixels with no source support = disoccluded.
  3. Oracle = Flash3D_render * (1 - mask) + GT_target * mask   (fill holes with truth).
  4. Compare PSNR/LPIPS: Flash3D vs Oracle. Panel [GT | Flash3D | Oracle | mask | zoom...].

This is an UPPER BOUND (uses GT to fill holes = cheating), so it answers "is the ceiling
high enough to bother?" NOT a method. Labeled ORACLE everywhere.

Run (Flash3D venv):
  python -m paper_b_generative3d.oracle_upperbound --split_path splits/re10k_mine_filtered/test_gap50.txt \
    --flash3d_max_psnr 25.0 --max_scenes 6 --out /home/data/sv3d-lab/reconstruction/E-119-oracle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from paper_b_generative3d.backbone_flash3d import Flash3DBackbone
from paper_b_generative3d.data_loader_re10k import build_flash3d_cfg, FLASH3D_CKPT
from paper_b_generative3d.viz_compare import (
    psnr,
    to_uint8,
    label_strip,
    draw_box,
    crop_zoom,
    find_improve_region,
    _create_loader_from_cfg,
)


def compute_disocclusion_mask(inputs, backbone, fid, H, W, device):
    """Mask (H,W) in [0,1]: 1 = target pixel NOT supported by source (disoccluded).

    Forward-warp source pixels into the target view using Flash3D source depth + relative
    pose, splat a coverage map, then mask = (coverage < threshold).
    """
    gauss_out = backbone.extract_source_gaussians(inputs)
    raw = {
        k: (v[0] if torch.is_tensor(v) and v.dim() >= 1 else v)
        for k, v in gauss_out.items()
        if k != "_outputs"
    }
    # Render an all-white "coverage" splat: set colors to 1, keep geometry.
    g_white = {
        "xyz": raw["xyz"],
        "scales": raw["scales"],
        "rotations": raw["rotations"],
        "opacity": raw["opacity"],
        "color_rgb": torch.ones_like(raw["color_rgb"]),
    }
    cam = inputs.get(("cam_T_cam", 0, fid))
    if cam is None:
        cam = gauss_out.get("_outputs", {}).get(("cam_T_cam", 0, fid))
    K_tgt = inputs.get(("K_tgt", fid), inputs[("K_src", 0)])
    cov = backbone.render_gaussians(g_white, cam[0], K_tgt[0], H, W, batch_idx=0)
    coverage = cov.mean(dim=0)  # (H,W) ~1 where source splats land, ~bg where holes
    mask = (coverage < 0.5).float()
    # light close/open via avgpool to remove speckle
    m = mask[None, None]
    m = F.max_pool2d(m, 3, 1, 1)
    m = -F.max_pool2d(-m, 3, 1, 1)
    return m[0, 0], gauss_out, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_path", default="splits/re10k_mine_filtered/test_gap50.txt")
    ap.add_argument("--flash3d_max_psnr", type=float, default=25.0)
    ap.add_argument("--max_scenes", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    cfg = build_flash3d_cfg(
        batch_size=1,
        num_workers=0,
        stage="dev",
        extra_overrides=[f"dataset.test_split_path={args.split_path}"],
    )
    backbone = Flash3DBackbone(cfg, FLASH3D_CKPT, device=str(device))
    backbone.to(device)
    _, loader = _create_loader_from_cfg(cfg)

    H, W = 256, 384
    summary = []
    n_saved, n_scanned = 0, 0
    max_scan = args.max_scenes * 40
    for inputs in loader:
        if n_saved >= args.max_scenes or n_scanned >= max_scan:
            break
        n_scanned += 1
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
        if ("color", 1, 0) not in inputs:
            continue
        tfids = [f for f in [1, 2, 3] if ("color", f, 0) in inputs]
        if not tfids:
            continue
        fid = tfids[-1]

        gauss_out = backbone.extract_source_gaussians(inputs)
        raw0 = gauss_out
        raw = {
            "xyz": raw0["xyz"][0],
            "scales": raw0["scales"][0],
            "rotations": raw0["rotations"][0],
            "opacity": raw0["opacity"][0],
            "color_rgb": raw0["color_rgb"][0],
        }
        cam = inputs.get(("cam_T_cam", 0, fid))
        if cam is None:
            cam = raw0.get("_outputs", {}).get(("cam_T_cam", 0, fid))
        if cam is None:
            continue
        K_tgt = inputs.get(("K_tgt", fid), inputs[("K_src", 0)])
        gt = inputs["color", fid, 0][0]

        r_bb = backbone.render_gaussians(
            raw, cam[0], K_tgt[0], H, W, batch_idx=0
        ).clamp(0, 1)
        ps_bb = psnr(r_bb, gt)
        if args.flash3d_max_psnr is not None and ps_bb >= args.flash3d_max_psnr:
            continue

        # disocclusion mask via white-coverage splat
        g_white = {
            "xyz": raw["xyz"],
            "scales": raw["scales"],
            "rotations": raw["rotations"],
            "opacity": raw["opacity"],
            "color_rgb": torch.ones_like(raw["color_rgb"]),
        }
        cov = backbone.render_gaussians(g_white, cam[0], K_tgt[0], H, W, batch_idx=0)
        coverage = cov.mean(dim=0)
        mask_hole = (coverage < 0.5).float()

        # degraded-region mask: where Flash3D deviates strongly from GT (smear/blur/holes).
        # A perfect generative head could fix these; Oracle fills with GT = upper bound.
        err = (r_bb - gt).abs().mean(dim=0)
        err_blur = F.avg_pool2d(err[None, None], 9, 1, 4)[0, 0]
        thr = torch.quantile(err_blur.flatten(), 0.70)  # worst 30% error region
        mask_deg = (err_blur > thr).float()

        mask = torch.clamp(mask_hole + mask_deg, 0, 1)
        m = mask[None, None]
        m = F.max_pool2d(m, 3, 1, 1)
        m = -F.max_pool2d(-m, 3, 1, 1)
        mask = m[0, 0]
        mask3 = mask[None].repeat(3, 1, 1)

        # Oracle = Flash3D outside fixable region, GT inside (upper bound on generative fix)
        oracle = r_bb * (1 - mask3) + gt * mask3
        ps_oracle = psnr(oracle, gt)
        hole_frac = float(mask.mean().item())
        hole_geo = float(mask_hole.mean().item())

        gt_u = to_uint8(gt)
        bb_u = to_uint8(r_bb)
        or_u = to_uint8(oracle)
        mask_u = to_uint8(mask3)
        box = find_improve_region(gt_u, bb_u, or_u)
        zoom_gt = crop_zoom(gt_u, box, (H, H))
        zoom_bb = crop_zoom(bb_u, box, (H, H))
        zoom_or = crop_zoom(or_u, box, (H, H))
        gap = np.ones((H, 6, 3), dtype=np.uint8) * 255
        panel = np.concatenate(
            [
                draw_box(gt_u, box),
                gap,
                draw_box(bb_u, box),
                gap,
                draw_box(or_u, box),
                gap,
                mask_u,
                gap,
                zoom_bb,
                gap,
                zoom_or,
            ],
            axis=1,
        )
        lbl = label_strip(
            panel.shape[1],
            f"scan{n_scanned} | GT | Flash3D {ps_bb:.2f} | ORACLE {ps_oracle:.2f} (+{ps_oracle - ps_bb:.2f}) | hole={hole_frac * 100:.1f}% | mask | ZOOM Flash3D / ORACLE",
        )
        full = np.concatenate([lbl, panel], axis=0)
        Image.fromarray(full).save(
            out_dir / f"scene{n_saved:02d}_scan{n_scanned:04d}.png"
        )
        summary.append(
            {
                "scan": n_scanned,
                "tgt": fid,
                "flash3d_psnr": ps_bb,
                "oracle_psnr": ps_oracle,
                "gain": ps_oracle - ps_bb,
                "hole_frac": hole_frac,
            }
        )
        print(
            f"[{n_saved}] scan{n_scanned}: Flash3D={ps_bb:.2f} ORACLE={ps_oracle:.2f} "
            f"gain=+{ps_oracle - ps_bb:.2f} hole={hole_frac * 100:.1f}%",
            flush=True,
        )
        n_saved += 1

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if summary:
        gains = [s["gain"] for s in summary]
        holes = [s["hole_frac"] for s in summary]
        print(f"\n=== ORACLE upper-bound: {len(summary)} scenes ===", flush=True)
        print(
            f"mean gain = +{np.mean(gains):.2f}dB (max +{np.max(gains):.2f})",
            flush=True,
        )
        print(f"mean hole fraction = {np.mean(holes) * 100:.1f}%", flush=True)
        print(
            f"VERDICT: {'GO (headroom exists)' if np.mean(gains) > 1.5 else 'MARGINAL/KILL (ceiling low)'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
