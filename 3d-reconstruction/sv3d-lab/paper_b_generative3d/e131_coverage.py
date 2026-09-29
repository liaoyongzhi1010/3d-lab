"""E-131 step 2: measure disocclusion COVERAGE at the target view.

For scene6 target f3, render an ALPHA/coverage map from Flash3D gaussians and find the
fraction of target pixels with NO source coverage (the true disocclusion / out-of-frame
region). Also split into: (a) inside source frustum but disoccluded (behind-edge),
(b) outside source frustum (out-of-frame reveal). This decides the method:
  - if dominated by out-of-frame -> need frustum-extrapolation generation
  - if behind-edge -> LDI-style spawn suffices

Renders a coverage heatmap overlay.

Run (flash3d venv):
  python -m paper_b_generative3d.e131_coverage --scene 6 \
    --out /home/data/sv3d-lab/reconstruction/E-131-coverage
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from paper_b_generative3d.backbone_flash3d import Flash3DBackbone
from paper_b_generative3d.data_loader_re10k import build_flash3d_cfg, FLASH3D_CKPT
from paper_b_generative3d.viz_compare import psnr, to_uint8, _create_loader_from_cfg


def save(t, path):
    Image.fromarray(to_uint8(t)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    cfg = build_flash3d_cfg(batch_size=1, num_workers=0, stage="dev")
    backbone = Flash3DBackbone(cfg, FLASH3D_CKPT, device=str(device))
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    _, loader = _create_loader_from_cfg(cfg)
    H, W = 256, 384

    for i, inputs in enumerate(loader):
        if i != args.scene:
            continue
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
        tfids = [f for f in [1, 2, 3] if ("color", f, 0) in inputs]
        T_c2w_s = inputs.get(("T_c2w", 0))
        K_src = inputs[("K_src", 0)][0]

        with torch.no_grad():
            g = backbone.extract_source_gaussians(inputs)
        anchor = {
            k: g[k][0] for k in ["xyz", "scales", "rotations", "opacity", "color_rgb"]
        }

        # Make an all-white-color copy with full opacity to render a COVERAGE map:
        # where gaussians project, coverage ~1; empty stays 0.
        cov_g = {kk: anchor[kk].clone() for kk in anchor}
        cov_g["color_rgb"] = torch.ones_like(cov_g["color_rgb"])
        cov_g["opacity"] = torch.ones_like(cov_g["opacity"])

        for fid in tfids:
            T_w2c_t = inputs.get(("T_w2c", fid))
            if T_w2c_t is None:
                continue
            cam = T_w2c_t[0] @ T_c2w_s[0]
            K_tgt = inputs.get(("K_tgt", fid), inputs[("K_src", 0)])
            if K_tgt.dim() == 3:
                K_tgt = K_tgt[0]
            gt = inputs["color", fid, 0][0]
            with torch.no_grad():
                cov = backbone.render_gaussians(cov_g, cam, K_tgt, H, W).clamp(0, 1)
            cov_map = cov.mean(0)  # [H,W] ~1 where covered, ~0 where empty (bg)
            empty = (cov_map < 0.3).float()
            frac_empty = float(empty.mean())

            # Determine which empty pixels are OUT-OF-FRAME wrt source: project target
            # pixel rays... simpler proxy: source covers a horizontal FOV; out-of-frame
            # reveal appears at image borders. Split empty into left/right/top/bottom bands
            # vs interior.
            Hh, Ww = empty.shape
            border = torch.zeros_like(empty)
            bw = int(Ww * 0.12)
            border[:, :bw] = 1
            border[:, -bw:] = 1
            border[: int(Hh * 0.12), :] = 1
            border[-int(Hh * 0.12) :, :] = 1
            empty_border = float((empty * border).sum() / (empty.sum() + 1e-6))
            empty_interior = 1.0 - empty_border

            print(
                f"scene{i} f{fid}: empty={100 * frac_empty:.1f}% of target | "
                f"of empty: {100 * empty_border:.0f}% at borders(out-of-frame) "
                f"{100 * empty_interior:.0f}% interior(behind-edge)",
                flush=True,
            )

            # save coverage visualization: GT with empty region tinted red
            gt_vis = gt.clone()
            gt_vis[0] = torch.maximum(gt_vis[0], empty)  # red where empty
            save(gt, out / f"f{fid}_gt.png")
            save(cov_map.unsqueeze(0).repeat(3, 1, 1), out / f"f{fid}_coverage.png")
            save(gt_vis, out / f"f{fid}_emptyoverlay.png")
        break
    print(f"=== saved to {out} ===", flush=True)


if __name__ == "__main__":
    main()
