"""E-131 step 1 (zero-training free win): does pruning/shrinking edge-straddling gaussians
remove the brown smear in the disocclusion region?

Mechanism (Shih-2020 "cut mesh at discontinuities", applied to Gaussians): Flash3D gaussians
whose footprint straddles a large depth jump get stretched and bleed into the disoccluded
region as a smear. If we detect gaussians at depth discontinuities and either (a) shrink their
scale or (b) drop them, the brown smear should reduce (leaving a clean hole = honest "I don't
know" rather than a confident wrong smear).

This is a diagnostic: no training. Renders Flash3D vs pruned at scene6 targets + right-crop.

Run (flash3d venv):
  python -m paper_b_generative3d.e131_prune_smear --scene 6 \
    --out /home/data/sv3d-lab/reconstruction/E-131-prune
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from paper_b_generative3d.backbone_flash3d import Flash3DBackbone
from paper_b_generative3d.data_loader_re10k import build_flash3d_cfg, FLASH3D_CKPT
from paper_b_generative3d.viz_compare import psnr, to_uint8, _create_loader_from_cfg


def save(t, path):
    Image.fromarray(to_uint8(t)).save(path)


def rightcrop(im, W, x_frac=0.6, scale=3):
    x0 = int(W * x_frac)
    crop = im[:, :, x0:]
    return Image.fromarray(to_uint8(crop)).resize(
        (int((W - x0) * scale), im.shape[1] * scale), Image.NEAREST
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--edge_thresh", type=float, default=0.15, help="log-depth grad threshold"
    )
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

        with torch.no_grad():
            g = backbone.extract_source_gaussians(inputs)
        anchor = {
            k: g[k][0] for k in ["xyz", "scales", "rotations", "opacity", "color_rgb"]
        }
        n = anchor["xyz"].shape[0]
        gpp = int(g["gaussians_per_pixel"])
        ph, pw = int(g["pixel_hw"][0]), int(g["pixel_hw"][1])
        area = ph * pw

        # Reconstruct per-pixel depth grid (layer 0) to find depth edges
        z = anchor["xyz"][:, 2]  # [n]
        # layer 0 = first `area` entries map to the (ph,pw) grid
        z0 = z[:area].reshape(ph, pw)
        logz = torch.log(z0.clamp(min=1e-3))
        gx = (logz[:, 1:] - logz[:, :-1]).abs()
        gy = (logz[1:, :] - logz[:-1, :]).abs()
        edge = torch.zeros(ph, pw, device=device)
        edge[:, 1:] = torch.maximum(edge[:, 1:], gx)
        edge[1:, :] = torch.maximum(edge[1:, :], gy)
        edge_mask_grid = edge > args.edge_thresh  # [ph,pw]
        n_edge = int(edge_mask_grid.sum())
        print(
            f"scene{i}: {n_edge}/{area} pixels are depth edges ({100 * n_edge / area:.1f}%)",
            flush=True,
        )

        # broadcast edge mask over all gpp layers
        edge_flat = edge_mask_grid.reshape(-1)  # [area]
        edge_all = edge_flat.repeat(gpp)  # [n]

        # Pruned variant: drop edge-straddling gaussians (set opacity 0)
        pruned = {k: anchor[k].clone() for k in anchor}
        pruned["opacity"] = pruned["opacity"].clone()
        pruned["opacity"][edge_all] = 0.0

        # Shrunk variant: shrink scales of edge gaussians by 4x (less bleed)
        shrunk = {k: anchor[k].clone() for k in anchor}
        shrunk["scales"] = shrunk["scales"].clone()
        shrunk["scales"][edge_all] = shrunk["scales"][edge_all] * 0.25

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
                r_flash = backbone.render_gaussians(anchor, cam, K_tgt, H, W).clamp(
                    0, 1
                )
                r_prune = backbone.render_gaussians(pruned, cam, K_tgt, H, W).clamp(
                    0, 1
                )
                r_shrink = backbone.render_gaussians(shrunk, cam, K_tgt, H, W).clamp(
                    0, 1
                )
            save(gt, out / f"f{fid}_gt.png")
            save(r_flash, out / f"f{fid}_flash.png")
            save(r_prune, out / f"f{fid}_prune.png")
            save(r_shrink, out / f"f{fid}_shrink.png")
            rightcrop(r_flash, W).save(out / f"f{fid}_flash_crop.png")
            rightcrop(r_prune, W).save(out / f"f{fid}_prune_crop.png")
            rightcrop(r_shrink, W).save(out / f"f{fid}_shrink_crop.png")
            rightcrop(gt, W).save(out / f"f{fid}_gt_crop.png")
            print(
                f"  f{fid}: flash={psnr(r_flash, gt):.2f} prune={psnr(r_prune, gt):.2f} "
                f"shrink={psnr(r_shrink, gt):.2f}",
                flush=True,
            )
        break
    print(f"=== saved to {out} ===", flush=True)


if __name__ == "__main__":
    main()
