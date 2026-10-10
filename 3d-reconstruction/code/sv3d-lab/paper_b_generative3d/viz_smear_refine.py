"""Qualitative comparison: GT | Flash3D | Refined, with red-box zoom on the smear region
that the refinement head fixes most.

Produces per-scene panels + a zoomed inset showing smear reduction.

Run (flash3d venv):
  python -m paper_b_generative3d.viz_smear_refine \
    --ckpt /home/data/sv3d-lab/reconstruction/E-128-smear-refine/refine_final.pt \
    --out /home/data/sv3d-lab/reconstruction/E-128-smear-refine/qual --n 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from paper_b_generative3d.backbone_flash3d import Flash3DBackbone
from paper_b_generative3d.data_loader_re10k import build_flash3d_cfg, FLASH3D_CKPT
from paper_b_generative3d.viz_compare import (
    psnr,
    to_uint8,
    label_strip,
    find_improve_region,
    draw_box,
    crop_zoom,
    _create_loader_from_cfg,
)
from paper_b_generative3d.model.smear_refine_head import SmearRefineHead
from paper_b_generative3d.train_smear_refine import sample_per_gaussian_feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--feat_dim", type=int, default=2048)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    cfg = build_flash3d_cfg(batch_size=1, num_workers=0, stage="dev")
    backbone = Flash3DBackbone(cfg, FLASH3D_CKPT, device=str(device))
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    head = SmearRefineHead(feat_dim=args.feat_dim, hidden=256, n_layers=4).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    head.load_state_dict(ckpt["head"])
    head.eval()

    _, loader = _create_loader_from_cfg(cfg)
    H, W = 256, 384
    saved = 0
    zh, zw = 160, 160
    for i, inputs in enumerate(loader):
        if saved >= args.n:
            break
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
        tfids = [f for f in [1, 2, 3] if ("color", f, 0) in inputs]
        T_c2w_s = inputs.get(("T_c2w", 0))
        if not tfids or T_c2w_s is None:
            continue
        with torch.no_grad():
            g = backbone.extract_source_gaussians(inputs)
            anchor = {
                k: g[k][0]
                for k in ["xyz", "scales", "rotations", "opacity", "color_rgb"]
            }
            n = anchor["xyz"].shape[0]
            feat = sample_per_gaussian_feats(
                g["source_features"], n, g["pixel_hw"], g["gaussians_per_pixel"], device
            )
            refined = head(anchor, feat)
            refined_g = {
                k: refined[k]
                for k in ["xyz", "scales", "rotations", "opacity", "color_rgb"]
            }

        # pick the target with the largest flash3d error (most smear)
        best_fid, best_err = None, -1
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
            err = (r_flash - gt).abs().mean().item()
            if err > best_err:
                best_err, best_fid = err, fid
        if best_fid is None:
            continue
        fid = best_fid
        cam = inputs[("T_w2c", fid)][0] @ T_c2w_s[0]
        K_tgt = inputs.get(("K_tgt", fid), inputs[("K_src", 0)])
        if K_tgt.dim() == 3:
            K_tgt = K_tgt[0]
        gt = inputs["color", fid, 0][0]
        with torch.no_grad():
            r_flash = backbone.render_gaussians(anchor, cam, K_tgt, H, W).clamp(0, 1)
            r_ref = backbone.render_gaussians(refined_g, cam, K_tgt, H, W).clamp(0, 1)

        gt_u, flash_u, ref_u = to_uint8(gt), to_uint8(r_flash), to_uint8(r_ref)
        box = find_improve_region(gt_u, flash_u, ref_u, box_frac=0.35)
        gt_b = draw_box(gt_u, box)
        flash_b = draw_box(flash_u, box)
        ref_b = draw_box(ref_u, box)
        # zoom row
        gt_z = crop_zoom(gt_u, box, (zh, zw))
        flash_z = crop_zoom(flash_u, box, (zh, zw))
        ref_z = crop_zoom(ref_u, box, (zh, zw))

        pf = psnr(r_flash, gt)
        pr = psnr(r_ref, gt)
        top = np.concatenate([gt_b, flash_b, ref_b], axis=1)
        labels = np.concatenate(
            [
                label_strip(W, "GT"),
                label_strip(W, f"Flash3D {pf:.1f}dB"),
                label_strip(W, f"Refined(ours) {pr:.1f}dB"),
            ],
            axis=1,
        )
        zoom = np.concatenate([gt_z, flash_z, ref_z], axis=1)
        # pad zoom row to full width
        pad = top.shape[1] - zoom.shape[1]
        if pad > 0:
            zoom = np.concatenate(
                [zoom, np.full((zoom.shape[0], pad, 3), 255, np.uint8)], axis=1
            )
        panel = np.concatenate([labels, top, zoom], axis=0)
        Image.fromarray(panel).save(out / f"panel_scene{i:03d}_f{fid}.png")
        saved += 1
        print(f"saved panel scene{i} f{fid} flash={pf:.2f} ref={pr:.2f}", flush=True)

    print(f"=== {saved} qualitative panels saved to {out} ===", flush=True)


if __name__ == "__main__":
    main()
