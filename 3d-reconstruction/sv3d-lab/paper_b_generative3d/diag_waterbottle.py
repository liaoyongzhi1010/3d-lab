"""Diagnose scene006 water-dispenser failure: is it visible in the SOURCE view?

Renders for a chosen scene:
  - source image (col0) : is the water dispenser present in the input at all?
  - GT novel target
  - Flash3D render at target
  - Refined (E-129) render at target
plus a high-zoom crop of the RIGHT side (where the dispenser is), saved separately.

Also dumps the Flash3D per-gaussian source-depth map so we can see whether any gaussians
exist near the dispenser / behind the occluding doorframe.

Run (flash3d venv):
  python -m paper_b_generative3d.diag_waterbottle \
    --ckpt /home/data/sv3d-lab/reconstruction/E-129-gated/refine_step1000.pt \
    --scene 6 --out /home/data/sv3d-lab/reconstruction/E-130-waterbottle-diag
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
from paper_b_generative3d.model.smear_refine_head import SmearRefineHead
from paper_b_generative3d.train_smear_refine import sample_per_gaussian_feats


def save(t, path):
    Image.fromarray(to_uint8(t)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene", type=int, default=6)
    ap.add_argument("--out", required=True)
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
    ck = torch.load(args.ckpt, map_location=device)
    head.load_state_dict(ck["head"])
    head.eval()

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
        src = inputs["color", 0, 0][0]
        save(src, out / "source.png")
        print(f"scene {i}: tfids={tfids}", flush=True)

        with torch.no_grad():
            g = backbone.extract_source_gaussians(inputs)
        anchor = {
            k: g[k][0] for k in ["xyz", "scales", "rotations", "opacity", "color_rgb"]
        }
        n = anchor["xyz"].shape[0]
        feat = sample_per_gaussian_feats(
            g["source_features"], n, g["pixel_hw"], g["gaussians_per_pixel"], device
        )
        with torch.no_grad():
            refined = head(anchor, feat)
        refined_g = {
            k: refined[k]
            for k in ["xyz", "scales", "rotations", "opacity", "color_rgb"]
        }

        # source-view depth map from anchor z
        az = anchor["xyz"][:, 2].detach().cpu().numpy()
        print(f"anchor z range: {az.min():.3f}..{az.max():.3f}, n={n}", flush=True)

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
                r_ref = backbone.render_gaussians(refined_g, cam, K_tgt, H, W).clamp(
                    0, 1
                )
            save(gt, out / f"f{fid}_gt.png")
            save(r_flash, out / f"f{fid}_flash.png")
            save(r_ref, out / f"f{fid}_refined.png")
            # right-side crop (dispenser region): rightmost 40% width
            x0 = int(W * 0.6)
            for name, im in [("gt", gt), ("flash", r_flash), ("ref", r_ref)]:
                crop = im[:, :, x0:]
                Image.fromarray(to_uint8(crop)).resize(
                    (int((W - x0) * 3), H * 3), Image.NEAREST
                ).save(out / f"f{fid}_{name}_rightcrop.png")
            print(
                f"  f{fid}: flash={psnr(r_flash, gt):.2f} ref={psnr(r_ref, gt):.2f}",
                flush=True,
            )
        break

    print(f"=== saved to {out} ===", flush=True)


if __name__ == "__main__":
    main()
