"""Stage 0a: Render Flash3D at target views and save (smear, source, gt) triples to disk.

Runs in the flash3d venv. Output feeds Stage 0b (Difix pseudo-GT gen in dr3d env).

For each scene: extract Flash3D gaussians, render to each target view (smeared novel
view), save the render + source image + GT target image as PNGs.

Run (flash3d venv):
  python -m paper_b_generative3d.gen_flash3d_renders \
    --max_scenes 100 --out /home/data/sv3d-lab/reconstruction/E-127-renders
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from paper_b_generative3d.backbone_flash3d import Flash3DBackbone
from paper_b_generative3d.data_loader_re10k import build_flash3d_cfg, FLASH3D_CKPT
from paper_b_generative3d.viz_compare import psnr, _create_loader_from_cfg


def save_png(t, path):
    arr = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_scenes", type=int, default=100)
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

    manifest = []
    n_done = 0
    for i, inputs in enumerate(loader):
        if n_done >= args.max_scenes:
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
            "xyz": g["xyz"][0],
            "scales": g["scales"][0],
            "rotations": g["rotations"][0],
            "opacity": g["opacity"][0],
            "color_rgb": g["color_rgb"][0],
        }
        source_img = inputs["color", 0, 0][0]
        sid = f"{i:06d}"
        save_png(source_img, out / f"{sid}_source.png")

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
                render = backbone.render_gaussians(anchor, cam, K_tgt, H, W)
            save_png(render, out / f"{sid}_f{fid}_smear.png")
            save_png(gt, out / f"{sid}_f{fid}_gt.png")
            manifest.append(
                {
                    "scene": i,
                    "sid": sid,
                    "fid": fid,
                    "smear": f"{sid}_f{fid}_smear.png",
                    "gt": f"{sid}_f{fid}_gt.png",
                    "source": f"{sid}_source.png",
                    "psnr_smear": float(psnr(render, gt)),
                }
            )
        n_done += 1
        if n_done % 20 == 0:
            print(f"[{n_done}/{args.max_scenes}] rendered", flush=True)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"\n=== Stage 0a done: {len(manifest)} target renders from {n_done} scenes ===",
        flush=True,
    )
    print(
        f"mean smear PSNR = {np.mean([m['psnr_smear'] for m in manifest]):.2f} dB",
        flush=True,
    )


if __name__ == "__main__":
    main()
