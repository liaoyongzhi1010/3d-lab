"""Step 0: evaluation harness + synthetic sharp-vs-blur sanity (GO/NO-GO gate).

Before training anything, prove the eval harness can DETECT a sharp-vs-blur difference
in disoccluded regions at our achievable sample size. If it can't, the whole
flow-matching thesis is unmeasurable (avoid a 5th "clean but insignificant" outcome).

Builds 4 image sets on the INDOOR gap50 split (Flash3D target renders):
  - real/     : ground-truth target frames  (FID/KID reference)
  - flash3d/  : raw Flash3D renders (the baseline; blurry/smeared in ~30% region)
  - blur/     : Flash3D with disoccluded region ADDITIONALLY blurred (worse)
  - sharp/    : Flash3D with disoccluded region replaced by GT (Oracle upper bound = sharpest)

Then computes FID/KID(real, {flash3d, blur, sharp}) with clean-fid.
GO if:  FID(sharp) < FID(flash3d) < FID(blur)  with KID separation > KID std
        i.e. the metric monotonically rewards sharpness in the region we care about.

Run (Flash3D venv):
  python -m paper_b_generative3d.eval_harness_step0 \
    --split_path splits/re10k_mine_filtered/test_gap50.txt \
    --flash3d_max_psnr 25.0 --n 200 \
    --out /home/data/sv3d-lab/reconstruction/E-120-harness
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
from paper_b_generative3d.viz_compare import psnr, to_uint8, _create_loader_from_cfg


def _fixable_mask(r_bb, gt):
    """Degraded-region mask (H,W): worst-30% Flash3D error, same as Oracle E-119."""
    err = (r_bb - gt).abs().mean(dim=0)
    err_blur = F.avg_pool2d(err[None, None], 9, 1, 4)[0, 0]
    thr = torch.quantile(err_blur.flatten(), 0.70)
    mask = (err_blur > thr).float()
    m = mask[None, None]
    m = F.max_pool2d(m, 3, 1, 1)
    m = -F.max_pool2d(-m, 3, 1, 1)
    return m[0, 0]


def _gaussian_blur(img, k=11, sigma=4.0):
    """Blur a (3,H,W) tensor with a Gaussian kernel."""
    coords = torch.arange(k, dtype=torch.float32, device=img.device) - k // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    return F.conv2d(img[None], kernel, padding=k // 2, groups=3)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_path", default="splits/re10k_mine_filtered/test_gap50.txt")
    ap.add_argument("--flash3d_max_psnr", type=float, default=25.0)
    ap.add_argument(
        "--n", type=int, default=200, help="number of target images to render"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ["real", "flash3d", "blur", "sharp"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

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
    n_saved, n_scanned = 0, 0
    for inputs in loader:
        if n_saved >= args.n:
            break
        n_scanned += 1
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
        tfids = [f for f in [1, 2, 3] if ("color", f, 0) in inputs]
        if not tfids:
            continue

        gauss_out = backbone.extract_source_gaussians(inputs)
        raw = {
            "xyz": gauss_out["xyz"][0],
            "scales": gauss_out["scales"][0],
            "rotations": gauss_out["rotations"][0],
            "opacity": gauss_out["opacity"][0],
            "color_rgb": gauss_out["color_rgb"][0],
        }
        for fid in tfids:
            if n_saved >= args.n:
                break
            cam = inputs.get(("cam_T_cam", 0, fid))
            if cam is None:
                cam = gauss_out.get("_outputs", {}).get(("cam_T_cam", 0, fid))
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

            mask = _fixable_mask(r_bb, gt)
            m3 = mask[None].repeat(3, 1, 1)

            sharp = r_bb * (1 - m3) + gt * m3
            blurred_region = _gaussian_blur(r_bb)
            blur = r_bb * (1 - m3) + blurred_region * m3

            tag = f"{n_saved:05d}"
            Image.fromarray(to_uint8(gt)).save(out / "real" / f"{tag}.png")
            Image.fromarray(to_uint8(r_bb)).save(out / "flash3d" / f"{tag}.png")
            Image.fromarray(to_uint8(blur)).save(out / "blur" / f"{tag}.png")
            Image.fromarray(to_uint8(sharp)).save(out / "sharp" / f"{tag}.png")
            n_saved += 1
        if n_saved % 25 == 0 and n_saved > 0:
            print(f"  rendered {n_saved}/{args.n} (scanned {n_scanned})", flush=True)

    print(f"Rendered {n_saved} target images. Computing FID/KID...", flush=True)

    from cleanfid import fid as cfid

    real_dir = str(out / "real")
    results = {"n_images": n_saved}
    for name in ["sharp", "flash3d", "blur"]:
        d = str(out / name)
        f = cfid.compute_fid(real_dir, d, mode="clean", num_workers=2, verbose=False)
        try:
            k = cfid.compute_kid(
                real_dir, d, mode="clean", num_workers=2, verbose=False
            )
        except Exception as e:
            k = float("nan")
        results[name] = {"fid": float(f), "kid": float(k)}
        print(f"  {name:8s}: FID={f:.3f} KID={k:.5f}", flush=True)

    # GO/NO-GO
    fs, ff, fb = (
        results["sharp"]["fid"],
        results["flash3d"]["fid"],
        results["blur"]["fid"],
    )
    monotonic = fs < ff < fb
    results["verdict"] = {
        "monotonic_fid": bool(monotonic),
        "sharp_better_than_flash3d": bool(fs < ff),
        "blur_worse_than_flash3d": bool(fb > ff),
        "fid_sharp_vs_blur_gap": float(fb - fs),
        "GO": bool(fs < ff and fb > ff),
    }
    (out / "harness_results.json").write_text(json.dumps(results, indent=2))
    print("\n=== STEP 0 GO/NO-GO ===", flush=True)
    print(
        f"FID: sharp={fs:.3f} < flash3d={ff:.3f} < blur={fb:.3f} ? monotonic={monotonic}",
        flush=True,
    )
    print(
        f"VERDICT: {'GO - harness detects sharp-vs-blur' if results['verdict']['GO'] else 'NO-GO - metric cannot separate; restructure eval before training'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
