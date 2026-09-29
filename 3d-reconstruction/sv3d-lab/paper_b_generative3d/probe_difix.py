"""Stage 0 Probe: Does Difix actually fix Flash3D smear?

Load nvidia/difix, render Flash3D at target views (produces smear), feed to Difix
with source image as reference, measure PSNR vs GT.

If Difix(smeared_render, source_ref) > smeared_render (PSNR vs GT), the approach works.
This is the make-or-break feasibility test before building the full training pipeline.

Run:
  python -m paper_b_generative3d.probe_difix \
    --max_scenes 20 --out /home/data/sv3d-lab/reconstruction/E-127-difix-probe
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
from paper_b_generative3d.viz_compare import psnr, _create_loader_from_cfg


def load_difix_pipeline(device):
    """Load the real DifixPipeline from /root/difix_src (native diffusers format).

    The nvidia/difix HF cache is a full DifixPipeline (model_index.json names it).
    We import the custom class and use from_pretrained so all weights (incl. custom
    VAE skip convs) load correctly. sd-turbo is downloaded once as needed.
    """
    import sys
    import os

    sys.path.insert(0, "/root/difix_src")
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    # Compat shim: nvidia/difix bundles custom modules written for diffusers 0.25.1
    # which import FromOriginalVAEMixin (renamed to FromOriginalModelMixin in 0.32).
    import diffusers.loaders as _dl

    if not hasattr(_dl, "FromOriginalVAEMixin"):
        _dl.FromOriginalVAEMixin = _dl.FromOriginalModelMixin

    from pipeline_difix import DifixPipeline

    pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def tensor_to_pil(t):
    """[3,H,W] float [0,1] -> PIL Image."""
    from PIL import Image

    arr = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_tensor(img, device):
    """PIL Image -> [3,H,W] float [0,1]."""
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_scenes", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=0.3)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    cfg = build_flash3d_cfg(batch_size=1, num_workers=0, stage="dev")
    backbone = Flash3DBackbone(cfg, FLASH3D_CKPT, device=str(device))
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    print("Loading Difix pipeline...", flush=True)
    pipe = load_difix_pipeline(device)
    print("Difix loaded.", flush=True)

    _, loader = _create_loader_from_cfg(cfg)
    H, W = 256, 384

    results = []
    for i, inputs in enumerate(loader):
        if i >= args.max_scenes:
            break
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
        tfids = [f for f in [1, 2, 3] if ("color", f, 0) in inputs]
        if not tfids:
            continue
        T_c2w_s = inputs.get(("T_c2w", 0))
        if T_c2w_s is None:
            continue

        with torch.no_grad():
            gauss = backbone.extract_source_gaussians(inputs)
        anchor = {
            "xyz": gauss["xyz"][0],
            "scales": gauss["scales"][0],
            "rotations": gauss["rotations"][0],
            "opacity": gauss["opacity"][0],
            "color_rgb": gauss["color_rgb"][0],
        }
        source_img = inputs["color", 0, 0][0]  # [3,H,W] reference for Difix

        for fid in tfids[:1]:  # just first target per scene for probe speed
            T_w2c_t = inputs.get(("T_w2c", fid))
            if T_w2c_t is None:
                continue
            cam = T_w2c_t[0] @ T_c2w_s[0]
            K_tgt = inputs.get(("K_tgt", fid), inputs[("K_src", 0)])
            if K_tgt.dim() == 3:
                K_tgt = K_tgt[0]
            gt = inputs["color", fid, 0][0]

            with torch.no_grad():
                render_flash3d = backbone.render_gaussians(anchor, cam, K_tgt, H, W)

            psnr_before = psnr(render_flash3d, gt)

            # Run Difix pipeline: fixes artifacts in the smeared render (1-step)
            smear_pil = tensor_to_pil(render_flash3d)
            with torch.no_grad():
                fixed_pil = pipe(
                    prompt="",
                    image=smear_pil,
                    height=512,
                    width=512,
                    num_inference_steps=1,
                    timesteps=[199],
                    guidance_scale=0.0,
                ).images[0]
            fixed_t = pil_to_tensor(fixed_pil.resize((W, H)), device)
            psnr_after = psnr(fixed_t, gt)
            delta = psnr_after - psnr_before

            results.append(
                {
                    "scene": i,
                    "fid": fid,
                    "psnr_before": float(psnr_before),
                    "psnr_after": float(psnr_after),
                    "delta": float(delta),
                }
            )
            if (i + 1) % 5 == 0:
                ds = [r["delta"] for r in results]
                print(
                    f"[{i + 1}/{args.max_scenes}] mean_delta={np.mean(ds):+.3f}dB "
                    f"psnr_before={np.mean([r['psnr_before'] for r in results]):.2f} "
                    f"psnr_after={np.mean([r['psnr_after'] for r in results]):.2f}",
                    flush=True,
                )

    ds = [r["delta"] for r in results]
    summary = {
        "mean_delta_dB": float(np.mean(ds)) if ds else 0,
        "mean_psnr_before": float(np.mean([r["psnr_before"] for r in results])),
        "mean_psnr_after": float(np.mean([r["psnr_after"] for r in results])),
        "pct_positive": float(np.mean([d > 0 for d in ds]) * 100),
        "n": len(ds),
        "strength": args.strength,
        "GO": bool(np.mean(ds) > 0.5) if ds else False,
    }
    (out / "probe_result.json").write_text(
        json.dumps({"summary": summary, "per_sample": results}, indent=2)
    )
    print(f"\n{'=' * 60}", flush=True)
    print(f"DIFIX PROBE RESULT (strength={args.strength}):", flush=True)
    print(f"  Before (Flash3D): {summary['mean_psnr_before']:.2f} dB", flush=True)
    print(f"  After (Difix):    {summary['mean_psnr_after']:.2f} dB", flush=True)
    print(f"  Delta:            {summary['mean_delta_dB']:+.3f} dB", flush=True)
    print(f"  % improved:       {summary['pct_positive']:.1f}%", flush=True)
    print(f"  GO: {'YES' if summary['GO'] else 'NO'} (threshold: +0.5dB)", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
