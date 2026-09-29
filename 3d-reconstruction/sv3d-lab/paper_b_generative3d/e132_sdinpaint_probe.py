"""E-132 probe: can SD-inpaint fill the novel-view smear region with plausible content
(doorframe + water dispenser) on scene006?

This is a FEASIBILITY probe (uses GT-error to define the mask, which is NOT
inference-legal -- just to test if the generative prior CAN fill plausibly). If it works,
a later method will derive the mask from source-visibility geometry (inference-legal).

Runs in dr3d env (diffusers 0.20.2, runwayml/stable-diffusion-inpainting cached).
Reads pre-rendered scene6 images from E-130-waterbottle-diag.

Run (dr3d env):
  python e132_sdinpaint_probe.py \
    --renders /home/data/sv3d-lab/reconstruction/E-130-waterbottle-diag \
    --out /home/data/sv3d-lab/reconstruction/E-132-inpaint-probe
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

sys.path.insert(0, "/root/difix_libs")

import importlib.util

_orig = importlib.util.find_spec


def _blk(name, *a, **k):
    if name == "bitsandbytes" or name.startswith("bitsandbytes."):
        return None
    return _orig(name, *a, **k)


importlib.util.find_spec = _blk

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fid", type=int, default=3)
    args = ap.parse_args()
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    out = args.out
    os.makedirs(out, exist_ok=True)

    from diffusers import StableDiffusionInpaintPipeline

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print("SD-inpaint loaded", flush=True)

    r = args.renders
    flash = Image.open(os.path.join(r, f"f{args.fid}_flash.png")).convert("RGB")
    gt = Image.open(os.path.join(r, f"f{args.fid}_gt.png")).convert("RGB")
    W, H = flash.size
    print(f"image size {W}x{H}", flush=True)

    # Probe mask = high flash-vs-gt error (the smear/disocc region). NOT inference-legal.
    fa = np.asarray(flash).astype(np.float32)
    ga = np.asarray(gt).astype(np.float32)
    err = np.abs(fa - ga).mean(2)  # [H,W]
    thr = np.percentile(err, 80)  # top 20% error = smear region
    mask = (err > thr).astype(np.uint8) * 255
    # dilate mask a bit
    from PIL import ImageFilter

    mask_img = Image.fromarray(mask).filter(ImageFilter.MaxFilter(9))
    mask_img.save(os.path.join(out, f"f{args.fid}_mask.png"))

    # SD-inpaint works at 512; resize
    def to512(im):
        return im.resize((512, 512), Image.LANCZOS)

    flash512 = to512(flash)
    mask512 = to512(mask_img)

    for prompt in ["a photo of a sunroom interior, sharp, realistic", "empty"]:
        gen = torch.Generator("cuda").manual_seed(0)
        with torch.no_grad():
            res = pipe(
                prompt=prompt,
                image=flash512,
                mask_image=mask512,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=gen,
            ).images[0]
        res = res.resize((W, H), Image.LANCZOS)
        tag = "prompt" if prompt != "empty" else "noprompt"
        res.save(os.path.join(out, f"f{args.fid}_inpaint_{tag}.png"))
        # right crop for inspection
        x0 = int(W * 0.6)
        res.crop((x0, 0, W, H)).resize((int((W - x0) * 3), H * 3), Image.NEAREST).save(
            os.path.join(out, f"f{args.fid}_inpaint_{tag}_crop.png")
        )
        # metrics vs gt
        ra = np.asarray(res).astype(np.float64)
        mse = np.mean((ra - ga.astype(np.float64)) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 99
        print(f"prompt='{prompt[:20]}': inpaint PSNR vs GT = {psnr:.2f}", flush=True)

    # also save flash & gt crops for side-by-side
    x0 = int(W * 0.6)
    flash.crop((x0, 0, W, H)).resize((int((W - x0) * 3), H * 3), Image.NEAREST).save(
        os.path.join(out, f"f{args.fid}_flash_crop.png")
    )
    gt.crop((x0, 0, W, H)).resize((int((W - x0) * 3), H * 3), Image.NEAREST).save(
        os.path.join(out, f"f{args.fid}_gt_crop.png")
    )
    print(f"=== saved to {out} ===", flush=True)


if __name__ == "__main__":
    main()
