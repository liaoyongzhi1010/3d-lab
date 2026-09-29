"""Stage 0b: Run Difix on Flash3D smear renders -> sharp pseudo-GT.

Runs in the dr3d conda env with /root/difix_libs (diffusers 0.25.1, peft 0.7.1).
Reads the (smear, source, gt) triples from Stage 0a, runs Difix single-step fix
with the source image as reference, saves fixed pseudo-GT, and measures LPIPS/PSNR
vs GT (LPIPS is the right metric — Difix improves perceptual quality with slight
spatial shift, so PSNR may be flat).

Run (dr3d env):
  source activate dr3d  # or the correct activation
  python gen_difix_pseudogt.py \
    --renders /home/data/sv3d-lab/reconstruction/E-127-renders \
    --out /home/data/sv3d-lab/reconstruction/E-127-difix-pgt
"""

import os
import sys
import json
import math
import argparse

sys.path.insert(0, "/root/difix_libs")
sys.path.insert(0, "/root/difix_src")

import importlib.util

_orig_find_spec = importlib.util.find_spec


def _blocked_find_spec(name, *a, **k):
    if name == "bitsandbytes" or name.startswith("bitsandbytes."):
        return None
    return _orig_find_spec(name, *a, **k)


importlib.util.find_spec = _blocked_find_spec
import peft.import_utils as _piu

_piu.is_bnb_available = lambda: False
_piu.is_bnb_4bit_available = lambda: False

import torch
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=100000)
    args = ap.parse_args()

    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    out = args.out
    os.makedirs(out, exist_ok=True)
    renders = args.renders
    manifest = json.load(open(os.path.join(renders, "manifest.json")))

    from pipeline_difix import DifixPipeline

    pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print("Difix pipeline loaded", flush=True)

    import lpips

    loss_fn = lpips.LPIPS(net="vgg").cuda()

    def to512(img):
        return img.resize((512, 512), Image.LANCZOS)

    def psnr(a, b):
        a = np.asarray(a.resize(b.size), dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        mse = np.mean((a - b) ** 2)
        return 20 * math.log10(255.0 / math.sqrt(mse)) if mse > 0 else 99.0

    def to_t(img, size):
        img = img.resize(size, Image.LANCZOS)
        return (
            torch.from_numpy(np.asarray(img).astype(np.float32) / 127.5 - 1)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .cuda()
        )

    def lp(a, b):
        with torch.no_grad():
            return float(loss_fn(to_t(a, (256, 256)), to_t(b, (256, 256))).item())

    results = []
    for idx, m in enumerate(manifest):
        if idx >= args.max:
            break
        smear = Image.open(os.path.join(renders, m["smear"])).convert("RGB")
        gt = Image.open(os.path.join(renders, m["gt"])).convert("RGB")
        source = Image.open(os.path.join(renders, m["source"])).convert("RGB")
        W, H = smear.size
        with torch.no_grad():
            fixed = pipe(
                prompt="remove degradation, sharp clean photo",
                image=to512(smear),
                ref_image=to512(source),
                num_inference_steps=1,
                timesteps=[199],
                guidance_scale=0.0,
                height=512,
                width=512,
            ).images[0]
        fixed = fixed.resize((W, H), Image.LANCZOS)
        fixed_name = m["smear"].replace("_smear.png", "_difix.png")
        fixed.save(os.path.join(out, fixed_name))

        r = {
            "sid": m["sid"],
            "fid": m["fid"],
            "psnr_smear": psnr(smear, gt),
            "psnr_difix": psnr(fixed, gt),
            "lpips_smear": lp(smear, gt),
            "lpips_difix": lp(fixed, gt),
            "difix": fixed_name,
            "gt": m["gt"],
            "source": m["source"],
            "smear": m["smear"],
        }
        r["psnr_delta"] = r["psnr_difix"] - r["psnr_smear"]
        r["lpips_delta"] = r["lpips_difix"] - r["lpips_smear"]  # negative = better
        results.append(r)
        if (idx + 1) % 20 == 0:
            print(
                f"[{idx + 1}/{len(manifest)}] "
                f"LPIPS {np.mean([x['lpips_smear'] for x in results]):.4f}->"
                f"{np.mean([x['lpips_difix'] for x in results]):.4f} "
                f"PSNR {np.mean([x['psnr_smear'] for x in results]):.2f}->"
                f"{np.mean([x['psnr_difix'] for x in results]):.2f}",
                flush=True,
            )

    summary = {
        "n": len(results),
        "lpips_smear": float(np.mean([x["lpips_smear"] for x in results])),
        "lpips_difix": float(np.mean([x["lpips_difix"] for x in results])),
        "lpips_delta": float(np.mean([x["lpips_delta"] for x in results])),
        "lpips_pct_improved": float(
            np.mean([x["lpips_delta"] < 0 for x in results]) * 100
        ),
        "psnr_smear": float(np.mean([x["psnr_smear"] for x in results])),
        "psnr_difix": float(np.mean([x["psnr_difix"] for x in results])),
        "GO": bool(np.mean([x["lpips_delta"] for x in results]) < -0.005),
    }
    json.dump(
        {"summary": summary, "per_sample": results},
        open(os.path.join(out, "difix_pgt_result.json"), "w"),
        indent=2,
    )
    print("\n" + "=" * 60, flush=True)
    print("STAGE 0b DIFIX PSEUDO-GT RESULT:", flush=True)
    print(
        f"  LPIPS(VGG): {summary['lpips_smear']:.4f} -> {summary['lpips_difix']:.4f} "
        f"(delta {summary['lpips_delta']:+.4f}, {summary['lpips_pct_improved']:.0f}% improved)",
        flush=True,
    )
    print(
        f"  PSNR:       {summary['psnr_smear']:.2f} -> {summary['psnr_difix']:.2f}",
        flush=True,
    )
    print(
        f"  GO (LPIPS improves >0.005): {'YES' if summary['GO'] else 'NO'}", flush=True
    )
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
