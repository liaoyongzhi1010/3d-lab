import os, sys, math

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

import torch, numpy as np
from PIL import Image, ImageDraw

OUT = "/home/data/E-099_C2_difix_splat"
os.makedirs(OUT, exist_ok=True)

import diffusers

print("diffusers", diffusers.__version__)

from pipeline_difix import DifixPipeline

pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe = pipe.to("cuda")
print("pipeline loaded")

GT = "/home/data/E-069_gen3r_realtraj/test_0a9f2831a3e73de8/gt_{:02d}.png"
F3D = "/home/data/E-069b_triptych/f3d_0a9f2831a3e73de8_{:02d}.png"
CLEAN = "/home/data/E-069_gen3r_realtraj/test_0a9f2831a3e73de8/gt_00.png"

ref = Image.open(CLEAN).convert("RGB")


def to512(img):
    return img.resize((512, 512), Image.LANCZOS)


results = {}
for frame in [36, 48]:
    deg = Image.open(F3D.format(frame)).convert("RGB")
    W, H = deg.size
    deg_in = to512(deg)
    ref_in = to512(ref)
    with torch.no_grad():
        out = pipe(
            prompt="remove degradation, sharp clean photo",
            image=deg_in,
            ref_image=ref_in,
            num_inference_steps=1,
            timesteps=[199],
            guidance_scale=0.0,
            height=512,
            width=512,
        ).images[0]
    out = out.resize((W, H), Image.LANCZOS)
    out.save(os.path.join(OUT, f"difix_refined_{frame}.png"))
    results[frame] = out
    print("refined frame", frame)

import lpips

loss_fn = lpips.LPIPS(net="alex").cuda()


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


metrics = {}
for frame in [36, 48]:
    gt = Image.open(GT.format(frame)).convert("RGB")
    f3d = Image.open(F3D.format(frame)).convert("RGB")
    ref_out = results[frame]
    metrics[frame] = {
        "f3d_psnr": psnr(f3d, gt),
        "f3d_lpips": lp(f3d, gt),
        "difix_psnr": psnr(ref_out, gt),
        "difix_lpips": lp(ref_out, gt),
    }
    print(frame, metrics[frame])


def label(img, txt):
    img = img.copy()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    d.text((4, 4), txt, fill=(255, 255, 255))
    return img


PH = 400
for frame in [36, 48]:
    gt = Image.open(GT.format(frame)).convert("RGB")
    f3d = Image.open(F3D.format(frame)).convert("RGB")
    rf = results[frame]
    tiles = []
    for img, name, extra in [
        (gt, "GT", ""),
        (
            f3d,
            "Flash3D",
            f"PSNR {metrics[frame]['f3d_psnr']:.2f} LPIPS {metrics[frame]['f3d_lpips']:.3f}",
        ),
        (
            rf,
            "Difix-refined",
            f"PSNR {metrics[frame]['difix_psnr']:.2f} LPIPS {metrics[frame]['difix_lpips']:.3f}",
        ),
    ]:
        im = img.resize((PH, PH), Image.LANCZOS)
        im = label(im, f"{name} f{frame}  {extra}")
        tiles.append(im)
    panel = Image.new("RGB", (PH * 3, PH), (255, 255, 255))
    for i, t in enumerate(tiles):
        panel.paste(t, (i * PH, 0))
    panel.save(os.path.join(OUT, f"panel_frame{frame}.png"))
    panel.convert("RGB").save(os.path.join(OUT, f"panel_frame{frame}.jpg"), quality=92)
    print("panel saved frame", frame)

import json

json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
print("DONE")
