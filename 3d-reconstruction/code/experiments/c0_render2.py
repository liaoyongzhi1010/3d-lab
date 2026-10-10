"""C0 Gen3R-Fuse render: build 3DGS from Gen3R reliable pcds using VGGT camera frame,
render frames 36/48 with the matching VGGT cameras. Runs in dr3d env (gsplat)."""

import os, math
import numpy as np
import torch
from PIL import Image, ImageDraw
from gsplat import rasterization

OUT = "/home/data/E-099_C0_gen3r_fuse"
SCENE = "0a9f2831a3e73de8"
GTDIR = "/home/data/E-069_gen3r_realtraj/test_%s" % SCENE
F3DDIR = "/home/data/E-069b_triptych"
DEVICE = "cuda"

pcds = np.load(os.path.join(OUT, "pcds.npy"))[0]  # [F,H,W,3]
masks = np.load(os.path.join(OUT, "point_masks.npy"))[0]  # [F,H,W]
rgbs = np.load(os.path.join(OUT, "rgbs.npy"))[0]  # [F,H,W,3] 0..1
extr = np.load(os.path.join(OUT, "vggt_extrinsics.npy"))[0]  # [F,3,4] w2c
intr = np.load(os.path.join(OUT, "vggt_intrinsics.npy"))[0]  # [F,3,3]
F, H, W, _ = pcds.shape
print("F,H,W", F, H, W, "mask mean", masks.mean())


def build(seed_frames):
    xs, cs, ss = [], [], []
    for f in seed_frames:
        m = masks[f].reshape(-1) > 0.5
        pw = pcds[f].reshape(-1, 3)[m]
        xs.append(pw)
        cs.append(rgbs[f].reshape(-1, 3)[m])
        # per-point scale = distance-to-camera * (1/focal) => projected pixel footprint
        cam_c = -extr[f][:3, :3].T @ extr[f][:3, 3]
        dist = np.linalg.norm(pw - cam_c[None], axis=1)
        fx = intr[f][0, 0]
        ss.append(dist / fx)
    return (
        np.concatenate(xs, 0).astype(np.float32),
        np.concatenate(cs, 0).astype(np.float32),
        np.concatenate(ss, 0).astype(np.float32),
    )


seed = [0, 6, 12, 18, 24, 30]
xyz, rgb, sc = build(seed)
print("reliable pts", xyz.shape[0])
MAXP = 1200000
if xyz.shape[0] > MAXP:
    idx = np.random.RandomState(0).choice(xyz.shape[0], MAXP, replace=False)
    xyz, rgb, sc = xyz[idx], rgb[idx], sc[idx]

means = torch.from_numpy(xyz).to(DEVICE)
colors = torch.from_numpy(rgb).to(DEVICE).clamp(0, 1)
N = means.shape[0]

scale_pp = torch.from_numpy(sc).to(DEVICE) * 0.8
scales = scale_pp[:, None].repeat(1, 3)
base_scale = float(scale_pp.median())
quats = torch.zeros((N, 4), device=DEVICE)
quats[:, 0] = 1.0
opac = torch.full((N,), 0.95, device=DEVICE)
print("base_scale", base_scale, "N", N)


def render(fidx):
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :4] = extr[fidx]
    viewmat = torch.from_numpy(vm).to(DEVICE)[None]
    K = torch.from_numpy(intr[fidx].astype(np.float32)).to(DEVICE)[None]
    out, _, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opac,
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=W,
        height=H,
        render_mode="RGB",
        packed=False,
    )
    return (out[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


import lpips

loss_fn = lpips.LPIPS(net="alex").to(DEVICE)


def psnr(a, b):
    a = np.asarray(a.resize(b.size), np.float64)
    b = np.asarray(b, np.float64)
    mse = np.mean((a - b) ** 2)
    return 20 * math.log10(255 / math.sqrt(mse)) if mse > 0 else 99.0


def to_t(img, s):
    img = img.resize(s, Image.LANCZOS)
    return (
        torch.from_numpy(np.asarray(img).astype(np.float32) / 127.5 - 1)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
    )


def lp(a, b):
    with torch.no_grad():
        return float(loss_fn(to_t(a, (256, 256)), to_t(b, (256, 256))).item())


def label(im, txt):
    im = im.copy()
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 20], fill=(0, 0, 0))
    d.text((3, 3), txt, fill=(255, 255, 255))
    return im


# sanity: render a seed frame (0) and compare to gen rgb -- should align well
r0 = Image.fromarray(render(0))
r0.save(os.path.join(OUT, "sanity_render_00.png"))

import json

metrics = {}
PH = 380
for frame in [36, 48]:
    r = Image.fromarray(render(frame))
    r.save(os.path.join(OUT, "gen3rfuse_%02d.png" % frame))
    gt = Image.open(os.path.join(GTDIR, "gt_%02d.png" % frame)).convert("RGB")
    f3d = Image.open(os.path.join(F3DDIR, "f3d_%s_%02d.png" % (SCENE, frame))).convert(
        "RGB"
    )
    metrics[frame] = {
        "f3d_psnr": psnr(f3d, gt),
        "f3d_lpips": lp(f3d, gt),
        "gen3rfuse_psnr": psnr(r, gt),
        "gen3rfuse_lpips": lp(r, gt),
    }
    print(frame, metrics[frame])
    tiles = []
    for im, name, ex in [
        (gt, "GT", ""),
        (
            f3d,
            "Flash3D",
            "P%.1f L%.3f" % (metrics[frame]["f3d_psnr"], metrics[frame]["f3d_lpips"]),
        ),
        (
            r,
            "Gen3R-Fuse(3DGS)",
            "P%.1f L%.3f"
            % (metrics[frame]["gen3rfuse_psnr"], metrics[frame]["gen3rfuse_lpips"]),
        ),
    ]:
        tiles.append(
            label(im.resize((PH, PH), Image.LANCZOS), "%s f%d %s" % (name, frame, ex))
        )
    panel = Image.new("RGB", (PH * 3, PH), (255, 255, 255))
    for i, t in enumerate(tiles):
        panel.paste(t, (i * PH, 0))
    panel.save(os.path.join(OUT, "panel_frame%d.png" % frame))
    panel.convert("RGB").save(
        os.path.join(OUT, "panel_frame%d.jpg" % frame), quality=92
    )
    print("panel saved", frame)

json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
print("DONE C0 render")
