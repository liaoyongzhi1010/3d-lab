import warnings

warnings.filterwarnings("ignore")
import os, sys, json, glob, math
import numpy as np
import torch
import imageio
from einops import rearrange
from torchvision.transforms.functional import resize

sys.path.insert(0, "/root/projects/Gen3R")
from gen3r.utils.data_utils import center_crop, compute_rays, preprocess_poses
from gen3r.pipeline import Gen3RPipeline

DEVICE = torch.device("cuda")
OUT = "/home/data/E-099_C0_gen3r_fuse"
os.makedirs(OUT, exist_ok=True)

SCENE = "/home/data/gen3r_re10k/re10k/test_0a9f2831a3e73de8"
CKPT = "/root/projects/Gen3R/checkpoints"


def rescale_K_for_crop(K, W, H, target=560):
    scale = target / min(W, H)
    newW, newH = round(W * scale), round(H * scale)
    K = K.copy()
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] *= scale
    K[1, 2] *= scale
    x0 = (newW - target) / 2.0
    y0 = (newH - target) / 2.0
    K[0, 2] -= x0
    K[1, 2] -= y0
    return K


def load_scene_cams(tf, n=49):
    t = json.load(open(tf))
    frames = t["frames"][:n]
    exts, Ks = [], []
    W = frames[0]["w"]
    H = frames[0]["h"]
    for fr in frames:
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        exts.append(np.linalg.inv(c2w))
        K = np.eye(3)
        K[0, 0] = fr["fl_x"]
        K[1, 1] = fr["fl_y"]
        K[0, 2] = fr["cx"]
        K[1, 2] = fr["cy"]
        Ks.append(K)
    return np.array(exts), np.array(Ks), W, H, frames


GEOM_NPZ = os.path.join(OUT, "gen3r_geom.npz")

if not os.path.exists(GEOM_NPZ):
    tf = os.path.join(SCENE, "transforms.json")
    exts, Ks, W, H, frames = load_scene_cams(tf, 49)
    img0 = os.path.join(SCENE, frames[0]["file_path"])
    frame = torch.from_numpy(imageio.v2.imread(img0))[..., :3]
    control_images = (
        frame[None].to(DEVICE, torch.bfloat16).permute(0, 3, 1, 2).unsqueeze(0).float()
        / 255.0
    )
    fh, fw = control_images.shape[3], control_images.shape[4]
    scale = 560 / min(fh, fw)
    nh, nw = round(fh * scale), round(fw * scale)
    control_images = resize(control_images[0], [nh, nw])
    control_images = center_crop(control_images, (560, 560))[None, ...].to(
        DEVICE, torch.bfloat16
    )

    Ks560 = np.array([rescale_K_for_crop(K, W, H, 560) for K in Ks])
    c2ws = np.array([np.linalg.inv(w2c) for w2c in exts])
    c2ws_t = torch.from_numpy(c2ws).float().to(DEVICE)
    c2ws_t = preprocess_poses(c2ws_t)[None, ...]
    Ks_t = torch.from_numpy(Ks560).float()[None].to(DEVICE)
    plk = []
    for i in range(c2ws_t.shape[0]):
        ro, rd = compute_rays(c2ws_t[i], Ks_t[i], h=560, w=560, device=DEVICE)
        plk.append(torch.cat([torch.cross(ro, rd, dim=1), rd], dim=1))
    plucker = torch.stack(plk, dim=0)

    print("loading Gen3R pipeline...", flush=True)
    pipeline = Gen3RPipeline.from_pretrained(CKPT)
    pipeline.to(DEVICE).to(torch.bfloat16)

    with torch.no_grad():
        sample = pipeline(
            prompt="A video walkthrough of an indoor real estate scene.",
            control_cameras=plucker,
            control_images=control_images,
            num_frames=49,
            negative_prompt="bad detailed",
            height=560,
            width=560,
            guidance_scale=5.0,
            return_dict=True,
            min_max_depth_mask=True,
        )
    pcds = sample.pcds[0].float().cpu().numpy()  # [F,H,W,3]
    masks = sample.point_masks[0].cpu().numpy()  # [F,H,W]
    rgbs = sample.rgbs[0].float().cpu().numpy()  # [F,H,W,3]
    extr = sample.cameras[0].squeeze(0).float().cpu().numpy()  # [F,3,4]
    intr = sample.cameras[1].squeeze(0).float().cpu().numpy()  # [F,3,3]
    np.savez_compressed(
        GEOM_NPZ, pcds=pcds, masks=masks, rgbs=rgbs, extr=extr, intr=intr
    )
    print(
        "saved geometry",
        pcds.shape,
        masks.shape,
        rgbs.shape,
        extr.shape,
        intr.shape,
        flush=True,
    )
    del pipeline
    torch.cuda.empty_cache()
else:
    print("geometry exists, loading", flush=True)

d = np.load(GEOM_NPZ)
pcds, masks, rgbs, extr, intr = d["pcds"], d["masks"], d["rgbs"], d["extr"], d["intr"]
F, H, W, _ = pcds.shape
print("F,H,W", F, H, W, "mask mean", masks.mean(), flush=True)

# Build 3DGS from reliable points across a subset of source frames (use 0,12,24 as seed views to fill geometry)
from gsplat import rasterization

seed_frames = [0, 12, 24]
xyz_all, rgb_all = [], []
for f in seed_frames:
    m = masks[f].astype(bool)
    xyz_all.append(pcds[f][m])
    rgb_all.append(rgbs[f][m])
xyz = np.concatenate(xyz_all, 0).astype(np.float32)
rgb = np.concatenate(rgb_all, 0).astype(np.float32)
print("total reliable pts", xyz.shape[0], flush=True)

MAXP = 400000
if xyz.shape[0] > MAXP:
    idx = np.random.RandomState(0).choice(xyz.shape[0], MAXP, replace=False)
    xyz = xyz[idx]
    rgb = rgb[idx]

xyz_t = torch.from_numpy(xyz).to(DEVICE)
rgb_t = torch.from_numpy(rgb).to(DEVICE).clamp(0, 1)
N = xyz_t.shape[0]

# nearest-neighbor scale estimate for gaussian size
with torch.no_grad():
    sub = xyz_t[torch.randperm(N, device=DEVICE)[: min(N, 20000)]]
    dmed = torch.cdist(sub[:2000], sub).topk(4, largest=False).values[:, 1:].mean()
scale_val = float(dmed) * 0.5
print("scale_val", scale_val, flush=True)

scales = torch.full((N, 3), scale_val, device=DEVICE)
quats = torch.zeros((N, 4), device=DEVICE)
quats[:, 0] = 1.0
opac = torch.full((N,), 0.9, device=DEVICE)
colors = rgb_t


def render(fidx):
    e = extr[fidx]  # [3,4] w2c
    vm = np.eye(4, dtype=np.float32)
    vm[:3, :4] = e
    viewmat = torch.from_numpy(vm).to(DEVICE)[None]
    K = torch.from_numpy(intr[fidx].astype(np.float32)).to(DEVICE)[None]
    out, _, _ = rasterization(
        means=xyz_t,
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
    img = out[0].clamp(0, 1).cpu().numpy()
    return (img * 255).astype(np.uint8)


from PIL import Image, ImageDraw

GT = "/home/data/E-069_gen3r_realtraj/test_0a9f2831a3e73de8/gt_{:02d}.png"
F3D = "/home/data/E-069b_triptych/f3d_0a9f2831a3e73de8_{:02d}.png"

import lpips

loss_fn = lpips.LPIPS(net="alex").to(DEVICE)


def psnr(a, b):
    a = np.asarray(a.resize(b.size), dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    return 20 * math.log10(255 / math.sqrt(mse)) if mse > 0 else 99.0


def to_t(img, size):
    img = img.resize(size, Image.LANCZOS)
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
    dd = ImageDraw.Draw(im)
    dd.rectangle([0, 0, im.width, 20], fill=(0, 0, 0))
    dd.text((3, 3), txt, fill=(255, 255, 255))
    return im


metrics = {}
PH = 380
for frame in [36, 48]:
    r = Image.fromarray(render(frame))
    r.save(os.path.join(OUT, f"gen3rfuse_{frame}.png"))
    gt = Image.open(GT.format(frame)).convert("RGB")
    f3d = Image.open(F3D.format(frame)).convert("RGB")
    metrics[frame] = {
        "f3d_psnr": psnr(f3d, gt),
        "f3d_lpips": lp(f3d, gt),
        "gen3rfuse_psnr": psnr(r, gt),
        "gen3rfuse_lpips": lp(r, gt),
    }
    print(frame, metrics[frame], flush=True)
    tiles = []
    for im, name, extra in [
        (gt, "GT", ""),
        (
            f3d,
            "Flash3D",
            f"P{metrics[frame]['f3d_psnr']:.1f} L{metrics[frame]['f3d_lpips']:.3f}",
        ),
        (
            r,
            "Gen3R-Fuse(3DGS)",
            f"P{metrics[frame]['gen3rfuse_psnr']:.1f} L{metrics[frame]['gen3rfuse_lpips']:.3f}",
        ),
    ]:
        t = im.resize((PH, PH), Image.LANCZOS)
        tiles.append(label(t, f"{name} f{frame} {extra}"))
    panel = Image.new("RGB", (PH * 3, PH), (255, 255, 255))
    for i, t in enumerate(tiles):
        panel.paste(t, (i * PH, 0))
    panel.save(os.path.join(OUT, f"panel_frame{frame}.png"))
    panel.convert("RGB").save(os.path.join(OUT, f"panel_frame{frame}.jpg"), quality=92)
    print("panel saved", frame, flush=True)

json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
print("DONE", flush=True)
