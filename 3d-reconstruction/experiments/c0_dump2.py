"""C0 Gen3R-Fuse: re-dump saving VGGT-estimated cameras (the frame the pcds live in).
Runs in .venv_gen3r."""

import warnings

warnings.filterwarnings("ignore")
import os, sys, json, math
import numpy as np
import torch
import imageio
from torchvision.transforms.functional import resize

sys.path.insert(0, "/root/projects/Gen3R")
from gen3r.utils.data_utils import center_crop, compute_rays, preprocess_poses
from gen3r.pipeline import Gen3RPipeline

DEVICE = torch.device("cuda")
SCENE = "0a9f2831a3e73de8"
DATA_ROOT = "/home/data/gen3r_re10k/re10k"
CKPT = "/root/projects/Gen3R/checkpoints"
OUT = "/home/data/E-099_C0_gen3r_fuse"
os.makedirs(OUT, exist_ok=True)


def rescale_K_for_crop(K, W, H, target=560):
    scale = target / min(W, H)
    newW, newH = round(W * scale), round(H * scale)
    K = K.copy()
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] *= scale
    K[1, 2] *= scale
    K[0, 2] -= (newW - target) / 2.0
    K[1, 2] -= (newH - target) / 2.0
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


scene = os.path.join(DATA_ROOT, "test_%s" % SCENE)
tf = os.path.join(scene, "transforms.json")
exts, Ks, W, H, frames = load_scene_cams(tf, 49)

pipeline = Gen3RPipeline.from_pretrained(CKPT)
pipeline.to(DEVICE).to(torch.bfloat16)

img0 = os.path.join(scene, frames[0]["file_path"])
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
c2ws_t = preprocess_poses(torch.from_numpy(c2ws).float().to(DEVICE))[None, ...]
Ks_t = torch.from_numpy(Ks560).float()[None].to(DEVICE)
plk = []
for i in range(c2ws_t.shape[0]):
    ro, rd = compute_rays(c2ws_t[i], Ks_t[i], h=560, w=560, device=DEVICE)
    plk.append(torch.cat([torch.cross(ro, rd, dim=1), rd], dim=1))
plucker = torch.stack(plk, dim=0)

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


def tn(x):
    return x.float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


pcds = tn(sample.pcds)
masks = tn(sample.point_masks)
rgbs = tn(sample.rgbs)
vggt_extr = tn(sample.cameras[0])  # [B,F,3,4] w2c OpenCV
vggt_intr = tn(sample.cameras[1])  # [B,F,3,3]
print("pcds", pcds.shape, "extr", vggt_extr.shape, "intr", vggt_intr.shape)

np.save(os.path.join(OUT, "pcds.npy"), pcds)
np.save(os.path.join(OUT, "point_masks.npy"), masks)
np.save(os.path.join(OUT, "rgbs.npy"), rgbs)
np.save(
    os.path.join(OUT, "vggt_extrinsics.npy"), vggt_extr
)  # THE camera frame pcds live in
np.save(os.path.join(OUT, "vggt_intrinsics.npy"), vggt_intr)
print("DUMP2 DONE with VGGT cameras")
