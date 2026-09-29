"""Diagnostic: dump intermediate geometry for ONE scene to find the render bug.
Compares our backproject+render against Flash3D's own model forward on the same input."""

import sys
import torch

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, "/root/sv3d-lab/paper_a_explicit3d/oracle")

from hydra import compose, initialize_config_dir
from oracle_core import (
    backproject,
    make_gaussians,
    set_scale_from_depth,
    render_gaussians_relpose,
    psnr,
    crop5,
)

device = "cuda"

with initialize_config_dir(
    config_dir="/root/projects/flash3d/configs", version_base=None
):
    cfg = compose(
        config_name="config",
        overrides=[
            "+experiment=layered_re10k",
            "+dataset.crop_border=true",
            "dataset.test_split_path=splits/re10k_mine_filtered/test_files_present.txt",
            "model.depth.version=v1",
        ],
    )

from datasets.re10k import Re10KDataset

ds = Re10KDataset(cfg, split="test")
inputs = ds[0]

color_src = inputs[("color", 0, 0)].to(device)
K_src = inputs[("K_tgt", 0)].to(device)
H, W = color_src.shape[1:]
print(f"H={H} W={W}")
print(f"K_src=\n{K_src}")
print(f"color range: {color_src.min():.3f} - {color_src.max():.3f}")

unidepth = (
    torch.hub.load(
        "lpiccinelli-eth/UniDepth",
        "UniDepth",
        version="v1",
        backbone="vitl14",
        pretrained=True,
        trust_repo=True,
    )
    .to(device)
    .eval()
)
with torch.no_grad():
    d = unidepth.infer(color_src.unsqueeze(0), intrinsics=K_src.unsqueeze(0))
    depth_src = d["depth"].squeeze()
print(
    f"depth shape={depth_src.shape} range={depth_src.min():.3f}-{depth_src.max():.3f} mean={depth_src.mean():.3f}"
)

inv_K = torch.linalg.inv(K_src)
pts_cam = backproject(depth_src, inv_K, device)
print(
    f"pts_cam shape={pts_cam.shape} x:[{pts_cam[:, 0].min():.2f},{pts_cam[:, 0].max():.2f}] "
    f"y:[{pts_cam[:, 1].min():.2f},{pts_cam[:, 1].max():.2f}] z:[{pts_cam[:, 2].min():.2f},{pts_cam[:, 2].max():.2f}]"
)

rgb = color_src.permute(1, 2, 0).reshape(-1, 3)
g = make_gaussians(pts_cam, rgb, K_src[0, 0].item())
set_scale_from_depth(g, depth_src.reshape(-1), K_src[0, 0].item())
print(
    f"scaling(log) range: {g['scaling'].min():.3f}-{g['scaling'].max():.3f} (exp: {g['scaling'].exp().min():.4f}-{g['scaling'].exp().max():.4f})"
)

T_id = torch.eye(4, device=device)
out = render_gaussians_relpose(g, K_src, T_id, H, W, device)
render = out["render"].clamp(0, 1)
print(f"render range: {render.min():.3f}-{render.max():.3f} mean={render.mean():.3f}")
if "rendered_alpha" in out:
    a = out["rendered_alpha"]
    print(f"alpha range: {a.min():.3f}-{a.max():.3f} mean={a.mean():.3f}")
p = psnr(crop5(render), crop5(color_src))
print(f"src recon PSNR={p:.2f}dB")

# Try WITHOUT proj transpose (hypothesis: double transpose)
from models.decoder.gauss_util import getProjectionMatrix, focal2fov, render_predicted

fx = K_src[0, 0].item()
fy = K_src[1, 1].item()
fovX = focal2fov(fx, W)
fovY = focal2fov(fy, H)


class _C:
    class model:
        renderer_w_pose = True


for label, transpose_proj in [("proj_T", True), ("proj_noT", False)]:
    proj = getProjectionMatrix(0.01, 100.0, fovX, fovY, 0.0, 0.0).to(device)
    if transpose_proj:
        proj = proj.transpose(0, 1).float()
    wvt = T_id.transpose(0, 1).float()
    cc = (-wvt[3, :3] @ wvt[:3, :3].transpose(0, 1)).float()
    fp = (wvt @ proj).float()
    o = render_predicted(
        _C(),
        g,
        wvt,
        fp,
        proj,
        cc,
        (fovX, fovY),
        (H, W),
        torch.zeros(3, device=device),
        0,
        override_color=g["rgb_direct"],
    )
    pp = psnr(crop5(o["render"].clamp(0, 1)), crop5(color_src))
    print(
        f"[{label}] PSNR={pp:.2f}dB alpha_mean={o.get('rendered_alpha', torch.zeros(1)).mean():.3f}"
    )

import os
import torchvision.utils as vutils

os.makedirs("/home/data/sv3d-lab/visualizations", exist_ok=True)
vutils.save_image(color_src, "/home/data/sv3d-lab/visualizations/diag_gt.png")
vutils.save_image(render, "/home/data/sv3d-lab/visualizations/diag_render.png")
print("saved diag_gt.png and diag_render.png")

# scale sweep: try scaling footprint by different factors
print("\n=== SCALE SWEEP (footprint * factor) ===")
base_fp = (depth_src.reshape(-1) / K_src[0, 0].item()).clamp(min=1e-4)
for factor in [1.0, 0.5, 0.25, 0.1, 0.05]:
    g["scaling"] = torch.log(base_fp * factor).unsqueeze(1).repeat(1, 3)
    o = render_gaussians_relpose(g, K_src, T_id, H, W, device)
    pp = psnr(crop5(o["render"].clamp(0, 1)), crop5(color_src))
    print(f"  factor={factor}: PSNR={pp:.2f}dB")
    vutils.save_image(
        o["render"].clamp(0, 1),
        f"/home/data/sv3d-lab/visualizations/diag_scale_{factor}.png",
    )
