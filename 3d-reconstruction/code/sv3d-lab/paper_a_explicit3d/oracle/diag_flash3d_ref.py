"""Ground-truth comparison: run Flash3D's OWN model.forward on scene 0, capture its
source-frame gaussian means + source render, and compare our construction to it.
This isolates exactly where our oracle geometry diverges from the working Flash3D path."""

import sys
import torch

sys.path.insert(0, "/root/projects/flash3d")

from hydra import compose, initialize_config_dir
from pathlib import Path

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
            "data_loader.batch_size=1",
        ],
    )

from datasets.re10k import Re10KDataset
from models.model import GaussianPredictor, to_device
from misc.util import add_source_frame_id

ds = Re10KDataset(cfg, split="test")
model = GaussianPredictor(cfg)
device_t = torch.device("cuda")
model.to(device_t)
ckpt = Path("/root/projects/flash3d/checkpoints/model_re10k_v2.pth")
model.load_model(ckpt, ckpt_ids=0)
model.set_eval()

inputs = ds[0]
# collate to batch of 1
batch = {}
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        batch[k] = v.unsqueeze(0)
    else:
        batch[k] = v
to_device(batch, device_t)
batch["target_frame_ids"] = [1, 2, 3]

with torch.no_grad():
    outputs = model(batch)

gm = outputs["gauss_means"]  # (B*gpp, 4, HW) or similar
print(f"gauss_means shape={gm.shape}")
print(
    f"gauss_means x:[{gm[:, 0].min():.2f},{gm[:, 0].max():.2f}] "
    f"y:[{gm[:, 1].min():.2f},{gm[:, 1].max():.2f}] z:[{gm[:, 2].min():.2f},{gm[:, 2].max():.2f}]"
)
depth = outputs[("depth", 0)]
print(
    f"flash3d depth shape={depth.shape} range={depth.min():.3f}-{depth.max():.3f} mean={depth.mean():.3f}"
)

# flash3d source render
from evaluation.evaluator import Evaluator

ev = Evaluator(crop_border=True).to(device_t)
pred_src = outputs[("color_gauss", 0, 0)]
gt_src = batch[("color", 0, 0)]
scores = ev(pred_src, gt_src)
print(f"Flash3D OWN src render: PSNR={scores['psnr']:.2f} SSIM={scores['ssim']:.3f}")

# Key facts to compare:
print(f"\nK_src (padded):\n{batch[('K_src', 0)][0]}")
print(f"K_tgt (unpadded):\n{batch[('K_tgt', 0)][0]}")
print(f"inv_K_src:\n{batch[('inv_K_src', 0)][0]}")
print(f"color_aug (padded) shape: {batch[('color_aug', 0, 0)].shape}")
print(f"color (unpadded) shape: {batch[('color', 0, 0)].shape}")
print(
    f"pad_border_aug={cfg.dataset.pad_border_aug}, gpp={cfg.model.gaussians_per_pixel}, "
    f"shift={cfg.model.shift_rays_half_pixel}, predict_offset={cfg.model.predict_offset}"
)

# === DECISIVE TEST: render Flash3D's OWN gaussians with OUR render function ===
import sys as _sys

_sys.path.insert(0, "/root/sv3d-lab/paper_a_explicit3d/oracle")
from oracle_core import render_gaussians_relpose, psnr, crop5

gpp = cfg.model.gaussians_per_pixel
gm = outputs["gauss_means"]  # (gpp, 4, HW) source frame
xyz0 = gm[0, :3, :].T
fdc = outputs["gauss_features_dc"]  # (gpp, 3, H, W)
rgb0 = fdc[0].reshape(3, -1).T  # SH DC coeffs
rgb0_rgb = (0.28209 * rgb0 + 0.5).clamp(0, 1)
opacity0 = outputs["gauss_opacity"][0].reshape(1, -1).T
scaling0 = outputs["gauss_scaling"][0].reshape(3, -1).T
rotation0 = outputs["gauss_rotation"][0].reshape(4, -1).T
g_flash = {
    "xyz": xyz0.contiguous().float(),
    "scaling": scaling0.contiguous().float(),
    "rotation": rotation0.contiguous().float(),
    "opacity": opacity0.contiguous().float(),
    "features_dc": rgb0.reshape(-1, 1, 3).contiguous().float(),
    "rgb_direct": rgb0_rgb.contiguous().float(),
}
K_tgt = batch[("K_tgt", 0)][0]
H, W = batch[("color", 0, 0)].shape[2:]
T_id = torch.eye(4, device=device_t)
o = render_gaussians_relpose(g_flash, K_tgt, T_id, H, W, device_t)
p_ours = psnr(crop5(o["render"].clamp(0, 1)), crop5(batch[("color", 0, 0)][0]))
print(f"\nDECISIVE: OUR render of Flash3D layer-0 gaussians: PSNR={p_ours:.2f}dB")
print("  (~38dB => our render/convention OK, bug is our backprojection/depth)")
print("  (low => our render function / K / proj convention is the bug)")
print(
    f"  flash3d scaling0(log) range {scaling0.min():.2f}..{scaling0.max():.2f} "
    f"(exp {scaling0.exp().min():.4f}..{scaling0.exp().max():.4f})"
)
print(f"  flash3d xyz0 z range {xyz0[:, 2].min():.2f}..{xyz0[:, 2].max():.2f}")

# Replicate Flash3D EXACT inline render to isolate matrix vs color handling
from models.decoder.gauss_util import (
    getProjectionMatrix,
    focal2fov,
    render_predicted as _rp,
)

focals = torch.diag(K_tgt)[:2]
fovY = focal2fov(focals[1].item(), H)
fovX = focal2fov(focals[0].item(), W)
proj_f = getProjectionMatrix(
    cfg.dataset.znear, cfg.dataset.zfar, fovX, fovY, pX=0, pY=0
).to(device_t)
wvt_f = torch.eye(4, device=device_t).transpose(0, 1).float()
cc_f = (-wvt_f[3, :3] @ wvt_f[:3, :3].transpose(0, 1)).float()
proj_f = proj_f.transpose(0, 1).float()
fp_f = (wvt_f @ proj_f).float()
pc = {
    "xyz": g_flash["xyz"],
    "opacity": g_flash["opacity"],
    "scaling": g_flash["scaling"],
    "rotation": g_flash["rotation"],
    "features_dc": g_flash["features_dc"],
}
out_f = _rp(
    cfg,
    pc,
    wvt_f,
    fp_f,
    proj_f,
    cc_f,
    (fovX, fovY),
    (H, W),
    torch.zeros(3, device=device_t),
    cfg.model.max_sh_degree,
)
p_inline = psnr(crop5(out_f["render"].clamp(0, 1)), crop5(batch[("color", 0, 0)][0]))
print(
    f"\nInline Flash3D-exact render (SH deg={cfg.model.max_sh_degree}): PSNR={p_inline:.2f}dB"
)
print(
    f"  znear={cfg.dataset.znear} zfar={cfg.dataset.zfar} fovX={fovX:.4f} fovY={fovY:.4f}"
)
