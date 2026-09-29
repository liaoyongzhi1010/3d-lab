"""E-591: Oracle upper-bound for selective generative refinement (Paper1/2 decision).

The plan: at large gaps Flash3D is blurry (LPIPS 0.321). A generative method that
REPLACES the disoccluded region with sharp plausible content should improve
LPIPS/FID there, while KEEPING Flash3D in the visible region (PSNR-faithful).
Prior disocc_target failed because it was judged on PSNR (generation is not
pixel-aligned). The right metric is LPIPS/FID.

Before building the generator, measure the ORACLE ceiling with zero cost: how much
do region metrics improve if the replacement is PERFECT (=GT)? This bounds the
achievable gain and tells us WHERE the headroom is.

Variants (per gap50 frame, 5% crop, VGG-LPIPS):
  base        : Flash3D render (the baseline)
  oracle_dis  : GT in disoccluded region, Flash3D in visible  (upper bound of "fix holes")
  oracle_vis  : GT in visible region, Flash3D in disoccluded  (upper bound of "fix visible")
  oracle_full : GT everywhere (sanity, LPIPS~0)

We report full-frame LPIPS for each variant (and PSNR). If oracle_dis >> base on
LPIPS, "selective generative refinement of the disoccluded region" has real
headroom -> worth building. If oracle_vis dominates instead, the blur is in the
visible region and generation of holes won't help much.

Run (flash3d .venv):
  cd /root/projects/flash3d
  .venv/bin/python _e591_oracle_refine.py +experiment=layered_re10k_v2_ft \
    dataset.test_split_path=splits/re10k_mine_filtered/test_gap50_smoke50.txt
"""

import os
import json
import math
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from PIL import Image

from models.model import GaussianPredictor
from evaluate import get_model_instance
from datasets.util import create_datasets
import lpips as lpips_lib
from skimage.metrics import structural_similarity as ssim_fn

_LP = None


def lpips_vgg(a, b, device):
    global _LP
    if _LP is None:
        _LP = lpips_lib.LPIPS(net="vgg").to(device).eval()
    pa = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device).float() * 2 - 1
    pb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).to(device).float() * 2 - 1
    with torch.no_grad():
        return float(_LP(pa, pb).item())


def crop5(x):
    H, W = x.shape[:2]
    ch, cw = int(math.ceil(0.05 * H)), int(math.ceil(0.05 * W))
    return x[ch : H - ch, cw : W - cw]


def psnr(a, b):
    v = ((a - b) ** 2).mean()
    return 100.0 if v < 1e-10 else float(-10 * np.log10(v))


def compute_visibility(depth_src, K_src, T_tgt, K_tgt, H, W):
    device = depth_src.device
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    z = depth_src.reshape(-1)
    x = (xs.reshape(-1).float() - K_src[0, 2]) / K_src[0, 0] * z
    y = (ys.reshape(-1).float() - K_src[1, 2]) / K_src[1, 1] * z
    pts = torch.stack([x, y, z, torch.ones_like(z)], 0)
    pc_t = T_tgt @ pts
    zt = pc_t[2].clamp(min=1e-6)
    u = (K_tgt[0, 0] * pc_t[0] / zt + K_tgt[0, 2]).round().long()
    v = (K_tgt[1, 1] * pc_t[1] / zt + K_tgt[1, 2]).round().long()
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (zt > 0)
    vis = torch.zeros(H * W, device=device)
    vis[(v[inb] * W + u[inb])] = 1.0
    return vis.reshape(H, W)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    os.chdir(hydra_cfg["runtime"]["output_dir"])
    cfg.data_loader.batch_size = 1
    cfg.data_loader.num_workers = 2
    device = torch.device("cuda:0")
    model = GaussianPredictor(cfg)
    model.to(device)
    if (ckpt_dir := model.checkpoint_dir()).exists():
        model.load_model(ckpt_dir, ckpt_ids=0)
    else:
        model.load_model(cfg.train.load_weights_folder, ckpt_ids=0)
    model_inst = get_model_instance(model)
    model_inst.set_eval()

    dataset, dataloader = create_datasets(cfg, split="test")
    gpp = cfg.model.gaussians_per_pixel

    variants = ["base", "oracle_dis", "oracle_vis", "oracle_full"]
    acc = {v: {"psnr": [], "lpips": []} for v in variants}
    disocc_fracs = []

    for inputs in dataloader:
        for k in list(inputs.keys()):
            if torch.is_tensor(inputs[k]):
                inputs[k] = inputs[k].to(device)
        inputs["target_frame_ids"] = [1, 2, 3]
        with torch.no_grad():
            outputs = model(inputs)
        depth = outputs[("depth", 0)]
        H, W = depth.shape[-2:]
        depth_src = depth.view(-1, gpp, 1, H, W)[0, 0, 0]
        K_src = (
            inputs[("K_src", 0)][0]
            if ("K_src", 0) in inputs
            else outputs[("K_src", 0)][0]
        )
        K_src = K_src[:3, :3] if K_src.shape[-1] == 4 else K_src

        f_id = 1
        pred = (
            outputs[("color_gauss", f_id, 0)][0]
            .permute(1, 2, 0)
            .clamp(0, 1)
            .cpu()
            .numpy()
        )
        gt = inputs[("color", f_id, 0)][0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        Hp, Wp = pred.shape[:2]
        gt = gt[:Hp, :Wp]
        if ("cam_T_cam", 0, f_id) in outputs:
            T = outputs[("cam_T_cam", 0, f_id)][0]
            K_tgt = inputs[("K_tgt", f_id)][0]
            K_tgt = K_tgt[:3, :3] if K_tgt.shape[-1] == 4 else K_tgt
            vis = compute_visibility(depth_src, K_src, T, K_tgt, H, W).cpu().numpy()
        else:
            vis = np.ones((H, W), np.float32)
        if vis.shape != (Hp, Wp):
            vis = (
                np.asarray(
                    Image.fromarray((vis * 255).astype(np.uint8)).resize(
                        (Wp, Hp), Image.NEAREST
                    )
                ).astype(np.float32)
                / 255.0
            )
        vis3 = vis[..., None]
        disocc_fracs.append(float(1 - vis.mean()))

        imgs = {
            "base": pred,
            "oracle_dis": pred * vis3 + gt * (1 - vis3),
            "oracle_vis": gt * vis3 + pred * (1 - vis3),
            "oracle_full": gt,
        }
        for v in variants:
            pc, gc = crop5(imgs[v]), crop5(gt)
            acc[v]["psnr"].append(psnr(pc, gc))
            acc[v]["lpips"].append(lpips_vgg(pc, gc, device))

    def mean(x):
        x = [v for v in x if not np.isnan(v)]
        return float(np.mean(x)) if x else float("nan")

    print(
        f"\n===== E-591 ORACLE refine ceiling ({len(disocc_fracs)} scenes, mean disocc {mean(disocc_fracs):.3f}) ====="
    )
    out = {"mean_disocc": mean(disocc_fracs)}
    for v in variants:
        out[v] = {"psnr": mean(acc[v]["psnr"]), "lpips": mean(acc[v]["lpips"])}
        print(f"  {v:12s}: psnr={out[v]['psnr']:.2f}  lpips={out[v]['lpips']:.4f}")
    base_lp = out["base"]["lpips"]
    print(
        f"\n  LPIPS headroom (base -> oracle_dis): {base_lp:.4f} -> {out['oracle_dis']['lpips']:.4f} "
        f"({100 * (base_lp - out['oracle_dis']['lpips']) / base_lp:.1f}% of blur is in DISOCCLUDED region)"
    )
    print(
        f"  LPIPS headroom (base -> oracle_vis): {base_lp:.4f} -> {out['oracle_vis']['lpips']:.4f} "
        f"({100 * (base_lp - out['oracle_vis']['lpips']) / base_lp:.1f}% of blur is in VISIBLE region)"
    )
    os.makedirs("/home/data/E-591_oracle", exist_ok=True)
    with open("/home/data/E-591_oracle/result.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
