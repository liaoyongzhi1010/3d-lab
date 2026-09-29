"""E-592: FID for oracle refinement variants (LPIPS said local blending hurts;
does FID -- a distribution metric insensitive to seam discontinuity -- show gain?).

E-591 finding: local GT replacement in a region IMPROVES PSNR but WORSENS VGG-LPIPS
(seam discontinuity artifacts). This is the known LPIPS-masking trap. The correct
metric for generated/replaced regions is FID (distribution-level). If oracle_dis
FID << base FID, then "generate the disoccluded region + report FID" is a valid
contribution path; the LPIPS regression is a metric artifact, not a real failure.

This dumps the 4 variants' full frames and computes cleanfid FID of each variant
set vs the GT set.

Run (flash3d .venv):
  cd /root/projects/flash3d
  .venv/bin/python _e592_oracle_fid.py +experiment=layered_re10k_v2_ft \
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


def crop5(x):
    H, W = x.shape[:2]
    ch, cw = int(math.ceil(0.05 * H)), int(math.ceil(0.05 * W))
    return x[ch : H - ch, cw : W - cw]


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
    get_model_instance(model).set_eval()

    dataset, dataloader = create_datasets(cfg, split="test")
    gpp = cfg.model.gaussians_per_pixel
    root = "/home/data/E-592_oracle_fid"
    variants = ["base", "oracle_dis", "oracle_vis", "gt"]
    for v in variants:
        os.makedirs(f"{root}/{v}", exist_ok=True)

    n = 0
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
        imgs = {
            "base": pred,
            "oracle_dis": pred * vis3 + gt * (1 - vis3),
            "oracle_vis": gt * vis3 + pred * (1 - vis3),
            "gt": gt,
        }
        for v in variants:
            Image.fromarray((crop5(imgs[v]) * 255).astype(np.uint8)).save(
                f"{root}/{v}/{n:04d}.png"
            )
        n += 1

    from cleanfid import fid

    print(f"\n===== E-592 FID (vs GT set), {n} imgs =====")
    out = {}
    for v in ["base", "oracle_dis", "oracle_vis"]:
        score = fid.compute_fid(
            f"{root}/{v}", f"{root}/gt", mode="clean", num_workers=0
        )
        out[v] = score
        print(f"  FID({v} vs gt) = {score:.3f}")
    with open(f"{root}/fid.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved", f"{root}/fid.json")


if __name__ == "__main__":
    main()
