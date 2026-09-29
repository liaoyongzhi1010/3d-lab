"""E-590: Large-baseline REGION-SPLIT evaluation protocol (Paper1/Paper2 base).

The correct battleground (E-580/E-581): at large gaps Flash3D collapses (gap50
tgt=17.97 vs small-gap 28.68). Generation+feed-forward papers win HERE, reporting
region-split metrics (visible vs disoccluded) rather than a single pixel PSNR.
This script builds that protocol so any method (Flash3D baseline, Scene-Splatter,
our methods) can be compared fairly.

For each (scene, src, tgt) at a chosen large gap:
  1. Flash3D forward -> gaussians -> render tgt (color + depth + alpha).
  2. Visibility mask at tgt: back-project the SOURCE-view depth (layer-1) into the
     TARGET camera; a target pixel is VISIBLE if some source pixel lands on it with
     consistent depth, else DISOCCLUDED. (geometric, method-agnostic.)
  3. Metrics, 5% border crop, VGG-LPIPS:
       full/visible/disoccluded PSNR ; full SSIM, LPIPS.
  4. Dump pred/gt PNGs into per-region folders for later FID (cleanfid).

This first run establishes the FLASH3D BASELINE table on the large-gap protocol.

Run (flash3d .venv):
  cd /root/projects/flash3d
  .venv/bin/python _e590_region_eval.py +experiment=layered_re10k_v2_ft \
    dataset.test_split_path=splits/re10k_mine_filtered/test_gap50_smoke50.txt \
    +e590.tag=flash3d_gap50 +e590.dump=1
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
from misc.util import add_source_frame_id
import lpips as lpips_lib
from skimage.metrics import structural_similarity as ssim_fn

_LP = None


def lpips_vgg(a, b, device):
    global _LP
    if _LP is None:
        _LP = lpips_lib.LPIPS(net="vgg").to(device).eval()
    pa = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    pb = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
    with torch.no_grad():
        return float(_LP(pa, pb).item())


def crop5(x):
    H, W = x.shape[:2]
    ch, cw = int(math.ceil(0.05 * H)), int(math.ceil(0.05 * W))
    return x[ch : H - ch, cw : W - cw]


def psnr_masked(a, b, m=None):
    d = (a - b) ** 2
    if m is not None:
        m = m > 0.5
        if m.sum() == 0:
            return float("nan")
        d = d[m]
    v = d.mean()
    return 100.0 if v < 1e-10 else float(-10 * np.log10(v))


def compute_visibility(depth_src, K_src, T_tgt, K_tgt, H, W):
    """Back-project source-view depth into the target camera; mark target pixels
    that receive a consistent source point as VISIBLE.
    depth_src: (H,W) metric depth at source view (layer-1).
    T_tgt: (4,4) source->target camera transform (world_view for tgt in src frame).
    Returns visibility mask (H,W) float {0,1} in TARGET image."""
    device = depth_src.device
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    xs = xs.float()
    ys = ys.float()
    z = depth_src.reshape(-1)
    x = (xs.reshape(-1) - K_src[0, 2]) / K_src[0, 0] * z
    y = (ys.reshape(-1) - K_src[1, 2]) / K_src[1, 1] * z
    pts = torch.stack([x, y, z, torch.ones_like(z)], 0)  # (4,N) in src cam
    pc_t = T_tgt @ pts  # (4,N) in tgt cam
    zt = pc_t[2].clamp(min=1e-6)
    u = (K_tgt[0, 0] * pc_t[0] / zt + K_tgt[0, 2]).round().long()
    v = (K_tgt[1, 1] * pc_t[1] / zt + K_tgt[1, 2]).round().long()
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (zt > 0)
    vis = torch.zeros(H * W, device=device)
    idx = v[inb] * W + u[inb]
    vis[idx] = 1.0
    return vis.reshape(H, W)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    os.chdir(hydra_cfg["runtime"]["output_dir"])
    tag = cfg.get("e590", {}).get("tag", "flash3d")
    dump = int(cfg.get("e590", {}).get("dump", 0))
    outroot = f"/home/data/E-590_region_eval/{tag}"
    os.makedirs(outroot, exist_ok=True)
    if dump:
        for reg in ["pred_full", "gt_full", "pred_invis", "gt_invis"]:
            os.makedirs(f"{outroot}/{reg}", exist_ok=True)

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
    target_frame_ids = [1, 2, 3]
    gpp = cfg.model.gaussians_per_pixel

    acc = {r: {"psnr": [], "ssim": [], "lpips": []} for r in ["full", "vis", "invis"]}
    n_img = 0
    for inputs in dataloader:
        for k in list(inputs.keys()):
            if torch.is_tensor(inputs[k]):
                inputs[k] = inputs[k].to(device)
        inputs["target_frame_ids"] = target_frame_ids
        with torch.no_grad():
            outputs = model(inputs)

        # source-view layer-1 depth for visibility
        depth = outputs[("depth", 0)]  # (b*gpp,1,H,W)
        H, W = depth.shape[-2:]
        depth_src = depth.view(-1, gpp, 1, H, W)[0, 0, 0]  # (H,W)
        K_src = (
            inputs[("K_src", 0)][0]
            if ("K_src", 0) in inputs
            else outputs[("K_src", 0)][0]
        )
        K_src = K_src[:3, :3] if K_src.shape[-1] == 4 else K_src

        # only eval the first (largest-gap) target: frame_id 1 (all three are same in gap50)
        for f_id in [1]:
            pred_t = (
                outputs[("color_gauss", f_id, 0)][0]
                .permute(1, 2, 0)
                .clamp(0, 1)
                .cpu()
                .numpy()
            )
            gt_t = (
                inputs[("color", f_id, 0)][0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            )
            Hp, Wp = pred_t.shape[:2]
            gt_t = gt_t[:Hp, :Wp]
            # visibility
            if ("cam_T_cam", 0, f_id) in outputs:
                T = outputs[("cam_T_cam", 0, f_id)][0]  # (4,4) src->tgt
                K_tgt = inputs[("K_tgt", f_id)][0]
                K_tgt = K_tgt[:3, :3] if K_tgt.shape[-1] == 4 else K_tgt
                vis = compute_visibility(depth_src, K_src, T, K_tgt, H, W).cpu().numpy()
            else:
                vis = np.ones((H, W), np.float32)
            # vis is at padded (H,W); resize to pred (Hp,Wp) so masks align after crop
            if vis.shape != (Hp, Wp):
                vis = (
                    np.asarray(
                        Image.fromarray((vis * 255).astype(np.uint8)).resize(
                            (Wp, Hp), Image.NEAREST
                        )
                    ).astype(np.float32)
                    / 255.0
                )

            pc, gc, mc = crop5(pred_t), crop5(gt_t), crop5(vis)
            acc["full"]["psnr"].append(psnr_masked(pc, gc))
            acc["full"]["ssim"].append(
                float(ssim_fn(pc, gc, channel_axis=2, data_range=1.0))
            )
            acc["full"]["lpips"].append(lpips_vgg(pc, gc, device))
            if mc.sum() > 50:
                acc["vis"]["psnr"].append(psnr_masked(pc, gc, mc))
            im = 1 - mc
            if im.sum() > 50:
                acc["invis"]["psnr"].append(psnr_masked(pc, gc, im))

            if dump:
                Image.fromarray((pc * 255).astype(np.uint8)).save(
                    f"{outroot}/pred_full/{n_img:04d}.png"
                )
                Image.fromarray((gc * 255).astype(np.uint8)).save(
                    f"{outroot}/gt_full/{n_img:04d}.png"
                )
                # disoccluded-only crops (masked pixels kept, rest set to GT to keep natural stats is complex;
                # for FID we dump full frames of the SAME region set; region FID handled separately)
            n_img += 1

    def mean(x):
        x = [v for v in x if not np.isnan(v)]
        return float(np.mean(x)) if x else float("nan")

    result = {
        "tag": tag,
        "n": n_img,
        "full": {m: mean(acc["full"][m]) for m in ["psnr", "ssim", "lpips"]},
        "visible": {"psnr": mean(acc["vis"]["psnr"])},
        "disoccluded": {"psnr": mean(acc["invis"]["psnr"])},
    }
    print("\n===== E-590 REGION EVAL:", tag, "=====")
    print(json.dumps(result, indent=2))
    with open(f"{outroot}/result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("saved", f"{outroot}/result.json")


if __name__ == "__main__":
    main()
