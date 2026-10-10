"""Render a qualitative comparison image from a trained UniDepthSpatial / Flash3D model.

Loads a checkpoint, runs one test scene, renders source->target novel views, and saves a
side-by-side PNG: [source | GT tgt5 | pred tgt5 | GT tgt10 | pred tgt10 | GT tgt_rand | pred tgt_rand]
with per-target PSNR printed. Uses the SAME eval protocol as Flash3D (crop_border, present split).

Usage (on server, flash3d venv):
  python render_compare.py --ckpt <path> --exp spatial --out /home/data/sv3d-lab/viz/spatial_v2.png --scene_idx 0
"""

import argparse
import os
import sys
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, "/root/projects/flash3d")
from hydra import compose, initialize_config_dir
from models.model import GaussianPredictor, to_device
from evaluation.evaluator import Evaluator
from datasets.util import create_datasets
from misc.util import add_source_frame_id


def to_img(t):
    t = t.detach().clamp(0, 1).cpu().numpy()
    if t.ndim == 4:
        t = t[0]
    return (t.transpose(1, 2, 0) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--exp", default="spatial", choices=["spatial", "baseline"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene_idx", type=int, default=0)
    ap.add_argument(
        "--split", default="splits/re10k_mine_filtered/test_files_present.txt"
    )
    args = ap.parse_args()

    exp_name = "layered_re10k_spatial" if args.exp == "spatial" else "layered_re10k"
    with initialize_config_dir(
        config_dir="/root/projects/flash3d/configs", version_base=None
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"+experiment={exp_name}",
                "+dataset.crop_border=true",
                f"dataset.test_split_path={args.split}",
                "model.depth.version=v1",
                "data_loader.batch_size=1",
                "data_loader.num_workers=1",
            ],
        )

    device = torch.device("cuda:0")
    model = GaussianPredictor(cfg)
    model.to(device)
    # load weights (strict=False: spatial branch / unidepth hub keys)
    sd = torch.load(args.ckpt, map_location="cpu")["model"]
    new = {}
    for k, v in sd.items():
        if "backproject_depth" in k:
            if k in model.state_dict():
                new[k] = model.state_dict()[k].clone()
        else:
            new[k] = v
    res = model.load_state_dict(new, strict=False)
    print(
        f"loaded; missing(non-unidepth)={[k for k in res.missing_keys if 'unidepth' not in k][:5]}"
    )
    model.set_eval()

    evaluator = Evaluator(crop_border=cfg.dataset.crop_border)
    evaluator.to(device)

    _, loader = create_datasets(cfg, split="test")
    it = iter(loader)
    for _ in range(args.scene_idx + 1):
        inputs = next(it)
    inputs = to_device(inputs, device)

    target_frame_ids = [1, 2, 3]
    inputs["target_frame_ids"] = target_frame_ids
    with torch.no_grad():
        outputs = model(inputs)

    names = ["tgt5", "tgt10", "tgt_rand"]
    src = to_img(inputs[("color", 0, 0)])
    panels = [src]
    labels = ["source"]
    for fid, nm in zip(target_frame_ids, names):
        gt = inputs[("color", fid, 0)]
        pred = outputs[("color_gauss", fid, 0)]
        # metrics (crop border applied inside evaluator)
        psnr = (
            evaluator.get_psnr_score(pred.clamp(0, 1), gt.clamp(0, 1))
            if hasattr(evaluator, "get_psnr_score")
            else None
        )
        panels.append(to_img(gt))
        panels.append(to_img(pred))
        labels.append(f"GT {nm}")
        labels.append(f"pred {nm}")

    # stack horizontally
    h = min(p.shape[0] for p in panels)
    panels = [p[:h] for p in panels]
    strip = np.concatenate(panels, axis=1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    Image.fromarray(strip).save(args.out)
    print(f"saved {args.out} | panels: {labels}")

    # also compute + print PSNR per target using evaluator on full batch
    for fid, nm in zip(target_frame_ids, names):
        gt = inputs[("color", fid, 0)].clamp(0, 1)
        pred = outputs[("color_gauss", fid, 0)].clamp(0, 1)
        mse = ((gt - pred) ** 2).mean().item()
        psnr = -10 * np.log10(mse) if mse > 0 else 99
        print(f"  {nm}: PSNR(no-crop)={psnr:.2f}")


if __name__ == "__main__":
    main()
