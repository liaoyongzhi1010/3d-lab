"""Full 620-scene RE10K eval for a given checkpoint, using Flash3D's own evaluate() +
Evaluator (identical protocol: crop_border, src/tgt5/tgt10/tgt_rand, VGG-LPIPS).

Usage (server, flash3d venv):
  python eval_full.py --ckpt <path.pth> --exp spatial --split splits/re10k_mine_filtered/test_files_present.txt --out <json>
"""

import argparse
import json
import os
import sys
import torch

sys.path.insert(0, "/root/projects/flash3d")
from hydra import compose, initialize_config_dir
from models.model import GaussianPredictor
from evaluation.evaluator import Evaluator
from datasets.util import create_datasets
from evaluate import evaluate  # reuse Flash3D's exact eval loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--exp", default="spatial", choices=["spatial", "baseline"])
    ap.add_argument(
        "--split", default="splits/re10k_mine_filtered/test_files_present.txt"
    )
    ap.add_argument("--out", required=True)
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
                "data_loader.num_workers=4",
                "++eval.save_vis=false",
            ],
        )

    device = torch.device("cuda:0")
    model = GaussianPredictor(cfg)
    model.to(device)

    sd = torch.load(args.ckpt, map_location="cpu")["model"]
    new = {}
    for k, v in sd.items():
        if "backproject_depth" in k:
            if k in model.state_dict():
                new[k] = model.state_dict()[k].clone()
        else:
            new[k] = v
    res = model.load_state_dict(new, strict=False)
    nonuni_missing = [k for k in res.missing_keys if "unidepth" not in k]
    print(
        f"loaded ckpt={args.ckpt}; non-unidepth missing={len(nonuni_missing)} unexpected={len(res.unexpected_keys)}"
    )

    evaluator = Evaluator(crop_border=cfg.dataset.crop_border)
    evaluator.to(device)

    _, loader = create_datasets(cfg, split="test")
    scores = evaluate(model, cfg, evaluator, loader, device=device, save_vis=False)

    # Flash3D's evaluate() returns scores[fid][metric] already reduced to a float mean,
    # plus scores[fid]["name"]. Handle both float and list defensively.
    def _val(x):
        if isinstance(x, (list, tuple)):
            return float(sum(x) / len(x)) if len(x) else None
        return float(x)

    summary = {}
    for fid, sc in scores.items():
        name = sc.get("name", str(fid)) if isinstance(sc, dict) else str(fid)
        summary[name] = {}
        for m in ["psnr", "ssim", "lpips"]:
            if m in sc:
                v = _val(sc[m])
                if v is not None:
                    summary[name][m] = v
    # target average (exclude src)
    for m in ["psnr", "ssim", "lpips"]:
        vals = [summary[n][m] for n in summary if n != "src" and m in summary[n]]
        if vals:
            summary.setdefault("target_avg", {})[m] = sum(vals) / len(vals)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
