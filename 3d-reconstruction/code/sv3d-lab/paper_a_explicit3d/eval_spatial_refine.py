"""
Paper A (D017) — SpatialRefine eval on the EXACT Flash3D repro protocol (present split, frames
[1,2,3] = tgt5/tgt10/tgt_rand, Flash3D Evaluator: 5% crop, VGG-LPIPS). Reports PSNR/SSIM/LPIPS so we
compare apples-to-apples against our reproduced Flash3D baseline (tgt5 28.68 / tgt10 26.09 / tgt_rand
25.10). Also supports --baseline to eval the RAW Flash3D backbone (refinement OFF) in the SAME harness.

Usage:
  python /root/sv3d-lab/paper_a_explicit3d/eval_spatial_refine.py \
     --ckpt /home/data/sv3d-lab/runs/sr_L2/ckpt_final.pt --n_scenes 500 \
     --split /root/projects/flash3d/splits/re10k_mine_filtered/test_files.txt \
     --out /home/data/sv3d-lab/evaluations/sr_L2
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, "/root/sv3d-lab")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.spatial_refine_model import SpatialRefineModel, SpatialRefineConfig  # noqa: E402
from train_paper_a import load_re10k, build_cfg, to_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument(
        "--baseline", action="store_true", help="eval raw Flash3D (refinement off)"
    )
    ap.add_argument("--n_scenes", type=int, default=500)
    ap.add_argument("--scene_start", type=int, default=0)
    ap.add_argument("--novel_frames", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    cfg = build_cfg(args.novel_frames, args.split)
    Re10KDataset = load_re10k()
    ds = Re10KDataset(cfg, split="test")
    s0 = args.scene_start
    ds._seq_key_src_idx_pairs = ds._seq_key_src_idx_pairs[s0 : s0 + args.n_scenes]
    ds.length = len(ds._seq_key_src_idx_pairs)

    from torch.utils.data import DataLoader
    from datasets.util import custom_collate
    from common.data.robust_dataset import RobustDataset
    from evaluation.evaluator import Evaluator

    ds = RobustDataset(ds)
    loader = DataLoader(ds, 1, shuffle=False, num_workers=2, collate_fn=custom_collate)
    evaluator = Evaluator(crop_border=True).to(device)

    model = SpatialRefineModel(cfg, SpatialRefineConfig()).to(device)
    fl_ckpt = "/root/projects/flash3d/checkpoints/model_re10k_v2.pth"
    model.load_visible_pretrained(fl_ckpt, device=device)
    if args.ckpt and not args.baseline:
        sd = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(sd["model"], strict=False)
        print(f"[eval] loaded refine ckpt {args.ckpt}", flush=True)
    else:
        # baseline: zero the refinement head so output == raw Flash3D
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
        print("[eval] BASELINE mode (refinement head zeroed = raw Flash3D)", flush=True)
    model.eval()

    names = {
        args.novel_frames[0]: "tgt5",
        args.novel_frames[1]: "tgt10",
        args.novel_frames[2]: "tgt_rand",
    }
    agg = {n: {"psnr": [], "ssim": [], "lpips": []} for n in names.values()}
    agg["src"] = {"psnr": [], "ssim": [], "lpips": []}

    for batch in loader:
        inputs = to_device(batch, device)
        target_ids = [f for f in args.novel_frames if ("color", f, 0) in inputs]
        if len(target_ids) < 3:
            continue
        with torch.no_grad():
            out = model(inputs, target_ids)
            out_src = model(inputs, [0]) if ("color", 0, 0) in inputs else None
        for fid in target_ids:
            pred = out[("render", fid)].clamp(0, 1)  # (1,3,H,W)
            gt = inputs[("color", fid, 0)]
            m = evaluator(pred, gt)
            nm = names[fid]
            for k in ("psnr", "ssim", "lpips"):
                agg[nm][k].append(float(m[k]))
        if out_src is not None and ("render", 0) in out_src:
            m = evaluator(out_src[("render", 0)].clamp(0, 1), inputs[("color", 0, 0)])
            for k in ("psnr", "ssim", "lpips"):
                agg["src"][k].append(float(m[k]))

    summary = {}
    for nm, d in agg.items():
        summary[nm] = {k: (float(np.mean(v)) if v else None) for k, v in d.items()}
    summary["n_evaluated"] = len(agg["tgt5"]["psnr"])
    summary["mode"] = "baseline" if (args.baseline or not args.ckpt) else "refine"
    summary["ckpt"] = args.ckpt
    with open(os.path.join(args.out, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
