"""E-129 Official MINE Protocol Eval.

Uses the official Flash3D MINE test split (3205 scenes), crop_border=5%, per-bucket
(tgt5=+5frame, tgt10=+10frame, tgt_rand=random) reporting. Renders Flash3D and Refined
side by side for a fair comparison under the exact official protocol.

Produces numbers directly comparable to Flash3D Table-2 (28.68/26.09/25.10 per bucket).

Run (flash3d venv):
  python -m paper_b_generative3d.eval_mine_protocol \
    --ckpt /home/data/sv3d-lab/reconstruction/E-129-gated/refine_step1000.pt \
    --max_scenes 200 --out /home/data/sv3d-lab/reconstruction/E-129-mine-eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/projects/flash3d")
from hydra import compose, initialize_config_dir
from datasets.util import create_datasets
from evaluation.evaluator import Evaluator

from paper_b_generative3d.backbone_flash3d import Flash3DBackbone
from paper_b_generative3d.data_loader_re10k import FLASH3D_CKPT
from paper_b_generative3d.model.smear_refine_head import SmearRefineHead
from paper_b_generative3d.train_smear_refine import sample_per_gaussian_feats


def crop_border(img, margin=0.05):
    """5% border crop matching Flash3D official."""
    _, _, H, W = img.shape
    ch = int(H * margin)
    cw = int(W * margin)
    return img[:, :, ch : H - ch, cw : W - cw]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_scenes", type=int, default=3205)
    ap.add_argument("--feat_dim", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")

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
                "data_loader.num_workers=4",
            ],
        )

    backbone = Flash3DBackbone(cfg, FLASH3D_CKPT, device=str(device))
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    head = SmearRefineHead(feat_dim=args.feat_dim, hidden=256, n_layers=4).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    head.load_state_dict(ck["head"])
    head.eval()

    evaluator = Evaluator(crop_border=True).to(device)
    _, loader = create_datasets(cfg, split="test")
    H, W = 256, 384

    # MINE buckets: fid 1=+5frame, fid 2=+10frame, fid 3=random
    bucket_names = {1: "tgt5", 2: "tgt10", 3: "tgt_rand"}
    scores_flash = {b: {"psnr": [], "lpips": []} for b in bucket_names.values()}
    scores_ref = {b: {"psnr": [], "lpips": []} for b in bucket_names.values()}

    n_done = 0
    for inputs in loader:
        if n_done >= args.max_scenes:
            break
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)

        with torch.no_grad():
            g = backbone.extract_source_gaussians(inputs)
        anchor = {
            k: g[k][0] for k in ["xyz", "scales", "rotations", "opacity", "color_rgb"]
        }
        n = anchor["xyz"].shape[0]
        feat = sample_per_gaussian_feats(
            g["source_features"], n, g["pixel_hw"], g["gaussians_per_pixel"], device
        )
        with torch.no_grad():
            refined = head(anchor, feat)
        refined_g = {
            k: refined[k]
            for k in ["xyz", "scales", "rotations", "opacity", "color_rgb"]
        }

        T_c2w_s = inputs.get(("T_c2w", 0))
        if T_c2w_s is None:
            n_done += 1
            continue

        for fid in [1, 2, 3]:
            gt_key = ("color", fid, 0)
            if gt_key not in inputs:
                continue
            T_w2c_t = inputs.get(("T_w2c", fid))
            if T_w2c_t is None:
                continue
            cam = T_w2c_t[0] @ T_c2w_s[0]
            K_tgt = inputs.get(("K_tgt", fid), inputs.get(("K_src", 0)))
            if K_tgt is not None and K_tgt.dim() == 3:
                K_tgt = K_tgt[0]
            gt = inputs[gt_key]  # [1,3,H,W]
            if gt.dim() == 3:
                gt = gt.unsqueeze(0)

            with torch.no_grad():
                r_flash = (
                    backbone.render_gaussians(anchor, cam, K_tgt, H, W)
                    .clamp(0, 1)
                    .unsqueeze(0)
                )
                r_ref = (
                    backbone.render_gaussians(refined_g, cam, K_tgt, H, W)
                    .clamp(0, 1)
                    .unsqueeze(0)
                )

            # official crop
            gt_c = crop_border(gt)
            rf_c = crop_border(r_flash)
            rr_c = crop_border(r_ref)

            out_f = evaluator(rf_c, gt_c)
            out_r = evaluator(rr_c, gt_c)
            bname = bucket_names[fid]
            scores_flash[bname]["psnr"].append(float(out_f["psnr"]))
            scores_flash[bname]["lpips"].append(float(out_f["lpips"]))
            scores_ref[bname]["psnr"].append(float(out_r["psnr"]))
            scores_ref[bname]["lpips"].append(float(out_r["lpips"]))

        n_done += 1
        if n_done % 50 == 0:
            print(
                f"[{n_done}/{args.max_scenes}] tgt5 flash={np.mean(scores_flash['tgt5']['psnr']):.2f} "
                f"ref={np.mean(scores_ref['tgt5']['psnr']):.2f}",
                flush=True,
            )

    summary = {}
    for bname in bucket_names.values():
        if scores_flash[bname]["psnr"]:
            summary[f"flash_{bname}_psnr"] = float(np.mean(scores_flash[bname]["psnr"]))
            summary[f"flash_{bname}_lpips"] = float(
                np.mean(scores_flash[bname]["lpips"])
            )
            summary[f"ref_{bname}_psnr"] = float(np.mean(scores_ref[bname]["psnr"]))
            summary[f"ref_{bname}_lpips"] = float(np.mean(scores_ref[bname]["lpips"]))
            summary[f"delta_{bname}_psnr"] = (
                summary[f"ref_{bname}_psnr"] - summary[f"flash_{bname}_psnr"]
            )
            summary[f"delta_{bname}_lpips"] = (
                summary[f"ref_{bname}_lpips"] - summary[f"flash_{bname}_lpips"]
            )
    summary["n_scenes"] = n_done

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\n{'=' * 60}", flush=True)
    print("MINE PROTOCOL EVAL (crop_border=5%):", flush=True)
    for bname in bucket_names.values():
        if f"flash_{bname}_psnr" in summary:
            print(
                f"  {bname}: Flash3D {summary[f'flash_{bname}_psnr']:.2f}/{summary[f'flash_{bname}_lpips']:.4f} "
                f"-> Ours {summary[f'ref_{bname}_psnr']:.2f}/{summary[f'ref_{bname}_lpips']:.4f} "
                f"(ΔPSNR {summary[f'delta_{bname}_psnr']:+.2f} ΔLPIPS {summary[f'delta_{bname}_lpips']:+.4f})",
                flush=True,
            )
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
