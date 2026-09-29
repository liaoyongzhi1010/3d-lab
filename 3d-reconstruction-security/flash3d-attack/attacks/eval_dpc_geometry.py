"""Evaluate the geometry-DPC white-box attack against frozen Flash3D (server-only).

Runs geometry-DPC on the source image, then evaluates the attacked image on the UNSEEN MINE
eval targets, reporting:
  * source-view PSNR (camouflage must stay high, drop <=3 dB per acceptance),
  * novel-view PSNR/SSIM/LPIPS drops (novel damage),
  * correspondence-aware geometry metrics on the explicit 3D primitives (aligned point
    displacement, reprojection displacement) plus the HARD near/far order-reversal rate with
    a frozen clean-eligible denominator.

The geometry metrics are computed in a COMMON target frame after global-scale alignment and
raster visibility filtering (see attacks/geometry_metrics.py). The optimization uses only the
SOFT differentiable objectives; the hard predicates are evaluation-only.

Run on server (Flash3D venv):
  cd /root/projects/flash3d && source .venv/bin/activate
  export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python /root/flash3d-attack/attacks/eval_dpc_geometry.py --max_scenes 30 --mode hybrid \
    --out /root/flash3d-attack/results/dpc_geometry_n30.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attacks.dpc_geometry_attack import GeometryAttackConfig, geometry_attack
from attacks.flash3d_geometry_adapter import Flash3DGeometryAdapter
from attacks.geometry_metrics import (
    OrderReversalConfig,
    aligned_point_displacement,
    order_reversal_rate,
    reprojection_displacement,
)


def _psnr(pred, gt):
    mse = (pred.clamp(0, 1) - gt.clamp(0, 1)).pow(2).mean().item()
    if mse <= 0:
        return 99.0
    return -10.0 * torch.log10(torch.tensor(mse)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split", default="splits/re10k_mine_filtered/test_files_present.txt"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_scenes", type=int, default=30)
    ap.add_argument(
        "--mode", choices=["primitive", "depth", "ordering", "hybrid"], default="hybrid"
    )
    ap.add_argument("--epsilon", type=float, default=8.0 / 255.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--step_size", type=float, default=1.0 / 255.0)
    ap.add_argument("--lambda_src", type=float, default=10.0)
    ap.add_argument("--n_aux_poses", type=int, default=4)
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir
    from datasets.util import create_datasets
    from evaluation.evaluator import Evaluator
    from models.model import GaussianPredictor

    device = torch.device("cuda:0")
    repo = "/root/projects/flash3d"
    with initialize_config_dir(config_dir=f"{repo}/configs", version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=layered_re10k",
                "+dataset.crop_border=true",
                f"dataset.test_split_path={args.split}",
                "model.depth.version=v1",
                "data_loader.batch_size=1",
                "data_loader.num_workers=1",
            ],
        )
    model = GaussianPredictor(cfg).to(device)
    state = torch.load(f"{repo}/checkpoints/model_re10k_v2.pth", map_location="cpu")
    sd = state["model"] if "model" in state else state
    current = model.state_dict()
    filtered = {}
    for k, v in sd.items():
        if "backproject_depth" in k:
            if k in current:
                filtered[k] = current[k].clone()
        else:
            filtered[k] = v
    model.load_state_dict(filtered, strict=False)
    model.set_eval()

    evaluator = Evaluator(crop_border=cfg.dataset.crop_border).to(device)
    _, loader = create_datasets(cfg, split="test")

    gcfg = GeometryAttackConfig(
        epsilon=args.epsilon,
        steps=args.steps,
        step_size=args.step_size,
        lambda_src=args.lambda_src,
        mode=args.mode,
        n_aux_poses=args.n_aux_poses,
    )
    order_cfg = OrderReversalConfig()

    agg = defaultdict(lambda: defaultdict(list))
    geom_rows = []
    names = {0: "src", 1: "tgt5", 2: "tgt10", 3: "tgt_rand"}
    n_done = 0

    for inputs in loader:
        if n_done >= args.max_scenes:
            break
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        if ("color", 1, 0) not in inputs:
            continue
        inputs["target_frame_ids"] = [1, 2, 3]

        adapter = Flash3DGeometryAdapter(model, cfg, inputs, device)
        clean_src = inputs[("color_aug", 0, 0)].detach()

        attacked_src, delta, info = geometry_attack(
            clean_src, adapter.build_geometry_state, gcfg
        )

        # Correspondence-aware geometry metrics (hard predicates, eval-only).
        with torch.no_grad():
            clean_state = adapter.build_geometry_state(clean_src)
            adv_state = adapter.build_geometry_state(attacked_src)
        T = torch.eye(4, device=device)
        pt_disp = aligned_point_displacement(
            adv_state.means,
            clean_state.means,
            adv_state.primitive_ids,
            clean_state.primitive_ids,
        )
        rep_disp = reprojection_displacement(
            adv_state.means,
            clean_state.means,
            clean_state.K_src,
            adv_state.primitive_ids,
            clean_state.primitive_ids,
        )
        order = order_reversal_rate(clean_state, adv_state, T, order_cfg)
        geom_rows.append(
            {
                "scene": n_done,
                "aligned_point_displacement": pt_disp,
                "reprojection_displacement": rep_disp,
                "order_reversal_rate": order.rate,
                "order_eligible_pairs": order.n_eligible_pairs,
                "order_reversed": order.n_reversed,
                "linf": info["linf"],
            }
        )

        for tag, src_img in [("clean", clean_src), ("attacked", attacked_src)]:
            eval_inputs = copy.deepcopy(inputs)
            eval_inputs[("color_aug", 0, 0)] = src_img.detach()
            eval_inputs["target_frame_ids"] = [1, 2, 3]
            with torch.no_grad():
                out = model(eval_inputs)
            for fid in [0, 1, 2, 3]:
                pred_key = ("color_gauss", fid, 0)
                gt_key = ("color", fid, 0)
                if pred_key not in out or gt_key not in eval_inputs:
                    continue
                pred = out[pred_key].clamp(0, 1)
                gt = eval_inputs[gt_key].clamp(0, 1)
                m = evaluator(pred, gt)
                for mk, mv in m.items():
                    agg[f"{tag}_{names[fid]}"][mk].append(float(mv))
        n_done += 1
        if n_done % 5 == 0:
            print(f"attacked {n_done} scenes; last linf={info['linf']:.4f}", flush=True)

    summary = {}
    for key, metrics in agg.items():
        summary[key] = {mk: float(sum(v) / len(v)) for mk, v in metrics.items() if v}
    summary["n_scenes"] = n_done
    summary["cfg"] = vars(args)
    summary["geometry_rows"] = geom_rows
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
