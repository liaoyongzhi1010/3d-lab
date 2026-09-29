"""Black-box transfer attack evaluation: Flash3D surrogate → VGGT target.

Zero-query transfer: craft adversarial perturbation using Flash3D (gradient access),
then measure how much VGGT's geometric predictions (depth, world points) are disrupted.

Key insight: If a perturbation optimized against Flash3D (CNN+UniDepth+3DGS) also
significantly disrupts VGGT (Transformer, completely different architecture), this proves
the vulnerability is at the "single-view 3D geometry assumption" level, not model-specific.

Metrics:
  - depth_mae: mean absolute depth change (attacked vs clean) normalized by clean depth
  - pointmap_displacement: mean L2 displacement of world points
  - depth_rank_reversal: fraction of pixel pairs whose relative depth ordering flips
  - source_psnr: quality of attacked image vs clean (must be high = imperceptible)

Run:
  cd /root/projects/flash3d && source .venv/bin/activate
  export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python /root/flash3d-attack/eval_vggt_transfer.py \
    --max_scenes 30 --epsilon 0.015686 --steps 40 \
    --out /root/flash3d-attack/results/BB-002-vggt-transfer.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, "/root/projects/vggt")
sys.path.insert(0, "/root/flash3d-attack")

from attacks.flash3d_geometry_adapter import Flash3DGeometryAdapter
from attacks.dpc_attack import DPCConfig, dpc_attack


def load_vggt(device):
    """Load VGGT model (requires ~4GB VRAM for 1B params in fp16)."""
    from vggt.models.vggt import VGGT

    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.to(device).eval()
    return model


def vggt_predict(model, image, device):
    """Run VGGT on a single image.

    Args:
        image: [3, H, W] in [0, 1]

    Returns:
        depth: [H', W'] predicted depth
        world_points: [H', W', 3] predicted world coordinates
        depth_conf: [H', W'] confidence
    """
    _, H, W = image.shape
    patch_size = 14
    new_H = (H // patch_size) * patch_size
    new_W = (W // patch_size) * patch_size
    img_resized = F.interpolate(
        image.unsqueeze(0), size=(new_H, new_W), mode="bilinear", align_corners=False
    )[0]
    img = img_resized.unsqueeze(0).unsqueeze(0)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        preds = model(img)

    depth = preds["depth"][0, 0, :, :, 0]
    world_pts = preds["world_points"][0, 0]
    depth_conf = preds["depth_conf"][0, 0]
    return depth, world_pts, depth_conf


def compute_transfer_metrics(
    clean_depth, clean_pts, attacked_depth, attacked_pts, clean_conf
):
    """Compute geometric disruption metrics."""
    valid = clean_conf > 0.5
    if valid.sum() < 100:
        valid = torch.ones_like(clean_conf, dtype=torch.bool)

    cd = clean_depth[valid]
    ad = attacked_depth[valid]
    depth_mae = (ad - cd).abs().mean().item()
    depth_rel_mae = ((ad - cd).abs() / (cd.abs() + 1e-6)).mean().item()

    cp = clean_pts[valid]
    ap = attacked_pts[valid]
    displacement = (ap - cp).norm(dim=-1).mean().item()

    n_sample = min(4096, valid.sum().item())
    indices = torch.randperm(valid.sum().item(), device=cd.device)[:n_sample]
    cd_sample = cd[indices]
    ad_sample = ad[indices]

    n_pairs = min(8192, n_sample * (n_sample - 1) // 2)
    i_idx = torch.randint(0, n_sample, (n_pairs,), device=cd.device)
    j_idx = torch.randint(0, n_sample, (n_pairs,), device=cd.device)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]

    clean_order = cd_sample[i_idx] > cd_sample[j_idx]
    atk_order = ad_sample[i_idx] > ad_sample[j_idx]
    reversals = (clean_order != atk_order).float().mean().item()

    return {
        "depth_mae": depth_mae,
        "depth_rel_mae": depth_rel_mae,
        "pointmap_displacement": displacement,
        "depth_rank_reversal": reversals,
        "n_valid_pixels": int(valid.sum().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_scenes", type=int, default=30)
    parser.add_argument("--epsilon", type=float, default=0.015686)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lambda_src", type=float, default=3.0)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print("Loading Flash3D surrogate...")
    adapter = Flash3DGeometryAdapter.from_default_checkpoint()

    print("Loading VGGT target...")
    vggt_model = load_vggt(device)

    dpc_cfg = DPCConfig(
        epsilon=args.epsilon,
        steps=args.steps,
        step_size=args.epsilon / 4,
        lambda_src=args.lambda_src,
        seed=0,
    )

    from hydra import compose, initialize_config_dir
    from datasets.util import create_datasets

    _, loader = create_datasets(adapter.cfg, split="test")

    all_results = []
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
        clean_src = inputs[("color_aug", 0, 0)].detach()

        def render_fn(image, pose):
            local = dict(adapter.example_inputs)
            local[("color_aug", 0, 0)] = image
            local["target_frame_ids"] = [1]
            outputs = adapter.model.models["unidepth_extended"](local)
            adapter.model.compute_gauss_means(local, outputs)
            B = image.shape[0]
            dtype = outputs["gauss_means"].dtype
            outputs[("cam_T_cam", 0, 1)] = (
                pose.to(device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
            )
            adapter.model.render_images(local, outputs)
            return outputs[("color_gauss", 1, 0)]

        print(f"[{n_done + 1}/{args.max_scenes}] Crafting on Flash3D...", flush=True)
        attacked_src, delta, info = dpc_attack(clean_src, render_fn, dpc_cfg)

        src_psnr = -10 * torch.log10(F.mse_loss(attacked_src, clean_src) + 1e-10).item()

        print(f"  src_psnr={src_psnr:.2f}dB, linf={info['linf']:.5f}", flush=True)
        print(f"  Evaluating on VGGT...", flush=True)

        clean_img = clean_src[0]
        atk_img = attacked_src[0]

        clean_depth, clean_pts, clean_conf = vggt_predict(vggt_model, clean_img, device)
        atk_depth, atk_pts, atk_conf = vggt_predict(vggt_model, atk_img, device)

        metrics = compute_transfer_metrics(
            clean_depth, clean_pts, atk_depth, atk_pts, clean_conf
        )
        metrics["src_psnr"] = src_psnr
        metrics["linf"] = info["linf"]
        metrics["scene_idx"] = n_done

        random_delta = (torch.rand_like(clean_src) * 2 - 1) * args.epsilon
        random_src = (clean_src + random_delta).clamp(0, 1)
        rand_depth, rand_pts, _ = vggt_predict(vggt_model, random_src[0], device)
        rand_metrics = compute_transfer_metrics(
            clean_depth, clean_pts, rand_depth, rand_pts, clean_conf
        )
        metrics["random_baseline"] = rand_metrics

        all_results.append(metrics)
        n_done += 1

        if n_done % 5 == 0:
            avg_disp = np.mean([r["pointmap_displacement"] for r in all_results])
            avg_rev = np.mean([r["depth_rank_reversal"] for r in all_results])
            rand_disp = np.mean(
                [r["random_baseline"]["pointmap_displacement"] for r in all_results]
            )
            print(
                f"  Running avg: displacement={avg_disp:.4f} (rand={rand_disp:.4f}), "
                f"rank_reversal={avg_rev:.4f}"
            )

    summary = {
        "n_scenes": len(all_results),
        "epsilon": args.epsilon,
        "steps": args.steps,
        "lambda_src": args.lambda_src,
        "surrogate": "Flash3D",
        "target": "VGGT-1B",
        "attack_method": "DPC (rendering divergence)",
    }
    for key in [
        "depth_mae",
        "depth_rel_mae",
        "pointmap_displacement",
        "depth_rank_reversal",
        "src_psnr",
    ]:
        vals = [r[key] for r in all_results]
        summary[f"mean_{key}"] = float(np.mean(vals))
        summary[f"std_{key}"] = float(np.std(vals))

    for key in ["pointmap_displacement", "depth_rank_reversal"]:
        vals = [r["random_baseline"][key] for r in all_results]
        summary[f"random_mean_{key}"] = float(np.mean(vals))

    summary["transfer_ratio_displacement"] = summary["mean_pointmap_displacement"] / (
        summary["random_mean_pointmap_displacement"] + 1e-10
    )
    summary["transfer_ratio_reversal"] = summary["mean_depth_rank_reversal"] / (
        summary["random_mean_depth_rank_reversal"] + 1e-10
    )

    summary["per_scene"] = all_results

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Transfer Attack Summary ({len(all_results)} scenes) ===")
    print(f"Surrogate: Flash3D | Target: VGGT-1B | eps={args.epsilon:.5f}")
    print(
        f"  Pointmap displacement: {summary['mean_pointmap_displacement']:.4f} "
        f"(random: {summary['random_mean_pointmap_displacement']:.4f}, "
        f"ratio: {summary['transfer_ratio_displacement']:.2f}x)"
    )
    print(
        f"  Depth rank reversal: {summary['mean_depth_rank_reversal']:.4f} "
        f"(random: {summary['random_mean_depth_rank_reversal']:.4f}, "
        f"ratio: {summary['transfer_ratio_reversal']:.2f}x)"
    )
    print(f"  Source PSNR: {summary['mean_src_psnr']:.2f}dB (imperceptibility)")


if __name__ == "__main__":
    main()
