"""Diagnostic: measure the QUALITY of the S0 teacher pseudo-GT directly.

Hypothesis (from S1 deletion showing disocc delta < 0): the Free layer faithfully
learns the teacher geometry, but the teacher itself (built with a crude median-depth
heuristic in S0) is low quality -> rendering teacher points to target views is WORSE
than showing background. If so, the teacher is the ceiling, not the S1 optimizer.

Test: for each scene with a teacher .pt, render:
  (a) anchor-only  (Flash3D)
  (b) anchor + TEACHER points (teacher xyz/color as isotropic gaussians)
to the real target views, measure PSNR in the disoccluded region (anchor renders bg).
If (b) is not > (a) in disocc region, the teacher pseudo-GT is the problem.

Run (Flash3D venv):
  python -m paper_b_generative3d.diagnose_teacher_quality \
    --teacher_dir /home/data/sv3d-lab/reconstruction/teacher_hidden_gs \
    --out /home/data/sv3d-lab/reconstruction/E-126-s1-route/teacher_quality.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from paper_b_generative3d.backbone_flash3d import Flash3DBackbone
from paper_b_generative3d.data_loader_re10k import build_flash3d_cfg, FLASH3D_CKPT
from paper_b_generative3d.viz_compare import psnr, _create_loader_from_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir", required=True)
    ap.add_argument("--max_scenes", type=int, default=80)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    cfg = build_flash3d_cfg(batch_size=1, num_workers=0, stage="dev")
    backbone = Flash3DBackbone(cfg, FLASH3D_CKPT, device=str(device))
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    teacher_dir = Path(args.teacher_dir)
    _, loader = _create_loader_from_cfg(cfg)
    H, W = 256, 384

    results = []
    for i, inputs in enumerate(loader):
        if i >= args.max_scenes:
            break
        tf = teacher_dir / f"{i:06d}.pt"
        if not tf.exists():
            continue
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
        tfids = [f for f in [1, 2, 3] if ("color", f, 0) in inputs]
        if not tfids:
            continue
        T_w2c_src = inputs.get(("T_w2c", 0))
        if T_w2c_src is None:
            continue
        T_w2c_src = T_w2c_src[0]

        teacher = torch.load(tf, map_location=device)
        tw = teacher["xyz"].float().to(device)  # world coords
        tc = teacher["color"].float().to(device)
        if tw.shape[0] < 10:
            continue
        # world -> source-cam
        ones = torch.ones(tw.shape[0], 1, device=device)
        t_cam = (T_w2c_src @ torch.cat([tw, ones], 1).T).T[:, :3]

        with torch.no_grad():
            g = backbone.extract_source_gaussians(inputs)
        anchor = {
            "xyz": g["xyz"][0],
            "scales": g["scales"][0],
            "rotations": g["rotations"][0],
            "opacity": g["opacity"][0],
            "color_rgb": g["color_rgb"][0],
        }
        # Teacher gaussians: isotropic, opaque, identity rotation
        M = t_cam.shape[0]
        t_gauss = {
            "xyz": t_cam,
            "scales": torch.full((M, 3), 0.02, device=device),
            "rotations": torch.tensor([1.0, 0, 0, 0], device=device)
            .expand(M, 4)
            .contiguous(),
            "opacity": torch.full((M, 1), 0.9, device=device),
            "color_rgb": tc.clamp(0, 1),
        }
        merged = {
            k: torch.cat([anchor[k], t_gauss[k]], 0)
            for k in ["xyz", "scales", "rotations", "opacity", "color_rgb"]
        }

        for fid in tfids:
            T_w2c_t = inputs.get(("T_w2c", fid))
            if T_w2c_t is None:
                continue
            cam = T_w2c_t[0] @ inputs[("T_c2w", 0)][0]
            K_tgt = inputs.get(("K_tgt", fid), inputs[("K_src", 0)])
            if K_tgt.dim() == 3:
                K_tgt = K_tgt[0]
            gt = inputs["color", fid, 0][0]
            with torch.no_grad():
                r_a = backbone.render_gaussians(anchor, cam, K_tgt, H, W)
                r_t = backbone.render_gaussians(merged, cam, K_tgt, H, W)
            anchor_is_bg = (r_a - 0.5).abs().mean(0) < 0.02
            n_diso = int(anchor_is_bg.sum().item())
            if n_diso <= 50:
                continue
            m = anchor_is_bg.unsqueeze(0).expand_as(gt)
            mse_a = ((r_a[m] - gt[m]) ** 2).mean().item()
            mse_t = ((r_t[m] - gt[m]) ** 2).mean().item()
            p_a = -10 * np.log10(mse_a + 1e-10)
            p_t = -10 * np.log10(mse_t + 1e-10)
            results.append(
                {
                    "scene": i,
                    "fid": fid,
                    "disocc_psnr_anchor": float(p_a),
                    "disocc_psnr_teacher": float(p_t),
                    "disocc_delta": float(p_t - p_a),
                    "n_disocc_px": n_diso,
                }
            )

    d = [r["disocc_delta"] for r in results]
    summary = {
        "mean_disocc_delta_teacher_dB": float(np.mean(d)) if d else None,
        "pct_positive": float(np.mean([x > 0 for x in d]) * 100) if d else None,
        "n_samples": len(d),
        "verdict": "TEACHER-USABLE"
        if (d and np.mean(d) > 0.5)
        else "TEACHER-IS-CEILING-PROBLEM",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"summary": summary, "per_sample": results}, indent=2)
    )
    print(f"\n{'=' * 60}")
    print("TEACHER PSEUDO-GT QUALITY (render teacher pts to target, disocc region):")
    print(
        f"  mean disocc delta (anchor+teacher vs anchor): {summary['mean_disocc_delta_teacher_dB']} dB"
    )
    print(f"  % positive: {summary['pct_positive']}  n={summary['n_samples']}")
    print(f"  VERDICT: {summary['verdict']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
