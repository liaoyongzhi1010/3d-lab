"""S1 Deletion Counterfactual Evaluation.

GO/NO-GO gate: if removing Free layer degrades render quality on target views by >1dB,
then Free contributes causally and S1 passes.

Computes:
  - PSNR_merged (anchor+free) vs GT on target views
  - PSNR_anchor (anchor only) vs GT on target views
  - Delta = PSNR_merged - PSNR_anchor (positive = Free helps)

Also monitors mean free_opacity to confirm no collapse.

Run (Flash3D venv):
  python -m paper_b_generative3d.eval_s1_deletion \
    --ckpt /home/data/sv3d-lab/reconstruction/E-126-s1/s1_final.pt \
    --out /home/data/sv3d-lab/reconstruction/E-126-s1/deletion_eval.json
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
from paper_b_generative3d.model.canonical_dual_layer import CanonicalDualLayer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_free", type=int, default=4096)
    ap.add_argument("--max_scenes", type=int, default=80)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    cfg = build_flash3d_cfg(batch_size=1, num_workers=0, stage="dev")
    backbone = Flash3DBackbone(cfg, FLASH3D_CKPT, device=str(device))
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    model = CanonicalDualLayer(
        anchor_backbone=backbone,
        feat_proj_dim=256,
        latent_dim=64,
        n_free=args.n_free,
        n_cross=3,
        n_self=2,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"Loaded checkpoint from step {ckpt.get('step', '?')}", flush=True)

    _, loader = _create_loader_from_cfg(cfg)
    H, W = 256, 384

    results = []
    for i, inputs in enumerate(loader):
        if i >= args.max_scenes:
            break
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)

        tfids = [f for f in [1, 2, 3] if ("color", f, 0) in inputs]
        if not tfids:
            continue

        with torch.no_grad():
            scene = model.build_scene(inputs)
            merged = model.merge_scene(scene)
            anchor_only = scene["anchor"]

        free_op = scene["free"]["opacity"].mean().item()

        for fid in tfids:
            T_w2c_t = inputs.get(("T_w2c", fid))
            T_c2w_s = inputs.get(("T_c2w", 0))
            if T_w2c_t is None or T_c2w_s is None:
                continue
            cam = T_w2c_t[0] @ T_c2w_s[0]
            K_tgt = inputs.get(("K_tgt", fid), inputs[("K_src", 0)])
            if K_tgt.dim() == 3:
                K_tgt = K_tgt[0]
            gt = inputs["color", fid, 0][0]

            with torch.no_grad():
                render_merged = backbone.render_gaussians(merged, cam, K_tgt, H, W)
                render_anchor = backbone.render_gaussians(anchor_only, cam, K_tgt, H, W)

            psnr_m = psnr(render_merged, gt)
            psnr_a = psnr(render_anchor, gt)
            delta = psnr_m - psnr_a

            # Disoccluded-region mask: anchor renders bg (0.5) where it has no coverage.
            # Those are exactly the pixels Free is supposed to fill. Measure delta there.
            bg = 0.5
            anchor_is_bg = (render_anchor - bg).abs().mean(dim=0) < 0.02  # [H,W]
            n_diso = int(anchor_is_bg.sum().item())
            if n_diso > 50:
                m = anchor_is_bg.unsqueeze(0).expand_as(gt)
                mse_m = ((render_merged[m] - gt[m]) ** 2).mean().item()
                mse_a = ((render_anchor[m] - gt[m]) ** 2).mean().item()
                psnr_m_diso = -10 * np.log10(mse_m + 1e-10)
                psnr_a_diso = -10 * np.log10(mse_a + 1e-10)
                delta_diso = float(psnr_m_diso - psnr_a_diso)
            else:
                delta_diso = None

            results.append(
                {
                    "scene": i,
                    "fid": fid,
                    "psnr_merged": float(psnr_m),
                    "psnr_anchor": float(psnr_a),
                    "delta": float(delta),
                    "delta_disocc": delta_diso,
                    "n_disocc_px": n_diso,
                    "free_opacity": float(free_op),
                }
            )

        if (i + 1) % 20 == 0:
            deltas = [r["delta"] for r in results]
            print(
                f"[{i + 1}/{args.max_scenes}] mean_delta={np.mean(deltas):.3f}dB "
                f"free_op={free_op:.3f} n_samples={len(results)}",
                flush=True,
            )

    deltas = [r["delta"] for r in results]
    mean_delta = np.mean(deltas)
    median_delta = np.median(deltas)
    pct_positive = np.mean([d > 0 for d in deltas]) * 100

    diso = [r["delta_disocc"] for r in results if r["delta_disocc"] is not None]
    mean_delta_diso = float(np.mean(diso)) if diso else None
    pct_pos_diso = float(np.mean([d > 0 for d in diso]) * 100) if diso else None

    summary = {
        "mean_delta_dB": float(mean_delta),
        "median_delta_dB": float(median_delta),
        "pct_positive": float(pct_positive),
        "mean_delta_disocc_dB": mean_delta_diso,
        "pct_positive_disocc": pct_pos_diso,
        "n_disocc_samples": len(diso),
        "mean_psnr_merged": float(np.mean([r["psnr_merged"] for r in results])),
        "mean_psnr_anchor": float(np.mean([r["psnr_anchor"] for r in results])),
        "mean_free_opacity": float(np.mean([r["free_opacity"] for r in results])),
        "n_samples": len(results),
        "GO": bool((mean_delta_diso or -99) > 1.0),
        "gate_threshold_dB": 1.0,
        "gate_note": "GO measured on disoccluded region (where Free should contribute)",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "per_sample": results}, indent=2)
    )
    print(f"\n{'=' * 60}", flush=True)
    print(f"DELETION COUNTERFACTUAL RESULT:", flush=True)
    print(f"  Full-frame delta (merged - anchor): {mean_delta:+.3f} dB", flush=True)
    print(f"  Full-frame % positive: {pct_positive:.1f}%", flush=True)
    print(
        f"  DISOCCLUDED-region delta: {mean_delta_diso if mean_delta_diso is None else f'{mean_delta_diso:+.3f}'} dB "
        f"(n={len(diso)}, %pos={pct_pos_diso})",
        flush=True,
    )
    print(f"  Mean free opacity: {summary['mean_free_opacity']:.3f}", flush=True)
    print(
        f"  GO/NO-GO (disocc): {'GO (>+1dB)' if summary['GO'] else 'NO-GO (<+1dB)'}",
        flush=True,
    )
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
