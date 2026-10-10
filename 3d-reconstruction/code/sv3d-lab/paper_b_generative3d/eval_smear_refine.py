"""Eval Stage 1 SmearRefineHead: does it beat Flash3D on novel views without breaking source?

Metrics (held-out dev scenes):
  - Novel-view PSNR/LPIPS: refined vs Flash3D (want LPIPS down = less smear)
  - Source-view PSNR: refined vs GT source (preservation; must not drop >0.2 dB vs Flash3D)
  - Deletion check: head-off render == Flash3D render (causally load-bearing)

GO: mean novel LPIPS improves >0.01 over Flash3D AND source PSNR not worse by >0.2 dB.

Run (flash3d venv):
  python -m paper_b_generative3d.eval_smear_refine \
    --ckpt /home/data/sv3d-lab/reconstruction/E-128-smear-refine/refine_final.pt \
    --out /home/data/sv3d-lab/reconstruction/E-128-smear-refine/eval.json
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
from paper_b_generative3d.model.smear_refine_head import SmearRefineHead
from paper_b_generative3d.train_real import get_lpips_fn
from paper_b_generative3d.train_smear_refine import sample_per_gaussian_feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_scenes", type=int, default=100)
    ap.add_argument("--feat_dim", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    cfg = build_flash3d_cfg(batch_size=1, num_workers=0, stage="dev")
    backbone = Flash3DBackbone(cfg, FLASH3D_CKPT, device=str(device))
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    head = SmearRefineHead(feat_dim=args.feat_dim, hidden=256, n_layers=4).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    head.load_state_dict(ckpt["head"])
    head.eval()
    lpips_fn = get_lpips_fn(device)
    print(f"Loaded head from step {ckpt.get('step', '?')}", flush=True)

    _, loader = _create_loader_from_cfg(cfg)
    H, W = 256, 384

    res = []
    for i, inputs in enumerate(loader):
        if i >= args.max_scenes:
            break
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
        tfids = [f for f in [1, 2, 3] if ("color", f, 0) in inputs]
        T_c2w_s = inputs.get(("T_c2w", 0))
        if not tfids or T_c2w_s is None:
            continue

        with torch.no_grad():
            g = backbone.extract_source_gaussians(inputs)
            anchor = {
                k: g[k][0]
                for k in ["xyz", "scales", "rotations", "opacity", "color_rgb"]
            }
            n = anchor["xyz"].shape[0]
            feat = sample_per_gaussian_feats(
                g["source_features"], n, g["pixel_hw"], g["gaussians_per_pixel"], device
            )
            refined = head(anchor, feat)
            refined_g = {
                k: refined[k]
                for k in ["xyz", "scales", "rotations", "opacity", "color_rgb"]
            }

        # source preservation
        K_src = inputs[("K_src", 0)][0]
        src_gt = inputs["color", 0, 0][0]
        cam_src = torch.eye(4, device=device)
        with torch.no_grad():
            r_src_flash = backbone.render_gaussians(anchor, cam_src, K_src, H, W).clamp(
                0, 1
            )
            r_src_ref = backbone.render_gaussians(
                refined_g, cam_src, K_src, H, W
            ).clamp(0, 1)
        src_psnr_flash = psnr(r_src_flash, src_gt)
        src_psnr_ref = psnr(r_src_ref, src_gt)

        for fid in tfids:
            T_w2c_t = inputs.get(("T_w2c", fid))
            if T_w2c_t is None:
                continue
            cam = T_w2c_t[0] @ T_c2w_s[0]
            K_tgt = inputs.get(("K_tgt", fid), inputs[("K_src", 0)])
            if K_tgt.dim() == 3:
                K_tgt = K_tgt[0]
            gt = inputs["color", fid, 0][0]
            with torch.no_grad():
                r_flash = backbone.render_gaussians(anchor, cam, K_tgt, H, W).clamp(
                    0, 1
                )
                r_ref = backbone.render_gaussians(refined_g, cam, K_tgt, H, W).clamp(
                    0, 1
                )
                lp_flash = lpips_fn(r_flash.unsqueeze(0), gt.unsqueeze(0)).mean().item()
                lp_ref = lpips_fn(r_ref.unsqueeze(0), gt.unsqueeze(0)).mean().item()
            res.append(
                {
                    "scene": i,
                    "fid": fid,
                    "psnr_flash": float(psnr(r_flash, gt)),
                    "psnr_ref": float(psnr(r_ref, gt)),
                    "lpips_flash": float(lp_flash),
                    "lpips_ref": float(lp_ref),
                    "src_psnr_flash": float(src_psnr_flash),
                    "src_psnr_ref": float(src_psnr_ref),
                }
            )
        if (i + 1) % 20 == 0:
            lpf = np.mean([r["lpips_flash"] for r in res])
            lpr = np.mean([r["lpips_ref"] for r in res])
            print(
                f"[{i + 1}] novel LPIPS flash={lpf:.4f} ref={lpr:.4f} (d={lpr - lpf:+.4f})",
                flush=True,
            )

    def m(k):
        return float(np.mean([r[k] for r in res]))

    summary = {
        "n": len(res),
        "novel_psnr_flash": m("psnr_flash"),
        "novel_psnr_ref": m("psnr_ref"),
        "novel_lpips_flash": m("lpips_flash"),
        "novel_lpips_ref": m("lpips_ref"),
        "novel_lpips_delta": m("lpips_ref") - m("lpips_flash"),
        "novel_lpips_pct_improved": float(
            np.mean([r["lpips_ref"] < r["lpips_flash"] for r in res]) * 100
        ),
        "src_psnr_flash": m("src_psnr_flash"),
        "src_psnr_ref": m("src_psnr_ref"),
        "src_psnr_drop": m("src_psnr_flash") - m("src_psnr_ref"),
    }
    summary["GO"] = bool(
        summary["novel_lpips_delta"] < -0.01 and summary["src_psnr_drop"] < 0.2
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"summary": summary, "per_sample": res}, indent=2)
    )
    print("\n" + "=" * 60, flush=True)
    print("SMEAR-REFINE EVAL:", flush=True)
    print(
        f"  Novel PSNR:  flash {summary['novel_psnr_flash']:.2f} -> ref {summary['novel_psnr_ref']:.2f}",
        flush=True,
    )
    print(
        f"  Novel LPIPS: flash {summary['novel_lpips_flash']:.4f} -> ref {summary['novel_lpips_ref']:.4f} "
        f"(delta {summary['novel_lpips_delta']:+.4f}, {summary['novel_lpips_pct_improved']:.0f}% improved)",
        flush=True,
    )
    print(
        f"  Source PSNR: flash {summary['src_psnr_flash']:.2f} -> ref {summary['src_psnr_ref']:.2f} "
        f"(drop {summary['src_psnr_drop']:+.2f})",
        flush=True,
    )
    print(
        f"  GO: {'YES' if summary['GO'] else 'NO'} (novel LPIPS -0.01 & src drop <0.2)",
        flush=True,
    )
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
