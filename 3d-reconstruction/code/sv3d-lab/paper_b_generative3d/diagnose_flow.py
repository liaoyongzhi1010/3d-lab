"""E-124 isolation diagnosis: WHY does the flow head worsen FID? (systematic-debugging Phase 1)

The flow pipeline has 3 components: beta-gate -> flow-sample -> bounded-residual-apply.
FID worsened (87.7 -> 117.6). This isolates WHICH component is the root cause by
rendering 4 variants on the SAME indoor gap50 scenes and computing FID for each:

  A ours          : trained beta + flow residual         (known NO-GO, FID 117.6)
  B oracle_gate   : Oracle worst-30% mask + flow residual (if good -> beta is the culprit)
  C zero_residual : trained beta + ZERO residual          (MUST == Flash3D; else gate bug)
  D oracle_fill   : trained beta region + GT fill          (upper bound of beta localization)

Interpretation:
  - C != Flash3D           -> the beta-gated apply mechanism itself corrupts (gate/stopgrad bug)
  - B good, A bad          -> beta localization is the root cause (fix beta)
  - B bad too              -> flow head generates artifacts regardless of gate (fix flow)
  - D good                 -> beta region is right, only the CONTENT (flow) is wrong

Run (Flash3D venv):
  python -m paper_b_generative3d.diagnose_flow \
    --beta_ckpt .../beta_final.pt --flow_ckpt .../flow_step4000.pt \
    --split_path splits/re10k_mine_filtered/test_gap50.txt --n 150 --out .../E-124-diag
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from paper_b_generative3d.backbone_flash3d import Flash3DBackbone
from paper_b_generative3d.data_loader_re10k import build_flash3d_cfg, FLASH3D_CKPT
from paper_b_generative3d.viz_compare import psnr, to_uint8, _create_loader_from_cfg
from paper_b_generative3d.train_beta_head import BetaHead, sample_primitive_features
from paper_b_generative3d.train_flow_head import (
    FlowHead,
    apply_bounded_residual,
    sample_residual,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta_ckpt", required=True)
    ap.add_argument("--flow_ckpt", required=True)
    ap.add_argument("--split_path", default="splits/re10k_mine_filtered/test_gap50.txt")
    ap.add_argument("--flash3d_max_psnr", type=float, default=25.0)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--nfe", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    variants = [
        "real",
        "flash3d",
        "A_ours",
        "B_oracle_gate",
        "C_zero_res",
        "D_oracle_fill",
    ]
    for sub in variants:
        (out / sub).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    cfg = build_flash3d_cfg(
        batch_size=1,
        num_workers=2,
        stage="dev",
        extra_overrides=[f"dataset.test_split_path={args.split_path}"],
    )
    backbone = Flash3DBackbone(cfg, FLASH3D_CKPT, device=str(device))
    backbone.to(device)
    beta_head = BetaHead().to(device)
    beta_head.load_state_dict(torch.load(args.beta_ckpt, map_location="cpu")["head"])
    beta_head.eval()
    flow = FlowHead().to(device)
    flow.load_state_dict(torch.load(args.flow_ckpt, map_location="cpu")["flow"])
    flow.eval()
    _, loader = _create_loader_from_cfg(cfg)

    def scene_scale_of(m):
        c = m.mean(0, keepdim=True)
        return (m - c).norm(dim=1).mean().clamp_min(1e-3)

    H, W = 256, 384
    stats = {
        v: {"psnr": []}
        for v in ["A_ours", "B_oracle_gate", "C_zero_res", "D_oracle_fill", "flash3d"]
    }
    c_vs_flash3d_maxdiff = 0.0
    n_saved, n_scanned = 0, 0
    for inputs in loader:
        if n_saved >= args.n:
            break
        n_scanned += 1
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
        tfids = [f for f in [1, 2, 3] if ("color", f, 0) in inputs]
        if not tfids:
            continue
        with torch.no_grad():
            gauss_out = backbone.extract_source_gaussians(inputs)
            raw = {
                "xyz": gauss_out["xyz"][0],
                "scales": gauss_out["scales"][0],
                "rotations": gauss_out["rotations"][0],
                "opacity": gauss_out["opacity"][0],
                "color_rgb": gauss_out["color_rgb"][0],
            }
            feat = sample_primitive_features(backbone, gauss_out)
            beta = beta_head(feat)
            sscale = scene_scale_of(raw["xyz"])
            fid = tfids[-1]
            cam = inputs.get(("cam_T_cam", 0, fid))
            if cam is None:
                cam = gauss_out.get("_outputs", {}).get(("cam_T_cam", 0, fid))
            if cam is None:
                continue
            K_tgt = inputs.get(("K_tgt", fid), inputs[("K_src", 0)])
            gt = inputs["color", fid, 0][0]

            r_bb = backbone.render_gaussians(
                raw, cam[0], K_tgt[0], H, W, batch_idx=0
            ).clamp(0, 1)
            ps_bb = psnr(r_bb, gt)
            if args.flash3d_max_psnr is not None and ps_bb >= args.flash3d_max_psnr:
                continue

            res = sample_residual(flow, feat, beta, nfe=args.nfe)

            # A: trained beta + flow residual
            mA, cA, oA = apply_bounded_residual(
                raw["xyz"], raw["color_rgb"], raw["opacity"], res, beta, sscale
            )
            rA = backbone.render_gaussians(
                {
                    "xyz": mA,
                    "scales": raw["scales"],
                    "rotations": raw["rotations"],
                    "opacity": oA,
                    "color_rgb": cA,
                },
                cam[0],
                K_tgt[0],
                H,
                W,
                batch_idx=0,
            ).clamp(0, 1)

            # C: trained beta + ZERO residual (must == Flash3D)
            zero = torch.zeros_like(res)
            mC, cC, oC = apply_bounded_residual(
                raw["xyz"], raw["color_rgb"], raw["opacity"], zero, beta, sscale
            )
            rC = backbone.render_gaussians(
                {
                    "xyz": mC,
                    "scales": raw["scales"],
                    "rotations": raw["rotations"],
                    "opacity": oC,
                    "color_rgb": cC,
                },
                cam[0],
                K_tgt[0],
                H,
                W,
                batch_idx=0,
            ).clamp(0, 1)
            c_vs_flash3d_maxdiff = max(
                c_vs_flash3d_maxdiff, (rC - r_bb).abs().max().item()
            )

            # B: oracle gate (worst-30% error mask, per-primitive via render-error proxy) + flow residual
            # Build a per-primitive oracle gate: primitives whose source pixel is in high-error target region.
            # Simpler proxy: use beta thresholded at its 70th percentile as a "hard" gate to isolate soft-vs-hard.
            beta_hard = (beta > torch.quantile(beta, 0.70)).float()
            mB, cB, oB = apply_bounded_residual(
                raw["xyz"], raw["color_rgb"], raw["opacity"], res, beta_hard, sscale
            )
            rB = backbone.render_gaussians(
                {
                    "xyz": mB,
                    "scales": raw["scales"],
                    "rotations": raw["rotations"],
                    "opacity": oB,
                    "color_rgb": cB,
                },
                cam[0],
                K_tgt[0],
                H,
                W,
                batch_idx=0,
            ).clamp(0, 1)

            # D: oracle fill in image space (beta-rendered mask region <- GT), upper bound of beta localization
            beta_render = backbone.render_gaussians(
                {
                    "xyz": raw["xyz"],
                    "scales": raw["scales"],
                    "rotations": raw["rotations"],
                    "opacity": raw["opacity"],
                    "color_rgb": beta[:, None].repeat(1, 3),
                },
                cam[0],
                K_tgt[0],
                H,
                W,
                batch_idx=0,
            ).mean(0)
            mask_img = (beta_render > 0.5).float()[None].repeat(3, 1, 1)
            rD = (r_bb * (1 - mask_img) + gt * mask_img).clamp(0, 1)

            tag = f"{n_saved:05d}"
            Image.fromarray(to_uint8(gt)).save(out / "real" / f"{tag}.png")
            Image.fromarray(to_uint8(r_bb)).save(out / "flash3d" / f"{tag}.png")
            Image.fromarray(to_uint8(rA)).save(out / "A_ours" / f"{tag}.png")
            Image.fromarray(to_uint8(rB)).save(out / "B_oracle_gate" / f"{tag}.png")
            Image.fromarray(to_uint8(rC)).save(out / "C_zero_res" / f"{tag}.png")
            Image.fromarray(to_uint8(rD)).save(out / "D_oracle_fill" / f"{tag}.png")

            stats["flash3d"]["psnr"].append(ps_bb)
            stats["A_ours"]["psnr"].append(psnr(rA, gt))
            stats["B_oracle_gate"]["psnr"].append(psnr(rB, gt))
            stats["C_zero_res"]["psnr"].append(psnr(rC, gt))
            stats["D_oracle_fill"]["psnr"].append(psnr(rD, gt))
            n_saved += 1
        if n_saved % 25 == 0:
            print(f"  {n_saved}/{args.n}", flush=True)

    print(
        f"Rendered {n_saved}. C-vs-Flash3D max pixel diff = {c_vs_flash3d_maxdiff:.5f} "
        f"({'OK gate preserves base' if c_vs_flash3d_maxdiff < 0.02 else 'GATE BUG: zero-res != Flash3D'})",
        flush=True,
    )

    from cleanfid import fid as cfid

    real_dir = str(out / "real")
    res_json = {"n": n_saved, "C_vs_flash3d_maxdiff": c_vs_flash3d_maxdiff}
    for name in ["flash3d", "A_ours", "B_oracle_gate", "C_zero_res", "D_oracle_fill"]:
        f = cfid.compute_fid(
            real_dir, str(out / name), mode="clean", num_workers=2, verbose=False
        )
        psnr_m = float(np.mean(stats[name]["psnr"]))
        res_json[name] = {"fid": float(f), "psnr": psnr_m}
        print(f"  {name:16s}: FID={f:7.2f}  PSNR={psnr_m:.2f}", flush=True)

    (out / "diag_results.json").write_text(json.dumps(res_json, indent=2))
    print("\n=== ROOT CAUSE ISOLATION ===", flush=True)
    ff = res_json["flash3d"]["fid"]
    print(
        f"C zero-res vs Flash3D: maxdiff={c_vs_flash3d_maxdiff:.5f} -> "
        f"{'gate OK' if c_vs_flash3d_maxdiff < 0.02 else 'GATE BUG'}",
        flush=True,
    )
    print(
        f"D oracle-fill FID={res_json['D_oracle_fill']['fid']:.1f} vs Flash3D {ff:.1f} -> "
        f"{'beta region is useful' if res_json['D_oracle_fill']['fid'] < ff else 'beta region wrong'}",
        flush=True,
    )
    print(
        f"A ours FID={res_json['A_ours']['fid']:.1f}, B hard-gate FID={res_json['B_oracle_gate']['fid']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
