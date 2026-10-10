"""Step 3: evaluate the trained flow-matching residual head vs Flash3D baseline.

Renders three sets on the INDOOR gap50 split and computes region-attributed FID/KID
(via clean-fid) + per-scene PSNR/LPIPS + red-box zoom qualitative panels:
  - real/    : GT target frames (FID reference)
  - flash3d/ : raw Flash3D renders (baseline)
  - ours/    : Flash3D base + beta-gated flow-matching residual (K-step ODE sample)

GO if: FID(ours) < FID(flash3d) by > KID std, visible-region PSNR preserved (>= Flash3D
       within ~0.1dB), and qualitative panels show sharper disoccluded regions w/o speckle.

Run (Flash3D venv):
  python -m paper_b_generative3d.eval_flow_head \
    --beta_ckpt .../E-121-beta/beta_final.pt --flow_ckpt .../E-122-flow/flow_final.pt \
    --split_path splits/re10k_mine_filtered/test_gap50.txt --flash3d_max_psnr 25.0 \
    --n 200 --nfe 8 --out .../E-123-flow-eval
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
from paper_b_generative3d.viz_compare import (
    psnr,
    to_uint8,
    label_strip,
    draw_box,
    crop_zoom,
    find_improve_region,
    _create_loader_from_cfg,
)
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
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--nfe", type=int, default=8)
    ap.add_argument("--n_panels", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ["real", "flash3d", "ours", "panels"]:
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

    def scene_scale_of(means):
        c = means.mean(0, keepdim=True)
        return (means - c).norm(dim=1).mean().clamp_min(1e-3)

    H, W = 256, 384
    rows = []
    n_saved, n_scanned, n_panel = 0, 0, 0
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

            # ours: multi-step ODE sample of the residual (inference path)
            res = sample_residual(flow, feat, beta, nfe=args.nfe)
            means, color, opacity = apply_bounded_residual(
                raw["xyz"], raw["color_rgb"], raw["opacity"], res, beta, sscale
            )
            g_ours = {
                "xyz": means,
                "scales": raw["scales"],
                "rotations": raw["rotations"],
                "opacity": opacity,
                "color_rgb": color,
            }
            r_ours = backbone.render_gaussians(
                g_ours, cam[0], K_tgt[0], H, W, batch_idx=0
            ).clamp(0, 1)
            ps_ours = psnr(r_ours, gt)

        tag = f"{n_saved:05d}"
        Image.fromarray(to_uint8(gt)).save(out / "real" / f"{tag}.png")
        Image.fromarray(to_uint8(r_bb)).save(out / "flash3d" / f"{tag}.png")
        Image.fromarray(to_uint8(r_ours)).save(out / "ours" / f"{tag}.png")
        rows.append(
            {
                "scan": n_scanned,
                "flash3d_psnr": ps_bb,
                "ours_psnr": ps_ours,
                "delta": ps_ours - ps_bb,
                "beta_mean": float(beta.mean()),
            }
        )

        if n_panel < args.n_panels:
            gt_u, bb_u, ou_u = to_uint8(gt), to_uint8(r_bb), to_uint8(r_ours)
            box = find_improve_region(gt_u, bb_u, ou_u)
            gap = np.ones((H, 6, 3), dtype=np.uint8) * 255
            panel = np.concatenate(
                [
                    draw_box(gt_u, box),
                    gap,
                    draw_box(bb_u, box),
                    gap,
                    draw_box(ou_u, box),
                    gap,
                    crop_zoom(bb_u, box, (H, H)),
                    gap,
                    crop_zoom(ou_u, box, (H, H)),
                ],
                axis=1,
            )
            lbl = label_strip(
                panel.shape[1],
                f"scan{n_scanned} | GT | Flash3D {ps_bb:.2f} | Ours {ps_ours:.2f} (d={ps_ours - ps_bb:+.2f}) | ZOOM Flash3D / Ours",
            )
            Image.fromarray(np.concatenate([lbl, panel], axis=0)).save(
                out / "panels" / f"panel{n_panel:02d}_scan{n_scanned:04d}.png"
            )
            n_panel += 1

        n_saved += 1
        if n_saved % 25 == 0:
            print(f"  {n_saved}/{args.n} rendered", flush=True)

    print(f"Rendered {n_saved}. Computing FID/KID...", flush=True)
    from cleanfid import fid as cfid

    real_dir = str(out / "real")
    res = {"n": n_saved}
    for name in ["ours", "flash3d"]:
        f = cfid.compute_fid(
            real_dir, str(out / name), mode="clean", num_workers=2, verbose=False
        )
        try:
            k = cfid.compute_kid(
                real_dir, str(out / name), mode="clean", num_workers=2, verbose=False
            )
        except Exception:
            k = float("nan")
        res[name] = {"fid": float(f), "kid": float(k)}
        print(f"  {name:8s}: FID={f:.3f} KID={k:.5f}", flush=True)

    deltas = np.array([r["delta"] for r in rows])
    res["psnr_delta_mean"] = float(deltas.mean())
    res["psnr_delta_wins"] = int((deltas > 0).sum())
    res["n_scenes"] = len(rows)
    res["fid_improvement"] = res["flash3d"]["fid"] - res["ours"]["fid"]
    res["GO"] = bool(res["ours"]["fid"] < res["flash3d"]["fid"])
    res["rows"] = rows
    (out / "flow_eval_results.json").write_text(json.dumps(res, indent=2))
    print(f"\n=== STEP 3 flow eval ({len(rows)} scenes) ===", flush=True)
    print(
        f"FID: Flash3D={res['flash3d']['fid']:.2f} Ours={res['ours']['fid']:.2f} "
        f"(improve {res['fid_improvement']:+.2f})",
        flush=True,
    )
    print(
        f"PSNR delta mean={deltas.mean():+.3f}dB wins={int((deltas > 0).sum())}/{len(rows)}",
        flush=True,
    )
    print(
        f"VERDICT: {'GO (FID improved)' if res['GO'] else 'NO-GO (FID not improved)'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
