"""E-125 Phase-3 test: does the flow head work IF the gate is correct?

Root cause from E-124: source-space per-primitive beta cannot predict target-space smear
(beta AUC 0.695; beta region filled with GT gave only +0.29 FID vs Oracle error-mask +9.82dB).

This isolates ONE variable: gate correctness. We DROP the learned beta gate and instead
blend in IMAGE SPACE using the target render-error mask (worst-30% Flash3D error, the SAME
mask that gave Oracle +9.82dB). Train-time GT is allowed for the mask (it's supervision).

  render_final = flow_render * error_mask + flash3d_render * (1 - error_mask)

If FID now IMPROVES over Flash3D -> hypothesis confirmed: the gate was the root cause, and
a correct (target-space) gate makes the flow head work. Then Step: build a target-view-
conditioned mask predictor for inference. If FID still bad -> flow head itself is the problem.

NOTE: this variant uses the error mask at train time only to LOCATE where to apply the
generated content; it is a diagnostic to isolate the gate. Not the final inference method.

Run (Flash3D venv):
  python -m paper_b_generative3d.train_flow_v2 --steps 4000 --out .../E-125-flowv2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from paper_b_generative3d.backbone_flash3d import Flash3DBackbone
from paper_b_generative3d.data_loader_re10k import (
    build_flash3d_cfg,
    build_re10k_dataloader,
    FLASH3D_CKPT,
)
from paper_b_generative3d.viz_compare import psnr
from paper_b_generative3d.train_beta_head import sample_primitive_features
from paper_b_generative3d.train_flow_head import (
    FlowHead,
    apply_bounded_residual,
    timestep_embed,
    RES_DIM,
)
from paper_b_generative3d.train_real import get_lpips_fn


def error_mask(r_bb, gt, pct=0.70):
    """Target-space worst-(1-pct) render-error mask (same as Oracle E-119)."""
    err = (r_bb - gt).abs().mean(dim=0)
    err_blur = F.avg_pool2d(err[None, None], 9, 1, 4)[0, 0]
    thr = torch.quantile(err_blur.flatten(), pct)
    m = (err_blur > thr).float()[None, None]
    m = F.max_pool2d(m, 3, 1, 1)
    m = -F.max_pool2d(-m, 3, 1, 1)
    return m[0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_lpips", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    cfg = build_flash3d_cfg(batch_size=1, num_workers=2, stage="train")
    backbone = Flash3DBackbone(cfg, FLASH3D_CKPT, device=str(device))
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    flow = FlowHead().to(device)
    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr)
    lpips_fn = get_lpips_fn(device)
    loader = build_re10k_dataloader(
        "train", batch_size=1, num_workers=2, max_scenes=None, flash3d_cfg=cfg
    )
    H, W = 256, 384

    def sscale_of(m):
        c = m.mean(0, keepdim=True)
        return (m - c).norm(dim=1).mean().clamp_min(1e-3)

    step = 0
    running = []
    di = iter(loader)
    while step < args.steps:
        try:
            inputs = next(di)
        except StopIteration:
            di = iter(loader)
            inputs = next(di)
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
            sscale = sscale_of(raw["xyz"])
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
            emask = error_mask(r_bb, gt)  # [H,W] target-space, worst-30%

        # flow residual applied to ALL primitives (gate=1), then IMAGE-SPACE blend by emask
        N = feat.shape[0]
        ones = torch.ones(N, device=device)
        x0r = torch.randn(N, RES_DIM, device=device)
        tr = torch.full((N,), 0.5, device=device)
        v0 = flow(x0r, tr, feat, ones)
        x_end = x0r + 0.5 * v0
        means, color, opacity = apply_bounded_residual(
            raw["xyz"], raw["color_rgb"], raw["opacity"], x_end, ones, sscale
        )
        r_flow = backbone.render_gaussians(
            {
                "xyz": means,
                "scales": raw["scales"],
                "rotations": raw["rotations"],
                "opacity": opacity,
                "color_rgb": color,
            },
            cam[0],
            K_tgt[0],
            H,
            W,
            batch_idx=0,
        )
        m3 = emask[None].repeat(3, 1, 1)
        render = r_flow * m3 + r_bb.detach() * (
            1 - m3
        )  # blend: flow only in error region

        loss_l1 = F.l1_loss(render.clamp(0, 1), gt)
        loss_lp = lpips_fn(render.clamp(0, 1).unsqueeze(0), gt.unsqueeze(0)).mean()
        # FM straightness regularizer (stable, detached endpoint)
        x_tgt = x_end.detach()
        x0f = torch.randn(N, RES_DIM, device=device)
        tf = torch.rand(N, device=device)
        x_tf = (1 - tf)[:, None] * x0f + tf[:, None] * x_tgt
        v_pred = flow(x_tf, tf, feat, ones)
        loss_fm = F.mse_loss(v_pred, (x_tgt - x0f))
        loss = loss_l1 + args.lambda_lpips * loss_lp + 0.1 * loss_fm

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step()
        running.append(loss.item())
        step += 1

        if step % args.log_every == 0:
            with torch.no_grad():
                ps_bb = psnr(r_bb, gt)
                ps_blend = psnr(render.clamp(0, 1), gt)
            print(
                f"[step {step}/{args.steps}] loss={np.mean(running[-args.log_every :]):.4f} "
                f"| Flash3D={ps_bb:.2f} Blend={ps_blend:.2f} d={ps_blend - ps_bb:+.2f} "
                f"mask={float(emask.mean()):.2f}",
                flush=True,
            )
        if step % args.ckpt_every == 0:
            torch.save(
                {"flow": flow.state_dict(), "step": step}, out / f"flowv2_step{step}.pt"
            )

    torch.save({"flow": flow.state_dict(), "step": step}, out / "flowv2_final.pt")
    print(f"\n=== E-125 flow-v2 done step {step} ===", flush=True)


if __name__ == "__main__":
    main()
