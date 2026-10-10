"""E-131: Disocclusion-focused GENERATIVE refinement of Flash3D gaussians.

Diagnosis (E-130/E-131): the water-dispenser / disocclusion smear is NOT a hole (target
coverage is ~100%) -- it is gaussians present but with WRONG color/position (brown streak).
The content is out-of-source-frame, so it must be GENERATED, not copied.

This extends the E-129 gated smear-refine head with:
  1. Free-color REGENERATION branch (absolute color, gated) so brown-smear gaussians can be
     recolored arbitrarily (not just +-0.3 residual).
  2. Larger position/scale bounds (max_dxyz=0.5, max_dlogscale=2.0) so smear gaussians can be
     relocated to the correct background depth.
  3. Disocclusion-weighted loss: weight the render loss by Flash3D's per-pixel error so the
     head spends capacity on smear/disocclusion regions (where Flash3D is wrong).
  4. Keep reliability gate + PatchGAN adversarial (sharpness) + real-neighbor NVS supervision.

Honest scope: out-of-frame content (the exact dispenser) cannot be pixel-recovered; goal is
to turn brown smear into PLAUSIBLE SHARP content -> better LPIPS/FID in disocclusion regions.

Run (flash3d venv):
  python -m paper_b_generative3d.train_disocc_refine \
    --steps 3000 --out /home/data/sv3d-lab/reconstruction/E-131-disocc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from paper_b_generative3d.backbone_flash3d import Flash3DBackbone
from paper_b_generative3d.data_loader_re10k import build_flash3d_cfg, FLASH3D_CKPT
from paper_b_generative3d.viz_compare import psnr, _create_loader_from_cfg
from paper_b_generative3d.model.smear_refine_head import (
    SmearRefineHead,
    PatchDiscriminator,
)
from paper_b_generative3d.train_real import get_lpips_fn
from paper_b_generative3d.train_smear_refine import sample_per_gaussian_feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_d", type=float, default=2e-4)
    ap.add_argument("--lambda_l1", type=float, default=1.0)
    ap.add_argument("--lambda_lpips", type=float, default=1.0)
    ap.add_argument("--lambda_adv", type=float, default=0.05)
    ap.add_argument("--lambda_src", type=float, default=1.0)
    ap.add_argument("--lambda_scale", type=float, default=0.05)
    ap.add_argument("--lambda_gate", type=float, default=0.02)
    ap.add_argument(
        "--disocc_weight",
        type=float,
        default=4.0,
        help="extra loss weight in high-Flash3D-error (smear/disocc) regions",
    )
    ap.add_argument("--adv_warmup", type=int, default=500)
    ap.add_argument("--adv_cap_step", type=int, default=2000)
    ap.add_argument("--max_dxyz", type=float, default=0.5)
    ap.add_argument("--max_dlogscale", type=float, default=2.0)
    ap.add_argument("--feat_dim", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=500)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    cfg = build_flash3d_cfg(batch_size=1, num_workers=0, stage="dev")
    backbone = Flash3DBackbone(cfg, FLASH3D_CKPT, device=str(device))
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    head = SmearRefineHead(
        feat_dim=args.feat_dim,
        hidden=256,
        n_layers=4,
        max_dxyz=args.max_dxyz,
        max_dlogscale=args.max_dlogscale,
        max_dcolor=0.3,
    ).to(device)
    disc = PatchDiscriminator().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr_d)
    lpips_fn = get_lpips_fn(device)
    print(
        f"Head {sum(p.numel() for p in head.parameters()) / 1e6:.1f}M | "
        f"Disc {sum(p.numel() for p in disc.parameters()) / 1e6:.1f}M | "
        f"max_dxyz={args.max_dxyz} regen-color ON disocc_w={args.disocc_weight}",
        flush=True,
    )

    _, loader = _create_loader_from_cfg(cfg)
    H, W = 256, 384

    step = 0
    run = {
        "l1": [],
        "lpips": [],
        "adv": [],
        "src": [],
        "dloss": [],
        "gate": [],
        "regen": [],
    }
    it = iter(loader)
    while step < args.steps:
        try:
            inputs = next(it)
        except StopIteration:
            it = iter(loader)
            inputs = next(it)
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
            k: g[k][0] for k in ["xyz", "scales", "rotations", "opacity", "color_rgb"]
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

        fid = tfids[torch.randint(len(tfids), (1,)).item()]
        T_w2c_t = inputs.get(("T_w2c", fid))
        if T_w2c_t is None:
            continue
        cam = T_w2c_t[0] @ T_c2w_s[0]
        K_tgt = inputs.get(("K_tgt", fid), inputs[("K_src", 0)])
        if K_tgt.dim() == 3:
            K_tgt = K_tgt[0]
        gt = inputs["color", fid, 0][0]

        with torch.no_grad():
            r_flash = backbone.render_gaussians(anchor, cam, K_tgt, H, W).clamp(0, 1)
            # disocclusion weight map: high where Flash3D is wrong (smear regions)
            err = (r_flash - gt).abs().mean(0, keepdim=True)  # [1,H,W]
            disocc_w = 1.0 + args.disocc_weight * (err / (err.max() + 1e-6))  # [1,H,W]

        render = backbone.render_gaussians(refined_g, cam, K_tgt, H, W).clamp(0, 1)

        cam_src = torch.eye(4, device=device)
        K_src = inputs[("K_src", 0)][0]
        src_gt = inputs["color", 0, 0][0]
        render_src = backbone.render_gaussians(refined_g, cam_src, K_src, H, W).clamp(
            0, 1
        )

        use_adv = step >= args.adv_warmup
        if use_adv:
            opt_d.zero_grad()
            with torch.no_grad():
                fake = render.unsqueeze(0)
            d_real = disc(gt.unsqueeze(0))
            d_fake = disc(fake)
            d_loss = 0.5 * (F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean())
            d_loss.backward()
            opt_d.step()
        else:
            d_loss = torch.tensor(0.0, device=device)

        opt.zero_grad()
        # disocclusion-weighted L1
        loss_l1 = (disocc_w * (render - gt).abs()).mean()
        loss_lpips = lpips_fn(render.unsqueeze(0), gt.unsqueeze(0)).mean()
        loss_src = (
            F.l1_loss(render_src, src_gt)
            + 0.5 * lpips_fn(render_src.unsqueeze(0), src_gt.unsqueeze(0)).mean()
        )
        loss_scale = refined["scales"].mean()
        loss_gate = refined["gate"].mean()
        if use_adv:
            g_adv = -disc(render.unsqueeze(0)).mean()
        else:
            g_adv = torch.tensor(0.0, device=device)
        if step < args.adv_warmup:
            adv_w = 0.0
        elif step < args.adv_cap_step:
            adv_w = (
                args.lambda_adv
                * (step - args.adv_warmup)
                / max(1, args.adv_cap_step - args.adv_warmup)
            )
        else:
            adv_w = args.lambda_adv
        loss = (
            args.lambda_l1 * loss_l1
            + args.lambda_lpips * loss_lpips
            + adv_w * g_adv
            + args.lambda_src * loss_src
            + args.lambda_scale * loss_scale
            + args.lambda_gate * loss_gate
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()

        run["l1"].append(loss_l1.item())
        run["lpips"].append(loss_lpips.item())
        run["adv"].append(float(g_adv))
        run["src"].append(loss_src.item())
        run["dloss"].append(float(d_loss))
        run["gate"].append(float(refined["gate"].mean()))
        run["regen"].append(float(refined["regen_w"].mean()))
        step += 1

        if step % args.log_every == 0:
            print(
                f"[step {step}/{args.steps}] l1={np.mean(run['l1'][-args.log_every :]):.4f} "
                f"lpips={np.mean(run['lpips'][-args.log_every :]):.4f} "
                f"adv={np.mean(run['adv'][-args.log_every :]):.3f} "
                f"src={np.mean(run['src'][-args.log_every :]):.4f} "
                f"dloss={np.mean(run['dloss'][-args.log_every :]):.3f} "
                f"gate={np.mean(run['gate'][-args.log_every :]):.3f} "
                f"regen={np.mean(run['regen'][-args.log_every :]):.3f}",
                flush=True,
            )
        if step % args.ckpt_every == 0:
            torch.save(
                {"head": head.state_dict(), "disc": disc.state_dict(), "step": step},
                out / f"refine_step{step}.pt",
            )

    torch.save(
        {"head": head.state_dict(), "disc": disc.state_dict(), "step": step},
        out / "refine_final.pt",
    )
    print(f"=== E-131 done at step {step} ===", flush=True)


if __name__ == "__main__":
    main()
