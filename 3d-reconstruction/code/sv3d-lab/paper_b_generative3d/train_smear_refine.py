"""Stage 1: Train the SmearRefineHead against REAL GT novel views.

Resolves the regression-vs-generation contradiction:
  - Regression side (stable geometry, one shared scene): frozen Flash3D backbone gives
    the initial gaussians; the refinement is view-independent (single scene per source).
  - Generation side (sharp, no smear): the head corrects scale/rotation/position (the
    smear cause) trained with LPIPS + a PatchGAN adversarial loss on rendered novel
    views, which breaks the mean-collapse blur that L1-only regression (E-114) fell into.

Losses:
  1. L1 on rendered novel view vs GT  (geometric anchoring)
  2. LPIPS(VGG) on rendered novel view vs GT  (perceptual sharpness)
  3. Adversarial: PatchGAN D distinguishes render vs real; head fools D  (anti-mean-collapse)
  4. Source preservation: render at source pose ~ source image  (don't break visible region)
  5. Scale regularizer: penalize growth (encourage shrinking smear, not adding it)

Anti-collapse / no-target-leakage: head sees only source features + own params; corrections
are computed once (view independent), rendered to multiple targets.

GO: novel-view LPIPS beats Flash3D by >0.01 AND source PSNR not worse by >0.2 dB.

Run (flash3d venv):
  python -m paper_b_generative3d.train_smear_refine \
    --steps 4000 --out /home/data/sv3d-lab/reconstruction/E-128-smear-refine
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
from paper_b_generative3d.data_loader_re10k import build_flash3d_cfg, FLASH3D_CKPT
from paper_b_generative3d.viz_compare import psnr, _create_loader_from_cfg
from paper_b_generative3d.model.smear_refine_head import (
    SmearRefineHead,
    PatchDiscriminator,
)
from paper_b_generative3d.train_real import get_lpips_fn


def sample_per_gaussian_feats(source_features, n, pixel_hw, gpp, device):
    """Map flat gaussian idx -> source pixel -> bilinear sample source_features [C,hf,wf].

    Returns [n, C]. Mirrors reconstruction_interface.per_primitive_features layout.
    """
    feats = source_features
    if feats.dim() == 4:
        feats = feats[0]
    C, hf, wf = feats.shape
    padded_h, padded_w = int(pixel_hw[0]), int(pixel_hw[1])
    area = padded_h * padded_w
    idx = torch.arange(n, device=device)
    pix = idx % area
    y = (pix // padded_w).float()
    x = (pix % padded_w).float()
    gx = x / max(padded_w - 1, 1) * 2.0 - 1.0
    gy = y / max(padded_h - 1, 1) * 2.0 - 1.0
    grid = torch.stack((gx, gy), dim=1).view(1, n, 1, 2)
    sampled = F.grid_sample(
        feats.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1)  # [n,C]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_d", type=float, default=2e-4)
    ap.add_argument("--lambda_l1", type=float, default=1.0)
    ap.add_argument("--lambda_lpips", type=float, default=1.0)
    ap.add_argument("--lambda_adv", type=float, default=0.05)
    ap.add_argument("--lambda_src", type=float, default=1.0)
    ap.add_argument("--lambda_scale", type=float, default=0.1)
    ap.add_argument("--lambda_gate", type=float, default=0.02)
    ap.add_argument("--adv_warmup", type=int, default=500)
    ap.add_argument("--adv_cap_step", type=int, default=2500)
    ap.add_argument("--feat_dim", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    cfg = build_flash3d_cfg(batch_size=1, num_workers=0, stage="dev")
    backbone = Flash3DBackbone(cfg, FLASH3D_CKPT, device=str(device))
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    head = SmearRefineHead(feat_dim=args.feat_dim, hidden=256, n_layers=4).to(device)
    disc = PatchDiscriminator().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr_d)
    lpips_fn = get_lpips_fn(device)
    print(
        f"Head params: {sum(p.numel() for p in head.parameters()) / 1e6:.1f}M | "
        f"Disc params: {sum(p.numel() for p in disc.parameters()) / 1e6:.1f}M",
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
        "rnorm": [],
        "gate": [],
    }
    data_iter = iter(loader)
    while step < args.steps:
        try:
            inputs = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inputs = next(data_iter)
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

        # ---- render novel views + source ----
        # pick one target for adversarial+render loss (random among tfids)
        fid = tfids[torch.randint(len(tfids), (1,)).item()]
        T_w2c_t = inputs.get(("T_w2c", fid))
        if T_w2c_t is None:
            continue
        cam = T_w2c_t[0] @ T_c2w_s[0]
        K_tgt = inputs.get(("K_tgt", fid), inputs[("K_src", 0)])
        if K_tgt.dim() == 3:
            K_tgt = K_tgt[0]
        gt = inputs["color", fid, 0][0]

        render = backbone.render_gaussians(refined_g, cam, K_tgt, H, W).clamp(0, 1)

        # source preservation
        cam_src = torch.eye(4, device=device)
        K_src = inputs[("K_src", 0)][0]
        src_gt = inputs["color", 0, 0][0]
        render_src = backbone.render_gaussians(refined_g, cam_src, K_src, H, W).clamp(
            0, 1
        )

        # ---- discriminator step ----
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

        # ---- generator (head) step ----
        opt.zero_grad()
        loss_l1 = F.l1_loss(render, gt)
        loss_lpips = lpips_fn(render.unsqueeze(0), gt.unsqueeze(0)).mean()
        loss_src = (
            F.l1_loss(render_src, src_gt)
            + 0.5 * lpips_fn(render_src.unsqueeze(0), src_gt.unsqueeze(0)).mean()
        )
        loss_scale = refined["scales"].mean()
        loss_gate = refined["gate"].mean()  # sparsity: only edit where it helps
        if use_adv:
            g_adv = -disc(render.unsqueeze(0)).mean()
        else:
            g_adv = torch.tensor(0.0, device=device)
        # Cap adversarial influence after adv_cap_step to prevent GAN over-training
        # (E-128 saw quality regress past step ~2500 as adv climbed). Ramp then hold.
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
        run["rnorm"].append(float(refined["_residual_norm"]))
        run["gate"].append(float(refined["gate"].mean()))
        step += 1

        if step % args.log_every == 0:
            print(
                f"[step {step}/{args.steps}] l1={np.mean(run['l1'][-args.log_every :]):.4f} "
                f"lpips={np.mean(run['lpips'][-args.log_every :]):.4f} "
                f"adv={np.mean(run['adv'][-args.log_every :]):.3f} "
                f"src={np.mean(run['src'][-args.log_every :]):.4f} "
                f"dloss={np.mean(run['dloss'][-args.log_every :]):.3f} "
                f"gate={np.mean(run['gate'][-args.log_every :]):.3f} "
                f"rnorm={np.mean(run['rnorm'][-args.log_every :]):.4f}",
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
    print(f"=== Stage 1 done at step {step} ===", flush=True)


if __name__ == "__main__":
    main()
