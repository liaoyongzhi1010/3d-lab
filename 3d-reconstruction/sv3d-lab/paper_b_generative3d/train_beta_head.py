"""Step 1: train the per-primitive confidence head beta on frozen Flash3D.

beta in [0,1] per source primitive: HIGH where the primitive is unreliable (its
region degrades under novel views), LOW where well-observed. Supervised by the
NOVEL-VIEW render error: splat beta as a color, render to a neighbor view, and match
the rendered-beta to the (detached) Flash3D render-error map on that neighbor.

Frozen: Flash3D backbone. Trainable: a small MLP head on per-primitive 2048-d features
(~2-4M params). This is Step 1 of the flow-matching plan; GO if beta-map correlates
with the disocclusion / degraded region (AUC vs the worst-30%-error mask > ~0.8).

Single-view inference (source only); neighbor frames are supervision, not input (no leak).

Run (Flash3D venv):
  python -m paper_b_generative3d.train_beta_head \
    --steps 3000 --out /home/data/sv3d-lab/reconstruction/E-121-beta
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


class BetaHead(nn.Module):
    """Per-primitive confidence: 2048-d feature -> scalar beta in [0,1]."""

    def __init__(self, in_dim=2048, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # small init so beta starts near 0.5 then learns
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feats):  # feats [N, in_dim]
        return torch.sigmoid(self.net(feats)).squeeze(-1)  # [N]


def sample_primitive_features(backbone, gauss_out, batch_idx=0):
    """Grid-sample Flash3D source features at each primitive's source pixel -> [N,2048]."""
    feats = gauss_out["source_features"][batch_idx]  # [C,hf,wf]
    C = feats.shape[0]
    pixel_hw = gauss_out["pixel_hw"]
    gpp = int(gauss_out["gaussians_per_pixel"])
    N = gauss_out["xyz"].shape[1]
    padded_h, padded_w = int(pixel_hw[0]), int(pixel_hw[1])
    # primitive i -> layer=i//(H*W), pix=i%(H*W), y=pix//W, x=pix%W  (padded grid)
    idx = torch.arange(N, device=feats.device)
    hw = padded_h * padded_w
    pix = idx % hw
    y = (pix // padded_w).float()
    x = (pix % padded_w).float()
    gx = x / max(padded_w - 1, 1) * 2.0 - 1.0
    gy = y / max(padded_h - 1, 1) * 2.0 - 1.0
    grid = torch.stack((gx, gy), dim=1).view(1, N, 1, 2)
    sampled = F.grid_sample(
        feats.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # [1,C,N,1]
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1)  # [N,C]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    cfg = build_flash3d_cfg(batch_size=1, num_workers=2, stage="train")
    backbone = Flash3DBackbone(cfg, FLASH3D_CKPT, device=str(device))
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    loader = build_re10k_dataloader(
        "train", batch_size=1, num_workers=2, max_scenes=None, flash3d_cfg=cfg
    )

    head = BetaHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    H, W = 256, 384

    step = 0
    running = []
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
        feats = sample_primitive_features(backbone, gauss_out)  # [N,2048]
        beta = head(feats)  # [N] in [0,1]

        # pick widest available target for strong disocclusion signal
        fid = tfids[-1]
        cam = inputs.get(("cam_T_cam", 0, fid))
        if cam is None:
            cam = gauss_out.get("_outputs", {}).get(("cam_T_cam", 0, fid))
        if cam is None:
            continue
        K_tgt = inputs.get(("K_tgt", fid), inputs[("K_src", 0)])
        gt = inputs["color", fid, 0][0]

        # render Flash3D + render beta-as-color to the SAME target view
        with torch.no_grad():
            r_bb = backbone.render_gaussians(
                raw, cam[0], K_tgt[0], H, W, batch_idx=0
            ).clamp(0, 1)
            err = (
                (r_bb - gt).abs().mean(dim=0)
            )  # [H,W] target-space error (target = supervision)
            err_blur = F.avg_pool2d(err[None, None], 9, 1, 4)[0, 0]
            err_norm = (err_blur - err_blur.min()) / (
                err_blur.max() - err_blur.min() + 1e-6
            )

        # splat beta as a grayscale color to target view (differentiable in beta)
        beta_rgb = beta[:, None].repeat(1, 3)  # [N,3]
        g_beta = {
            "xyz": raw["xyz"],
            "scales": raw["scales"],
            "rotations": raw["rotations"],
            "opacity": raw["opacity"],
            "color_rgb": beta_rgb,
        }
        beta_render = backbone.render_gaussians(
            g_beta, cam[0], K_tgt[0], H, W, batch_idx=0
        )
        beta_map = beta_render.mean(dim=0)  # [H,W] rendered beta

        # loss: rendered-beta should match normalized render-error (both in [0,1])
        loss = F.binary_cross_entropy(beta_map.clamp(1e-4, 1 - 1e-4), err_norm.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()

        running.append(loss.item())
        step += 1

        if step % args.log_every == 0:
            # AUC of rendered-beta predicting the worst-30% error region
            with torch.no_grad():
                thr = torch.quantile(err_norm.flatten(), 0.70)
                lbl = (err_norm > thr).float().flatten()
                score = beta_map.flatten()
                auc = _fast_auc(score, lbl)
            print(
                f"[step {step}/{args.steps}] loss={np.mean(running[-args.log_every :]):.4f} "
                f"beta_mean={beta.mean().item():.3f} AUC={auc:.3f}",
                flush=True,
            )

        if step % args.ckpt_every == 0:
            torch.save(
                {"head": head.state_dict(), "step": step}, out / f"beta_step{step}.pt"
            )

    torch.save({"head": head.state_dict(), "step": step}, out / "beta_final.pt")
    # final AUC over a few batches
    aucs = []
    for _ in range(30):
        try:
            inputs = next(data_iter)
        except StopIteration:
            break
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
            feats = sample_primitive_features(backbone, gauss_out)
            beta = head(feats)
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
            err = (r_bb - gt).abs().mean(dim=0)
            err_blur = F.avg_pool2d(err[None, None], 9, 1, 4)[0, 0]
            err_norm = (err_blur - err_blur.min()) / (
                err_blur.max() - err_blur.min() + 1e-6
            )
            beta_rgb = beta[:, None].repeat(1, 3)
            g_beta = {
                "xyz": raw["xyz"],
                "scales": raw["scales"],
                "rotations": raw["rotations"],
                "opacity": raw["opacity"],
                "color_rgb": beta_rgb,
            }
            beta_map = backbone.render_gaussians(
                g_beta, cam[0], K_tgt[0], H, W, batch_idx=0
            ).mean(dim=0)
            thr = torch.quantile(err_norm.flatten(), 0.70)
            aucs.append(
                _fast_auc(beta_map.flatten(), (err_norm > thr).float().flatten())
            )
    final_auc = float(np.mean(aucs)) if aucs else float("nan")
    result = {"final_auc": final_auc, "steps": step, "GO": bool(final_auc > 0.8)}
    (out / "beta_results.json").write_text(json.dumps(result, indent=2))
    print(f"\n=== STEP 1 beta head: final AUC={final_auc:.3f} ===", flush=True)
    print(
        f"VERDICT: {'GO (beta aligns with disocclusion)' if final_auc > 0.8 else 'MARGINAL (AUC<0.8)'}",
        flush=True,
    )


def _fast_auc(score, label):
    """AUC via rank statistic. score,label: 1D tensors."""
    label = label.long()
    n_pos = int(label.sum().item())
    n_neg = int((1 - label).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(score)
    ranks = torch.zeros_like(score)
    ranks[order] = torch.arange(
        1, len(score) + 1, device=score.device, dtype=score.dtype
    )
    auc = (ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc.item())


if __name__ == "__main__":
    main()
