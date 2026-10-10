"""Step 2: flow-matching residual sampling head on frozen Flash3D (beta-gated).

The generative core. In the beta-flagged unreliable region, a Rectified-Flow velocity
field SAMPLES bounded residual Gaussian attributes (offset, color, opacity), instead of
regressing the (blurry) mean. Final gaussian = Flash3D base + beta * bounded_residual(sample).

Trained by REAL multi-view neighbor render loss (L1 + LPIPS-VGG), region-weighted toward
the disoccluded/beta region, jointly with the flow-matching velocity objective. This is
3D-native (residual on explicit gaussians), single-view inference (source only; neighbor
frames are supervision, not input), and starts == Flash3D (zero-init velocity -> zero residual).

Anti-speckle: every residual channel is tanh-bounded (offset in fraction of scene scale,
color/opacity in logit space); beta gate limits where residual applies; base is stop-grad.

Run (Flash3D venv, after Step 1 beta head is trained):
  python -m paper_b_generative3d.train_flow_head \
    --beta_ckpt /home/data/sv3d-lab/reconstruction/E-121-beta/beta_final.pt \
    --steps 8000 --out /home/data/sv3d-lab/reconstruction/E-122-flow
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
from paper_b_generative3d.train_beta_head import BetaHead, sample_primitive_features
from paper_b_generative3d.train_real import get_lpips_fn

# residual dims: offset(3) + color(3) + opacity(1) = 7
RES_DIM = 7


def timestep_embed(t, dim=128):
    """Sinusoidal embedding of t in [0,1]. t: [N] -> [N,dim]."""
    half = dim // 2
    freqs = torch.exp(
        -np.log(10000) * torch.arange(half, device=t.device).float() / half
    )
    ang = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class FlowHead(nn.Module):
    """Per-primitive Rectified-Flow velocity field.

    Input:  x_t [N,7] (noisy residual), t [N], cond feat [N,Cf], beta [N]
    Output: velocity v [N,7]
    """

    def __init__(self, feat_dim=2048, cond=256, hidden=512, tdim=128, blocks=4):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, cond)
        self.in_proj = nn.Linear(RES_DIM + cond + tdim + 1, hidden)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
                )
                for _ in range(blocks)
            ]
        )
        self.out = nn.Linear(hidden, RES_DIM)
        self.tdim = tdim
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)  # zero-init: residual starts at 0 -> == Flash3D

    def forward(self, x_t, t, feat, beta):
        c = torch.tanh(self.feat_proj(feat))
        temb = timestep_embed(t, self.tdim)
        h = torch.cat([x_t, c, temb, beta[:, None]], dim=-1)
        h = self.in_proj(h)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


def apply_bounded_residual(
    base_means,
    base_color,
    base_opacity,
    res,
    beta,
    scene_scale,
    frac=0.05,
    color_logit=2.0,
    opacity_logit=2.0,
):
    """Apply beta-gated tanh-bounded residual (anti-speckle). base is stop-grad."""
    bm = base_means.detach()
    bc = base_color.detach()
    bo = base_opacity.detach()
    g = beta[:, None]
    mean_delta = frac * scene_scale * torch.tanh(res[:, 0:3])
    means = bm + g * mean_delta
    color_l = torch.logit(bc.clamp(1e-4, 1 - 1e-4))
    color = torch.sigmoid(color_l + g * color_logit * torch.tanh(res[:, 3:6]))
    op_l = torch.logit(bo.clamp(1e-4, 1 - 1e-4))
    opacity = torch.sigmoid(op_l + g * opacity_logit * torch.tanh(res[:, 6:7])).clamp(
        0, 1
    )
    return means, color, opacity


@torch.no_grad()
def sample_residual(flow, feat, beta, nfe=8):
    """Integrate the RF ODE from noise to residual sample. Euler, nfe steps."""
    N = feat.shape[0]
    x = torch.randn(N, RES_DIM, device=feat.device)
    dt = 1.0 / nfe
    for i in range(nfe):
        t = torch.full((N,), i * dt, device=feat.device)
        v = flow(x, t, feat, beta)
        x = x + dt * v
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta_ckpt", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_photo", type=float, default=1.0)
    ap.add_argument("--lambda_fm", type=float, default=1.0)
    ap.add_argument("--lambda_lpips", type=float, default=0.5)
    ap.add_argument("--nfe", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log_every", type=int, default=100)
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

    beta_head = BetaHead().to(device)
    bsd = torch.load(args.beta_ckpt, map_location="cpu")
    beta_head.load_state_dict(bsd["head"])
    beta_head.eval()
    for p in beta_head.parameters():
        p.requires_grad_(False)

    flow = FlowHead().to(device)
    opt = torch.optim.AdamW(flow.parameters(), lr=args.lr)
    lpips_fn = get_lpips_fn(device)

    loader = build_re10k_dataloader(
        "train", batch_size=1, num_workers=2, max_scenes=None, flash3d_cfg=cfg
    )
    H, W = 256, 384

    def scene_scale_of(means):
        c = means.mean(0, keepdim=True)
        return (means - c).norm(dim=1).mean().clamp_min(1e-3)

    step = 0
    running = {"loss": [], "fm": [], "photo": []}
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
            feat = sample_primitive_features(backbone, gauss_out)  # [N,2048]
            beta = beta_head(feat)  # [N]
            sscale = scene_scale_of(raw["xyz"])

        # --- render-loss branch FIRST: one differentiable residual draw (memory-safe) ---
        # A single flow eval from t=0.5 gives an endpoint estimate; the render loss
        # backprops into the velocity field to teach WHAT residual to produce. This is
        # the real learning signal.
        N = feat.shape[0]
        x0r = torch.randn(N, RES_DIM, device=device)
        tr = torch.full((N,), 0.5, device=device)
        v0 = flow(x0r, tr, feat, beta)
        x_endpoint = x0r + 0.5 * v0  # endpoint estimate from midpoint
        means, color, opacity = apply_bounded_residual(
            raw["xyz"], raw["color_rgb"], raw["opacity"], x_endpoint, beta, sscale
        )
        g_ref = {
            "xyz": means,
            "scales": raw["scales"],
            "rotations": raw["rotations"],
            "opacity": opacity,
            "color_rgb": color,
        }

        fid = tfids[-1]
        cam = inputs.get(("cam_T_cam", 0, fid))
        if cam is None:
            cam = gauss_out.get("_outputs", {}).get(("cam_T_cam", 0, fid))
        if cam is None:
            continue
        K_tgt = inputs.get(("K_tgt", fid), inputs[("K_src", 0)])
        gt = inputs["color", fid, 0][0]
        render = backbone.render_gaussians(g_ref, cam[0], K_tgt[0], H, W, batch_idx=0)

        loss_l1 = F.l1_loss(render.clamp(0, 1), gt)
        loss_lp = lpips_fn(render.clamp(0, 1).unsqueeze(0), gt.unsqueeze(0)).mean()
        loss_photo = loss_l1 + args.lambda_lpips * loss_lp

        # --- FM straightness regularizer (STABLE): transport noise -> the render-driven
        # endpoint (detached). Target is bounded by the render loss, so no self-feedback
        # blow-up (the earlier self-conditioned x1 diverged). Keeps the velocity field a
        # valid few-step sampler for inference without dominating the learning signal.
        x_tgt = x_endpoint.detach()
        x0f = torch.randn(N, RES_DIM, device=device)
        tf = torch.rand(N, device=device)
        x_tf = (1 - tf)[:, None] * x0f + tf[:, None] * x_tgt
        v_pred = flow(x_tf, tf, feat, beta)
        loss_fm = F.mse_loss(v_pred, (x_tgt - x0f))

        loss = args.lambda_photo * loss_photo + args.lambda_fm * loss_fm
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step()

        running["loss"].append(loss.item())
        running["fm"].append(loss_fm.item())
        running["photo"].append(loss_photo.item())
        step += 1

        if step % args.log_every == 0:
            with torch.no_grad():
                r_bb = backbone.render_gaussians(
                    raw, cam[0], K_tgt[0], H, W, batch_idx=0
                ).clamp(0, 1)
                ps_bb = psnr(r_bb, gt)
                ps_ours = psnr(render.clamp(0, 1), gt)
            print(
                f"[step {step}/{args.steps}] loss={np.mean(running['loss'][-args.log_every :]):.4f} "
                f"fm={np.mean(running['fm'][-args.log_every :]):.4f} "
                f"photo={np.mean(running['photo'][-args.log_every :]):.4f} "
                f"| Flash3D={ps_bb:.2f} Ours={ps_ours:.2f} d={ps_ours - ps_bb:+.2f}",
                flush=True,
            )

        if step % args.ckpt_every == 0:
            torch.save(
                {"flow": flow.state_dict(), "step": step}, out / f"flow_step{step}.pt"
            )

    torch.save({"flow": flow.state_dict(), "step": step}, out / "flow_final.pt")
    print(f"\n=== STEP 2 flow head done at step {step} ===", flush=True)


if __name__ == "__main__":
    main()
