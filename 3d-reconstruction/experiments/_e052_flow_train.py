"""E-052 v3: conditional FLOW-MATCHING generator for hole-region 3D Gaussians.

MODERN generative prior (rectified flow, like SD3/Flux/TRELLIS/Gen3R) — NOT GAN, NOT 2D.
Generates EXPLICIT per-pixel 3D-Gaussian G-buffer at hole pixels, conditioned on the
visible Flash3D context. Output is composited onto Flash3D base -> true 3D hole gaussians.

WHY FLOW (vs v1 regression / GAN):
  - p(hole gaussians | visible context) is MULTI-MODAL (many plausible completions).
  - Regression collapses to the mean -> blur/flat slab (v1 failure).
  - Flow matching learns the full conditional distribution -> sharp SAMPLES, not averages.
  - Deterministic teacher labels give a clean transport target.

Target x1 (per-pixel, normalized): [z, sh_dc(3), logscale(3), opa(1), rot(4)] = 12ch @ 256x384
  normalized by teacher-corpus per-channel mean/std (measured):
    z    mean 3.41 std 3.09
    shdc mean ~0   std 0.76
    logs mean -4.43 std 0.78
    opa  mean 2.43 std 0.32
    rot  mean [1,0,0,0] std ~0.02
Condition c: base_render(3) + hole_mask(1) + depth_filled(1) = 5ch
Path: x_t = (1-t) x0 + t x1,  x0~N(0,I);  target velocity u = x1 - x0
Loss: || v_theta(x_t, t, c) - u ||^2  over HOLE pixels (masked)
Net: spatial UNet, in = [x_t(12) + c(5) + t_embed(broadcast)] -> velocity(12)
Sample: Euler from noise, N steps, conditioned on c.

Run:
  python _e052_flow_train.py --labels /home/data/E-052_teacher_cache/labels \
    --epochs 200 --batch 8 --out /home/data/E-052_flow
"""

import os, sys, math, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/root/projects/flash3d")
os.chdir("/root/projects/flash3d")

DEVICE = "cuda:0"

# teacher-corpus per-channel normalization (measured over 800 labels, hole pixels)
CH_MEAN = torch.tensor(
    [3.41, 0.087, -0.029, -0.243, -4.43, -4.43, -4.45, 2.43, 1.0, 0.0, 0.0, 0.0]
)
CH_STD = torch.tensor(
    [3.09, 0.764, 0.757, 0.773, 0.787, 0.785, 0.764, 0.321, 0.05, 0.05, 0.05, 0.05]
)


def sinusoidal_embedding(t, dim=128):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class FlowUNet(nn.Module):
    """Spatial UNet velocity field: in=[x_t(12)+cond(5)]=17 (+t via FiLM) -> velocity(12)."""

    def __init__(self, x_ch=12, cond_ch=5, base=96, t_dim=128):
        super().__init__()
        self.x_ch = x_ch
        in_ch = x_ch + cond_ch
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, base * 4), nn.SiLU(), nn.Linear(base * 4, base * 4)
        )
        self.t_dim = t_dim

        def cbr(i, o, s=1):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.SiLU()
            )

        self.e0 = cbr(in_ch, base)  # base
        self.e1 = cbr(base, base * 2, 2)  # base*2
        self.e2 = cbr(base * 2, base * 4, 2)  # base*4
        self.e3 = cbr(base * 4, base * 4, 2)  # base*4
        self.mid = cbr(base * 4, base * 4)  # base*4
        # FiLM from t at bottleneck
        self.film = nn.Linear(base * 4, base * 4 * 2)
        self.d2 = cbr(base * 4 + base * 4, base * 4)  # up(mid)+e3 = 4+4 -> 4
        self.d1 = cbr(base * 4 + base * 4, base * 2)  # up(d2)+e2 = 4+4 -> 2
        self.d0 = cbr(base * 2 + base * 2, base)  # up(d1)+e1 = 2+2 -> 1
        self.final = cbr(base + base, base)  # up(d0)+e0 = 1+1 -> 1
        self.head = nn.Conv2d(base, x_ch, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_t, cond, t):
        temb = self.t_mlp(sinusoidal_embedding(t, self.t_dim))
        h = torch.cat([x_t, cond], dim=1)
        e0 = self.e0(h)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        m = self.mid(e3)
        scale, shift = self.film(temb).chunk(2, dim=1)
        m = m * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        def up_to(t_, ref):
            return F.interpolate(
                t_, size=ref.shape[-2:], mode="bilinear", align_corners=False
            )

        d2 = self.d2(torch.cat([up_to(m, e3), e3], 1))
        d1 = self.d1(torch.cat([up_to(d2, e2), e2], 1))
        d0 = self.d0(torch.cat([up_to(d1, e1), e1], 1))
        f = self.final(torch.cat([up_to(d0, e0), e0], 1))
        return self.head(f)


class FlowLabelDataset(Dataset):
    def __init__(self, label_dir, min_hole=0.03):
        self.files = [f for f in sorted(Path(label_dir).glob("*.pt"))]
        self.min_hole = min_hole

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], map_location="cpu")
        target = torch.cat(
            [
                d["gbuf_z"].float(),
                d["gbuf_shdc"].float(),
                d["gbuf_logscale"].float(),
                d["gbuf_opa"].float(),
                d["gbuf_rot"].float(),
            ],
            dim=0,
        )  # [12,H,W]
        # normalize
        target = (target - CH_MEAN[:, None, None]) / CH_STD[:, None, None]
        cond = torch.cat(
            [
                d["base_render"].float(),
                d["hole_mask"].float().unsqueeze(0),
                d["depth_filled"].float(),
            ],
            dim=0,
        )  # [5,H,W]
        return {"target": target, "cond": cond, "hole_mask": d["hole_mask"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="/home/data/E-052_teacher_cache/labels")
    ap.add_argument("--out", default="/home/data/E-052_flow")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base", type=int, default=96)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = FlowLabelDataset(args.labels)
    print(f"Dataset: {len(ds)} labels", flush=True)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    net = FlowUNet(x_ch=12, cond_ch=5, base=args.base).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(dl))
    print(
        f"Flow net params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M",
        flush=True,
    )

    step = 0
    for epoch in range(args.epochs):
        net.train()
        for batch in dl:
            x1 = batch["target"].to(DEVICE)
            cond = batch["cond"].to(DEVICE)
            mask = batch["hole_mask"].to(DEVICE).float().unsqueeze(1)
            B = x1.shape[0]
            x0 = torch.randn_like(x1)
            t = torch.rand(B, device=DEVICE)
            t_b = t[:, None, None, None]
            x_t = (1 - t_b) * x0 + t_b * x1
            u = x1 - x0  # target velocity (rectified flow)
            v = net(x_t, cond, t)
            loss = (((v - u) ** 2) * mask).sum() / (mask.sum() * 12 + 1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % args.log_every == 0:
                print(
                    f"[ep{epoch} step{step}] flow_loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.6f}",
                    flush=True,
                )

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            torch.save(
                net.state_dict(), os.path.join(args.out, f"flow_ep{epoch + 1:03d}.pt")
            )
            print(f"  saved flow_ep{epoch + 1:03d}.pt", flush=True)

    torch.save(net.state_dict(), os.path.join(args.out, "flow_final.pt"))
    print(f"DONE -> {args.out}/flow_final.pt", flush=True)


if __name__ == "__main__":
    main()
