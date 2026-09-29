"""E-052 student: feed-forward hole G-buffer predictor.

INPUT (all at target frame, 256x384):
  base_render[3] + hole_mask[1] + depth_filled[1] = 5ch

OUTPUT (per-pixel G-buffer at hole pixels):
  gbuf_z(1) + gbuf_shdc(3) + gbuf_logscale(3) + gbuf_opa(1) + gbuf_rot(4) = 12ch

LOSS: (all masked to hole pixels)
  1. param L1: regress teacher G-buffer channels
  2. render LPIPS + L1: composite pred_hole_gaussians + base_pc -> render vs teacher_render

ARCHITECTURE: UNet (same as CG.DirectUNet from paper-1, proven to work at this resolution).

WHY this avoids mean-regression blur (the E-006/E-040 trap):
  - Teacher provides a SINGLE sharp deterministic target (SD-anchor is deterministic for the
    same input, multiview fit converges to one solution).
  - Student regresses that single target → well-posed, no averaging over modes.
  - At test time: student runs once (no SD, no optimization) → training-式前馈, legit.

Run:
  python _e052_student_train.py --labels /home/data/E-052_teacher_cache/labels \
    --epochs 30 --lr 2e-4 --batch 4 --out outputs/e052_student
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

from models.decoder.gauss_util import render_predicted
from types import SimpleNamespace

DEVICE = "cuda:0"

# minimal cfg stub for render_predicted (needs cfg.model.renderer_w_pose)
RENDER_CFG = SimpleNamespace(
    model=SimpleNamespace(renderer_w_pose=True, max_sh_degree=1)
)


class PatchGAN(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()

        def block(i, o, s=2, norm=True):
            layers = [nn.Conv2d(i, o, 4, s, 1)]
            if norm:
                layers.append(nn.InstanceNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_ch, base, norm=False),
            *block(base, base * 2),
            *block(base * 2, base * 4),
            *block(base * 4, base * 8, s=1),
            nn.Conv2d(base * 8, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class HoleGBufUNet(nn.Module):
    """Pixel-aligned UNet: 5ch input -> 12ch G-buffer output (masked to hole)."""

    def __init__(self, in_ch=5, out_ch=12, base=64):
        super().__init__()

        def cbr(i, o, s=1):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.SiLU()
            )

        self.e0 = cbr(in_ch, base)
        self.e1 = cbr(base, base * 2, 2)
        self.e2 = cbr(base * 2, base * 4, 2)
        self.e3 = cbr(base * 4, base * 8, 2)
        self.mid = cbr(base * 8, base * 8)
        self.d2 = cbr(base * 8 + base * 8, base * 4)
        self.d1 = cbr(base * 4 + base * 4, base * 2)
        self.d0 = cbr(base * 2 + base * 2, base)
        self.final = cbr(base + base, base)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.head = nn.Conv2d(base, out_ch, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        m = self.mid(e3)

        def up_to(t, ref):
            return F.interpolate(
                t, size=ref.shape[-2:], mode="bilinear", align_corners=False
            )

        d2 = self.d2(torch.cat([up_to(m, e3), e3], 1))
        d1 = self.d1(torch.cat([up_to(d2, e2), e2], 1))
        d0 = self.d0(torch.cat([up_to(d1, e1), e1], 1))
        f = self.final(torch.cat([up_to(d0, e0), e0], 1))
        return self.head(f)


class TeacherLabelDataset(Dataset):
    def __init__(self, label_dir):
        self.files = sorted(Path(label_dir).glob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], map_location="cpu")
        inp = torch.cat(
            [
                d["base_render"].float(),
                d["hole_mask"].float().unsqueeze(0),
                d["depth_filled"].float(),
            ],
            dim=0,
        )
        target = torch.cat(
            [
                d["gbuf_z"].float(),
                d["gbuf_shdc"].float(),
                d["gbuf_logscale"].float(),
                d["gbuf_opa"].float(),
                d["gbuf_rot"].float(),
            ],
            dim=0,
        )
        return {
            "input": inp,
            "target": target,
            "hole_mask": d["hole_mask"],
            "teacher_render": d["teacher_render"].float(),
            "base_pc": d["base_pc"],
            "cam": d["cam"],
            "K_tgt": d["K_tgt"],
            "T_src_from_tgt": d["T_src_from_tgt"],
        }


def assemble_and_render(
    pred_gbuf,
    hole_mask,
    base_pc,
    cam,
    K_tgt,
    T_src_from_tgt,
    cfg_znear=0.01,
    cfg_zfar=100.0,
):
    """Convert predicted G-buffer + base_pc to a rendered image (differentiable)."""
    H, W = hole_mask.shape
    ys, xs = torch.where(hole_mask)
    if ys.numel() == 0:
        return None
    z = pred_gbuf[0, ys, xs].clamp(min=1e-3)
    sh_dc = pred_gbuf[1:4, ys, xs].T
    logscale = pred_gbuf[4:7, ys, xs].T
    opa_logit = pred_gbuf[7, ys, xs][:, None]
    rot_raw = pred_gbuf[8:12, ys, xs].T

    fx, fy = K_tgt[0, 0], K_tgt[1, 1]
    cx, cy = K_tgt[0, 2], K_tgt[1, 2]
    x = (xs.float() + 0.5 - cx) / fx * z
    y = (ys.float() + 0.5 - cy) / fy * z
    pts_tgt = torch.stack([x, y, z, torch.ones_like(z)], dim=-1)
    pts_src = (T_src_from_tgt @ pts_tgt.T).T[:, :3]

    M = pts_src.shape[0]
    hole_pc = {
        "xyz": pts_src,
        "opacity": torch.sigmoid(opa_logit).clamp(1e-4, 1 - 1e-4),
        "scaling": torch.exp(logscale.clamp(-8, 2)),
        "rotation": F.normalize(rot_raw, dim=-1),
        "features_dc": sh_dc[:, None, :],
        "features_rest": torch.zeros(M, 3, 3, device=pts_src.device),
    }
    merged = {
        k: torch.cat(
            [base_pc[k].float().to(pts_src.device), hole_pc[k]], dim=0
        ).contiguous()
        for k in hole_pc
    }
    bg_color = torch.tensor([0.0, 0.0, 0.0], device=pts_src.device, dtype=torch.float32)
    out = render_predicted(
        RENDER_CFG,
        merged,
        cam["world_view_transform"].float().to(pts_src.device),
        cam["full_proj_transform"].float().to(pts_src.device),
        cam["proj_mtrx"].float().to(pts_src.device),
        cam["camera_center"].float().to(pts_src.device),
        (cam["fovX"], cam["fovY"]),
        (H, W),
        bg_color,
        1,
    )
    return out["render"].clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="/home/data/E-052_teacher_cache/labels")
    ap.add_argument("--out", default="outputs/e052_student")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--w_param", type=float, default=1.0)
    ap.add_argument("--w_render", type=float, default=2.0)
    ap.add_argument("--w_lpips", type=float, default=1.0)
    ap.add_argument("--w_gan", type=float, default=0.5)
    ap.add_argument(
        "--render_every", type=int, default=1, help="compute render loss every N steps"
    )
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    import lpips as _lpips

    lpips_fn = _lpips.LPIPS(net="vgg").to(DEVICE).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    ds = TeacherLabelDataset(args.labels)
    print(f"Dataset: {len(ds)} labels", flush=True)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    net = HoleGBufUNet(in_ch=5, out_ch=12, base=64).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(dl))

    D = PatchGAN(in_ch=3, base=64).to(DEVICE) if args.w_gan > 0 else None
    opt_D = (
        torch.optim.AdamW(D.parameters(), lr=args.lr * 0.5, weight_decay=1e-4)
        if D
        else None
    )

    print(
        f"Student params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M",
        flush=True,
    )
    step = 0
    for epoch in range(args.epochs):
        net.train()
        for batch in dl:
            inp = batch["input"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            mask = batch["hole_mask"].to(DEVICE)

            pred = net(inp)

            mask_f = mask.float().unsqueeze(1)
            # per-channel-group normalized L1 so z (~3.5) & logscale (~-4) don't dominate
            # sh/opa/rot (~O(1)). weights = 1/typical_scale per group.
            # groups: z[0], shdc[1:4], logscale[4:7], opa[7], rot[8:12]
            ch_w = torch.tensor(
                [0.5] + [1.0] * 3 + [1.0] * 3 + [1.0] + [1.0] * 4,
                device=DEVICE,
            ).view(1, 12, 1, 1)
            loss_param = ((pred - target).abs() * ch_w * mask_f).sum() / (
                mask_f.sum() * 12 + 1e-6
            )
            loss = args.w_param * loss_param

            loss_render = torch.tensor(0.0, device=DEVICE)
            loss_lpips = torch.tensor(0.0, device=DEVICE)
            if step % args.render_every == 0 and args.w_render > 0:
                r_terms = 0
                lr_sum = torch.tensor(0.0, device=DEVICE)
                ll_sum = torch.tensor(0.0, device=DEVICE)
                for b in range(inp.shape[0]):
                    pc = {k: v[b].to(DEVICE) for k, v in batch["base_pc"].items()}
                    cam = {
                        k: (
                            v[b]
                            if torch.is_tensor(v)
                            else v[b]
                            if isinstance(v, list)
                            else v
                        )
                        for k, v in batch["cam"].items()
                    }
                    render = assemble_and_render(
                        pred[b],
                        mask[b],
                        pc,
                        cam,
                        batch["K_tgt"][b].to(DEVICE),
                        batch["T_src_from_tgt"][b].to(DEVICE),
                    )
                    if render is None:
                        continue
                    tr = batch["teacher_render"][b].to(DEVICE)
                    lr_sum = lr_sum + F.l1_loss(render, tr)
                    if args.w_lpips > 0:
                        ll_sum = (
                            ll_sum
                            + lpips_fn(render[None] * 2 - 1, tr[None] * 2 - 1).mean()
                        )
                    r_terms += 1
                if r_terms > 0:
                    loss_render = lr_sum / r_terms
                    loss_lpips = ll_sum / r_terms
                    loss = (
                        loss + args.w_render * loss_render + args.w_lpips * loss_lpips
                    )

                    # GAN loss on the last render
                    if D is not None and render is not None:
                        # D step: real=teacher_render, fake=student render
                        d_real = D(tr[None])
                        d_fake = D(render[None].detach())
                        loss_d = (
                            F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
                        ) * 0.5
                        opt_D.zero_grad(set_to_none=True)
                        loss_d.backward()
                        opt_D.step()
                        # G step: fool D
                        d_fake_g = D(render[None])
                        loss_g = -d_fake_g.mean()
                        loss = loss + args.w_gan * loss_g

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1

            if step % args.log_every == 0:
                print(
                    f"[ep{epoch} step{step}] loss={loss.item():.4f} param={loss_param.item():.4f} "
                    f"render={loss_render.item():.4f} lpips={loss_lpips.item():.4f} lr={sched.get_last_lr()[0]:.6f}",
                    flush=True,
                )

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            torch.save(
                net.state_dict(),
                os.path.join(args.out, f"student_ep{epoch + 1:03d}.pt"),
            )
            print(f"  saved student_ep{epoch + 1:03d}.pt", flush=True)

    torch.save(net.state_dict(), os.path.join(args.out, "student_final.pt"))
    print(f"DONE training -> {args.out}/student_final.pt", flush=True)


if __name__ == "__main__":
    main()
