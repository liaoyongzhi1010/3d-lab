"""Step S1: train the Canonical Free layer (Anchor FROZEN) with teacher hidden-GS + render loss.

The crucial difference from D009/D010: the Free layer is supervised by a CANONICAL TEACHER
pseudo-GT (real multi-view hidden points, built in S0), not just render loss. This is what
prevents opacity collapse (D009's failure: render-loss-alone lets Free learn to do nothing).

Losses:
  1. Teacher geometry matching: Chamfer(Free.xyz, teacher.xyz) + color L1 (nearest-neighbor)
  2. Multi-view render loss: render merged scene to real neighbor views, L1 + LPIPS
  3. Routing penalty: Free opacity penalized where anchor already covers (visibility)
  4. KL on latent z
  5. Scale/opacity regularization (anti-billboard)

GO/NO-GO: deletion counterfactual > +1dB (remove Free -> render worsens = Free contributes).

Run (Flash3D venv):
  python -m paper_b_generative3d.train_s1_free \
    --teacher_dir /home/data/sv3d-lab/reconstruction/teacher_hidden_gs \
    --steps 4000 --out /home/data/sv3d-lab/reconstruction/E-126-s1
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
from paper_b_generative3d.model.canonical_dual_layer import CanonicalDualLayer
from paper_b_generative3d.train_real import get_lpips_fn


def chamfer_l1(pred_xyz, tgt_xyz, max_pts=2048):
    """One-directional Chamfer (pred -> nearest tgt) + return nn indices for color.

    pred_xyz: [P, 3], tgt_xyz: [T, 3]. Subsample for memory.
    """
    P = pred_xyz.shape[0]
    T = tgt_xyz.shape[0]
    if P > max_pts:
        pidx = torch.randperm(P, device=pred_xyz.device)[:max_pts]
        pred_s = pred_xyz[pidx]
    else:
        pidx = torch.arange(P, device=pred_xyz.device)
        pred_s = pred_xyz
    if T > max_pts:
        tidx = torch.randperm(T, device=tgt_xyz.device)[:max_pts]
        tgt_s = tgt_xyz[tidx]
    else:
        tidx = torch.arange(T, device=tgt_xyz.device)
        tgt_s = tgt_xyz
    # pairwise dist [Ps, Ts]
    d = torch.cdist(pred_s, tgt_s)  # [Ps, Ts]
    min_d, nn_idx = d.min(dim=1)  # pred -> nearest tgt
    return min_d.mean(), pidx, tidx[nn_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n_free", type=int, default=4096)
    ap.add_argument("--lambda_teacher", type=float, default=1.0)
    ap.add_argument("--lambda_render", type=float, default=1.0)
    ap.add_argument("--lambda_lpips", type=float, default=0.5)
    ap.add_argument("--lambda_kl", type=float, default=0.001)
    ap.add_argument("--lambda_routing", type=float, default=1.0)
    ap.add_argument("--lambda_reg", type=float, default=0.01)
    ap.add_argument("--lambda_alive", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    teacher_dir = Path(args.teacher_dir)
    teacher_files = sorted(teacher_dir.glob("[0-9]*.pt"))
    print(f"Found {len(teacher_files)} teacher pseudo-GT files", flush=True)

    cfg = build_flash3d_cfg(batch_size=1, num_workers=0, stage="dev")
    backbone = Flash3DBackbone(cfg, FLASH3D_CKPT, device=str(device))
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    model = CanonicalDualLayer(
        anchor_backbone=backbone,
        feat_proj_dim=256,
        latent_dim=64,
        n_free=args.n_free,
        n_cross=3,
        n_self=2,
    ).to(device)
    # Freeze anchor (it's inside backbone, already frozen); train free + proj + latent
    trainable = [
        p for n, p in model.named_parameters() if "anchor" not in n and p.requires_grad
    ]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    lpips_fn = get_lpips_fn(device)
    print(
        f"Trainable params: {sum(p.numel() for p in trainable) / 1e6:.1f}M", flush=True
    )

    _, loader = _create_loader_from_cfg(cfg)
    H, W = 256, 384

    step = 0
    running = {
        "loss": [],
        "teacher": [],
        "render": [],
        "kl": [],
        "alive": [],
        "route": [],
    }
    scene_idx = 0
    data_iter = iter(loader)
    while step < args.steps:
        try:
            inputs = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            scene_idx = 0
            inputs = next(data_iter)
        cur_scene = scene_idx
        scene_idx += 1

        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
        tfids = [f for f in [1, 2, 3] if ("color", f, 0) in inputs]
        if not tfids:
            continue

        # Load teacher pseudo-GT for this scene (by index, matching S0 order)
        tf = teacher_dir / f"{cur_scene:06d}.pt"
        if not tf.exists():
            continue
        teacher = torch.load(tf, map_location=device)
        teacher_xyz_world = (
            teacher["xyz"].float().to(device)
        )  # [M, 3] WORLD coords (S0 backprojected with T_c2w_neighbor)
        teacher_color = teacher["color"].float().to(device)
        if teacher_xyz_world.shape[0] < 10:
            continue

        # COORDINATE-FRAME FIX: anchor/free gaussians live in SOURCE-CAMERA coords
        # (Flash3D extract_source_gaussians). Teacher pts are WORLD coords. Transform
        # teacher into source-camera frame so the Chamfer target is comparable.
        T_w2c_src = inputs.get(("T_w2c", 0))
        if T_w2c_src is None:
            continue
        T_w2c_src = T_w2c_src[0]  # [4,4]
        ones = torch.ones(teacher_xyz_world.shape[0], 1, device=device)
        tw_h = torch.cat([teacher_xyz_world, ones], dim=1)  # [M,4]
        teacher_xyz = (T_w2c_src @ tw_h.T).T[:, :3]  # [M,3] source-camera frame

        # Build scene (anchor frozen, free trainable)
        scene = model.build_scene(inputs)
        free = scene["free"]

        # 1. Teacher geometry matching: free xyz -> teacher xyz (Chamfer) + color L1
        # Both now in source-camera coords.
        cham_d, pidx, nn_tidx = chamfer_l1(free["xyz"], teacher_xyz)
        color_l1 = F.l1_loss(free["color_rgb"][pidx], teacher_color[nn_tidx])
        loss_teacher = cham_d + color_l1

        # 2. Multi-view render loss
        merged = model.merge_scene(scene)
        loss_render = torch.tensor(0.0, device=device)
        n_rendered = 0
        for fid in tfids:
            cam = inputs.get(("cam_T_cam", 0, fid))
            if cam is None:
                # dev split has T_w2c/T_c2w; compute relative
                T_w2c_t = inputs.get(("T_w2c", fid))
                T_c2w_s = inputs.get(("T_c2w", 0))
                if T_w2c_t is None or T_c2w_s is None:
                    continue
                cam = (T_w2c_t[0] @ T_c2w_s[0]).unsqueeze(0)
            K_tgt = inputs.get(("K_tgt", fid), inputs[("K_src", 0)])
            gt = inputs["color", fid, 0][0]
            render = backbone.render_gaussians(
                merged, cam[0], K_tgt[0], H, W, batch_idx=0
            ).clamp(0, 1)
            loss_render = (
                loss_render
                + F.l1_loss(render, gt)
                + args.lambda_lpips
                * lpips_fn(render.unsqueeze(0), gt.unsqueeze(0)).mean()
            )
            n_rendered += 1
        if n_rendered > 0:
            loss_render = loss_render / n_rendered

        # 3. KL
        mu, logvar = scene["mu"], scene["logvar"]
        loss_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

        # 4. Scale reg only (anti-billboard). Do NOT penalize opacity: that drives the
        # D009 opacity-collapse. Instead we ADD an anti-collapse term below that rewards
        # opacity on Free points that matched the teacher geometry (the pidx set).
        loss_reg = free["scales"].mean()

        # 5. Anti-collapse: Free points matched to teacher should be opaque (BCE toward 1).
        # This is the mechanism that makes teacher supervision prevent opacity collapse:
        # teacher positions the points AND requires they be visible where geometry exists.
        matched_op = free["opacity"][pidx].clamp(1e-4, 1 - 1e-4)
        loss_alive = F.binary_cross_entropy(matched_op, torch.ones_like(matched_op))

        # 6. Visibility routing: penalize Free opacity that pollutes anchor-visible rays.
        # Fixes the -4dB full-frame regression (Free splattering over visible pixels).
        loss_routing = model.visibility_routing_loss(
            scene, inputs, inputs[("K_src", 0)], H=H, W=W
        )

        loss = (
            args.lambda_teacher * loss_teacher
            + args.lambda_render * loss_render
            + args.lambda_kl * loss_kl
            + args.lambda_reg * loss_reg
            + args.lambda_alive * loss_alive
            + args.lambda_routing * loss_routing
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        running["loss"].append(loss.item())
        running["teacher"].append(loss_teacher.item())
        running["render"].append(float(loss_render) if n_rendered > 0 else 0.0)
        running["kl"].append(loss_kl.item())
        running["alive"].append(loss_alive.item())
        running["route"].append(loss_routing.item())
        step += 1

        if step % args.log_every == 0:
            print(
                f"[step {step}/{args.steps}] loss={np.mean(running['loss'][-args.log_every :]):.4f} "
                f"teacher={np.mean(running['teacher'][-args.log_every :]):.4f} "
                f"render={np.mean(running['render'][-args.log_every :]):.4f} "
                f"kl={np.mean(running['kl'][-args.log_every :]):.4f} "
                f"alive={np.mean(running['alive'][-args.log_every :]):.4f} "
                f"route={np.mean(running['route'][-args.log_every :]):.4f} "
                f"free_op={free['opacity'].mean().item():.3f}",
                flush=True,
            )
        if step % args.ckpt_every == 0:
            torch.save(
                {"model": model.state_dict(), "step": step}, out / f"s1_step{step}.pt"
            )

    torch.save({"model": model.state_dict(), "step": step}, out / "s1_final.pt")
    print(
        f"\n=== S1 done step {step}, free_opacity_mean={free['opacity'].mean().item():.3f} ===",
        flush=True,
    )
    print(
        f"(opacity collapse check: if free_op < 0.01 -> D009 failure repeated)",
        flush=True,
    )


if __name__ == "__main__":
    main()
