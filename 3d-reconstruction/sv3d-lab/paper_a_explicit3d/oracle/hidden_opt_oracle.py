"""
D026 go/no-go oracle: per-scene FREE OPTIMIZATION of added hidden Gaussians.

Distinct from E002 (representation oracle, which CHEATS with target geometry and does NOT optimize).
Here we FREEZE the visible Gaussians (Flash3D/UniDepth-v1 source back-projection) and optimize ONLY a
set of added hidden Gaussians per scene, supervised by held-out WIDE targets, under a HARD source-null
constraint. This answers the one question that sits between E002 (representable) and E006/8/9 (not
amortizable): is the far-target RGB signal even IDENTIFIABLE for load-bearing hidden 3D content?

Oracle = per-scene upper bound. NEVER reported as a model result (charter §7).
Thresholds are PRE-REGISTERED in decision_log.md D026 (G1..G5). Do NOT lower them.

Run on server (flash3d venv):
  cd /root/projects/flash3d && source .venv/bin/activate
  export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python /root/sv3d-lab/paper_a_explicit3d/oracle/hidden_opt_oracle.py \
      --n_scenes 100 --split splits/re10k_mine_filtered/test_files_wide700.txt \
      --out /home/data/sv3d-lab/evaluations/hidden_opt_oracle_l2 --loss l2
"""

import os
import sys
import json
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, "/root/sv3d-lab/paper_a_explicit3d/oracle")
sys.path.insert(0, "/root/sv3d-lab")

from oracle_core import (
    backproject,
    make_gaussians,
    set_scale_from_depth,
    merge_gaussians,
    render_gaussians_relpose,
    psnr,
    crop5,
)
from common.geometry.visibility import visibility_partition
from run_oracle import build_cfg


# ---------------------------------------------------------------------------
# Hidden Gaussian set (the ONLY thing optimized). All params raw; activated in .gauss().
# ---------------------------------------------------------------------------
def _inv_sigmoid(x):
    x = x.clamp(1e-4, 1 - 1e-4)
    return torch.log(x / (1 - x))


class HiddenGaussians(nn.Module):
    def __init__(self, xyz_init, rgb_init, scale_init, opacity_init=0.05):
        super().__init__()
        N = xyz_init.shape[0]
        dev = xyz_init.device
        self.xyz = nn.Parameter(xyz_init.clone())
        self.rgb_logit = nn.Parameter(_inv_sigmoid(rgb_init.clone()))
        self.log_scale = nn.Parameter(torch.log(scale_init.clone().clamp(min=1e-6)))
        rot = torch.zeros(N, 4, device=dev)
        rot[:, 0] = 1.0
        self.rot = nn.Parameter(rot)
        self.opacity_logit = nn.Parameter(
            torch.full(
                (N, 1), float(_inv_sigmoid(torch.tensor(opacity_init))), device=dev
            )
        )

    def gauss(self):
        N = self.xyz.shape[0]
        return {
            "xyz": self.xyz,
            "scaling": torch.exp(self.log_scale),
            "rotation": F.normalize(self.rot, dim=1),
            "opacity": torch.sigmoid(self.opacity_logit),
            "features_dc": torch.zeros(N, 1, 3, device=self.xyz.device),
            "rgb_direct": torch.sigmoid(self.rgb_logit),
        }


# ---------------------------------------------------------------------------
# Hidden-Gaussian initialization (GEOMETRY-AGNOSTIC: no target depth used).
# Occluded-behind seeds: source pixels back-projected at multiple depth multipliers.
# OOF seeds: a padded border ring back-projected at median depth multipliers.
# ---------------------------------------------------------------------------
def init_hidden(
    depth_src,
    color_src,
    K,
    device,
    stride=2,
    occ_mult=(1.3, 1.8, 2.6),
    oof_pad_frac=0.4,
    oof_mult=(1.0, 1.8),
    footprint=0.15,
):
    H, W = depth_src.shape
    inv_K = torch.linalg.inv(K)
    fx = K[0, 0].item()

    # --- occluded-behind seeds (behind the visible surface, at source pixels) ---
    ys, xs = torch.meshgrid(
        torch.arange(0, H, stride, device=device, dtype=torch.float32),
        torch.arange(0, W, stride, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], 0)  # (3, M)
    rays = inv_K @ pix  # (3, M)
    yi = ys.long().clamp(0, H - 1)
    xi = xs.long().clamp(0, W - 1)
    d0 = depth_src[yi, xi]  # (M,)
    col = color_src[:, yi, xi].T  # (M,3)

    xyz_list, rgb_list, scale_list = [], [], []
    for m in occ_mult:
        d = d0 * m
        pts = (rays * d.unsqueeze(0)).T  # (M,3) source cam
        xyz_list.append(pts)
        rgb_list.append(col)
        scale_list.append(
            (d / fx * footprint).clamp(min=1e-5).unsqueeze(1).repeat(1, 3)
        )

    # --- OOF seeds: padded border ring (outside [0,W)x[0,H)) ---
    padW = int(oof_pad_frac * W)
    padH = int(oof_pad_frac * H)
    yy, xx = torch.meshgrid(
        torch.arange(-padH, H + padH, stride, device=device, dtype=torch.float32),
        torch.arange(-padW, W + padW, stride, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    outside = (xx < 0) | (xx >= W) | (yy < 0) | (yy >= H)
    xx = xx[outside]
    yy = yy[outside]
    ring = torch.stack([xx, yy, torch.ones_like(xx)], 0)
    ring_rays = inv_K @ ring
    med = float(depth_src.median())
    gray = torch.full((xx.shape[0], 3), 0.5, device=device)
    for m in oof_mult:
        d = med * m
        pts = (ring_rays * d).T
        xyz_list.append(pts)
        rgb_list.append(gray)
        sc = torch.full((xx.shape[0], 3), d / fx * footprint, device=device).clamp(
            min=1e-5
        )
        scale_list.append(sc)

    xyz = torch.cat(xyz_list, 0)
    rgb = torch.cat(rgb_list, 0)
    scale = torch.cat(scale_list, 0)
    return xyz, rgb, scale


# ---------------------------------------------------------------------------
# Held-out evaluation helper (no-grad). Used for the final metrics AND, in --profile mode, for the
# #optimization-steps curve (evaluated along the SAME trajectory). split_regions=True additionally
# reports occluded-only and OOF-only deletion Δ (region-causal deletion, per region).
# ---------------------------------------------------------------------------
def _eval_heldout(
    g_vis,
    gh,
    tgt,
    eval_frames,
    depth_src,
    K,
    H,
    W,
    device,
    lpips_fn,
    split_regions=False,
):
    g_merged = merge_gaussians([g_vis, gh])
    accum = {
        "delta_overall": [],
        "delta_hidden": [],
        "lpips_gain": [],
        "delta_occ": [],
        "delta_oof": [],
    }
    with torch.no_grad():
        for f in eval_frames:
            gt = tgt[f]["gt"]
            T_rel = tgt[f]["T_rel"]
            merged = render_gaussians_relpose(g_merged, K, T_rel, H, W, device)[
                "render"
            ].clamp(0, 1)
            visonly = render_gaussians_relpose(g_vis, K, T_rel, H, W, device)[
                "render"
            ].clamp(0, 1)
            visible, occluded, oof = visibility_partition(
                depth_src, K, T_rel, H, W, device, dilate=2
            )
            hid = occluded | oof
            pm, pv = psnr(crop5(merged), crop5(gt)), psnr(crop5(visonly), crop5(gt))
            accum["delta_overall"].append(pm - pv)
            pmh, pvh = psnr(merged, gt, mask=hid), psnr(visonly, gt, mask=hid)
            if pmh is not None and pvh is not None:
                accum["delta_hidden"].append(pmh - pvh)
            if lpips_fn is not None:
                lp_m = lpips_fn(
                    merged.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1
                ).item()
                lp_v = lpips_fn(
                    visonly.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1
                ).item()
                accum["lpips_gain"].append(lp_v - lp_m)
            if split_regions:
                pmo, pvo = (
                    psnr(merged, gt, mask=occluded),
                    psnr(visonly, gt, mask=occluded),
                )
                if pmo is not None and pvo is not None:
                    accum["delta_occ"].append(pmo - pvo)
                pmf, pvf = psnr(merged, gt, mask=oof), psnr(visonly, gt, mask=oof)
                if pmf is not None and pvf is not None:
                    accum["delta_oof"].append(pmf - pvf)
    return {k: (float(np.mean(v)) if v else None) for k, v in accum.items()}


# ---------------------------------------------------------------------------
# Per-scene optimization
# ---------------------------------------------------------------------------
def process_scene(unidepth, inputs, device, args, lpips_fn=None):
    color_src = inputs[("color", 0, 0)].to(device)
    K = inputs[("K_tgt", 0)].to(device)
    c2w_src = inputs[("T_c2w", 0)].to(device)
    w2c_src = inputs[("T_w2c", 0)].to(device)
    H, W = color_src.shape[1:]

    with torch.no_grad():
        depth_src = unidepth.infer(color_src.unsqueeze(0), intrinsics=K.unsqueeze(0))[
            "depth"
        ].squeeze()
    inv_K = torch.linalg.inv(K)
    pts_src = backproject(depth_src, inv_K, device)
    rgb_src = color_src.permute(1, 2, 0).reshape(-1, 3)
    g_vis = make_gaussians(pts_src, rgb_src, K[0, 0].item())
    set_scale_from_depth(g_vis, depth_src.reshape(-1), K[0, 0].item(), args.footprint)
    for k in g_vis:
        g_vis[k] = g_vis[k].detach()

    # target frames + relative poses (src-cam -> tgt-cam)
    target_frames = [f for f in [1, 2, 3] if ("color", f, 0) in inputs]
    # HOLD-OUT protocol (leakage-free): optimize hidden Gaussians only on `opt_frames`, evaluate
    # deletion/PSNR/LPIPS on `eval_frames` (DISJOINT). If content generalizes to an unseen view, the
    # win is genuine 3D completion, not target-fitting. args.holdout_frame = the frame id held out.
    if (
        getattr(args, "holdout_frame", 0)
        and args.holdout_frame in target_frames
        and len(target_frames) > 1
    ):
        opt_frames = [f for f in target_frames if f != args.holdout_frame]
        eval_frames = [args.holdout_frame]
    else:
        opt_frames = target_frames
        eval_frames = target_frames
    tgt = {}
    for f in target_frames:
        c2w_t = inputs[("T_c2w", f)].to(device)
        w2c_t = inputs[("T_w2c", f)].to(device)
        tgt[f] = {
            "gt": inputs[("color", f, 0)].to(device),
            "T_rel": (w2c_t @ c2w_src).float(),
        }

    # init hidden gaussians (geometry-agnostic)
    xyz0, rgb0, scale0 = init_hidden(
        depth_src, color_src, K, device, stride=args.stride, footprint=args.footprint
    )
    hidden = HiddenGaussians(xyz0, rgb0, scale0, opacity_init=args.opacity_init).to(
        device
    )
    # Per-param LRs near 3DGS conventions: xyz must move slowly (metric units), opacity/rgb faster
    # (they start near-off). Uniform high LR diverges.
    opt = torch.optim.Adam(
        [
            {"params": [hidden.xyz], "lr": args.lr_xyz},
            {"params": [hidden.log_scale], "lr": args.lr_scale},
            {"params": [hidden.rot], "lr": args.lr_rot},
            {"params": [hidden.opacity_logit], "lr": args.lr_opacity},
            {"params": [hidden.rgb_logit], "lr": args.lr_rgb},
        ]
    )

    # frozen source render of visible-only (target for source-null constraint)
    vis_src_render = (
        render_gaussians_relpose(g_vis, K, torch.eye(4, device=device), H, W, device)[
            "render"
        ]
        .clamp(0, 1)
        .detach()
    )

    T_id = torch.eye(4, device=device)
    loss_traj = []
    # --profile: capture per-step wall time (opt only) + #steps curve along the SAME trajectory.
    profile = getattr(args, "profile", False)
    steps_curve = {}
    curve_at = set(
        int(s) for s in getattr(args, "curve_steps", []) if 0 < int(s) <= args.steps
    )
    opt_wall = 0.0
    for step in range(args.steps):
        if profile:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()
        opt.zero_grad()
        gh = hidden.gauss()
        g_merged = merge_gaussians([g_vis, gh])
        loss = 0.0
        for f in opt_frames:
            pred = render_gaussians_relpose(g_merged, K, tgt[f]["T_rel"], H, W, device)[
                "render"
            ].clamp(0, 1)
            gt = tgt[f]["gt"]
            if args.loss == "l2":
                loss = loss + F.mse_loss(pred, gt)
            elif args.loss == "l1":
                loss = loss + F.l1_loss(pred, gt)
            elif args.loss == "lpips":
                loss = (
                    loss
                    + F.mse_loss(pred, gt)
                    + args.w_lpips
                    * lpips_fn(
                        pred.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1
                    ).mean()
                )
        # source-null: merged source render must match visible-only source render (=> hidden
        # Gaussians must not corrupt the source view). Legitimately-occluded Gaussians (behind the
        # visible surface) are z-buffered away in the merged render => zero penalty, correctly allowed.
        merged_src = render_gaussians_relpose(g_merged, K, T_id, H, W, device)[
            "render"
        ].clamp(0, 1)
        loss = loss + args.w_null * F.mse_loss(merged_src, vis_src_render)
        # additional hidden-ONLY source-alpha penalty: the hidden set alone should be ~invisible at
        # source (mean alpha -> 0). Complements the merged-MSE term (which can be masked by z-buffer).
        hid_src_alpha = render_gaussians_relpose(gh, K, T_id, H, W, device).get(
            "rendered_alpha", None
        )
        if hid_src_alpha is not None:
            loss = loss + args.w_null_alpha * hid_src_alpha.mean()
        # hidden-Gaussian regularizers (D038, default 0.0 => byte-identical to E018/E021).
        # (1) opacity L1 sparsity: cull the fog of near-transparent specks; keep fewer committed Gaussians.
        if args.w_opa_sparse > 0:
            loss = loss + args.w_opa_sparse * torch.sigmoid(hidden.opacity_logit).mean()
        # (2) scale penalty: discourage oversized/degenerate Gaussians that render as streaks/needles.
        if args.w_scale > 0:
            sc = torch.exp(hidden.log_scale)
            over = (sc - args.max_scale).clamp(min=0.0) if args.max_scale > 0 else sc
            loss = loss + args.w_scale * over.mean()
        loss.backward()
        opt.step()
        # hard projected clamp: hidden Gaussians that project IN-FRAME and IN-FRONT of the visible
        # surface at the source (frustum pokethrough) are illegal -> push their opacity down.
        if args.w_null > 0:
            with torch.no_grad():
                xyz = hidden.xyz
                z = xyz[:, 2].clamp(min=1e-6)
                u = xyz[:, 0] / z * K[0, 0] + K[0, 2]
                v = xyz[:, 1] / z * K[1, 1] + K[1, 2]
                in_frame = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0.05)
                if in_frame.any():
                    ui = u[in_frame].long().clamp(0, W - 1)
                    vi = v[in_frame].long().clamp(0, H - 1)
                    src_d = depth_src[vi, ui]
                    infront = z[in_frame] < src_d * 0.95
                    idx = torch.where(in_frame)[0][infront]
                    if idx.numel() > 0:
                        hidden.opacity_logit.data[idx] -= 0.5
        if profile:
            torch.cuda.synchronize()
            opt_wall += time.perf_counter() - _t0
            # #steps curve: evaluate held-out metrics at requested checkpoints along THIS trajectory.
            if (step + 1) in curve_at:
                steps_curve[step + 1] = _eval_heldout(
                    g_vis,
                    hidden.gauss(),
                    tgt,
                    eval_frames,
                    depth_src,
                    K,
                    H,
                    W,
                    device,
                    lpips_fn,
                    split_regions=False,
                )
        if args.verbose and (
            step % max(1, args.steps // 8) == 0 or step == args.steps - 1
        ):
            loss_traj.append(
                (
                    step,
                    float(loss.detach()),
                    float(torch.sigmoid(hidden.opacity_logit).mean()),
                )
            )

    if args.verbose:
        print(
            "  loss_traj (step,loss,opacity):",
            " ".join(f"{s}:{l:.4f}/{o:.3f}" for s, l, o in loss_traj),
        )

    # ---- metrics (no grad) ----
    with torch.no_grad():
        gh = hidden.gauss()
        g_merged = merge_gaussians([g_vis, gh])
        # source-null check
        src_merged = render_gaussians_relpose(g_merged, K, T_id, H, W, device)[
            "render"
        ].clamp(0, 1)
        src_vis = render_gaussians_relpose(g_vis, K, T_id, H, W, device)[
            "render"
        ].clamp(0, 1)
        src_gt = color_src
        src_psnr_merged = psnr(crop5(src_merged), crop5(src_gt))
        src_psnr_vis = psnr(crop5(src_vis), crop5(src_gt))

        per_target = {}
        for f in eval_frames:
            gt = tgt[f]["gt"]
            T_rel = tgt[f]["T_rel"]
            merged = render_gaussians_relpose(g_merged, K, T_rel, H, W, device)[
                "render"
            ].clamp(0, 1)
            visonly = render_gaussians_relpose(g_vis, K, T_rel, H, W, device)[
                "render"
            ].clamp(0, 1)
            visible, occluded, oof = visibility_partition(
                depth_src, K, T_rel, H, W, device, dilate=2
            )
            hid = occluded | oof
            m = {
                "psnr_merged": psnr(crop5(merged), crop5(gt)),
                "psnr_vis": psnr(crop5(visonly), crop5(gt)),
                "psnr_merged_hidden": psnr(merged, gt, mask=hid),
                "psnr_vis_hidden": psnr(visonly, gt, mask=hid),
                "hidden_frac": float(hid.float().mean()),
            }
            m["delta_overall"] = m["psnr_merged"] - m["psnr_vis"]
            if m["psnr_merged_hidden"] is not None and m["psnr_vis_hidden"] is not None:
                m["delta_hidden"] = m["psnr_merged_hidden"] - m["psnr_vis_hidden"]
            else:
                m["delta_hidden"] = None
            # region-split causal deletion (occluded-only vs OOF-only) + hold-out distance
            if profile:
                pmo, pvo = (
                    psnr(merged, gt, mask=occluded),
                    psnr(visonly, gt, mask=occluded),
                )
                m["delta_occ"] = (
                    (pmo - pvo) if (pmo is not None and pvo is not None) else None
                )
                m["occ_frac"] = float(occluded.float().mean())
                pmf, pvf = psnr(merged, gt, mask=oof), psnr(visonly, gt, mask=oof)
                m["delta_oof"] = (
                    (pmf - pvf) if (pmf is not None and pvf is not None) else None
                )
                m["oof_frac"] = float(oof.float().mean())
                m["baseline_t"] = float(torch.linalg.norm(T_rel[:3, 3]))
            if lpips_fn is not None:
                lp_m = lpips_fn(
                    merged.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1
                ).item()
                lp_v = lpips_fn(
                    visonly.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1
                ).item()
                m["lpips_merged"] = lp_m
                m["lpips_vis"] = lp_v
                m["lpips_gain"] = lp_v - lp_m
            per_target[str(f)] = m

    return {
        "n_hidden": int(hidden.xyz.shape[0]),
        "src_psnr_merged": src_psnr_merged,
        "src_psnr_vis": src_psnr_vis,
        "src_delta": src_psnr_merged - src_psnr_vis,
        "final_hidden_opacity_mean": float(torch.sigmoid(hidden.opacity_logit).mean()),
        "opt_wall_sec": opt_wall if profile else None,
        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1e9) if profile else None,
        "steps_curve": steps_curve if profile else None,
        "per_target": per_target,
        "_hidden_module": hidden,
        "_depth_src": depth_src.detach(),
        "_g_vis": g_vis,
        "_tgt": tgt,
        "_K": K,
        "_HW": (H, W),
        "_color_src": color_src.detach(),
        "_eval_frames": eval_frames,
        "_opt_frames": opt_frames,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_scenes", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--split", default="splits/re10k_mine_filtered/test_files_wide700.txt"
    )
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lr_xyz", type=float, default=2e-4)
    ap.add_argument("--lr_scale", type=float, default=5e-3)
    ap.add_argument("--lr_rot", type=float, default=1e-3)
    ap.add_argument("--lr_opacity", type=float, default=5e-2)
    ap.add_argument("--lr_rgb", type=float, default=1e-2)
    ap.add_argument("--loss", default="l2", choices=["l2", "l1", "lpips"])
    ap.add_argument("--w_lpips", type=float, default=1.0)
    ap.add_argument("--w_null", type=float, default=10.0)
    ap.add_argument("--w_null_alpha", type=float, default=5.0)
    ap.add_argument(
        "--w_opa_sparse",
        type=float,
        default=0.0,
        help="(D038) opacity L1 sparsity on hidden Gaussians (cull speck fog). 0=off=paper-E018 config.",
    )
    ap.add_argument(
        "--w_scale",
        type=float,
        default=0.0,
        help="(D038) penalty on hidden-Gaussian scale (anti-streak/needle). 0=off.",
    )
    ap.add_argument(
        "--max_scale",
        type=float,
        default=0.0,
        help="(D038) if >0, only penalize scale ABOVE this ceiling; else penalize all scale.",
    )
    ap.add_argument("--footprint", type=float, default=0.15)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--opacity_init", type=float, default=0.05)
    ap.add_argument(
        "--holdout_frame",
        type=int,
        default=0,
        help="if set (e.g. 3), optimize on other targets and EVALUATE only on this held-out frame (leakage-free)",
    )
    ap.add_argument("--appearance_lr_mult", type=float, default=3.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--profile",
        action="store_true",
        help="additive instrumentation: per-scene opt wall-time, peak VRAM, region-split (occ/oof) "
        "deletion, hold-out translation distance, and a #steps curve. Does NOT change optimization.",
    )
    ap.add_argument(
        "--curve_steps",
        type=int,
        nargs="+",
        default=[100, 200, 300, 400, 500],
        help="(profile mode) checkpoints at which to eval held-out metrics along the same trajectory",
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    print("Loading UniDepth v1...")
    unidepth = (
        torch.hub.load(
            "lpiccinelli-eth/UniDepth",
            "UniDepth",
            version="v1",
            backbone="vitl14",
            pretrained=True,
            trust_repo=True,
        )
        .to(device)
        .eval()
    )

    lpips_fn = None
    if args.loss == "lpips":
        import lpips as lpips_lib

        lpips_fn = lpips_lib.LPIPS(net="vgg").to(device).eval()
    else:
        try:
            import lpips as lpips_lib

            lpips_fn = lpips_lib.LPIPS(net="vgg").to(device).eval()
        except Exception as e:
            print(f"lpips unavailable ({e}); skipping LPIPS metrics")

    cfg = build_cfg(args.split)
    from datasets.re10k import Re10KDataset

    ds = Re10KDataset(cfg, split="test")
    print(f"Dataset length: {len(ds)}; running {min(args.n_scenes, len(ds))} scenes")

    all_res = []
    agg = {
        "src_delta": [],
        "delta_overall": [],
        "delta_hidden": [],
        "lpips_gain": [],
        "hidden_frac": [],
        "opacity": [],
    }
    prof = {
        "opt_wall_sec": [],
        "peak_vram_gb": [],
        "delta_occ": [],
        "delta_oof": [],
        "occ_frac": [],
        "oof_frac": [],
        "n_hidden": [],
        "dist_pairs": [],  # (baseline_t, delta_overall) for distance bucketing
        "scene_wall_sec": [],  # total per-scene time incl. UniDepth depth + init + opt
    }
    curve_agg = {}  # step -> list of delta_overall/delta_hidden/lpips_gain dicts
    for i in range(min(args.n_scenes, len(ds))):
        try:
            if args.profile:
                torch.cuda.synchronize()
                _scene_t0 = time.perf_counter()
            r = process_scene(unidepth, ds[i], device, args, lpips_fn)
            if args.profile:
                torch.cuda.synchronize()
                prof["scene_wall_sec"].append(time.perf_counter() - _scene_t0)
        except Exception as e:
            print(f"scene {i} FAILED: {e}")
            continue
        all_res.append(r)
        agg["src_delta"].append(r["src_delta"])
        agg["opacity"].append(r["final_hidden_opacity_mean"])
        if args.profile:
            if r.get("opt_wall_sec") is not None:
                prof["opt_wall_sec"].append(r["opt_wall_sec"])
            if r.get("peak_vram_gb") is not None:
                prof["peak_vram_gb"].append(r["peak_vram_gb"])
            prof["n_hidden"].append(r["n_hidden"])
            for st, d in (r.get("steps_curve") or {}).items():
                curve_agg.setdefault(int(st), []).append(d)
        for f, m in r["per_target"].items():
            agg["delta_overall"].append(m["delta_overall"])
            if m.get("delta_hidden") is not None:
                agg["delta_hidden"].append(m["delta_hidden"])
            if m.get("lpips_gain") is not None:
                agg["lpips_gain"].append(m["lpips_gain"])
            agg["hidden_frac"].append(m["hidden_frac"])
            if args.profile:
                if m.get("delta_occ") is not None:
                    prof["delta_occ"].append(m["delta_occ"])
                if m.get("delta_oof") is not None:
                    prof["delta_oof"].append(m["delta_oof"])
                if m.get("occ_frac") is not None:
                    prof["occ_frac"].append(m["occ_frac"])
                if m.get("oof_frac") is not None:
                    prof["oof_frac"].append(m["oof_frac"])
                if m.get("baseline_t") is not None:
                    prof["dist_pairs"].append([m["baseline_t"], m["delta_overall"]])
        if (i + 1) % 5 == 0:
            print(
                f"[{i + 1}] src_delta={np.mean(agg['src_delta']):+.3f} "
                f"delta_overall={np.mean(agg['delta_overall']):+.3f} "
                f"delta_hidden={np.mean(agg['delta_hidden']) if agg['delta_hidden'] else float('nan'):+.3f} "
                f"lpips_gain={np.mean(agg['lpips_gain']) if agg['lpips_gain'] else float('nan'):+.4f} "
                f"opacity={np.mean(agg['opacity']):.3f}"
            )

    summary = {k: (float(np.mean(v)) if v else None) for k, v in agg.items()}
    summary["n_scenes"] = len(all_res)
    summary["loss"] = args.loss
    summary["steps"] = args.steps
    if args.profile:
        summary["profile"] = {
            "opt_wall_sec_mean": float(np.mean(prof["opt_wall_sec"]))
            if prof["opt_wall_sec"]
            else None,
            "opt_wall_sec_std": float(np.std(prof["opt_wall_sec"]))
            if prof["opt_wall_sec"]
            else None,
            "scene_wall_sec_mean": float(np.mean(prof["scene_wall_sec"]))
            if prof["scene_wall_sec"]
            else None,
            "peak_vram_gb_max": float(np.max(prof["peak_vram_gb"]))
            if prof["peak_vram_gb"]
            else None,
            "n_hidden_mean": float(np.mean(prof["n_hidden"]))
            if prof["n_hidden"]
            else None,
            "delta_occ_mean": float(np.mean(prof["delta_occ"]))
            if prof["delta_occ"]
            else None,
            "delta_oof_mean": float(np.mean(prof["delta_oof"]))
            if prof["delta_oof"]
            else None,
            "occ_frac_mean": float(np.mean(prof["occ_frac"]))
            if prof["occ_frac"]
            else None,
            "oof_frac_mean": float(np.mean(prof["oof_frac"]))
            if prof["oof_frac"]
            else None,
        }
        # #steps curve: mean over scenes at each checkpoint
        curve_out = {}
        for st in sorted(curve_agg):
            ds_list = curve_agg[st]

            def _m(key):
                vals = [d[key] for d in ds_list if d.get(key) is not None]
                return float(np.mean(vals)) if vals else None

            curve_out[str(st)] = {
                "delta_overall": _m("delta_overall"),
                "delta_hidden": _m("delta_hidden"),
                "lpips_gain": _m("lpips_gain"),
            }
        summary["steps_curve"] = curve_out
        # distance buckets (terciles of baseline translation norm)
        if prof["dist_pairs"]:
            arr = np.array(prof["dist_pairs"])  # (N,2): t, delta_overall
            ts = arr[:, 0]
            q1, q2 = np.quantile(ts, [1 / 3, 2 / 3])
            buckets = {"near": [], "mid": [], "far": []}
            for t, d in arr:
                if t <= q1:
                    buckets["near"].append(d)
                elif t <= q2:
                    buckets["mid"].append(d)
                else:
                    buckets["far"].append(d)
            summary["distance_buckets"] = {
                "q1_t": float(q1),
                "q2_t": float(q2),
                "near_delta_overall": float(np.mean(buckets["near"]))
                if buckets["near"]
                else None,
                "mid_delta_overall": float(np.mean(buckets["mid"]))
                if buckets["mid"]
                else None,
                "far_delta_overall": float(np.mean(buckets["far"]))
                if buckets["far"]
                else None,
                "near_n": len(buckets["near"]),
                "mid_n": len(buckets["mid"]),
                "far_n": len(buckets["far"]),
            }
    with open(os.path.join(args.out, "summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)
    all_res_clean = [
        {k: v for k, v in r.items() if not k.startswith("_")} for r in all_res
    ]
    with open(os.path.join(args.out, "per_scene.json"), "w") as fp:
        json.dump(all_res_clean, fp, indent=2)

    print("\n=== HIDDEN-OPT ORACLE SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\n=== PRE-REGISTERED GATES (D026) ===")
    g1 = summary["src_delta"] is not None and abs(summary["src_delta"]) <= 0.05
    g2 = summary["delta_overall"] is not None and summary["delta_overall"] >= 1.0
    g3 = summary["delta_hidden"] is not None and summary["delta_hidden"] >= 2.0
    g4 = summary["lpips_gain"] is not None and summary["lpips_gain"] >= 0.02
    print(
        f"G1 source-null |src_delta|<=0.05: {summary['src_delta']} -> {'PASS' if g1 else 'FAIL'}"
    )
    print(
        f"G2 overall delta>=+1.0dB: {summary['delta_overall']} -> {'PASS' if g2 else 'FAIL'}"
    )
    print(
        f"G3 hidden-region delta>=+2.0dB: {summary['delta_hidden']} -> {'PASS' if g3 else 'FAIL'}"
    )
    print(
        f"G4 hidden LPIPS gain>=0.02: {summary['lpips_gain']} -> {'PASS' if g4 else 'FAIL'}"
    )
    verdict = "PASS" if (g1 and (g2 or g3 or g4)) else ("PARTIAL/FAIL")
    print(f"\nVERDICT (G1 and (G2 or G3 or G4)): {verdict}")
    summary["gates"] = {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "verdict": verdict}
    with open(os.path.join(args.out, "summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)


if __name__ == "__main__":
    main()
