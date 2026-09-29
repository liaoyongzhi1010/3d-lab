"""
Paper A Phase 4b — DISTILLATION trainer (D030). Trains the amortized VisibleAnchoredHiddenHead by
distilling the per-scene oracle teachers (gen_oracle_teacher.py) instead of regressing raw multi-modal
target RGB (which collapsed 4x: E006/E008/E009/E015).

Per scene: load teacher G*_hidden (source-frame Gaussians), render it merged-with-visible to each target
= the "achievable" supervision image. Train the student so its merged render matches the TEACHER render
(L1 + LPIPS) on the hidden region, plus source-null. The teacher target is a single concrete 3D
explanation => non-gray, achievable => the opacity->0 mean-collapse optimum is removed.

Iterates scenes by explicit dataset INDEX so teacher_{idx}.npz aligns with the scene (no shuffle).

Run on server (flash3d venv):
  python /root/sv3d-lab/paper_a_explicit3d/train_distill.py \
      --teacher_dir /home/data/sv3d-lab/teachers/wide700_l2 \
      --split /root/projects/flash3d/splits/re10k_mine_filtered/test_files_wide700.txt \
      --steps 8000 --out /home/data/sv3d-lab/runs/paperA_distill_pilot
"""

import os
import sys
import json
import time
import glob
import argparse

import numpy as np
import torch

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, "/root/sv3d-lab")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "oracle"))

from model import PaperAModel, PaperAConfig
from train_paper_a import build_cfg, load_re10k, to_device
from trainer_lib import compute_region_masks, psnr
from oracle_core import render_gaussians_relpose, merge_gaussians
from datasets.util import custom_collate


def load_teacher(teacher_dir, idx, device):
    fn = os.path.join(teacher_dir, f"teacher_{idx:06d}.npz")
    if not os.path.exists(fn):
        return None
    d = np.load(fn)
    rgb = torch.from_numpy(d["rgb"]).float().to(device)
    return {
        "xyz": torch.from_numpy(d["xyz"]).float().to(device),
        "scaling": torch.from_numpy(d["scaling"]).float().to(device),
        "rotation": torch.from_numpy(d["rotation"]).float().to(device),
        "opacity": torch.from_numpy(d["opacity"]).float().to(device),
        "rgb_direct": rgb,
        "features_dc": rgb.reshape(-1, 1, 3),
    }


def get_inputs(ds, idx, device):
    return to_device(custom_collate([ds[idx]]), device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir", required=True)
    ap.add_argument(
        "--split",
        default="/root/projects/flash3d/splits/re10k_mine_filtered/test_files_wide700.txt",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k_hidden", type=int, default=2)
    ap.add_argument("--opacity_init", type=float, default=0.3)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--hidden_depth", type=int, default=3)
    ap.add_argument(
        "--w_distill",
        type=float,
        default=1.0,
        help="L1 on merged render vs teacher render",
    )
    ap.add_argument("--w_lpips", type=float, default=0.5)
    ap.add_argument("--lpips_after", type=int, default=500)
    ap.add_argument(
        "--w_hidden",
        type=float,
        default=5.0,
        help="extra weight on hidden-region pixels",
    )
    ap.add_argument("--w_null", type=float, default=5.0)
    ap.add_argument(
        "--w_gt",
        type=float,
        default=0.25,
        help="direct GT term on visible-covered region",
    )
    ap.add_argument("--novel_frames", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument(
        "--eval_start",
        type=int,
        default=0,
        help="held-out eval scenes [eval_start:+eval_n]",
    )
    ap.add_argument("--eval_n", type=int, default=40)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=2)

    cfg = build_cfg(args.novel_frames, args.split)
    Re10K = load_re10k()
    ds = Re10K(cfg, split="test")
    n_ds = len(ds._seq_key_src_idx_pairs)
    pad = cfg.dataset.pad_border_aug

    teacher_files = glob.glob(os.path.join(args.teacher_dir, "teacher_*.npz"))
    teacher_idxs = sorted(
        int(os.path.basename(f).split("_")[1].split(".")[0]) for f in teacher_files
    )
    teacher_idxs = [i for i in teacher_idxs if i < n_ds]
    print(f"[distill] {len(teacher_idxs)} teachers; dataset {n_ds} scenes", flush=True)
    if not teacher_idxs:
        print("[distill] NO teachers found; abort", flush=True)
        return

    pcfg = PaperAConfig(
        anchor_mode="visible",
        freeze_visible=True,
        k_hidden=args.k_hidden,
        opacity_init=args.opacity_init,
        hidden_dim=args.hidden_dim,
        hidden_depth=args.hidden_depth,
    )
    model = PaperAModel(cfg, pcfg).to(device)
    ckpt = "/root/projects/flash3d/checkpoints/model_re10k_v2.pth"
    model.load_visible_pretrained(ckpt, device=device)
    print("[distill] loaded official visible backbone (frozen)", flush=True)
    model.train()

    try:
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

        lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
    except Exception as e:
        print(f"[distill] LPIPS unavailable ({e})", flush=True)
        lpips = None

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    print(
        f"[distill] trainable params {sum(p.numel() for p in params) / 1e6:.2f}M",
        flush=True,
    )
    logf = open(os.path.join(args.out, "train_log.jsonl"), "a")

    def render_teacher(g_teacher, inputs, outputs, target_ids, H, W):
        """Render teacher(vis+G*) to each target using the SAME visible Gaussians the student uses."""
        gvis = {
            k: outputs["vis_gaussians"][0][k]
            for k in [
                "xyz",
                "scaling",
                "rotation",
                "opacity",
                "rgb_direct",
                "features_dc",
            ]
        }
        merged = merge_gaussians([gvis, g_teacher])
        K = inputs[("K_src", 0)][0]
        imgs = {}
        for fid in target_ids:
            T = outputs[("cam_T_cam", 0, fid)][0]
            imgs[fid] = (
                render_gaussians_relpose(merged, K, T, H, W, device)["render"]
                .clamp(0, 1)
                .detach()
            )
        return imgs

    def run_eval(eval_idxs):
        model.eval()
        dels, opas = [], []
        with torch.no_grad():
            for idx in eval_idxs:
                try:
                    inputs = get_inputs(ds, idx, device)
                except Exception:
                    continue
                tids = [f for f in args.novel_frames if ("color", f, 0) in inputs]
                if not tids:
                    continue
                try:
                    outs = model(inputs, tids, hidden_ray_fid=tids[-1], z_mode=None)
                except Exception:
                    continue
                H = inputs["color", 0, 0].shape[2]
                W = inputs["color", 0, 0].shape[3]
                dsrc = outs[("depth", 0)][0, 0]
                if pad and pad > 0:
                    dsrc = dsrc[pad : dsrc.shape[0] - pad, pad : dsrc.shape[1] - pad]
                K0 = inputs[("K_src", 0)][0]
                for fid in tids:
                    gt = inputs[("color", fid, 0)][0]
                    merged = outs[("render_merged", fid)][0]
                    vonly = outs[("render_vis", fid)][0]
                    T = outs[("cam_T_cam", 0, fid)][0]
                    _, occ, oof = compute_region_masks(dsrc, K0, T, H, W, device)
                    hid = occ | oof
                    pm = psnr(merged, gt, hid)
                    pv = psnr(vonly, gt, hid)
                    if pm is not None and pv is not None:
                        dels.append(pm - pv)
                opas.append(
                    float(
                        torch.stack(
                            [g["opacity"].mean() for g in outs["hidden_gaussians"]]
                        ).mean()
                    )
                )
        model.train()
        return (
            float(np.mean(dels)) if dels else None,
            float(np.mean(opas)) if opas else None,
        )

    eval_idxs = list(range(args.eval_start, args.eval_start + args.eval_n))
    step = 0
    t0 = time.time()
    order = list(teacher_idxs)
    ptr = 0
    while step < args.steps:
        if ptr >= len(order):
            np.random.shuffle(order)
            ptr = 0
        idx = order[ptr]
        ptr += 1
        g_teacher = load_teacher(args.teacher_dir, idx, device)
        if g_teacher is None:
            continue
        try:
            inputs = get_inputs(ds, idx, device)
        except Exception:
            continue
        target_ids = [f for f in args.novel_frames if ("color", f, 0) in inputs]
        if not target_ids:
            continue

        outputs = model(inputs, target_ids, hidden_ray_fid=target_ids[-1], z_mode=None)
        H = inputs["color", 0, 0].shape[2]
        W = inputs["color", 0, 0].shape[3]
        dsrc_full = outputs[("depth", 0)].detach()[0, 0]
        if pad and pad > 0:
            dsrc = dsrc_full[
                pad : dsrc_full.shape[0] - pad, pad : dsrc_full.shape[1] - pad
            ]
        else:
            dsrc = dsrc_full
        K0 = inputs[("K_src", 0)][0]

        teacher_imgs = render_teacher(g_teacher, inputs, outputs, target_ids, H, W)

        total = 0.0
        logs = {}
        for fid in target_ids:
            student = outputs[("render_merged", fid)][0]
            teach = teacher_imgs[fid]
            gt = inputs[("color", fid, 0)][0]
            T = outputs[("cam_T_cam", 0, fid)][0]
            _, occ, oof = compute_region_masks(dsrc, K0, T, H, W, device)
            hid = (occ | oof).float()
            w = (1.0 + args.w_hidden * hid).unsqueeze(0)
            total = total + args.w_distill * ((student - teach).abs() * w).mean()
            if args.w_gt > 0:
                vism = (1.0 - hid).unsqueeze(0)
                total = total + args.w_gt * ((student - gt).abs() * vism).mean()
            if lpips is not None and step >= args.lpips_after:
                lp = lpips(
                    (student.unsqueeze(0) * 2 - 1).clamp(-1, 1),
                    (teach.unsqueeze(0) * 2 - 1).clamp(-1, 1),
                )
                total = total + args.w_lpips * lp
                logs["lpips"] = float(lp)

        if (
            args.w_null > 0
            and ("render_merged", 0) in outputs
            and ("render_vis", 0) in outputs
        ):
            msrc = outputs[("render_merged", 0)][0]
            vsrc = outputs[("render_vis", 0)][0].detach()
            total = total + args.w_null * ((msrc - vsrc) ** 2).mean()

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % args.log_every == 0:
            opa = float(
                torch.stack(
                    [g["opacity"].mean() for g in outputs["hidden_gaussians"]]
                ).mean()
            )
            rec = {
                "step": step,
                "loss": float(total),
                "hidden_opacity": opa,
                "t": time.time() - t0,
            }
            rec.update(logs)
            print("[distill] " + json.dumps(rec), flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()

        if args.eval_every and step > 0 and step % args.eval_every == 0:
            d, o = run_eval(eval_idxs)
            rec = {"step": step, "EVAL_heldout_deletion": d, "EVAL_opacity": o}
            print("[distill] " + json.dumps(rec), flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()

        if args.save_every and step > 0 and step % args.save_every == 0:
            torch.save(
                {"model": model.state_dict(), "step": step},
                os.path.join(args.out, f"ckpt_{step:06d}.pt"),
            )
        step += 1

    torch.save(
        {"model": model.state_dict(), "step": step},
        os.path.join(args.out, "ckpt_final.pt"),
    )
    d, o = run_eval(eval_idxs)
    print(f"[distill] FINAL heldout deletion={d} opacity={o}", flush=True)
    logf.write(
        json.dumps({"step": step, "FINAL_deletion": d, "FINAL_opacity": o}) + "\n"
    )
    logf.close()
    print(f"[distill] DONE {step} steps in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
