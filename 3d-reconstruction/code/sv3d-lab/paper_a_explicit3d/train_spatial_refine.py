"""
Paper A (D017) — SpatialRefine training. Standard single-image NVS objective on RE10K:
  L = L1 + w_ssim*SSIM + w_lpips*LPIPS(vgg, after warmup), on target renders vs GT (5% crop).
Residual head is zero-init -> step 0 == Flash3D baseline, so the metric can only improve if the
spatial-guided refinement helps. Compare deltas in OUR harness (Flash3D repro tgt5=28.68).

Usage (server):
  python /root/sv3d-lab/paper_a_explicit3d/train_spatial_refine.py --level L1 --n_scenes 20 \
     --steps 4000 --split .../test_files_wide700.txt --out /home/data/sv3d-lab/runs/sr_L1
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, "/root/sv3d-lab")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.spatial_refine_model import SpatialRefineModel, SpatialRefineConfig  # noqa: E402
from trainer_lib import SSIM, crop5, psnr  # noqa: E402
from train_paper_a import load_re10k, build_cfg, to_device  # noqa: E402


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["L0", "L1", "L2"], default="L0")
    ap.add_argument("--n_scenes", type=int, default=20)
    ap.add_argument("--scene_start", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w_ssim", type=float, default=0.15)
    ap.add_argument("--w_lpips", type=float, default=0.25)
    ap.add_argument("--lpips_after", type=int, default=1000)
    ap.add_argument("--freeze_visible", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--resume", default="")
    ap.add_argument("--novel_frames", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    return ap


def main():
    args = build_args().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(
        f"[sr] level={args.level} n_scenes={args.n_scenes} steps={args.steps}",
        flush=True,
    )

    cfg = build_cfg(args.novel_frames, args.split)
    Re10KDataset = load_re10k()
    ds = Re10KDataset(cfg, split="test")
    s0 = args.scene_start
    ds._seq_key_src_idx_pairs = ds._seq_key_src_idx_pairs[s0 : s0 + args.n_scenes]
    ds.length = len(ds._seq_key_src_idx_pairs)
    print(f"[sr] dataset scenes used: {ds.length} (start={s0})", flush=True)

    from torch.utils.data import DataLoader
    from datasets.util import custom_collate
    from common.data.robust_dataset import RobustDataset

    ds = RobustDataset(ds)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        persistent_workers=args.num_workers > 0,
    )

    rcfg = SpatialRefineConfig(freeze_visible=args.freeze_visible)
    model = SpatialRefineModel(cfg, rcfg).to(device)
    ckpt = "/root/projects/flash3d/checkpoints/model_re10k_v2.pth"
    if os.path.exists(ckpt):
        model.load_visible_pretrained(ckpt, device=device)
        print("[sr] loaded visible pretrained", flush=True)
    model.train()

    ssim = SSIM().to(device)
    try:
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

        lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
    except Exception as e:
        print(f"[sr] LPIPS unavailable ({e}); disabling", flush=True)
        lpips = None

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location=device)
        model.load_state_dict(sd["model"], strict=False)
        start_step = int(sd.get("step", 0))
        if "opt" in sd:
            try:
                opt.load_state_dict(sd["opt"])
            except Exception as e:
                print(f"[sr] opt not restored ({e})", flush=True)
        print(f"[sr] RESUMED from {args.resume} at {start_step}", flush=True)
    n_train = sum(p.numel() for p in params)
    n_ref = (
        sum(p.numel() for p in model.head.parameters())
        + sum(p.numel() for p in model.blocks.parameters())
        + sum(p.numel() for p in model.point_proj.parameters())
        + sum(p.numel() for p in model.feat_proj.parameters())
    )
    print(
        f"[sr] trainable params: {n_train / 1e6:.2f}M (refine {n_ref / 1e6:.2f}M)",
        flush=True,
    )

    novel = args.novel_frames
    logf = open(os.path.join(args.out, "train_log.jsonl"), "a")

    step = start_step
    t0 = time.time()
    data_iter = iter(loader)
    while step < args.steps:
        try:
            inputs = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inputs = next(data_iter)
        inputs = to_device(inputs, device)
        target_ids = [fid for fid in novel if ("color", fid, 0) in inputs]
        if not target_ids:
            step += 1
            continue

        outputs = model(inputs, target_ids)
        lpips_on = lpips is not None and step >= args.lpips_after
        total = 0.0
        psnrs = []
        for fid in target_ids:
            gt = inputs[("color", fid, 0)][0]
            pred = outputs[("render", fid)][0]
            predc, gtc = crop5(pred), crop5(gt)
            l1 = (predc - gtc).abs().mean()
            ssim_l = ssim(predc.unsqueeze(0), gtc.unsqueeze(0)).mean()
            loss = l1 + args.w_ssim * ssim_l
            if lpips_on:
                loss = loss + args.w_lpips * lpips(
                    predc.unsqueeze(0).clamp(0, 1) * 2 - 1,
                    gtc.unsqueeze(0).clamp(0, 1) * 2 - 1,
                )
            total = total + loss
            with torch.no_grad():
                p = psnr(predc, gtc)
                if p is not None:
                    psnrs.append(p)

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % args.log_every == 0:
            rec = {
                "step": step,
                "loss": float(total),
                "psnr": float(np.mean(psnrs)) if psnrs else None,
                "lpips_on": lpips_on,
                "t": time.time() - t0,
            }
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"[sr] {rec}", flush=True)

        if step > 0 and step % args.save_every == 0:
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(), "step": step},
                os.path.join(args.out, f"ckpt_{step:06d}.pt"),
            )
        step += 1

    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict(), "step": step},
        os.path.join(args.out, "ckpt_final.pt"),
    )
    logf.close()
    print("[sr] done", flush=True)


if __name__ == "__main__":
    main()
