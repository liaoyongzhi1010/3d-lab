"""
Paper A (reframed, D015) — training entrypoint for calibrated aleatoric uncertainty.

Model = Flash3D visible backbone + per-Gaussian log-variance head (uncertainty_model.py).
Loss  = photometric recon on the MEAN (L1 + SSIM, + LPIPS after warmup) to keep NVS quality on par
        with Flash3D, PLUS heteroscedastic Gaussian NLL that learns the per-pixel variance:
          L_nll = 0.5 * mean[ (Î - I_gt)^2 / σ^2 + log σ^2 ]   (over the 5%-cropped image)
        β_nll anneals in so σ^2 is not driven to extremes before the mean stabilises.

Single-image inference: variance is a function of source features only; targets used only as
supervision. Uncertainty is splatted through the shared 3D Gaussians -> multi-view consistent.

Usage (server):
  cd /root/projects/flash3d && source .venv/bin/activate
  export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python /root/sv3d-lab/paper_a_explicit3d/train_uncertainty.py --level L0 --n_scenes 2 --steps 60 \
      --out /home/data/sv3d-lab/runs/unc_L0
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

from model.uncertainty_model import UncertaintyModel, UncertaintyConfig  # noqa: E402
from trainer_lib import SSIM, compute_region_masks, crop5, psnr  # noqa: E402
from train_paper_a import load_re10k, build_cfg, to_device  # noqa: E402


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["L0", "L1", "L2"], default="L0")
    ap.add_argument("--n_scenes", type=int, default=2)
    ap.add_argument("--scene_start", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--w_ssim", type=float, default=0.15)
    ap.add_argument("--w_lpips", type=float, default=0.25)
    ap.add_argument("--lpips_after", type=int, default=3000)
    ap.add_argument("--w_nll", type=float, default=1.0)
    ap.add_argument("--nll_anneal", type=int, default=1500)
    ap.add_argument("--freeze_visible", action="store_true")
    ap.add_argument(
        "--unc_only",
        action="store_true",
        help="freeze visible, train ONLY the uncertainty head (cheap sanity)",
    )
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=2500)
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
        f"[unc] level={args.level} n_scenes={args.n_scenes} steps={args.steps}",
        flush=True,
    )

    cfg = build_cfg(args.novel_frames, args.split)
    Re10KDataset = load_re10k()
    ds = Re10KDataset(cfg, split="test")
    s0 = args.scene_start
    ds._seq_key_src_idx_pairs = ds._seq_key_src_idx_pairs[s0 : s0 + args.n_scenes]
    ds.length = len(ds._seq_key_src_idx_pairs)
    print(f"[unc] dataset scenes used: {ds.length} (start={s0})", flush=True)

    from torch.utils.data import DataLoader
    from datasets.util import custom_collate
    from common.data.robust_dataset import RobustDataset

    ds = RobustDataset(ds)  # skip corrupt RE10K gzips instead of crashing

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        persistent_workers=args.num_workers > 0,
    )

    ucfg = UncertaintyConfig(freeze_visible=(args.freeze_visible or args.unc_only))
    model = UncertaintyModel(cfg, ucfg).to(device)
    ckpt = "/root/projects/flash3d/checkpoints/model_re10k_v2.pth"
    if os.path.exists(ckpt):
        model.load_visible_pretrained(ckpt, device=device)
        print("[unc] loaded visible pretrained", flush=True)
    model.train()

    ssim = SSIM().to(device)
    try:
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

        lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
    except Exception as e:
        print(f"[unc] LPIPS unavailable ({e}); disabling", flush=True)
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
                print(f"[unc] opt not restored ({e})", flush=True)
        print(f"[unc] RESUMED from {args.resume} at {start_step}", flush=True)
    n_train = sum(p.numel() for p in params)
    n_unc = sum(p.numel() for p in model.logvar_head.parameters())
    print(
        f"[unc] trainable params: {n_train / 1e6:.2f}M (unc head {n_unc / 1e6:.3f}M)",
        flush=True,
    )

    novel = args.novel_frames
    logf = open(os.path.join(args.out, "train_log.jsonl"), "a")
    pad = cfg.dataset.pad_border_aug

    def unpad(img):
        if pad and pad > 0:
            return img[..., pad : img.shape[-2] - pad, pad : img.shape[-1] - pad]
        return img

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

        H = inputs["color", 0, 0].shape[2]
        W = inputs["color", 0, 0].shape[3]
        # source depth for region masks (padded -> unpadded)
        depth_src_full = outputs.get(("depth", 0), None)
        total = 0.0
        logs = {}
        beta = args.w_nll * min(1.0, step / max(1, args.nll_anneal))
        lpips_on = lpips is not None and step >= args.lpips_after

        agg_err_vis, agg_err_hid, agg_var_vis, agg_var_hid = [], [], [], []
        for fid in target_ids:
            gt = inputs[("color", fid, 0)][0]  # (3,H,W)
            pred = outputs[("render", fid)][0]  # (3,H,W)
            var = outputs[("var", fid)][0]  # (1,H,W)

            predc, gtc, varc = crop5(pred), crop5(gt), crop5(var)
            # recon on the mean
            l1 = (predc - gtc).abs().mean()
            ssim_l = ssim(predc.unsqueeze(0), gtc.unsqueeze(0)).mean()
            recon = l1 + args.w_ssim * ssim_l
            if lpips_on:
                lp = lpips(
                    predc.unsqueeze(0).clamp(0, 1) * 2 - 1,
                    gtc.unsqueeze(0).clamp(0, 1) * 2 - 1,
                )
                recon = recon + args.w_lpips * lp
            # heteroscedastic Gaussian NLL. DETACH the mean so the NLL trains ONLY the variance
            # (the recon term above trains the mean); otherwise the mean "cheats" by inflating error
            # to match a large predicted variance, destabilising the pretrained backbone.
            se = ((predc.detach() - gtc) ** 2).mean(dim=0, keepdim=True)  # (1,h,w)
            nll = 0.5 * (se / varc + torch.log(varc)).mean()
            total = total + recon + beta * nll

            # diagnostics: does variance track error, split by region?
            if depth_src_full is not None:
                gpp = cfg.model.gaussians_per_pixel
                dpp = depth_src_full[0, 0]
                dsrc = (
                    dpp[pad : dpp.shape[0] - pad, pad : dpp.shape[1] - pad]
                    if pad
                    else dpp
                )
                K0 = inputs[("K_src", 0)][0]
                T = outputs[("cam_T_cam", 0, fid)][0]
                vis_m, occ_m, oof_m = compute_region_masks(dsrc, K0, T, H, W, device)
                hid_m = occ_m | oof_m
                err = ((pred - gt) ** 2).mean(dim=0)  # (H,W)
                v = var[0]  # (H,W)
                if vis_m.sum() > 10:
                    agg_err_vis.append(float(err[vis_m].mean()))
                    agg_var_vis.append(float(v[vis_m].mean()))
                if hid_m.sum() > 10:
                    agg_err_hid.append(float(err[hid_m].mean()))
                    agg_var_hid.append(float(v[hid_m].mean()))

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % args.log_every == 0:
            with torch.no_grad():
                p0 = psnr(
                    crop5(outputs[("render", target_ids[0])][0]),
                    crop5(inputs[("color", target_ids[0], 0)][0]),
                )
            rec = {
                "step": step,
                "loss": float(total),
                "beta": beta,
                "psnr": float(p0) if p0 is not None else None,
                "var_vis": float(np.mean(agg_var_vis)) if agg_var_vis else None,
                "var_hid": float(np.mean(agg_var_hid)) if agg_var_hid else None,
                "err_vis": float(np.mean(agg_err_vis)) if agg_err_vis else None,
                "err_hid": float(np.mean(agg_err_hid)) if agg_err_hid else None,
                "t": time.time() - t0,
            }
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"[unc] {rec}", flush=True)

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
    print("[unc] done", flush=True)


if __name__ == "__main__":
    main()
