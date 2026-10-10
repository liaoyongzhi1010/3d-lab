"""
Paper A — training entrypoint (Phase 4). Wires Flash3D Re10KDataset (whitelist reader) + hydra cfg +
PaperAModel + trainer_lib losses. Runs L0 smoke / L1 overfit / L2 pilot.

Usage (server):
  cd /root/projects/flash3d && source .venv/bin/activate
  export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python /root/sv3d-lab/paper_a_explicit3d/train_paper_a.py --level L0 --n_scenes 2 --steps 60 \
      --out /home/data/sv3d-lab/runs/paper_a_L0

Design: single-GPU, batch=1, GT poses (train.use_gt_poses=true), no EMA at L0/L1 (keep it minimal
and debuggable per D009 — correctness first, no extra machinery).
"""

import os
import sys
import json
import time
import argparse
import importlib.util

import numpy as np
import torch

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, "/root/sv3d-lab")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hydra import initialize_config_dir, compose

from model import PaperAModel, PaperAConfig
from trainer_lib import SSIM, compute_region_masks, crop5, psnr, build_argparser


def load_re10k():
    """Load Flash3D's Re10KDataset via importlib to avoid HF `datasets` shadowing."""
    spec = importlib.util.spec_from_file_location(
        "flash3d_re10k", "/root/projects/flash3d/datasets/re10k.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Re10KDataset


def build_cfg(novel_frames, split_path):
    with initialize_config_dir(
        config_dir="/root/projects/flash3d/configs", version_base=None
    ):
        overrides = [
            "+experiment=layered_re10k",
            "data_loader.batch_size=1",
            "data_loader.num_workers=0",
            f"dataset.test_split_path={split_path}",
            "hydra.job.chdir=false",
        ]
        cfg = compose(config_name="config", overrides=overrides)
    # set the novel frame ids used by the model config path
    return cfg


def to_device(inputs, device):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs


def main():
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(
        f"[paper_a] level={args.level} n_scenes={args.n_scenes} steps={args.steps}",
        flush=True,
    )

    cfg = build_cfg(args.novel_frames, args.split)

    # dataset: use TEST split loader (deterministic src+targets from the wide split file).
    Re10KDataset = load_re10k()
    ds = Re10KDataset(cfg, split="test")
    # restrict to a scene slice [scene_start : scene_start+n_scenes] (L2 uses disjoint train/eval)
    s0 = getattr(args, "scene_start", 0)
    ds._seq_key_src_idx_pairs = ds._seq_key_src_idx_pairs[s0 : s0 + args.n_scenes]
    ds.length = len(ds._seq_key_src_idx_pairs)
    print(f"[paper_a] dataset scenes used: {ds.length} (start={s0})", flush=True)

    from torch.utils.data import DataLoader
    from datasets.util import custom_collate

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=getattr(args, "num_workers", 0),
        collate_fn=custom_collate,
        persistent_workers=getattr(args, "num_workers", 0) > 0,
    )

    # model
    pcfg = PaperAConfig(
        n_queries=args.n_queries,
        freeze_visible=args.freeze_visible,
        anchor_mode=args.anchor_mode,
        behind_offset=args.behind_offset,
        latent_dim=args.latent_dim,
        latent_mode=args.latent_mode,
        k_hidden=getattr(args, "k_hidden", 2),
        opacity_init=getattr(args, "opacity_init", 0.3),
    )
    model = PaperAModel(cfg, pcfg).to(device)
    # init visible from official ckpt (fair common base)
    ckpt = "/root/projects/flash3d/checkpoints/model_re10k_v2.pth"
    if os.path.exists(ckpt):
        model.load_visible_pretrained(ckpt, device=device)
        print("[paper_a] loaded visible pretrained", flush=True)

    model.train()
    ssim = SSIM().to(device)
    try:
        from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

        lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
    except Exception as e:
        print(f"[paper_a] LPIPS unavailable ({e}); disabling", flush=True)
        lpips = None

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    start_step = 0
    resume = getattr(args, "resume", "")
    if resume and os.path.exists(resume):
        sd = torch.load(resume, map_location=device)
        model.load_state_dict(sd["model"], strict=False)
        start_step = int(sd.get("step", 0))
        if "opt" in sd:
            try:
                opt.load_state_dict(sd["opt"])
            except Exception as e:
                print(f"[paper_a] opt state not restored ({e})", flush=True)
        print(f"[paper_a] RESUMED from {resume} at step {start_step}", flush=True)
    n_train = sum(p.numel() for p in params)
    n_hidden = sum(p.numel() for p in model.hidden.parameters())
    print(
        f"[paper_a] trainable params: {n_train / 1e6:.2f}M (hidden {n_hidden / 1e6:.2f}M)",
        flush=True,
    )

    novel = args.novel_frames
    log_path = os.path.join(args.out, "train_log.jsonl")
    logf = open(log_path, "a")

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

        # ensure K_tgt exists for each novel frame; re10k gives K_tgt per loaded frame id.
        target_ids = [
            fid
            for fid in novel
            if ("K_tgt", fid) in inputs or ("color", fid, 0) in inputs
        ]
        if not target_ids:
            # fall back to whatever novel frames the dataset actually loaded
            target_ids = [fid for fid in novel if ("color", fid, 0) in inputs]
        if not target_ids:
            print(
                f"[paper_a] step {step}: no target frames in batch, skipping",
                flush=True,
            )
            step += 1
            continue

        outputs = model(
            inputs,
            target_ids,
            hidden_ray_fid=target_ids[-1],
            z_mode=("posterior" if args.latent_dim > 0 else None),
            post_tgt_fid=target_ids[-1],
        )

        # source depth (for region masks) — from visible backbone. NOTE: backbone depth is at
        # PADDED resolution (pad_border_aug); crop to the unpadded HxW that GT/renders use.
        depth_src_full = outputs[("depth", 0)].detach()  # (B*gpp,1,Hpad,Wpad)
        gpp = cfg.model.gaussians_per_pixel
        depth_src_p = depth_src_full[0, 0]  # first layer, batch 0 (B=1), (Hpad,Wpad)
        if pad and pad > 0:
            depth_src = depth_src_p[
                pad : depth_src_p.shape[0] - pad, pad : depth_src_p.shape[1] - pad
            ]
        else:
            depth_src = depth_src_p
        K0 = inputs[("K_src", 0)][0]

        total = 0.0
        logs = {}
        H = inputs["color", 0, 0].shape[2]
        W = inputs["color", 0, 0].shape[3]

        for fid in target_ids:
            gt = inputs[("color", fid, 0)][0]  # (3,H,W) unpadded GT
            merged = outputs[("render_merged", fid)][0]
            vonly = outputs[("render_vis", fid)][0]
            honly = outputs[("render_hidden", fid)][0]
            # renders come out at padded res? No: our renderer uses H,W of color[0,0] (unpadded).
            # region masks
            T = outputs[("cam_T_cam", 0, fid)][0]
            vis_m, occ_m, oof_m = compute_region_masks(depth_src, K0, T, H, W, device)
            hidden_m = occ_m | oof_m

            # region-weighted photometric on merged
            w = 1.0 + args.w_hidden * hidden_m.float()  # (H,W)
            w3 = w.unsqueeze(0)
            l1 = ((merged - gt).abs() * w3).mean()
            ssim_l = ssim(merged.unsqueeze(0), gt.unsqueeze(0)).mean()
            photo = l1 + 0.15 * ssim_l
            if lpips is not None and step >= args.lpips_after:
                lp = lpips(
                    (merged.unsqueeze(0) * 2 - 1).clamp(-1, 1),
                    (gt.unsqueeze(0) * 2 - 1).clamp(-1, 1),
                )
                photo = photo + args.w_lpips * lp
                logs["lpips"] = float(lp)

            # causal / load-bearing: hidden-only should explain (gt - vonly) residual in hidden region
            causal = torch.tensor(0.0, device=device)
            if hidden_m.sum() > 10:
                resid = gt - vonly.detach()
                # hidden render should match gt in hidden region (drive it to be load-bearing there)
                causal = ((honly - gt).abs()[:, hidden_m]).mean()

            total = total + photo + args.w_causal * causal

        # opacity sparsity on hidden (discourage smearing over visible)
        opa = torch.stack(
            [g["opacity"].mean() for g in outputs["hidden_gaussians"]]
        ).mean()
        # scale reg (avoid huge splats)
        scl = torch.stack(
            [g["scaling"].mean() for g in outputs["hidden_gaussians"]]
        ).mean()
        total = total + args.w_opa_sparse * opa + args.w_scale_reg * scl

        # source-null constraint (D027/4b): merged source render must match visible-only source
        # render => hidden Gaussians may not corrupt the source view. Frozen backbone => vis source
        # render is the Flash3D source; hidden that pokes into the source is penalized. w_null=0 keeps
        # legacy behavior unchanged.
        if (
            args.w_null > 0
            and ("render_merged", 0) in outputs
            and ("render_vis", 0) in outputs
        ):
            msrc = outputs[("render_merged", 0)][0]
            vsrc = outputs[("render_vis", 0)][0].detach()
            src_null = ((msrc - vsrc) ** 2).mean()
            total = total + args.w_null * src_null
            logs["src_null"] = float(src_null)

        # CVAE KL(q||p) (D013), linearly annealed. Only when latent_dim>0 and posterior present.
        kl_val = 0.0
        if args.latent_dim > 0 and "z_out" in outputs and "mu_q" in outputs["z_out"]:
            zo = outputs["z_out"]
            mu_q, logvar_q = zo["mu_q"], zo["logvar_q"]
            mu_p, logvar_p = zo["mu_p"], zo["logvar_p"]
            # KL(N(mu_q,var_q) || N(mu_p,var_p)) closed form
            kl = (
                0.5
                * (
                    (logvar_p - logvar_q)
                    + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
                    - 1.0
                )
                .sum(dim=1)
                .mean()
            )
            beta = args.w_kl * min(1.0, step / max(1, args.kl_anneal))
            total = total + beta * kl
            kl_val = float(kl)

        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if not torch.isfinite(total):
            print(
                f"[paper_a] step {step}: NON-FINITE loss {total.item()} — stopping",
                flush=True,
            )
            torch.save(
                {"inputs_keys": list(inputs.keys())},
                os.path.join(args.out, "nan_batch.pt"),
            )
            break

        if step % args.log_every == 0:
            # quick eval on last target: region PSNRs + deletion delta
            with torch.no_grad():
                fid = target_ids[-1]
                gt = inputs[("color", fid, 0)][0]
                merged = outputs[("render_merged", fid)][0]
                vonly = outputs[("render_vis", fid)][0]
                T = outputs[("cam_T_cam", 0, fid)][0]
                vis_m, occ_m, oof_m = compute_region_masks(
                    depth_src, K0, T, H, W, device
                )
                hidden_m = occ_m | oof_m
                p_all_m = psnr(crop5(merged.unsqueeze(0)), crop5(gt.unsqueeze(0)))
                p_all_v = psnr(crop5(merged.unsqueeze(0)), crop5(gt.unsqueeze(0)))
                p_hid_m = psnr(merged, gt, hidden_m)
                p_hid_v = psnr(vonly, gt, hidden_m)
                delta = (
                    (p_hid_m - p_hid_v)
                    if (p_hid_m is not None and p_hid_v is not None)
                    else None
                )
            rec = {
                "step": step,
                "loss": float(total),
                "psnr_all_merged": p_all_m,
                "psnr_hidden_merged": p_hid_m,
                "psnr_hidden_vis": p_hid_v,
                "deletion_delta": delta,
                "hidden_opacity_mean": float(opa),
                "hidden_scale_mean": float(scl),
                "hidden_frac": float(hidden_m.float().mean()),
                "kl": kl_val,
                "t": time.time() - t0,
            }
            rec.update(logs)
            print("[paper_a] " + json.dumps(rec), flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()

        if args.save_every and step > 0 and step % args.save_every == 0:
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(), "step": step},
                os.path.join(args.out, f"ckpt_{step:06d}.pt"),
            )

        step += 1

    # final save + summary
    torch.save(
        {"model": model.state_dict(), "step": step},
        os.path.join(args.out, "ckpt_final.pt"),
    )
    logf.close()
    print(f"[paper_a] DONE {step} steps in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
