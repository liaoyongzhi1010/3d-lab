"""E-052: cache sharp 3D-native teacher labels for feed-forward student training.

This is NOT the final method. It converts the previously successful per-scene
SD-anchor + multi-view fitting pipeline (E-042) into an offline teacher that
emits supervision targets:
  - source/target metadata
  - cleaned hole mask
  - visible render, SD anchor, naive-lift render, final teacher render, GT
  - optimized hole Gaussian parameters (explicit 3DGS label)

The final student will be feed-forward and trained on these labels; inference
will not run SD or per-scene optimization.
"""

import argparse
import importlib.util
import json
import os
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEVICE = "cuda:0"
RES = 256

spec42 = importlib.util.spec_from_file_location(
    "e042", "/root/projects/flash3d/_e042_sdprior_sharp.py"
)
E42 = importlib.util.module_from_spec(spec42)
spec42.loader.exec_module(E42)

CG = E42.CG
VGGT = E42.VGGT
E22 = E42.E22


def save_img(t, path):
    arr = E42.to_np(t)
    Image.fromarray(arr).save(path)


def masked_psnr(pred, gt, mask):
    if mask.sum() < 10:
        return None
    mse = ((pred - gt) ** 2).mean(0)[mask].mean().item()
    return -10 * math.log10(max(mse, 1e-10))


def load_sd_pipeline(use_controlnet):
    from diffusers import StableDiffusionInpaintPipeline

    pipe = None
    cn = None
    if use_controlnet:
        try:
            from diffusers import (
                StableDiffusionControlNetInpaintPipeline,
                ControlNetModel,
            )

            cn = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
            )
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                controlnet=cn,
                torch_dtype=torch.float16,
                safety_checker=None,
            ).to(DEVICE)
            print("SD-inpaint + ControlNet-depth loaded", flush=True)
        except Exception as e:
            print(f"ControlNet failed ({e}); fallback plain SD-inpaint", flush=True)
            pipe = None
    if pipe is None:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(DEVICE)
        cn = None
    pipe.set_progress_bar_config(disable=True)
    return pipe, cn


def load_models(noroute_ckpt):
    import lpips as _lpips

    lpips_fn = _lpips.LPIPS(net="vgg").to(DEVICE).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    vggt = VGGT.from_pretrained("facebook/VGGT-1B").to(DEVICE).eval()
    for p in vggt.parameters():
        p.requires_grad = False

    net_vis = CG.DirectUNet(in_ch=5, base=48, K=1).to(DEVICE)
    net_vis.load_state_dict(torch.load(noroute_ckpt, map_location=DEVICE))
    net_vis.eval()
    for p in net_vis.parameters():
        p.requires_grad = False
    return lpips_fn, vggt, net_vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/data/E-052_teacher_cache")
    ap.add_argument("--n_candidates", type=int, default=64)
    ap.add_argument("--max_keep", type=int, default=12)
    ap.add_argument("--gap", type=int, default=50)
    ap.add_argument("--n_aux", type=int, default=3)
    ap.add_argument("--fit_iters", type=int, default=200)
    ap.add_argument("--min_hole", type=float, default=0.06)
    ap.add_argument("--use_controlnet", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--noroute_ckpt",
        default="/home/data/archive_v0.0.1_disocfill_accv/results/crossgs_v3/noroute.pt",
    )
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ["labels", "vis", "teacher", "anchor", "base", "gt", "mask"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pipe, cn = load_sd_pipeline(bool(args.use_controlnet))
    lpips_fn, vggt, net_vis = load_models(args.noroute_ckpt)

    eval_seqs = CG.load_re10k_index(args.n_candidates + 200)[200:]
    manifest = []
    kept = 0

    for si, rows in enumerate(eval_seqs[: args.n_candidates]):
        if kept >= args.max_keep:
            break
        try:
            i = len(rows) // 3
            j = min(i + args.gap, len(rows) - 1)
            s, t = rows[i], rows[j]
            pair = {
                "src_pil": Image.open(s["img"]).convert("RGB"),
                "gt": torch.tensor(
                    cv2.resize(
                        np.array(Image.open(t["img"]).convert("RGB")).astype(np.float32)
                        / 255.0,
                        (RES, RES),
                    ),
                    device=DEVICE,
                ).permute(2, 0, 1),
                "w2c_s": E42.view_params(s)["w2c"],
                "w2c_t": E42.view_params(t)["w2c"],
                "fx": E42.view_params(t)["fx"],
                "fy": E42.view_params(t)["fy"],
                "cx": E42.view_params(t)["cx"],
                "cy": E42.view_params(t)["cy"],
            }

            hi = min(len(rows) - 1, j + args.gap)
            cand = [k for k in range(i + 1, hi + 1) if k != j]
            aux_idx = (
                sorted(
                    np.random.choice(
                        cand, min(args.n_aux, len(cand)), replace=False
                    ).tolist()
                )
                if cand
                else []
            )
            aux = []
            for k in aux_idx:
                r = rows[k]
                gt_k = torch.tensor(
                    cv2.resize(
                        np.array(Image.open(r["img"]).convert("RGB")).astype(np.float32)
                        / 255.0,
                        (RES, RES),
                    ),
                    device=DEVICE,
                ).permute(2, 0, 1)
                aux.append((gt_k, E42.view_params(r)))

            with torch.no_grad():
                _, depth_src, _ = E22.M.vggt_geom(vggt, pair["src_pil"], RES)
                scale_k = float(depth_src.median().item())
                g_vis, _ = E22.build_visible_gaussians(vggt, net_vis, pair, scale_k)
                ctx_rgb, hole_mask, boundary_depth = E22.detect_holes_and_context(
                    g_vis, pair, scale_k, vggt
                )
                hole_mask = E42.clean_hole_mask(hole_mask, 0.01)
            hole_frac = float(hole_mask.float().mean().item())
            if hole_frac < args.min_hole or hole_mask.sum() < 50:
                print(f"cand{si}: hole={hole_frac:.3f}, skip", flush=True)
                continue

            rp = (pair["w2c_t"], pair["fx"], pair["fy"], pair["cx"], pair["cy"])
            r_vis = E22.render_rgb(g_vis, *rp)

            depth_ctrl = None
            if cn is not None:
                with torch.no_grad():
                    _, dvis, _ = E22.M.vggt_geom(vggt, pair["src_pil"], RES)
                dn = (dvis - dvis.min()) / (dvis.max() - dvis.min() + 1e-6)
                dctrl = (dn.squeeze().cpu().numpy() * 255).astype(np.uint8)
                depth_ctrl = Image.fromarray(np.stack([dctrl] * 3, -1)).resize(
                    (512, 512)
                )

            with torch.no_grad():
                col_fill = E42.sd_inpaint(pipe, None, r_vis, hole_mask, depth_ctrl)

            bd_val = float(boundary_depth[0, 0, 0].item())
            dep_flat = torch.full((RES, RES), bd_val, device=DEVICE)
            gh = E42.init_hole_gaussians(
                col_fill.permute(1, 2, 0), dep_flat, hole_mask, pair
            )
            for k in gh:
                gh[k] = gh[k].detach().requires_grad_(True)
            opt = torch.optim.Adam(
                [
                    gh["xyz"],
                    gh["logscale"],
                    gh["sh_dc"],
                    gh["opacity_logit"],
                    gh["rot"],
                ],
                lr=0.02,
            )

            targets = [
                (
                    col_fill,
                    {
                        "w2c": pair["w2c_t"],
                        "fx": pair["fx"],
                        "fy": pair["fy"],
                        "cx": pair["cx"],
                        "cy": pair["cy"],
                    },
                    hole_mask,
                )
            ]
            for gt_k, vp in aux:
                targets.append((gt_k, vp, None))

            for _ in range(args.fit_iters):
                g_full = E22.merge_gaussians(g_vis, E42.assemble(gh))
                loss = 0.0
                for tgt, vp, _m in targets:
                    r = E42.render_at(g_full, vp)
                    loss = (
                        loss
                        + 0.2 * F.l1_loss(r, tgt)
                        + 0.8
                        * lpips_fn(
                            r.unsqueeze(0) * 2 - 1, tgt.unsqueeze(0) * 2 - 1
                        ).mean()
                    )
                loss = loss / len(targets)
                opt.zero_grad()
                loss.backward()
                opt.step()

            with torch.no_grad():
                g_hole = E42.assemble(gh)
                g_full = E22.merge_gaussians(g_vis, g_hole)
                r_final = E42.render_at(
                    g_full,
                    {
                        "w2c": pair["w2c_t"],
                        "fx": pair["fx"],
                        "fy": pair["fy"],
                        "cx": pair["cx"],
                        "cy": pair["cy"],
                    },
                )
                g_naive = E22.merge_gaussians(
                    g_vis,
                    E42.assemble(
                        E42.init_hole_gaussians(
                            col_fill.permute(1, 2, 0), dep_flat, hole_mask, pair
                        )
                    ),
                )
                r_naive = E42.render_at(
                    g_naive,
                    {
                        "w2c": pair["w2c_t"],
                        "fx": pair["fx"],
                        "fy": pair["fy"],
                        "cx": pair["cx"],
                        "cy": pair["cy"],
                    },
                )

            name = f"teacher_{kept:04d}_cand{si:04d}_h{hole_frac:.3f}"
            label = {
                "scene_candidate": si,
                "src_img": s["img"],
                "tgt_img": t["img"],
                "src_index": int(i),
                "tgt_index": int(j),
                "aux_indices": [int(x) for x in aux_idx],
                "hole_frac": hole_frac,
                "psnr_teacher": masked_psnr(r_final, pair["gt"], hole_mask),
                "psnr_naive": masked_psnr(r_naive, pair["gt"], hole_mask),
                "psnr_base": masked_psnr(r_vis, pair["gt"], hole_mask),
                "num_hole_gaussians": int(gh["xyz"].shape[0]),
                "label_path": str(out / "labels" / f"{name}.pt"),
            }
            torch.save(
                {
                    "raw": {k: v.detach().cpu() for k, v in gh.items()},
                    "assembled": {k: v.detach().cpu() for k, v in g_hole.items()},
                    "hole_mask": hole_mask.detach().cpu(),
                    "w2c_t": pair["w2c_t"],
                    "intrinsics_t": {
                        "fx": pair["fx"],
                        "fy": pair["fy"],
                        "cx": pair["cx"],
                        "cy": pair["cy"],
                    },
                    "meta": label,
                },
                out / "labels" / f"{name}.pt",
            )
            save_img(r_final, out / "teacher" / f"{name}.png")
            save_img(col_fill, out / "anchor" / f"{name}.png")
            save_img(r_vis, out / "base" / f"{name}.png")
            save_img(pair["gt"], out / "gt" / f"{name}.png")
            Image.fromarray((hole_mask.cpu().numpy().astype(np.uint8) * 255)).save(
                out / "mask" / f"{name}.png"
            )
            strip = np.concatenate(
                [
                    E42.to_np(r_vis),
                    E42.to_np(col_fill),
                    E42.to_np(r_naive),
                    E42.to_np(r_final),
                    E42.to_np(pair["gt"]),
                ],
                axis=1,
            )
            Image.fromarray(strip).save(out / "vis" / f"{name}.jpg")

            manifest.append(label)
            kept += 1
            print(
                f"SAVE {name}: hole={hole_frac:.3f} psnr base={label['psnr_base']:.2f} naive={label['psnr_naive']:.2f} teacher={label['psnr_teacher']:.2f} nG={label['num_hole_gaussians']}",
                flush=True,
            )
        except Exception as e:
            print(f"cand{si}: ERROR {type(e).__name__}: {e}", flush=True)
            continue

    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"DONE kept={kept}/{args.n_candidates} -> {out}", flush=True)


if __name__ == "__main__":
    main()
