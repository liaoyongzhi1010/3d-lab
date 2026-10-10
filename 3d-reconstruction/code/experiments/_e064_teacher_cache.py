"""E-064 teacher cache: generate SHARP hole-fill targets via plain SD-inpaint (NO
ControlNet-depth -- probe v2 proved depth-cond hallucinates pillars) on MEDIUM holes
(0.05-0.20, the meaningful regime), with K-seed CONSISTENCY FILTERING to drop ambiguous/
hallucination-prone samples.

WHY (E-064 design, root-caused):
  - E-063 mean-only blurs because it regresses hole colour toward GT_tgt, which is TRULY
    occluded / unpredictable from the input view -> hedging -> blur (E-005 single-GT trap).
  - SD-inpaint fill is DERIVED FROM VISIBLE CONTEXT -> it IS a learnable function of the
    input -> a student CAN regress to it -> sharp (converts ill-posed -> well-posed, the
    E-022 insight but with a 2D generative teacher instead of a 3D-GT teacher).
  - Teacher != GT, so downstream we report FID/LPIPS/visual wins + honest PSNR tie/loss.

CONFIDENCE FILTER: run SD-inpaint K times with different seeds; compute per-pixel stddev
across the K fills INSIDE the hole. Low std = confident/context-determined (keep, use the
median as the consensus target). High std = ambiguous / hallucination-prone (drop). This
directly attacks probe_07's "bare wall -> invented painting" failure.

Output per kept sample (leak-safe TRAIN scenes):
  <OUT>/<key>.npz : teacher_rgb(H,W,3 uint8, hole filled w/ consensus), hole(H,W uint8),
                    base_rgb(H,W,3), tgt frame id, scene, hole_frac, hole_std
  <OUT>/vis/<key>.png : [GT | Flash3D(gray hole) | teacher-consensus] for eyeballing

Run (cd /root/projects/flash3d && source .venv/bin/activate, CUDA 11.8):
  python _e064_teacher_cache.py +experiment=layered_re10k +dataset.crop_border=true \
    dataset.data_path=data/RealEstate10K model.depth.version=v1 \
    ++tc.out=/home/data/E-064_teacher_cache ++tc.n_want=1200
"""

import os, sys, json
from pathlib import Path

sys.path.insert(0, "/root/projects/flash3d")
os.chdir("/root/projects/flash3d")

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
import scipy.ndimage as ndi
from PIL import Image
from models.model import GaussianPredictor, to_device
from datasets.util import create_datasets
from diffusers import StableDiffusionInpaintPipeline

DEVICE = "cuda:0"
HOLE_LO, HOLE_HI = 0.05, 0.20
K_SEED = 3
STD_KEEP = 0.10  # keep if in-hole cross-seed stddev (0..1 rgb) below this


def clean_mask(m, open_i=1, close_i=3, min_area=0.01):
    m = ndi.binary_opening(m > 0.5, iterations=open_i)
    m = ndi.binary_closing(m, iterations=close_i)
    lbl, n = ndi.label(m)
    if n == 0:
        return m.astype(np.float32)
    for i in range(1, n + 1):
        if (lbl == i).mean() < min_area:
            m[lbl == i] = 0
    return m.astype(np.float32)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    tc = cfg.get("tc", {})
    out = Path(tc.get("out", "/home/data/E-064_teacher_cache"))
    (out / "vis").mkdir(parents=True, exist_ok=True)
    n_want = int(tc.get("n_want", 1200))
    max_scan = int(tc.get("max_scan", 20000))

    cfg.data_loader.batch_size = 1
    cfg.data_loader.num_workers = int(tc.get("workers", 4))
    print("[1] Flash3D frozen backbone...", flush=True)
    model = GaussianPredictor(cfg).to(DEVICE)
    model.load_model(
        "/root/projects/flash3d/checkpoints/model_re10k_v2.pth", ckpt_ids=0
    )
    model.set_eval()

    print("[2] SD-inpaint teacher (no controlnet)...", flush=True)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None  # NSFW filter returns black imgs -> corrupts cache

    _, dl = create_datasets(cfg, split="train")
    novel = list(cfg.model.gauss_novel_frames)
    PROMPT = "a photo, realistic, high detail, consistent scene"

    kept = 0
    scanned = 0
    dropped_std = 0
    manifest = []
    for inputs in dl:
        if kept >= n_want or scanned >= max_scan:
            break
        scanned += 1
        try:
            frame_tag = inputs[("frame_id", 0)][0]
            scene = frame_tag.split("+")[1] if "+" in frame_tag else f"{scanned:06d}"
        except Exception:
            scene = f"{scanned:06d}"
        to_device(inputs, DEVICE)
        inputs["target_frame_ids"] = novel
        with torch.no_grad():
            try:
                out_m = model(inputs)
            except Exception as e:
                continue

            # largest-parallax neighbor = target view whose hole we fill
            def cam_dist(fid):
                T = out_m[("cam_T_cam", 0, fid)][0].float()
                return torch.norm(T[:3, 3]).item()

            fids = sorted(novel, key=cam_dist, reverse=True)
            tgt = fids[0]
            pred = out_m[("color_gauss", tgt, 0)][0].clamp(0, 1)
            gt = inputs[("color", tgt, 0)][0].clamp(0, 1)
            alpha = out_m[("alpha_gauss", tgt, 0)][0, 0]
            depth_tgt = out_m[("depth_gauss", tgt, 0)][0, 0]  # Flash3D depth @ tgt
            K_tgt = inputs[("K_tgt", tgt)][0]  # 3x3 intrinsics @ tgt
        invis = clean_mask((alpha < 0.5).float().cpu().numpy())
        if not (HOLE_LO <= invis.mean() <= HOLE_HI):
            continue

        H, W = pred.shape[1], pred.shape[2]
        img = (pred.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        msk = (invis * 255).astype(np.uint8)
        img512 = Image.fromarray(img).resize((512, 512))
        msk512 = Image.fromarray(msk).resize((512, 512))

        fills = []
        for k in range(K_SEED):
            g = torch.Generator(device=DEVICE).manual_seed(1000 + k)
            r = (
                pipe(
                    prompt=PROMPT,
                    image=img512,
                    mask_image=msk512,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    generator=g,
                )
                .images[0]
                .resize((W, H))
            )
            fills.append(np.asarray(r).astype(np.float32) / 255.0)
        fills = np.stack(fills, 0)  # (K,H,W,3)

        # confidence: cross-seed stddev INSIDE the hole
        hole_b = invis > 0.5
        if hole_b.sum() < 16:
            continue
        in_hole_std = float(fills.std(0)[hole_b].mean())
        consensus = np.median(fills, 0)  # robust to a single hallucinated seed

        # composite: keep Flash3D visible pixels, put teacher consensus in the hole
        base01 = img.astype(np.float32) / 255.0
        teacher = base01.copy()
        teacher[hole_b] = consensus[hole_b]
        teacher_u8 = (teacher * 255).clip(0, 255).astype(np.uint8)

        keep = in_hole_std <= STD_KEEP
        gtimg = (gt.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        predimg = img.copy()
        predimg[hole_b] = 128
        tag = "KEEP" if keep else "DROP"
        trip = np.concatenate([gtimg, predimg, teacher_u8], axis=1)
        key = f"{scene}_t{tgt}_h{invis.mean():.3f}_s{in_hole_std:.3f}_{tag}"
        Image.fromarray(trip).save(out / "vis" / f"{key}.png")

        if not keep:
            dropped_std += 1
            continue

        np.savez_compressed(
            out / f"{key}.npz",
            teacher_rgb=teacher_u8,
            hole=(hole_b.astype(np.uint8)),
            base_rgb=img,
            gt_rgb=gtimg,
            depth_tgt=depth_tgt.cpu().numpy().astype(np.float32),
            K_tgt=K_tgt.cpu().numpy().astype(np.float32),
            tgt=int(tgt),
            scene=scene,
            hole_frac=float(invis.mean()),
            hole_std=in_hole_std,
        )
        manifest.append(
            {
                "key": key,
                "scene": scene,
                "tgt": int(tgt),
                "hole_frac": float(invis.mean()),
                "hole_std": in_hole_std,
            }
        )
        kept += 1
        if kept % 25 == 0:
            print(
                f"  kept={kept}/{n_want} scanned={scanned} dropped_std={dropped_std} "
                f"(last std={in_hole_std:.3f} hole={invis.mean():.3f})",
                flush=True,
            )

    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"DONE: kept={kept} scanned={scanned} dropped_std={dropped_std} -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
