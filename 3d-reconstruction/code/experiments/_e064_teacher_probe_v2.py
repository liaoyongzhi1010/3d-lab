"""Teacher-quality audit v2: MEDIUM holes (0.05-0.20) + DEPTH-CONDITIONED inpaint.

Why v2 (lesson from user seeing images): wide700 h=0.25-0.37 holes are a PSEUDO
benchmark -- nobody can reconstruct 25-37% of a frame to match GT, so "winning" there
is meaningless. The MEANINGFUL regime is medium holes where visible context strongly
constrains the fill. Also: plain SD-inpaint hallucinates outdoors (wood pillar). Test if
ControlNet-DEPTH conditioning curbs that by forcing geometric consistency.

For medium-hole scenes: take Flash3D render + hole mask + Flash3D depth (EDT-filled in
hole), run BOTH plain SD-inpaint and depth-conditioned ControlNet-inpaint. Save 4-panel:
[GT | Flash3D(hole gray) | SD-inpaint | Depth-SD-inpaint]. Eyeball: sharpness, GT-fidelity,
hallucination. Distillation ceiling = teacher quality, so audit BEFORE building pipeline.

Run (flash3d venv, CUDA 11.8): python _e064_teacher_probe_v2.py
"""

import os, sys

sys.path.insert(0, "/root/projects/flash3d")
os.chdir("/root/projects/flash3d")
import numpy as np
import torch
from PIL import Image
import scipy.ndimage as ndi
import hydra
from omegaconf import DictConfig
from models.model import GaussianPredictor, to_device
from datasets.util import create_datasets
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
)

DEVICE = "cuda:0"
OUT = "/home/data/E-064_teacher_probe_v2"
os.makedirs(OUT, exist_ok=True)

HOLE_LO, HOLE_HI = 0.05, 0.20  # medium regime
N_WANT = 8


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


def depth_to_control(depth_hw, hole_np):
    """Flash3D metric depth -> MiDaS-style disparity control image (near=bright).
    Fill hole via EDT-nearest so control is defined everywhere."""
    d = depth_hw.detach().cpu().numpy().astype(np.float32)
    valid = ~(hole_np > 0.5)
    if valid.any():
        _, (iy, ix) = ndi.distance_transform_edt(
            ~valid, return_distances=True, return_indices=True
        )
        d = d[iy, ix]
    disp = 1.0 / np.clip(d, 1e-3, None)
    lo, hi = np.percentile(disp, 2), np.percentile(disp, 98)
    disp = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
    img = (disp * 255).astype(np.uint8)
    return Image.fromarray(np.stack([img, img, img], -1))


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cfg.data_loader.batch_size = 1
    cfg.data_loader.num_workers = 4
    model = GaussianPredictor(cfg).to(DEVICE)
    model.load_model(
        "/root/projects/flash3d/checkpoints/model_re10k_v2.pth", ckpt_ids=0
    )
    model.set_eval()

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    cnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
    )
    dpipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=cnet,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    dpipe.set_progress_bar_config(disable=True)

    _, dl = create_datasets(cfg, split="test")
    PROMPT = "a photo, realistic, high detail, consistent scene"
    done = 0
    for inputs in dl:
        if done >= N_WANT:
            break
        to_device(inputs, DEVICE)
        inputs["target_frame_ids"] = [1, 2, 3]
        with torch.no_grad():
            out = model(inputs)
        # scan tgt frames for a MEDIUM hole
        for tgt in (1, 2, 3):
            with torch.no_grad():
                pred = out[("color_gauss", tgt, 0)][0].clamp(0, 1)
                gt = inputs[("color", tgt, 0)][0].clamp(0, 1)
                alpha = out[("alpha_gauss", tgt, 0)][0, 0]
                depth_tgt = out[("depth_gauss", tgt, 0)][0, 0]
            invis = clean_mask((alpha < 0.5).float().cpu().numpy())
            if HOLE_LO <= invis.mean() <= HOLE_HI:
                break
        else:
            continue

        H, W = pred.shape[1], pred.shape[2]
        img = (pred.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        msk = (invis * 255).astype(np.uint8)
        img512 = Image.fromarray(img).resize((512, 512))
        msk512 = Image.fromarray(msk).resize((512, 512))
        ctrl512 = depth_to_control(depth_tgt, invis).resize((512, 512))

        g = torch.Generator(device=DEVICE).manual_seed(0)
        res_sd = (
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

        g = torch.Generator(device=DEVICE).manual_seed(0)
        res_dep = (
            dpipe(
                prompt=PROMPT,
                image=img512,
                mask_image=msk512,
                control_image=ctrl512,
                num_inference_steps=30,
                guidance_scale=7.5,
                controlnet_conditioning_scale=1.0,
                generator=g,
            )
            .images[0]
            .resize((W, H))
        )

        gtimg = (gt.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        predimg = img.copy()
        predimg[invis > 0.5] = 128
        trip = np.concatenate(
            [gtimg, predimg, np.array(res_sd), np.array(res_dep)], axis=1
        )
        Image.fromarray(trip).save(
            f"{OUT}/probe_{done:02d}_t{tgt}_h{invis.mean():.3f}.png"
        )
        print(f"saved probe {done} tgt={tgt} hole={invis.mean():.3f}", flush=True)
        done += 1
    print(
        f"DONE ({done}) -> {OUT}  [GT|Flash3D|SD-inpaint|Depth-SD-inpaint]", flush=True
    )


if __name__ == "__main__":
    main()
