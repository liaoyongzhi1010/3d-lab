"""Teacher-quality audit: does SD-inpainting produce SHARP, PLAUSIBLE hole content on
our large-parallax disocclusion cases? Distillation ceiling = teacher quality, so audit
BEFORE building a distillation pipeline (lesson: sdfit-teacher was worse than baseline).

For a few wide700 hole scenes: take Flash3D render + hole mask, run SD-inpainting,
save [GT | Flash3D(hole gray) | SD-inpainted] triptych. Eyeball sharpness + correctness.

Run (flash3d venv, CUDA 11.8): python _e064_teacher_probe.py
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
from diffusers import StableDiffusionInpaintPipeline

DEVICE = "cuda:0"
OUT = "/home/data/E-064_teacher_probe"
os.makedirs(OUT, exist_ok=True)


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
    cfg.data_loader.batch_size = 1
    cfg.data_loader.num_workers = 4
    cfg.dataset.test_split_path = "splits/re10k_mine_filtered/test_files_wide700.txt"
    model = GaussianPredictor(cfg).to(DEVICE)
    model.load_model(
        "/root/projects/flash3d/checkpoints/model_re10k_v2.pth", ckpt_ids=0
    )
    model.set_eval()

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    _, dl = create_datasets(cfg, split="test")
    tgt = 3
    done = 0
    for inputs in dl:
        if done >= 6:
            break
        to_device(inputs, DEVICE)
        inputs["target_frame_ids"] = [1, 2, 3]
        with torch.no_grad():
            out = model(inputs)
            pred = out[("color_gauss", tgt, 0)][0].clamp(0, 1)
            gt = inputs[("color", tgt, 0)][0].clamp(0, 1)
            alpha = out[("alpha_gauss", tgt, 0)][0, 0]
        invis = clean_mask((alpha < 0.5).float().cpu().numpy())
        if invis.mean() < 0.08:  # only big holes
            continue
        H, W = pred.shape[1], pred.shape[2]
        # SD wants multiple-of-8, square-ish; resize to 512
        img = (pred.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        msk = (invis * 255).astype(np.uint8)
        img512 = Image.fromarray(img).resize((512, 512))
        msk512 = Image.fromarray(msk).resize((512, 512))
        res = pipe(
            prompt="a photo of an indoor room, realistic, high detail",
            image=img512,
            mask_image=msk512,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images[0]
        res = res.resize((W, H))
        gtimg = (gt.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        predimg = img.copy()
        predimg[invis > 0.5] = 128  # gray the hole for viz
        trip = np.concatenate([gtimg, predimg, np.array(res)], axis=1)
        Image.fromarray(trip).save(f"{OUT}/probe_{done:02d}_h{invis.mean():.3f}.png")
        print(f"saved probe {done} hole={invis.mean():.3f}", flush=True)
        done += 1
    print(f"DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
