"""DPC with perceptual (LPIPS) novel-view destruction for visually striking results.

The standard L2 DPC produces distributed pixel shifts that drop PSNR but look subtle.
This variant maximizes LPIPS (perceptual distance) on novel views, which concentrates
damage on edges, textures, and structures — producing visible ghosting, warping, and
artifacts that are immediately obvious to the human eye.

Run on server (Flash3D venv):
  python /root/flash3d-attack/attacks/make_dpc_perceptual.py --scenes 7 12 3 \
    --out_dir /root/flash3d-attack/results/qual_perceptual
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hydra import compose, initialize_config_dir
from models.model import GaussianPredictor
from datasets.util import create_datasets

from attacks.dpc_attack import sample_auxiliary_poses, DPCConfig

FLASH3D_CKPT = "/root/projects/flash3d/checkpoints/model_re10k_v2.pth"


def _load(cfg, device):
    model = GaussianPredictor(cfg).to(device)
    state = torch.load(FLASH3D_CKPT, map_location="cpu")
    sd = state["model"] if "model" in state else state
    current = model.state_dict()
    filtered = {}
    for k, v in sd.items():
        if "backproject_depth" in k:
            if k in current:
                filtered[k] = current[k].clone()
        else:
            filtered[k] = v
    model.load_state_dict(filtered, strict=False)
    model.set_eval()
    return model


def _make_cfg(split):
    with initialize_config_dir(
        config_dir="/root/projects/flash3d/configs", version_base=None
    ):
        return compose(
            config_name="config",
            overrides=[
                "+experiment=layered_re10k",
                "+dataset.crop_border=true",
                f"dataset.test_split_path={split}",
                "model.depth.version=v1",
                "data_loader.batch_size=1",
                "data_loader.num_workers=1",
            ],
        )


def _render_single(model, inputs, src_image, relative_pose):
    local = dict(inputs)
    local[("color_aug", 0, 0)] = src_image
    local["target_frame_ids"] = [1]
    outputs = model.models["unidepth_extended"](local)
    model.compute_gauss_means(local, outputs)
    B = src_image.shape[0]
    device = src_image.device
    dtype = outputs["gauss_means"].dtype
    outputs[("cam_T_cam", 0, 1)] = (
        relative_pose.to(device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    )
    model.render_images(local, outputs)
    return outputs[("color_gauss", 1, 0)]


class LPIPSLoss(torch.nn.Module):
    """Lightweight LPIPS using VGG features (no external lpips package needed)."""

    def __init__(self, device):
        super().__init__()
        import torchvision.models as models

        vgg = models.vgg16(pretrained=True).features.to(device).eval()
        self.slices = torch.nn.ModuleList(
            [
                vgg[:4],  # relu1_2
                vgg[4:9],  # relu2_2
                vgg[9:16],  # relu3_3
                vgg[16:23],  # relu4_3
            ]
        )
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, x, y):
        x = self._normalize(x)
        y = self._normalize(y)
        loss = 0.0
        for s in self.slices:
            x = s(x)
            y = s(y)
            loss = loss + F.l1_loss(x, y)
        return loss


def dpc_perceptual_attack(
    clean_src, render_fn, lpips_fn, epsilon, steps, lambda_src, n_aux_poses, device
):
    """DPC attack with LPIPS-based novel destruction (visually striking)."""
    cfg = DPCConfig(epsilon=epsilon, n_aux_poses=n_aux_poses)
    aux_poses = sample_auxiliary_poses(cfg, device=device)

    # Get clean renders (targets)
    with torch.no_grad():
        clean_src_render = render_fn(clean_src, torch.eye(4, device=device))
        clean_novel_renders = [render_fn(clean_src, p) for p in aux_poses]

    # Random init in L-inf ball
    delta = (torch.rand_like(clean_src) * 2 - 1) * epsilon
    delta = delta.clone().detach().requires_grad_(True)

    momentum = torch.zeros_like(clean_src)

    for step in range(steps):
        adv = (clean_src + delta).clamp(0, 1)

        # Source camouflage: L2 (keep source looking the same)
        src_render = render_fn(adv, torch.eye(4, device=device))
        loss_src = F.mse_loss(src_render, clean_src_render)

        # Novel destruction: maximize LPIPS (perceptual distance)
        loss_novel = 0.0
        for i, pose in enumerate(aux_poses):
            novel_render = render_fn(adv, pose)
            # Maximize perceptual distance from clean
            loss_novel = loss_novel - lpips_fn(novel_render, clean_novel_renders[i])
        loss_novel = loss_novel / len(aux_poses)

        loss = lambda_src * loss_src + loss_novel

        loss.backward()

        with torch.no_grad():
            grad = delta.grad.clone()
            momentum = 0.9 * momentum + grad / (grad.abs().mean() + 1e-8)
            delta.data -= (epsilon / steps * 2) * momentum.sign()
            delta.data.clamp_(-epsilon, epsilon)
            delta.data = (clean_src + delta.data).clamp(0, 1) - clean_src
            delta.grad.zero_()

    attacked = (clean_src + delta.detach()).clamp(0, 1)
    return attacked


def _to_img(t):
    if t.dim() == 4:
        t = t[0]
    return (t.detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(
        np.uint8
    )


def _psnr(a, b):
    mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean() / (255.0**2)
    if mse < 1e-10:
        return 99.0
    return -10.0 * np.log10(mse)


def _add_label(img, text):
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([(0, 0), (im.width, 26)], fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255), font=font)
    return np.array(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split", default="splits/re10k_mine_filtered/test_files_present.txt"
    )
    ap.add_argument("--scenes", type=int, nargs="+", default=[7, 12, 3])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epsilon", type=float, default=4.0 / 255.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lambda_src", type=float, default=20.0)
    ap.add_argument("--n_aux_poses", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0")
    cfg = _make_cfg(args.split)
    model = _load(cfg, device)
    _, loader = create_datasets(cfg, split="test")
    lpips_fn = LPIPSLoss(device).eval()

    scene_set = set(args.scenes)
    max_scene = max(args.scenes)
    idx = -1
    for inputs in loader:
        idx += 1
        if idx > max_scene:
            break
        if idx not in scene_set:
            continue
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        if ("color", 1, 0) not in inputs:
            continue
        inputs["target_frame_ids"] = [1, 2, 3]

        clean_src = inputs[("color_aug", 0, 0)].detach()

        def render_fn(image, pose):
            return _render_single(model, inputs, image, pose)

        print(
            f"scene {idx}: attacking (eps={args.epsilon:.3f}, {args.steps} steps, LPIPS)...",
            flush=True,
        )
        attacked_src = dpc_perceptual_attack(
            clean_src,
            render_fn,
            lpips_fn,
            epsilon=args.epsilon,
            steps=args.steps,
            lambda_src=args.lambda_src,
            n_aux_poses=args.n_aux_poses,
            device=device,
        )

        # Full forward for both: render all targets
        with torch.no_grad():
            clean_inputs = dict(inputs)
            clean_inputs[("color_aug", 0, 0)] = clean_src
            clean_inputs["target_frame_ids"] = [1, 2, 3]
            out_c = model(clean_inputs)

            att_inputs = dict(inputs)
            att_inputs[("color_aug", 0, 0)] = attacked_src
            att_inputs["target_frame_ids"] = [1, 2, 3]
            out_a = model(att_inputs)

        # Pick widest target (usually most damage)
        fid_names = {1: "tgt5", 2: "tgt10", 3: "tgt_rand"}
        best_fid = None
        best_diff = -1
        for fid in [1, 2, 3]:
            pk = ("color_gauss", fid, 0)
            if pk in out_c and pk in out_a:
                rc = _to_img(out_c[pk])
                ra = _to_img(out_a[pk])
                p = _psnr(rc, ra)
                if best_fid is None or p < best_diff or best_diff < 0:
                    best_fid = fid
                    best_diff = p

        # Generate comparison figure: 2 rows x 3 cols
        # Row 1: Source clean | Novel clean | Novel clean (another)
        # Row 2: Source attacked | Novel attacked | Novel attacked (another)
        src_c = _to_img(out_c[("color_gauss", 0, 0)])
        src_a = _to_img(out_a[("color_gauss", 0, 0)])
        src_psnr_hdr = _psnr(src_c, src_a)

        fids_show = [1, 2, 3]
        row1_imgs = [_add_label(src_c, "Source (clean)")]
        row2_imgs = [_add_label(src_a, f"Source (attacked) {src_psnr_hdr:.1f}dB")]
        for fid in fids_show:
            pk = ("color_gauss", fid, 0)
            if pk in out_c:
                rc = _to_img(out_c[pk])
                ra = _to_img(out_a[pk])
                p = _psnr(rc, ra)
                row1_imgs.append(_add_label(rc, f"{fid_names[fid]} (clean)"))
                row2_imgs.append(
                    _add_label(ra, f"{fid_names[fid]} (attacked) {p:.1f}dB")
                )

        # Make same size
        H = min(im.shape[0] for im in row1_imgs + row2_imgs)
        W = min(im.shape[1] for im in row1_imgs + row2_imgs)
        row1 = np.concatenate([im[:H, :W] for im in row1_imgs], axis=1)
        row2 = np.concatenate([im[:H, :W] for im in row2_imgs], axis=1)
        grid = np.concatenate([row1, row2], axis=0)

        out_path = os.path.join(args.out_dir, f"scene{idx}_perceptual.png")
        Image.fromarray(grid).save(out_path, quality=95)
        src_psnr = _psnr(src_c[:H, :W], src_a[:H, :W])
        print(f"  -> saved {out_path} | src_psnr={src_psnr:.1f}", flush=True)


if __name__ == "__main__":
    main()
