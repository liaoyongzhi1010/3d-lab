"""Generate improved qualitative DPC figures with strong visual impact.

Improvements over v1:
  - eps=16/255 by default (much more visible degradation: novel -9.2 dB)
  - Diff row uses JET heatmap (not raw RGB diff) with normalized colorbar
  - Adds a zoom-in crop panel of the most-damaged region
  - Higher diff amplification factor
  - Larger output resolution

Layout per scene: 4 rows x 4 columns
  row 1: clean renders        [ src | tgt5 | tgt10 | tgt_rand ]
  row 2: DPC-attacked renders [ src | tgt5 | tgt10 | tgt_rand ]
  row 3: heatmap diff (jet)   [ src | tgt5 | tgt10 | tgt_rand ]
  row 4: zoom-in crop of most-damaged target (clean vs attacked side-by-side x2)

Run on server (Flash3D venv):
  python /root/flash3d-attack/attacks/make_dpc_qualitative_v2.py --scenes 0 3 7 12 20 \
    --out_dir /root/flash3d-attack/results/qual_v2
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hydra import compose, initialize_config_dir
from models.model import GaussianPredictor
from datasets.util import create_datasets

from attacks.dpc_attack import DPCConfig, dpc_attack

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


def _to_img(t):
    t = t.detach().clamp(0, 1)[0].cpu().numpy()
    return (t.transpose(1, 2, 0) * 255).astype(np.uint8)


def _jet_heatmap(diff_rgb, vmax=None):
    """Convert absolute RGB diff to a JET-colored heatmap (more visually striking).
    diff_rgb: (H,W,3) uint8. Returns (H,W,3) uint8 jet-colored."""
    gray = diff_rgb.astype(np.float32).mean(axis=2)
    if vmax is None:
        vmax = max(gray.max(), 1.0)
    norm = np.clip(gray / vmax, 0, 1)
    # manual jet colormap (avoid matplotlib dependency)
    r = np.clip(1.5 - np.abs(norm * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(norm * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(norm * 4 - 1), 0, 1)
    jet = np.stack([r, g, b], axis=2)
    # make zero-diff black (not blue) for clarity
    mask = gray < 2.0
    jet[mask] = 0
    return (jet * 255).astype(np.uint8)


def _find_most_damaged_crop(clean_img, att_img, crop_size=128):
    """Find the crop_size x crop_size region with the largest mean absolute diff."""
    diff = np.abs(clean_img.astype(np.float32) - att_img.astype(np.float32)).mean(
        axis=2
    )
    H, W = diff.shape
    cs = min(crop_size, H, W)
    # use a stride for speed
    stride = max(cs // 4, 1)
    best_val = -1
    best_y, best_x = 0, 0
    for y in range(0, H - cs + 1, stride):
        for x in range(0, W - cs + 1, stride):
            val = diff[y : y + cs, x : x + cs].mean()
            if val > best_val:
                best_val = val
                best_y, best_x = y, x
    return best_y, best_x, cs


def _add_red_border(img, width=3):
    """Add a red border to an image."""
    out = img.copy()
    out[:width, :] = [255, 0, 0]
    out[-width:, :] = [255, 0, 0]
    out[:, :width] = [255, 0, 0]
    out[:, -width:] = [255, 0, 0]
    return out


def _psnr_val(clean, att):
    diff = (clean.astype(np.float32) / 255.0 - att.astype(np.float32) / 255.0) ** 2
    mse = diff.mean()
    if mse < 1e-10:
        return 99.0
    return -10.0 * np.log10(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split", default="splits/re10k_mine_filtered/test_files_present.txt"
    )
    ap.add_argument("--scenes", type=int, nargs="+", default=[0, 3, 7, 12, 20])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epsilon", type=float, default=16.0 / 255.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lambda_src", type=float, default=3.0)
    ap.add_argument("--crop_size", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0")
    cfg = _make_cfg(args.split)
    model = _load(cfg, device)
    _, loader = create_datasets(cfg, split="test")
    dpc_cfg = DPCConfig(
        epsilon=args.epsilon, steps=args.steps, lambda_src=args.lambda_src
    )

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

        attacked_src, delta, info = dpc_attack(clean_src, render_fn, dpc_cfg)

        panels = {"clean": [], "attacked": []}
        for tag, src_img in [("clean", clean_src), ("attacked", attacked_src)]:
            eval_inputs = {
                kk: (vv.clone() if isinstance(vv, torch.Tensor) else vv)
                for kk, vv in inputs.items()
            }
            eval_inputs[("color_aug", 0, 0)] = src_img.detach()
            eval_inputs["target_frame_ids"] = [1, 2, 3]
            with torch.no_grad():
                out = model(eval_inputs)
            row = [_to_img(out[("color_gauss", 0, 0)])]
            for fid in [1, 2, 3]:
                row.append(_to_img(out[("color_gauss", fid, 0)]))
            panels[tag] = row

        # Compute per-view PSNR for annotation
        view_names = ["src", "tgt5", "tgt10", "tgt_rand"]
        psnrs = []
        for c, a in zip(panels["clean"], panels["attacked"]):
            psnrs.append(_psnr_val(c, a))

        # Heatmap diff row (normalize across all views for fair comparison)
        diffs_raw = []
        for c, a in zip(panels["clean"], panels["attacked"]):
            diffs_raw.append(
                np.abs(c.astype(np.int16) - a.astype(np.int16)).astype(np.uint8)
            )
        # Use a shared vmax across all views for the same scene
        global_max = max(d.astype(np.float32).mean(axis=2).max() for d in diffs_raw)
        vmax = max(global_max * 0.8, 10.0)  # slight headroom
        heatmaps = [_jet_heatmap(d, vmax=vmax) for d in diffs_raw]

        # Find zoom-in crop on the most-damaged novel view (tgt_rand = idx 3)
        worst_idx = 3  # tgt_rand typically has max parallax
        y, x, cs = _find_most_damaged_crop(
            panels["clean"][worst_idx], panels["attacked"][worst_idx], args.crop_size
        )
        crop_clean = panels["clean"][worst_idx][y : y + cs, x : x + cs]
        crop_att = panels["attacked"][worst_idx][y : y + cs, x : x + cs]
        crop_heat = heatmaps[worst_idx][y : y + cs, x : x + cs]
        # Scale crops up for visibility
        scale = 2
        crop_clean_big = np.array(
            Image.fromarray(crop_clean).resize((cs * scale, cs * scale), Image.LANCZOS)
        )
        crop_att_big = np.array(
            Image.fromarray(_add_red_border(crop_att)).resize(
                (cs * scale, cs * scale), Image.LANCZOS
            )
        )
        crop_heat_big = np.array(
            Image.fromarray(crop_heat).resize((cs * scale, cs * scale), Image.NEAREST)
        )

        # Assemble grid
        # Rows 1-3: clean / attacked / heatmap (4 cols)
        h = min(
            p.shape[0]
            for row in [panels["clean"], panels["attacked"], heatmaps]
            for p in row
        )
        w = min(
            p.shape[1]
            for row in [panels["clean"], panels["attacked"], heatmaps]
            for p in row
        )
        row1 = np.concatenate([p[:h, :w] for p in panels["clean"]], axis=1)
        row2 = np.concatenate([p[:h, :w] for p in panels["attacked"]], axis=1)
        row3 = np.concatenate([p[:h, :w] for p in heatmaps], axis=1)

        # Row 4: zoom-in crops (clean | attacked | heatmap) centered in same width
        crop_row = np.concatenate([crop_clean_big, crop_att_big, crop_heat_big], axis=1)
        # Pad crop_row to match grid width
        grid_w = row1.shape[1]
        crop_w = crop_row.shape[1]
        if crop_w < grid_w:
            pad_left = (grid_w - crop_w) // 2
            pad_right = grid_w - crop_w - pad_left
            crop_row = np.pad(
                crop_row, ((0, 0), (pad_left, pad_right), (0, 0)), constant_values=0
            )
        else:
            crop_row = crop_row[:, :grid_w]

        grid = np.concatenate([row1, row2, row3, crop_row], axis=0)

        out_path = os.path.join(args.out_dir, f"scene{idx}_dpc_v2.png")
        Image.fromarray(grid).save(out_path, quality=95)
        print(
            f"scene {idx}: saved {out_path} | "
            f"PSNR(clean vs att): src={psnrs[0]:.1f} tgt5={psnrs[1]:.1f} "
            f"tgt10={psnrs[2]:.1f} trand={psnrs[3]:.1f} | linf={info['linf']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
