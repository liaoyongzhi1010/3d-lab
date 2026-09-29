"""Generate qualitative DPC figures: source-view camouflage vs novel-view collapse.

For a few scenes, saves a panel:
  row 1: clean   [ source | novel tgt5 | novel tgt10 | novel tgt_rand ]
  row 2: attacked[ source | novel tgt5 | novel tgt10 | novel tgt_rand ]
  row 3: |diff| amplified

The visual story: row-2 source looks ~identical to row-1 source (camouflage), but row-2 novel
views are visibly broken. Requires the white-box DPC (uses the model renderer).

Run on server (Flash3D venv):
  python /root/flash3d-attack/attacks/make_dpc_qualitative.py --scenes 0 3 7 \
    --out_dir /root/flash3d-attack/results/qual
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split", default="splits/re10k_mine_filtered/test_files_present.txt"
    )
    ap.add_argument("--scenes", type=int, nargs="+", default=[0, 3, 7])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epsilon", type=float, default=8.0 / 255.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lambda_src", type=float, default=3.0)
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

        # Full forward on clean and attacked to render the real eval targets.
        panels = {"clean": [], "attacked": [], "diff": []}
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

        for a, b in zip(panels["clean"], panels["attacked"]):
            d = np.abs(a.astype(np.int16) - b.astype(np.int16))
            d = np.clip(d * 4, 0, 255).astype(np.uint8)
            panels["diff"].append(d)

        rows = []
        for tag in ["clean", "attacked", "diff"]:
            h = min(p.shape[0] for p in panels[tag])
            rows.append(np.concatenate([p[:h] for p in panels[tag]], axis=1))
        h = min(r.shape[0] for r in rows)
        grid = np.concatenate([r[:h] for r in rows], axis=0)
        out_path = os.path.join(args.out_dir, f"scene{idx}_dpc.png")
        Image.fromarray(grid).save(out_path)
        print(
            f"scene {idx}: saved {out_path} "
            f"(cols: src|tgt5|tgt10|tgt_rand; rows: clean|attacked|diffx4) linf={info['linf']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
