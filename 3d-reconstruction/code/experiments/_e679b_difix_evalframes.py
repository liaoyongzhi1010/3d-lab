"""E-679b: Difix-repair the eval-frame flash3d renders from _e679.

For each scene: read baseline_{idx}.png (flash3d render of the eval pose), use the
scene source frame images/0.png as reference, run nvidia/difix_ref, save
fixed_{idx}.png. Same Difix invocation as _e642_step2_difix.py.

Run (difix env, cwd=/root/projects/Difix3D/src, HF_HOME=/home/data/hf_cache):
  python _e679b_difix_evalframes.py --baselines <dir> --source <images/0.png> --out <dir>
"""

import argparse
import glob
import os
import re

from PIL import Image
import torch

EVAL_IDXS = [15, 30, 50, 69, 84, 99]


def frame_index(path):
    match = re.search(r"baseline_(\d+)\.png$", path)
    if not match:
        raise ValueError(f"Unexpected filename: {path}")
    return int(match.group(1))


def match_reference_size(reference, size):
    if reference.size == size:
        return reference
    return reference.resize(size, Image.Resampling.LANCZOS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="nvidia/difix_ref")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from pipeline_difix import DifixPipeline

    pipe = DifixPipeline.from_pretrained(args.model, trust_remote_code=True)
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")
    reference = Image.open(args.source).convert("RGB")

    paths = sorted(glob.glob(os.path.join(args.baselines, "baseline_*.png")))
    n = 0
    for input_path in paths:
        idx = frame_index(input_path)
        image = Image.open(input_path).convert("RGB")
        sized_reference = match_reference_size(reference, image.size)
        with torch.no_grad():
            fixed = pipe(
                "remove degradation",
                image=image,
                ref_image=sized_reference,
                num_inference_steps=1,
                timesteps=[199],
                guidance_scale=0.0,
                height=image.height,
                width=image.width,
            ).images[0]
        output_path = os.path.join(args.out, f"fixed_{idx:03d}.png")
        fixed.save(output_path)
        n += 1
        print(f"[e679b] fixed frame {idx}: {output_path}", flush=True)

    print(f"[e679b] saved {n} fixed frames to {args.out}", flush=True)


if __name__ == "__main__":
    main()
