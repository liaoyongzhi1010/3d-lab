"""Cross-model DPC: native attack + transfer across feed-forward single-image 3DGS backbones.

Supports Flash3D and CATSplat (both expose the same GaussianPredictor interface: forward(inputs),
compute_gauss_means, render_images, cam_T_cam). Two modes:

  --mode native   : craft DPC on the target model, evaluate on the same model.
  --mode transfer : craft DPC on the SOURCE model, evaluate the crafted image on the TARGET model.

The transfer mode is the key architectural-vulnerability test: if a perturbation optimized against
Flash3D also degrades CATSplat (a different architecture sharing only the single-image + monocular
depth assumption), the vulnerability is architectural, not model-specific.

Run on server (flash3d venv works for both repos):
  cd /root/projects/flash3d && source .venv/bin/activate
  export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  # native on CATSplat
  python /root/flash3d-attack/attacks/eval_dpc_crossmodel.py --mode native \
    --target catsplat --max_scenes 50 --out /root/flash3d-attack/results/cm_native_catsplat.json

  # transfer Flash3D -> CATSplat
  python /root/flash3d-attack/attacks/eval_dpc_crossmodel.py --mode transfer \
    --source flash3d --target catsplat --max_scenes 50 \
    --out /root/flash3d-attack/results/cm_transfer_f3d_to_cat.json
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
from collections import defaultdict

import torch

REPO = {
    "flash3d": {
        "path": "/root/projects/flash3d",
        "ckpt": "/root/projects/flash3d/checkpoints/model_re10k_v2.pth",
        "configs": "/root/projects/flash3d/configs",
    },
    "catsplat": {
        "path": "/root/projects/catsplat",
        "ckpt": "/home/data/sv3d-lab/checkpoints/catsplat/CATSplat.pth",
        "configs": "/root/projects/catsplat/configs",
        "data_path": "/root/projects/catsplat/data/RealEstate10K",
    },
}

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from attacks.dpc_attack import DPCConfig, dpc_attack  # noqa: E402


def _fresh_import(repo_path):
    """Import GaussianPredictor / Evaluator / create_datasets from a specific repo path.

    Both repos use identical module names, so we manipulate sys.path and purge cached modules.
    """
    for mod in list(sys.modules):
        if mod.split(".")[0] in {
            "models",
            "evaluation",
            "datasets",
            "misc",
            "config",
            "hydra",
            "omegaconf",
        }:
            # keep hydra/omegaconf; only purge repo-local modules
            if mod.split(".")[0] in {
                "models",
                "evaluation",
                "datasets",
                "misc",
                "config",
            }:
                del sys.modules[mod]
    while repo_path in sys.path:
        sys.path.remove(repo_path)
    sys.path.insert(0, repo_path)
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    import importlib as _il

    model_mod = _il.import_module("models.model")
    eval_mod = _il.import_module("evaluation.evaluator")
    ds_mod = _il.import_module("datasets.util")
    _il.reload(model_mod)
    _il.reload(eval_mod)
    _il.reload(ds_mod)
    return {
        "GaussianPredictor": model_mod.GaussianPredictor,
        "Evaluator": eval_mod.Evaluator,
        "create_datasets": ds_mod.create_datasets,
        "compose": compose,
        "initialize_config_dir": initialize_config_dir,
        "GlobalHydra": GlobalHydra,
    }


def _load_model(name, api, device):
    cfgs = REPO[name]["configs"]
    data_path_override = REPO[name].get("data_path")
    api["GlobalHydra"].instance().clear()
    with api["initialize_config_dir"](config_dir=cfgs, version_base=None):
        overrides = [
            "+experiment=layered_re10k",
            "+dataset.crop_border=true",
            "dataset.test_split_path=splits/re10k_mine_filtered/test_files_present.txt",
            "model.depth.version=v1",
            "data_loader.batch_size=1",
            "data_loader.num_workers=1",
        ]
        if data_path_override:
            overrides.append(f"dataset.data_path={data_path_override}")
        cfg = api["compose"](config_name="config", overrides=overrides)
    model = api["GaussianPredictor"]
    # CATSplat loads a pointnet ckpt via a cwd-relative path '../../../pointnet/ckpt/save.pth'
    # (resolves from models/encoder/). chdir there during construction so it is found.
    prev_cwd = os.getcwd()
    ctor_cwd = os.path.join(REPO[name]["path"], "models", "encoder")
    if os.path.isdir(ctor_cwd):
        os.chdir(ctor_cwd)
    try:
        model = model(cfg).to(device)
    finally:
        os.chdir(prev_cwd)
    state = torch.load(REPO[name]["ckpt"], map_location="cpu")
    sd = state["model"] if "model" in state else state
    current = model.state_dict()
    filtered = {}
    for k, v in sd.items():
        if "backproject_depth" in k:
            if k in current:
                filtered[k] = current[k].clone()
        else:
            filtered[k] = v
    res = model.load_state_dict(filtered, strict=False)
    non_ud = [k for k in res.missing_keys if "unidepth" not in k]
    print(
        f"[{name}] loaded; non-unidepth missing={len(non_ud)} unexpected={len(res.unexpected_keys)}"
    )
    model.set_eval()
    return model, cfg


def _render_single(model, inputs, src_image, relative_pose):
    local = dict(inputs)
    local[("color_aug", 0, 0)] = src_image
    local["target_frame_ids"] = [1]
    outputs = model.models["unidepth_extended"](local)
    model.compute_gauss_means(local, outputs)
    B = src_image.shape[0]
    dtype = outputs["gauss_means"].dtype
    outputs[("cam_T_cam", 0, 1)] = (
        relative_pose.to(device=src_image.device, dtype=dtype)
        .unsqueeze(0)
        .repeat(B, 1, 1)
    )
    model.render_images(local, outputs)
    return outputs[("color_gauss", 1, 0)]


def _eval_full(model, inputs, src_img, evaluator, names):
    out_metrics = {}
    eval_inputs = copy.deepcopy(inputs)
    eval_inputs[("color_aug", 0, 0)] = src_img.detach()
    eval_inputs["target_frame_ids"] = [1, 2, 3]
    with torch.no_grad():
        out = model(eval_inputs)
    for fid in [0, 1, 2, 3]:
        pk, gk = ("color_gauss", fid, 0), ("color", fid, 0)
        if pk in out and gk in eval_inputs:
            m = evaluator(out[pk].clamp(0, 1), eval_inputs[gk].clamp(0, 1))
            out_metrics[names[fid]] = {k: float(v) for k, v in m.items()}
    return out_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["native", "transfer"], required=True)
    ap.add_argument("--target", choices=list(REPO), required=True)
    ap.add_argument("--source", choices=list(REPO), default="flash3d")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_scenes", type=int, default=50)
    ap.add_argument("--epsilon", type=float, default=8.0 / 255.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lambda_src", type=float, default=3.0)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    dpc_cfg = DPCConfig(
        epsilon=args.epsilon, steps=args.steps, lambda_src=args.lambda_src
    )
    names = {0: "src", 1: "tgt5", 2: "tgt10", 3: "tgt_rand"}

    # Load target model + its dataloader (defines the scenes/eval).
    tgt_api = _fresh_import(REPO[args.target]["path"])
    target_model, tgt_cfg = _load_model(args.target, tgt_api, device)
    evaluator = tgt_api["Evaluator"](crop_border=tgt_cfg.dataset.crop_border).to(device)
    _, loader = tgt_api["create_datasets"](tgt_cfg, split="test")

    # For transfer, load the source (attacker) model too.
    source_model = None
    if args.mode == "transfer" and args.source != args.target:
        src_api = _fresh_import(REPO[args.source]["path"])
        source_model, _ = _load_model(args.source, src_api, device)
        # re-import target modules so the loader/model classes stay consistent
        _fresh_import(REPO[args.target]["path"])

    agg = defaultdict(lambda: defaultdict(list))
    n_done = 0
    for inputs in loader:
        if n_done >= args.max_scenes:
            break
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        if ("color", 1, 0) not in inputs:
            continue
        inputs["target_frame_ids"] = [1, 2, 3]
        clean_src = inputs[("color_aug", 0, 0)].detach()

        attack_model = (
            source_model
            if (args.mode == "transfer" and source_model is not None)
            else target_model
        )

        def render_fn(image, pose):
            return _render_single(attack_model, inputs, image, pose)

        attacked_src, delta, info = dpc_attack(clean_src, render_fn, dpc_cfg)

        clean_m = _eval_full(target_model, inputs, clean_src, evaluator, names)
        att_m = _eval_full(
            target_model, inputs, attacked_src.detach(), evaluator, names
        )
        for view in clean_m:
            for mk in clean_m[view]:
                agg[f"clean_{view}"][mk].append(clean_m[view][mk])
        for view in att_m:
            for mk in att_m[view]:
                agg[f"attacked_{view}"][mk].append(att_m[view][mk])
        n_done += 1
        if n_done % 5 == 0:
            print(
                f"[{args.mode} {args.source}->{args.target}] {n_done} scenes; linf={info['linf']:.4f}",
                flush=True,
            )

    summary = {
        k: {mk: float(sum(v) / len(v)) for mk, v in d.items() if v}
        for k, d in agg.items()
    }
    summary["n_scenes"] = n_done
    summary["config"] = vars(args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
