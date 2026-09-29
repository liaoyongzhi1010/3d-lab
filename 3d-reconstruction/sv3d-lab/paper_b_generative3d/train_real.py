"""Real-data training for Paper B candidates on RE10K with Flash3D backbone.

Usage:
  cd /root/projects/flash3d && source .venv/bin/activate
  export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  cd ~/sv3d-lab
  nohup python -m paper_b_generative3d.train_real \
    --candidate C1 --stage smoke --seed 0 \
    --out /home/data/sv3d-lab/runs/paperb_C1_smoke_s0 \
    > /home/data/sv3d-lab/runs/paperb_C1_smoke_s0.log 2>&1 &

Extends train.py for real data:
  - Loads Flash3D backbone (frozen)
  - Loads RE10K data via data_loader_re10k
  - Runs train_step with real renders + MSE + LPIPS loss
  - Writes manifest/curves/checkpoints/VRAM
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/projects/flash3d")

from paper_b_generative3d.model.candidate_registry import build_candidate
from paper_b_generative3d.model.reconstruction_interface import (
    GaussianScene,
    SourceInput,
    TargetCameras,
)
from paper_b_generative3d.backbone_flash3d import Flash3DBackbone
from paper_b_generative3d.data_loader_re10k import (
    build_re10k_dataloader,
    batch_to_supervision,
    FLASH3D_CKPT,
    QUALITATIVE_IDS,
    RESOLUTION,
)
from paper_b_generative3d.train import save_checkpoint

STAGE_CONFIGS = {
    "smoke": {"steps": 500, "scenes": 8, "batch_size": 2, "lr": 1e-3, "log_every": 50},
    "overfit": {
        "steps": 5000,
        "scenes": 20,
        "batch_size": 2,
        "lr": 5e-4,
        "log_every": 100,
    },
    "pilot": {
        "steps": 30000,
        "scenes": 1000,
        "batch_size": 4,
        "lr": 3e-4,
        "log_every": 500,
    },
    "full": {
        "steps": 100000,
        "scenes": None,
        "batch_size": 4,
        "lr": 2e-4,
        "log_every": 1000,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Paper B real-data training")
    p.add_argument("--candidate", required=True, help="C0-C5")
    p.add_argument("--stage", required=True, choices=list(STAGE_CONFIGS.keys()))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="Output directory (must not exist)")
    p.add_argument("--ckpt", default=FLASH3D_CKPT, help="Flash3D checkpoint path")
    p.add_argument(
        "--flash3d_cfg", default="/root/projects/flash3d/configs/model/re10k.yaml"
    )
    p.add_argument("--resume", default=None, help="Resume from checkpoint path")
    p.add_argument("--lpips_weight", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=None, help="Override learning rate")
    p.add_argument("--steps", type=int, default=None, help="Override step count")
    return p.parse_args()


def build_flash3d_cfg(cfg_path: str):
    from paper_b_generative3d.data_loader_re10k import build_flash3d_cfg as _bcfg

    return _bcfg()


def _batch_item(gauss_out: dict, b: int) -> dict:
    """Slice a batched extract_source_gaussians dict to a single item.

    Keeps a leading batch dim of size 1 so candidate ``build_scene_from_backbone``
    (which reads ``batch_idx=0``) sees exactly one scene. Non-tensor / unrelated
    keys are dropped so no target signal can ride along.
    """

    keys = ("xyz", "scales", "rotations", "opacity", "color_rgb", "source_features")
    item = {}
    for k in keys:
        v = gauss_out.get(k)
        if v is None:
            continue
        item[k] = v[b : b + 1] if torch.is_tensor(v) and v.dim() >= 1 else v
    return item


def get_lpips_fn(device):
    import lpips

    fn = lpips.LPIPS(net="vgg").to(device)
    fn.eval()
    for p in fn.parameters():
        p.requires_grad_(False)
    return fn


def real_train_step(
    candidate,
    backbone: Flash3DBackbone,
    inputs: dict,
    optimizer,
    lpips_fn,
    lpips_weight: float,
    device,
):
    """One training step with real Flash3D backbone + RE10K data.

    Returns loss dict with mse, lpips, total.
    """
    candidate.train()
    optimizer.zero_grad()

    with torch.no_grad():
        gauss_out = backbone.extract_source_gaussians(inputs)

    source = backbone.build_source_input_from_dataloader(inputs)

    B = source.batch
    H, W = RESOLUTION
    bb_outputs = gauss_out.get("_outputs", {})
    target_frame_ids = [fid for fid in [1, 2, 3] if ("color", fid, 0) in inputs]
    if not target_frame_ids:
        target_frame_ids = [1]

    scenes = [
        candidate.build_scene_from_backbone(_batch_item(gauss_out, b), source)
        for b in range(B)
    ]

    renders = []
    targets = []
    for fid in target_frame_ids:
        cam_T_cam = inputs.get(("cam_T_cam", 0, fid))
        if cam_T_cam is None and bb_outputs is not None:
            cam_T_cam = bb_outputs.get(("cam_T_cam", 0, fid))
        if cam_T_cam is None:
            continue
        K_tgt = inputs.get(("K_tgt", fid), inputs[("K_src", 0)])
        tgt_rgb = inputs["color", fid, 0]

        for b in range(B):
            scene = scenes[b]
            gauss_b = {
                "xyz": scene.means,
                "scales": scene.scales,
                "rotations": scene.rotations,
                "opacity": scene.opacity,
                "color_rgb": scene.color,
            }
            rendered = backbone.render_gaussians(
                gauss_b, cam_T_cam[b], K_tgt[b], H, W, batch_idx=0
            )
            renders.append(rendered)
            targets.append(tgt_rgb[b])

    if not renders:
        return {"loss": 0.0, "mse": 0.0, "lpips": 0.0}

    pred = torch.stack(renders)  # (N, 3, H, W)
    gt = torch.stack(targets)  # (N, 3, H, W)

    mse_loss = F.mse_loss(pred, gt)

    lpips_loss = torch.tensor(0.0, device=device)
    if lpips_fn is not None and lpips_weight > 0:
        lpips_loss = lpips_fn(pred * 2 - 1, gt * 2 - 1).mean()

    total = mse_loss + lpips_weight * lpips_loss
    total.backward()
    optimizer.step()

    return {
        "loss": total.item(),
        "mse": mse_loss.item(),
        "lpips": lpips_loss.item(),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out)
    if out_dir.exists():
        if args.resume is None:
            raise FileExistsError(f"Output directory exists: {out_dir}. Use --resume.")
    else:
        out_dir.mkdir(parents=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage_cfg = STAGE_CONFIGS[args.stage]
    lr = args.lr or stage_cfg["lr"]
    total_steps = args.steps or stage_cfg["steps"]
    batch_size = stage_cfg["batch_size"]
    max_scenes = stage_cfg["scenes"]

    config = {
        "candidate": args.candidate,
        "stage": args.stage,
        "seed": args.seed,
        "lr": lr,
        "steps": total_steps,
        "batch_size": batch_size,
        "max_scenes": max_scenes,
        "lpips_weight": args.lpips_weight,
        "ckpt": args.ckpt,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "qualitative_ids.json").write_text(json.dumps(QUALITATIVE_IDS))

    flash3d_cfg = build_flash3d_cfg(args.flash3d_cfg)
    backbone = Flash3DBackbone(flash3d_cfg, args.ckpt, device=str(device))
    backbone.to(device)

    candidate_cfg = {"num_points": 24, "hidden": 8, "seed": args.seed}
    candidate = build_candidate(args.candidate, candidate_cfg)
    candidate.to(device)

    # The per-primitive feature adapter is now built EAGERLY in the candidate
    # __init__ (reconstruction_interface.SOURCE_FEATURE_DIM), so it is present in
    # .parameters() BEFORE the first forward and is therefore captured by the
    # optimizer. Assert the adapter is trainable and in the optimizer set — the
    # old lazy adapter was created on the first forward, AFTER this line, and was
    # silently never optimized (checkpoint showed _feat_adapters.2048.weight
    # essentially untrained).
    trainable = [p for p in candidate.parameters() if p.requires_grad]
    if not trainable:
        print(f"WARNING: candidate {args.candidate} has no trainable parameters!")
    adapter = getattr(candidate, "_feat_adapter", None)
    if adapter is not None:
        adapter_params = list(adapter.parameters())
        adapter_ids = {id(p) for p in adapter_params}
        trainable_ids = {id(p) for p in trainable}
        assert adapter_ids <= trainable_ids, (
            "eager feature adapter params missing from optimizer set — "
            "adapter would never train"
        )
        print(
            f"optimizer: {len(trainable)} trainable tensors "
            f"(feature adapter included: {len(adapter_params)} tensors)"
        )
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

    start_step = 0
    if args.resume:
        from paper_b_generative3d.train import load_checkpoint

        start_step = load_checkpoint(args.resume, candidate, optimizer)
        print(f"Resumed from step {start_step}")

    lpips_fn = get_lpips_fn(device)

    dataloader = build_re10k_dataloader(
        "train",
        batch_size=batch_size,
        num_workers=4,
        seed=args.seed,
        max_scenes=max_scenes,
        flash3d_cfg=flash3d_cfg,
    )

    curves = {"loss": [], "mse": [], "lpips": []}
    vram_peak = 0
    step = start_step
    epoch = 0

    print(f"=== Paper B Real Training: {args.candidate} / {args.stage} ===")
    print(f"Steps: {total_steps}, LR: {lr}, Batch: {batch_size}, Scenes: {max_scenes}")
    print(f"Device: {device}, VRAM limit: 44GB")
    t0 = time.time()

    while step < total_steps:
        epoch += 1
        for batch_inputs in dataloader:
            if step >= total_steps:
                break

            batch_inputs = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch_inputs.items()
            }

            try:
                result = real_train_step(
                    candidate,
                    backbone,
                    batch_inputs,
                    optimizer,
                    lpips_fn,
                    args.lpips_weight,
                    device,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM at step {step}! Clearing cache...")
                    torch.cuda.empty_cache()
                    continue
                raise

            step += 1
            curves["loss"].append(result["loss"])
            curves["mse"].append(result["mse"])
            curves["lpips"].append(result["lpips"])

            if torch.cuda.is_available():
                vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
                vram_peak = max(vram_peak, vram_mb)

            if step % stage_cfg["log_every"] == 0:
                elapsed = time.time() - t0
                print(
                    f"[step {step}/{total_steps}] "
                    f"loss={result['loss']:.4f} mse={result['mse']:.4f} "
                    f"lpips={result['lpips']:.4f} "
                    f"vram={vram_peak:.0f}MB elapsed={elapsed:.0f}s"
                )

            if step % (total_steps // 5) == 0 or step == total_steps:
                save_checkpoint(
                    out_dir / "checkpoints" / f"step_{step:06d}.pt",
                    candidate,
                    optimizer,
                    step,
                )

    elapsed = time.time() - t0
    (out_dir / "curves.json").write_text(json.dumps(curves))
    (out_dir / "vram.json").write_text(
        json.dumps(
            {
                "peak_mb": vram_peak,
                "device": str(device),
            }
        )
    )

    manifest = {
        "candidate": args.candidate,
        "stage": args.stage,
        "seed": args.seed,
        "steps": step,
        "final_loss": curves["loss"][-1] if curves["loss"] else None,
        "final_mse": curves["mse"][-1] if curves["mse"] else None,
        "final_lpips": curves["lpips"][-1] if curves["lpips"] else None,
        "vram_peak_mb": vram_peak,
        "elapsed_s": elapsed,
        "timestamp": time.time(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n=== DONE ===")
    print(f"Final loss: {manifest['final_loss']:.4f}")
    print(f"VRAM peak: {vram_peak:.0f} MB")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"Artifacts: {out_dir}")


if __name__ == "__main__":
    main()
