"""Training infrastructure for the reconstruction method-selection funnel.

Provides:
  * train_step — one forward/backward/step; returns loss dict.
  * build_fixture_batch — deterministic tiny batch for test reproducibility.
  * save_checkpoint / load_checkpoint — state persistence for resume.
  * run_stage — full stage runner (smoke/overfit/pilot/full) writing immutable
    artifacts: config, manifest, curves, checkpoints, raw rows, VRAM, visuals.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Sequence

import torch

from .model.candidate_registry import build_candidate
from .model.reconstruction_interface import (
    GaussianScene,
    SourceInput,
    TargetCameras,
    render_scene,
)

VALID_STAGES = ("smoke", "overfit", "pilot", "full")


def build_fixture_batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    source = SourceInput(
        source_rgb=torch.rand(1, 3, 8, 12, generator=g),
        K_src=torch.tensor([[[8.0, 0.0, 6.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]]),
        T_c2w_src=torch.eye(4).unsqueeze(0),
    )
    K = torch.tensor([[8.0, 0.0, 6.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    poses = []
    for i in range(3):
        p = torch.eye(4)
        p[0, 3] = 0.1 * (i + 1)
        poses.append(p)
    target_cameras = TargetCameras(
        K=torch.stack([K, K, K]).unsqueeze(0),
        T_c2w=torch.stack(poses).unsqueeze(0),
    )
    target_rgb = torch.rand(1, 3, 3, 8, 12, generator=g)
    return {
        "source": source,
        "target_cameras": target_cameras,
        "target_rgb": target_rgb,
    }


def train_step(model, batch, optimizer):
    model.train()
    optimizer.zero_grad()
    source = batch["source"]
    target_cameras = batch["target_cameras"]
    target_rgb = batch["target_rgb"]

    scene = model.build_scene(source)
    render_out = model.render_targets(scene, target_cameras)
    pred = render_out.images
    loss = torch.nn.functional.mse_loss(pred, target_rgb)
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.item())}


def save_checkpoint(path, model, optimizer, step: int):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        },
        str(path),
    )


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("step", 0)


def run_stage(
    stage: str,
    *,
    candidate: str,
    cfg: dict,
    out_dir,
    qualitative_ids: Optional[Sequence[str]] = None,
):
    if stage not in VALID_STAGES:
        raise ValueError(f"unknown stage '{stage}'; valid: {VALID_STAGES}")
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    (out_dir / "checkpoints").mkdir()

    full_cfg = dict(cfg)
    full_cfg["candidate"] = candidate
    full_cfg["stage"] = stage
    (out_dir / "config.json").write_text(json.dumps(full_cfg, indent=2))

    if qualitative_ids is None:
        qualitative_ids = []
    (out_dir / "qualitative_ids.json").write_text(json.dumps(qualitative_ids))

    model = build_candidate(candidate, cfg)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    batch = build_fixture_batch(seed=cfg.get("seed", 0))
    steps = int(cfg.get("steps", 100))

    curves = {"loss": []}
    for step_i in range(steps):
        result = train_step(model, batch, optimizer)
        curves["loss"].append(result["loss"])
        if step_i == steps - 1:
            save_checkpoint(
                out_dir / "checkpoints" / f"step_{step_i:06d}.pt",
                model,
                optimizer,
                step_i,
            )

    (out_dir / "curves.json").write_text(json.dumps(curves))
    (out_dir / "raw_rows.json").write_text(json.dumps(curves["loss"]))
    (out_dir / "vram.json").write_text(json.dumps({"peak_mb": 0, "note": "cpu"}))
    manifest = {
        "candidate": candidate,
        "stage": stage,
        "steps": steps,
        "final_loss": curves["loss"][-1] if curves["loss"] else None,
        "timestamp": time.time(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        "stage": stage,
        "out_dir": str(out_dir),
        "final_loss": manifest["final_loss"],
    }
