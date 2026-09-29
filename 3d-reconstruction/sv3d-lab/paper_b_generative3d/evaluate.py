"""Evaluation utilities: export MINE rows and run deletion tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from .model.candidate_registry import build_candidate
from .model.reconstruction_interface import (
    SourceInput,
    TargetCameras,
)
from .train import build_fixture_batch


def export_mine_rows(
    *,
    candidate: str,
    cfg: dict,
    out_dir,
    num_rows: int = 2,
    checkpoint: Optional[str] = None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_candidate(candidate, cfg)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rows = []
    for i in range(num_rows):
        batch = build_fixture_batch(seed=i)
        source = batch["source"]
        target_cameras = batch["target_cameras"]
        with torch.no_grad():
            scene = model.build_scene(source)
            render_out = model.render_targets(scene, target_cameras)
        row = {
            "row_index": i,
            "scene_id": scene.scene_id,
            "num_primitives": len(scene),
            "render_shape": list(render_out.images.shape),
        }
        rows.append(row)
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2))
    return rows


def run_deletion_test(
    *,
    candidate: str,
    cfg: dict,
    checkpoint: Optional[str] = None,
):
    model = build_candidate(candidate, cfg)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    batch = build_fixture_batch(seed=0)
    source = batch["source"]
    target_cameras = batch["target_cameras"]

    with torch.no_grad():
        scene = model.build_scene(source)
        full_render = model.render_targets(scene, target_cameras).images

        added_tags = set(scene.provenance) - {"backbone"}
        backbone_only = scene
        for tag in added_tags:
            backbone_only = backbone_only.delete_by_provenance(tag)
        restored_render = model.render_targets(backbone_only, target_cameras).images

        ref_scene = model.build_scene(source)
        ref_backbone = ref_scene
        for tag in added_tags:
            ref_backbone = ref_backbone.delete_by_provenance(tag)
        ref_render = model.render_targets(ref_backbone, target_cameras).images

    delta = float((full_render - restored_render).abs().mean())
    matches = bool(torch.allclose(restored_render, ref_render, atol=1e-5))

    return {
        "delta": delta,
        "restored_matches_backbone": matches,
        "num_full_primitives": len(scene),
        "num_backbone_primitives": len(backbone_only),
    }
