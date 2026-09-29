"""E-112: real-data candidate evaluation on held-out dev scenes.

For each trained candidate checkpoint, measures on N dev scenes:
  - target-view PSNR of the full candidate scene
  - target-view PSNR of the backbone-only scene (delete trainable provenance)
  - deletion delta (candidate PSNR - backbone PSNR): the causal contribution
Compares against the raw Flash3D backbone render as the baseline anchor.

Single-view inference only; uses the same real backbone + RE10K loader as training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from paper_b_generative3d.backbone_flash3d import Flash3DBackbone
from paper_b_generative3d.data_loader_re10k import (
    build_flash3d_cfg,
    build_re10k_dataloader,
)
from paper_b_generative3d.model.candidate_registry import build_candidate
from paper_b_generative3d.train_real import _batch_item, FLASH3D_CKPT


def psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2).clamp_min(1e-10)
    return float(10.0 * torch.log10(1.0 / mse))


def eval_candidate(candidate, ckpt_path, backbone, loader, device, max_scenes):
    cfg = {"num_points": 24, "hidden": 8, "seed": 0}
    model = build_candidate(candidate, cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state_dict"]
    # The feature adapter is now built EAGERLY in the candidate __init__ at the
    # FIXED key ``_feat_adapter.{weight,bias}`` (reconstruction_interface.
    # SOURCE_FEATURE_DIM), so it exists BEFORE load_state_dict and the trained
    # adapter weights load correctly. Previously the adapter was created lazily on
    # the first forward — so at load time it did not exist yet and the trained
    # ``_feat_adapters.2048.*`` key was silently dropped (strict=False), collapsing
    # the candidate render to the backbone render (delta PSNR ~0). We still use
    # strict=False for the unused fake ``backbone.*`` (FrozenBackbone) keys, but
    # assert the adapter + heads actually loaded.
    load_result = model.load_state_dict(state, strict=False)
    missing = set(load_result.missing_keys)
    adapter_keys = {"_feat_adapter.weight", "_feat_adapter.bias"}
    dropped_adapter = [k for k in state if k in adapter_keys and k in missing]
    assert not dropped_adapter, (
        f"feature adapter keys failed to load: {dropped_adapter}"
    )
    ckpt_adapter_keys = adapter_keys & set(state)
    assert ckpt_adapter_keys, (
        "checkpoint has no eager _feat_adapter.* keys — was it trained with the "
        "lazy adapter? re-train with the eager adapter fix"
    )
    model.to(device).eval()

    rows = []
    n = 0
    for inputs in loader:
        if n >= max_scenes:
            break
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
        if ("color", 1, 0) not in inputs:
            continue
        with torch.no_grad():
            gauss_out = backbone.extract_source_gaussians(inputs)
            bb_outputs = gauss_out.get("_outputs", {})
            source = backbone.build_source_input_from_dataloader(inputs)
            tfids = [f for f in [1, 2, 3] if ("color", f, 0) in inputs]
            if not tfids:
                continue
            scene = model.build_scene_from_backbone(_batch_item(gauss_out, 0), source)
            # Baseline = RAW Flash3D backbone Gaussians (unmodified), not the
            # candidate scene minus its extra set: the in-place refinement is
            # tagged "backbone", so deleting the candidate tag would leave the
            # refinements in place and make the baseline identical to the candidate.
            raw = _batch_item(gauss_out, 0)
            g_bb = {
                "xyz": raw["xyz"][0],
                "scales": raw["scales"][0],
                "rotations": raw["rotations"][0],
                "opacity": raw["opacity"][0],
                "color_rgb": raw["color_rgb"][0],
            }
            H, W = 256, 384
            cand_ps, bb_ps = [], []
            for fid in tfids:
                cam = inputs.get(("cam_T_cam", 0, fid))
                if cam is None and bb_outputs:
                    cam = bb_outputs.get(("cam_T_cam", 0, fid))
                if cam is None:
                    continue
                K_tgt = inputs.get(("K_tgt", fid), inputs[("K_src", 0)])
                gt = inputs["color", fid, 0][0]
                g_full = {
                    "xyz": scene.means,
                    "scales": scene.scales,
                    "rotations": scene.rotations,
                    "opacity": scene.opacity,
                    "color_rgb": scene.color,
                }
                r_full = backbone.render_gaussians(
                    g_full, cam[0], K_tgt[0], H, W, batch_idx=0
                )
                r_bb = backbone.render_gaussians(
                    g_bb, cam[0], K_tgt[0], H, W, batch_idx=0
                )
                cand_ps.append(psnr(r_full.clamp(0, 1), gt))
                bb_ps.append(psnr(r_bb.clamp(0, 1), gt))
            if cand_ps:
                rows.append(
                    {
                        "scene": n,
                        "candidate_psnr": sum(cand_ps) / len(cand_ps),
                        "backbone_psnr": sum(bb_ps) / len(bb_ps),
                        "delta": sum(cand_ps) / len(cand_ps) - sum(bb_ps) / len(bb_ps),
                        "num_full": len(scene),
                        "num_backbone": int(raw["xyz"].shape[1]),
                    }
                )
        n += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_scenes", type=int, default=16)
    ap.add_argument("--step", default="step_000800.pt")
    args = ap.parse_args()

    device = torch.device("cuda")
    cfg = build_flash3d_cfg(batch_size=1, num_workers=0, stage="dev")
    backbone = Flash3DBackbone(cfg, FLASH3D_CKPT, device=str(device))
    backbone.to(device)
    loader = build_re10k_dataloader(
        "dev", batch_size=1, num_workers=0, max_scenes=args.max_scenes, flash3d_cfg=cfg
    )

    summary = {}
    for cand in ["C0", "C1", "C2", "C3", "C4", "C5"]:
        ckpt = Path(args.root) / cand / "checkpoints" / args.step
        if not ckpt.exists():
            summary[cand] = {"error": "no checkpoint"}
            continue
        rows = eval_candidate(
            cand, str(ckpt), backbone, loader, device, args.max_scenes
        )
        if rows:
            import statistics

            summary[cand] = {
                "n": len(rows),
                "candidate_psnr": statistics.mean(r["candidate_psnr"] for r in rows),
                "backbone_psnr": statistics.mean(r["backbone_psnr"] for r in rows),
                "delta_psnr": statistics.mean(r["delta"] for r in rows),
                "num_full": rows[0]["num_full"],
                "num_backbone": rows[0]["num_backbone"],
                "rows": rows,
            }
        else:
            summary[cand] = {"error": "no rows"}
        print(cand, summary[cand].get("delta_psnr", summary[cand]))

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
