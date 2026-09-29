"""Ablation runner: systematically disable components and measure impact.

Each ablation trains a variant with one component removed/modified, then
compares against the full model under the same budget and data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .train import build_fixture_batch, run_stage, train_step
from .model.candidate_registry import build_candidate


def run_ablation(
    *,
    candidate: str,
    base_cfg: dict,
    ablations: Dict[str, dict],
    out_dir,
    steps: int = 50,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    base_cfg_full = dict(base_cfg)
    base_cfg_full["steps"] = steps

    baseline = run_stage(
        "smoke",
        candidate=candidate,
        cfg=base_cfg_full,
        out_dir=out_dir / "baseline",
    )
    results["baseline"] = baseline

    for ablation_name, ablation_cfg in ablations.items():
        merged = dict(base_cfg_full)
        merged.update(ablation_cfg)
        try:
            result = run_stage(
                "smoke",
                candidate=candidate,
                cfg=merged,
                out_dir=out_dir / ablation_name,
            )
            results[ablation_name] = result
        except Exception as e:
            results[ablation_name] = {"error": str(e)}

    (out_dir / "ablation_summary.json").write_text(json.dumps(results, indent=2))
    return results
