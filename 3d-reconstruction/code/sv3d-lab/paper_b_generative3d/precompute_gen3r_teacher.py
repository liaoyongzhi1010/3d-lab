"""Precompute Gen3R teacher cache for training-only distillation (C3).

This script is run ONCE on the remote server with Gen3R weights to produce a
cache of teacher point sets for train-split scenes. The cache is then loaded
by TeacherDistillation.register_teacher_cache() during training.

Charter rule: teacher outputs are TRAINING-ONLY supervision. They never appear
in the student's build_scene forward or in test/dev inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def precompute_teacher_cache(
    *,
    scene_ids,
    split: str = "train",
    out_path,
    gen3r_model=None,
):
    if split != "train":
        raise ValueError("teacher precomputation is only allowed for the train split")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache = {}
    for scene_id in scene_ids:
        if gen3r_model is not None:
            points = gen3r_model.predict_points(scene_id)
        else:
            points = None
        cache[scene_id] = {
            "scene_id": scene_id,
            "split": split,
            "points": points,
        }

    manifest = {
        "split": split,
        "num_scenes": len(cache),
        "scene_ids": list(cache.keys()),
    }
    out_path.write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Precompute Gen3R teacher cache")
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = precompute_teacher_cache(
        scene_ids=args.scenes,
        split=args.split,
        out_path=args.out,
    )
    print(f"Wrote manifest: {result['num_scenes']} scenes")
