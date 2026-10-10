#!/usr/bin/env python3
"""Regenerate release manifests deterministically from released result JSONs."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENE_SOURCE = "paper1_selective_generation/results/E-142_combined_N169.json"
FRAME_SOURCE = "paper3_reliability_learning/results/E-149_frames_b5678_aug.json"
SCENE_MANIFEST = "manifests/re10k_scene_manifest.json"
FRAME_MANIFEST = "manifests/frame_reliability_manifest.json"
FRAME_FEATURES = [
    "f3d_vis",
    "f3d_inv",
    "vis_frac",
    "vis_gap",
    "cam_trans",
    "cam_rot_deg",
    "disocc_frac",
    "f3d_vi_ratio",
    "f3d_vi_prod",
]


def load_json(root, relative_path):
    with (root / relative_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def scene_splits(rows):
    by_split = defaultdict(list)
    seen = {}
    for row in rows:
        scene_id = row["sid"]
        split = row["split"]
        if scene_id in seen:
            raise ValueError(f"duplicate scene row: {scene_id}")
        seen[scene_id] = split
        by_split[split].append(scene_id)
    return {split: sorted(by_split[split]) for split in sorted(by_split)}


def frame_scene_splits(rows):
    by_split = defaultdict(list)
    seen = {}
    for row in rows:
        scene_id = row["sid"]
        split = row["split"]
        previous = seen.setdefault(scene_id, split)
        if previous != split:
            raise ValueError(f"inconsistent split for {scene_id}: {previous}, {split}")
    for scene_id, split in seen.items():
        by_split[split].append(scene_id)
    return {split: sorted(by_split[split]) for split in sorted(by_split)}


def build_manifests(root=ROOT):
    scenes = load_json(root, SCENE_SOURCE)
    frames = load_json(root, FRAME_SOURCE)
    split_scene_ids = scene_splits(scenes)
    frame_split_scene_ids = frame_scene_splits(frames)
    frame_counts = Counter(row["sid"] for row in frames)

    scene_manifest = {
        "schema_version": 1,
        "protocol_label": "released_long_sequence_disocclusion_diagnostic",
        "source_json": SCENE_SOURCE,
        "row_count": len(scenes),
        "unique_scene_count": sum(len(ids) for ids in split_scene_ids.values()),
        "seed": None,
        "seed_note": "Not present in the released source JSON.",
        "scene_ids_by_source_split": split_scene_ids,
    }
    frame_manifest = {
        "schema_version": 1,
        "protocol_label": "frame_reliability_long_sequence_diagnostic",
        "source_json": FRAME_SOURCE,
        "frame_count": len(frames),
        "unique_scene_count": len(frame_counts),
        "scene_ids_by_source_split": frame_split_scene_ids,
        "feature_version": "E-149 augmented features as released",
        "features": FRAME_FEATURES,
        "grouping_rule": "grouped leave-one-scene-out cross-validation by scene_id",
        "seed": None,
        "seed_note": "Not present in the released source JSON.",
        "frame_count_by_scene_id": dict(sorted(frame_counts.items())),
    }
    return {SCENE_MANIFEST: scene_manifest, FRAME_MANIFEST: frame_manifest}


def generated_manifest_text(root=ROOT):
    return {
        path: json.dumps(manifest, indent=2, ensure_ascii=True) + "\n"
        for path, manifest in build_manifests(root).items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail if checked-in manifests differ"
    )
    args = parser.parse_args()

    stale = []
    for relative_path, expected in generated_manifest_text(ROOT).items():
        path = ROOT / relative_path
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != expected:
                stale.append(relative_path)
        else:
            path.write_text(expected, encoding="utf-8")
            print(f"wrote {relative_path}")
    if stale:
        raise SystemExit(f"stale manifests: {', '.join(stale)}")


if __name__ == "__main__":
    main()
