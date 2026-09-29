"""Pillow-only export preflight; images are inspected but never resized."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from common.data.evaluation_contract import parse_mine_split, row_export_name


FRAMES = ("source", "tgt5", "tgt10", "tgt_rand")
EXPECTED_SIZE = (384, 256)


def _expected_files():
    return {f"{kind}_{frame}.png" for kind in ("pred", "gt") for frame in FRAMES}


def inspect_export(root, rows):
    root = Path(root)
    expected_directories = {
        row_export_name(index, row) for index, row in enumerate(rows)
    }
    actual_directories = {path.name for path in root.iterdir() if path.is_dir()}
    if actual_directories != expected_directories:
        raise ValueError(
            f"row directories mismatch: missing={sorted(expected_directories - actual_directories)}, "
            f"extra={sorted(actual_directories - expected_directories)}"
        )
    expected_files = _expected_files()
    image_count = 0
    for directory_name in sorted(expected_directories):
        directory = root / directory_name
        actual_files = {path.name for path in directory.iterdir() if path.is_file()}
        if actual_files != expected_files:
            raise ValueError(
                f"files mismatch in {directory_name}: missing={sorted(expected_files - actual_files)}, "
                f"extra={sorted(actual_files - expected_files)}"
            )
        for filename in sorted(expected_files):
            with Image.open(directory / filename) as image:
                if image.size != EXPECTED_SIZE:
                    raise ValueError(
                        f"{directory_name}/{filename} must be 256x384, got {image.height}x{image.width}"
                    )
            image_count += 1
    return {"image_count": image_count, "row_count": len(rows), "shape": [256, 384]}


def compare_ground_truth(candidate_root, reference_root, rows, threshold=1e-3):
    inspect_export(candidate_root, rows)
    inspect_export(reference_root, rows)
    total_absolute_error = 0.0
    total_values = 0
    for index, row in enumerate(rows):
        directory = row_export_name(index, row)
        for frame in FRAMES:
            candidate = np.asarray(
                Image.open(
                    Path(candidate_root) / directory / f"gt_{frame}.png"
                ).convert("RGB"),
                dtype=np.float64,
            )
            reference = np.asarray(
                Image.open(
                    Path(reference_root) / directory / f"gt_{frame}.png"
                ).convert("RGB"),
                dtype=np.float64,
            )
            total_absolute_error += np.abs(candidate - reference).sum() / 255.0
            total_values += candidate.size
    mae = total_absolute_error / total_values
    if mae > threshold:
        raise ValueError(f"GT-MAE {mae:.8g} exceeds threshold {threshold:.8g}")
    return {"mae": mae, "threshold": threshold, "value_count": total_values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("export")
    parser.add_argument("split")
    parser.add_argument("--reference-gt")
    parser.add_argument("--gt-mae-threshold", type=float, default=1e-3)
    args = parser.parse_args()
    rows = parse_mine_split(args.split)
    report = {"export": inspect_export(args.export, rows)}
    if args.reference_gt:
        report["ground_truth"] = compare_ground_truth(
            args.export, args.reference_gt, rows, args.gt_mae_threshold
        )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
