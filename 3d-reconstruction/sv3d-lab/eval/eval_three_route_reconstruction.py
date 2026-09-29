"""Official Flash3D evaluator wrapper for frozen three-route exports."""

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from common.data.evaluation_contract import parse_mine_split, row_export_name
from common.data.run_manifest import write_manifest_once
from eval.verify_export_alignment import FRAMES, inspect_export


def _canonical_write(path, value):
    Path(path).write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    )


class OfficialEvaluatorAdapter:
    def __init__(self):
        import torch

        evaluator_class = importlib.import_module("evaluation.evaluator").Evaluator
        self.torch = torch
        self.evaluator = evaluator_class(crop_border=True).eval()
        self.metric_names = tuple(self.evaluator.metric_names())

    def __call__(self, pred, gt):
        pred_tensor = (
            self.torch.from_numpy(np.asarray(pred, dtype=np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        gt_tensor = (
            self.torch.from_numpy(np.asarray(gt, dtype=np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        with self.torch.no_grad():
            return self.evaluator(pred_tensor, gt_tensor)


def evaluate_export(export_root, rows, output_dir, *, evaluator=None, manifest=None):
    inspect_export(export_root, rows)
    evaluator = evaluator or OfficialEvaluatorAdapter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    export_root = Path(export_root)
    for index, row in enumerate(rows):
        directory_name = row_export_name(index, row)
        for bucket in FRAMES:
            pred = Image.open(
                export_root / directory_name / f"pred_{bucket}.png"
            ).convert("RGB")
            gt = Image.open(export_root / directory_name / f"gt_{bucket}.png").convert(
                "RGB"
            )
            metrics = {
                name: float(value) for name, value in evaluator(pred, gt).items()
            }
            records.append(
                {
                    "bucket": bucket,
                    "metrics": metrics,
                    "row": index,
                    "row_name": directory_name,
                    "scene": row.scene,
                }
            )
    metric_names = tuple(records[0]["metrics"])
    summary = {}
    for bucket in FRAMES:
        bucket_records = [record for record in records if record["bucket"] == bucket]
        summary[bucket] = {
            metric: float(
                np.mean([record["metrics"][metric] for record in bucket_records])
            )
            for metric in metric_names
        }
    novel_records = [record for record in records if record["bucket"] != "source"]
    summary["novel_mean"] = {
        metric: float(np.mean([record["metrics"][metric] for record in novel_records]))
        for metric in metric_names
    }
    source_psnr = summary["source"]["psnr"]
    if source_psnr < 35.0:
        raise ValueError(
            f"source PSNR {source_psnr:.6g} dB is below the 35 dB validity gate"
        )
    result_summary = {
        "buckets": summary,
        **summary,
        "row_count": len(rows),
        "valid": True,
    }
    _canonical_write(output_dir / "rows.json", records)
    _canonical_write(output_dir / "summary.json", result_summary)
    write_manifest_once(output_dir / "manifest.json", manifest or {})
    return {"rows": records, "summary": result_summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("export")
    parser.add_argument("split")
    parser.add_argument("output")
    parser.add_argument("manifest")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    evaluate_export(
        args.export, parse_mine_split(args.split), args.output, manifest=manifest
    )


if __name__ == "__main__":
    main()
