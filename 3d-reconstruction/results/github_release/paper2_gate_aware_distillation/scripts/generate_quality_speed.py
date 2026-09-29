#!/usr/bin/env python3
"""Generate the E-211 quality-speed scatter from released metrics."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "gen3r": "Gen3R baseline",
    "flash3d": "Flash3D evidence",
    "teacher": "Teacher",
    "student": "Student",
}
COLORS = {
    "gen3r": "#6b7280",
    "flash3d": "#2563eb",
    "teacher": "#7c3aed",
    "student": "#ea580c",
}
OFFSETS = {
    "gen3r": (-5, 7),
    "flash3d": (5, 7),
    "teacher": (-5, -14),
    "student": (5, -14),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT / "results/E-211_student_standard_metrics.json",
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "paper/figs/fig_quality_speed.pdf"
    )
    args = parser.parse_args()

    if not args.metrics.is_file():
        raise FileNotFoundError(
            f"E-211 metrics JSON not found: {args.metrics}. "
            "Pass --metrics to the released E-211 JSON."
        )
    try:
        metrics = json.loads(args.metrics.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid E-211 metrics JSON at {args.metrics}: {error}"
        ) from error
    methods = list(LABELS)
    required = {method: {"time_s", "lpips", "fid"} for method in methods}
    missing = {
        method: sorted(keys - set(metrics.get(method, {})))
        for method, keys in required.items()
        if keys - set(metrics.get(method, {}))
    }
    if missing:
        raise ValueError(f"E-211 metrics are missing required method fields: {missing}")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)

    for axis, metric, title in zip(
        axes, ("lpips", "fid"), ("LPIPS-VGG (lower is better)", "FID (lower is better)")
    ):
        for method in methods:
            values = metrics[method]
            axis.scatter(
                values["time_s"], values[metric], color=COLORS[method], s=34, zorder=3
            )
            axis.annotate(
                LABELS[method],
                (values["time_s"], values[metric]),
                xytext=OFFSETS[method],
                textcoords="offset points",
                ha="left" if OFFSETS[method][0] > 0 else "right",
                fontsize=7,
            )
        axis.set_xscale("log")
        axis.set_xlabel("Runtime per scene (s, log scale)")
        axis.set_ylabel(title)
        axis.grid(alpha=0.25, linewidth=0.5)

    fig.suptitle(
        f"Fixed custom {metrics['n_frames']}-frame diagnostic (E-211; not official Flash3D protocol)",
        fontsize=10,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
