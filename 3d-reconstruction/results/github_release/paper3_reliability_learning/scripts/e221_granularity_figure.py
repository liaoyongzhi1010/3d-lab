"""Plot ROC-AUC diagnostics with evidence and population caveats.

Frame GT quality probe and camera-only observable results use 74 scenes. E-223
RGB + oracle-visibility and patch diagnostics use 21 selected high-gain scenes,
so the figure is not a controlled leaderboard. Precision-recall metrics are
omitted because E-223 reports AP while E-145b/E-224 use trapezoidal PR-AUC.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=root / "paper/figs/fig_granularity.pdf"
    )
    args = parser.parse_args()

    diagnostic = load_json(root / "results/E-145b_N2898_ext.json")
    camera = load_json(root / "results/E-224_camera_only_observable.json")
    rgb = load_json(root / "results/E-223_observable_frame_reliability.json")
    patch = load_json(root / "results/E-208_pixel_reliability.json")

    labels = [
        "GT-quality\nprobe",
        "Camera-only\nobservable",
        "RGB + oracle\nvisibility*",
        "Patch diagnostic*",
    ]
    aucs = [
        diagnostic["roc_auc"],
        camera["metrics"]["roc_auc"],
        rgb["metrics"]["roc_auc"],
        patch["roc_auc"],
    ]
    counts = [
        diagnostic["n"],
        camera["frame_count"],
        rgb["frame_count"],
        patch["n_patches"],
    ]
    colors = ["#d97706", "#2f855a", "#4f78a8", "#8c96a3"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.1))
    bars = axes[0].bar(labels, aucs, color=colors, width=0.68)
    axes[0].axhline(0.5, color="#333333", linestyle="--", linewidth=0.8)
    axes[0].set_ylim(0.35, 1.0)
    axes[0].set_ylabel("Grouped-CV ROC-AUC")
    axes[0].set_title("(a) Different evidence and populations")
    for bar, value in zip(bars, aucs):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.3f}",
            ha="center",
            fontsize=9,
        )

    bars = axes[1].bar(labels, counts, color=colors, width=0.68)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Units (log scale)")
    axes[1].set_title("(b) Frame and patch units differ")
    for bar, value in zip(bars, counts):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            f"{value:,}",
            ha="center",
            fontsize=9,
        )

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelsize=8)
    fig.text(
        0.5,
        0.01,
        "ROC-AUC only; PR metrics differ (E-223 AP, E-145b/E-224 trapezoidal PR-AUC). *21 selected scenes; E-223 uses VGGT complete-clip visibility.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"aucs={','.join(f'{value:.6f}' for value in aucs)}")
    print(f"counts={','.join(map(str, counts))}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
