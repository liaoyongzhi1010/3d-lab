"""Build an honest limitations composite from released and aggregate evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--failure",
        type=Path,
        default=root / "paper" / "figs" / "panel_failure_boxed.png",
    )
    parser.add_argument(
        "--acid", type=Path, default=root / "results" / "E-215_acid_metrics.json"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "paper" / "figs" / "fig_failures_limitations.pdf",
    )
    args = parser.parse_args()

    acid = json.loads(args.acid.read_text())["mechanism"]
    failure = Image.open(args.failure).convert("RGB")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(10.8, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0])
    axes = [
        fig.add_subplot(grid[:, :2]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[0, 3]),
        fig.add_subplot(grid[1, 2:]),
    ]

    axes[0].imshow(failure)
    axes[0].axis("off")
    axes[0].set_title("(a) Released injected failure", fontweight="bold")
    axes[0].text(
        0.5,
        -0.035,
        "Released asset; no fallback decision.",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    ax = axes[1]
    temporal = [0.0251, 0.0267]
    bars = ax.bar([0, 1], temporal, color=["#6b7c93", "#b55245"], width=0.62)
    ax.set_xticks([0, 1], ["Gen3R", "Injected\noutput"])
    ax.set_ylim(0, 0.036)
    ax.set_ylabel("Second-order energy\n(lower is better)")
    ax.set_title("(b) Temporal limitation", fontweight="bold")
    for bar, value in zip(bars, temporal):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.0006,
            f"{value:.4f}",
            ha="center",
        )
    ax.text(
        0.5,
        0.0335,
        "9/21 scenes favor injected",
        ha="center",
        va="top",
        fontsize=9,
    )

    ax = axes[2]
    mean_gain = float(acid["mean_gain"])
    wins = int(acid["win"])
    scenes = int(acid["n_scenes"])
    bar = ax.bar([0], [mean_gain], color="#b55245", width=0.55)[0]
    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.62, 0.18)
    ax.set_xticks([0], ["Always injected\ncandidate"])
    ax.set_ylabel("Mean invisible PSNR\nchange (dB)")
    ax.set_title("(c) ACID failed transfer", fontweight="bold")
    ax.text(
        0,
        mean_gain - 0.035,
        f"{mean_gain:.3f} dB",
        ha="center",
        va="top",
        fontweight="bold",
    )
    ax.text(
        0,
        0.13,
        f"{scenes} scenes; {wins}/{scenes} wins",
        ha="center",
        va="top",
        fontweight="bold",
    )
    ax.text(
        0,
        0.055,
        "Aggregate only; no raw images",
        ha="center",
        va="top",
        fontsize=8.5,
    )

    ax = axes[3]
    ax.axis("off")
    ax.set_title("(d) Single-image boundary leak", fontweight="bold")
    ax.text(
        0.5,
        0.72,
        "Complete target clip",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#f4c7c3",
            "edgecolor": "#b55245",
        },
    )
    ax.annotate(
        "",
        xy=(0.5, 0.47),
        xytext=(0.5, 0.64),
        arrowprops={"arrowstyle": "->", "color": "#b55245", "lw": 1.8},
    )
    ax.text(
        0.5,
        0.39,
        "VGGT -> visibility.npy  |  ORACLE VISIBILITY",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.14,
        "Crosses the single-image inference boundary.\nDeployment needs a source-only visibility estimator.",
        ha="center",
        va="center",
        fontsize=10,
    )

    for ax in axes[1:3]:
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(
        f"saved {args.out}; temporal=0.0251/0.0267 (9/21); "
        f"ACID={scenes} scenes, {mean_gain:.3f} dB, {wins}/{scenes} wins"
    )


if __name__ == "__main__":
    main()
