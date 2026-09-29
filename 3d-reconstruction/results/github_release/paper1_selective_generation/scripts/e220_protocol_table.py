"""Generate the mechanism figure, including a target-GT quality-gap oracle diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#3572b0"
VIOLET = "#7755aa"
ORANGE = "#d97922"
GREEN = "#2f8f5b"
RED = "#b53a3a"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=root / "results" / "E-142_combined_N169.json"
    )
    parser.add_argument(
        "--gate", type=Path, default=root / "results" / "E-216_gate_sensitivity.json"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "paper" / "figs" / "fig_mechanism_combined.pdf",
    )
    args = parser.parse_args()

    rows = json.loads(args.data.read_text())
    gate = json.loads(args.gate.read_text())
    base = np.array([row["base_inv"] for row in rows], dtype=float)
    gap = np.array([row["f3d_inv"] - row["base_inv"] for row in rows], dtype=float)
    benefit = np.array(
        [row["teacher_inv"] - row["base_inv"] for row in rows], dtype=float
    )
    correlation = float(np.corrcoef(gap, benefit)[0, 1])

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25), constrained_layout=True)

    ax = axes[0]
    scatter = ax.scatter(
        gap, benefit, c=base, cmap="viridis_r", s=16, alpha=0.82, linewidth=0
    )
    ax.axhline(0, color="0.45", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Geometry - generative invisible PSNR (dB)")
    ax.set_ylabel("Always-inject benefit (dB)")
    ax.set_title(
        f"(a) Evidence gap predicts benefit\n$r={correlation:.3f}$, $N={len(rows)}$"
    )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.02)
    colorbar.set_label("Baseline invisible PSNR")

    ax = axes[1]
    definitions = [
        ("Hard\n<12", base < 12),
        ("Mid\n12-20", (base >= 12) & (base < 20)),
        ("Easy\n>=20", base >= 20),
    ]
    means = [float(benefit[mask].mean()) for _, mask in definitions]
    counts = [int(mask.sum()) for _, mask in definitions]
    bars = ax.bar(range(3), means, color=[GREEN, BLUE, RED], width=0.68)
    ax.axhline(0, color="0.2", linewidth=0.7)
    ax.set_xticks(range(3), ["Hard\n<12 dB", "Mid\n12-20 dB", "Easy\n>=20 dB"])
    ax.set_ylim(-3.25, 3.35)
    ax.set_ylabel("Mean always-inject benefit (dB)")
    ax.set_title("(b) Benefit depends on baseline difficulty")
    for bar, value, count in zip(bars, means, counts):
        if value >= 0:
            y, color, va = value + 0.10, "black", "bottom"
        else:
            y, color, va = value + 0.18, "white", "bottom"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:+.2f}; $n={count}$",
            ha="center",
            va=va,
            color=color,
            fontsize=7.5,
        )

    ax = axes[2]
    sweep = gate["sweep"]
    thresholds = np.array([entry["tau"] for entry in sweep], dtype=float)
    means = np.array([entry["mean"] for entry in sweep], dtype=float)
    worst = np.array([entry["worst"] for entry in sweep], dtype=float)
    ax.plot(thresholds, means, color=VIOLET, linewidth=1.8, label="Mean benefit")
    ax.plot(thresholds, worst, color=ORANGE, linewidth=1.8, label="Worst case")
    default = int(np.argmin(np.abs(thresholds - 5.0)))
    ax.scatter(
        [thresholds[default]],
        [means[default]],
        color=VIOLET,
        edgecolor="white",
        linewidth=0.6,
        s=40,
        zorder=3,
    )
    ax.scatter(
        [thresholds[default]],
        [worst[default]],
        color=ORANGE,
        edgecolor="white",
        linewidth=0.6,
        s=40,
        zorder=3,
    )
    ax.axvline(5.0, color="0.35", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="0.45", linewidth=0.7)
    ax.annotate(
        f"$\\tau=5$\n{means[default]:+.3f} dB mean",
        (5.0, means[default]),
        xytext=(12, -24),
        textcoords="offset points",
        fontsize=8,
        ha="left",
        arrowprops={"arrowstyle": "-", "color": VIOLET, "lw": 0.8},
    )
    ax.set_xlim(-0.25, 10.25)
    ax.set_xlabel("Quality-gap oracle threshold $\\tau$ (dB)")
    ax.set_ylabel("Oracle-selected benefit (dB)")
    ax.set_title("(c) Oracle selectivity\nMean and worst-case benefit", pad=6)
    ax.legend(loc="lower right", frameon=False)

    for ax in axes:
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(
        f"saved {args.out}; r={correlation:.3f}; quality-gap oracle={means[default]:+.3f} dB"
    )


if __name__ == "__main__":
    main()
