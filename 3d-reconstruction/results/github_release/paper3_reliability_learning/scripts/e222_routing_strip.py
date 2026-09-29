"""Generate a quality-probe diagnostic strip from released frame rows.

The scores use GT-derived PSNR quality probes and are not observable routing.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_frame_module(path):
    spec = importlib.util.spec_from_file_location("frame_net", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def representative_indices(probabilities):
    order = np.argsort(probabilities)
    targets = [
        0,
        len(order) // 8,
        len(order) // 2 - 1,
        len(order) // 2,
        7 * len(order) // 8,
        len(order) - 1,
    ]
    return [int(order[target]) for target in targets]


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=root / "results/E-149_frames_b5678_aug.json"
    )
    parser.add_argument(
        "--out", type=Path, default=root / "paper/figs/fig_routing_strip.pdf"
    )
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as handle:
        rows = [
            row
            for row in json.load(handle)
            if row.get("vis_gap") is not None and row.get("delta") is not None
        ]
    frame_net = load_frame_module(root / "scripts/e145b_frame_net_full.py")
    columns = frame_net.BASE_FEATS + frame_net.EXT_FEATS
    probabilities = frame_net.group_cv(rows, columns)
    selected = representative_indices(probabilities)

    fig, axes = plt.subplots(1, len(selected), figsize=(11.2, 2.35), sharey=True)
    for axis, index in zip(axes, selected):
        row = rows[index]
        score = probabilities[index]
        call = score > 0.5
        gain = row["delta"]
        favorable = bool(row["label"])
        color = "#d95f3d" if call else "#8c96a3"
        axis.barh([0], [score], color=color, height=0.33)
        axis.axvline(0.5, color="#222222", linewidth=0.8, linestyle="--")
        axis.set_xlim(0, 1)
        axis.set_ylim(-0.52, 0.52)
        axis.set_yticks([])
        axis.set_xticks([0, 0.5, 1])
        axis.tick_params(axis="x", labelsize=7)
        axis.set_title(f"{row['sid'][6:14]}\nf{row['frame_idx']}", fontsize=8)
        axis.text(0.5, 0.37, f"score {score:.3f}", ha="center", fontsize=8)
        axis.text(
            0.5,
            -0.28,
            "CALL" if call else "SKIP",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=color,
        )
        axis.text(
            0.5,
            -0.43,
            f"true gain {gain:+.2f} dB | oracle {'call' if favorable else 'skip'}",
            ha="center",
            fontsize=7,
        )
        axis.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle(
        "Quality-probe diagnostic: grouped-CV scores use GT-derived PSNR fields",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Diagnostic only, not observable routing. True gain and oracle labels are evaluation annotations.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.88), w_pad=0.6)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")

    for index in selected:
        row = rows[index]
        print(
            f"sid={row['sid']} frame={row['frame_idx']} score={probabilities[index]:.6f} "
            f"decision={'call' if probabilities[index] > 0.5 else 'skip'} "
            f"delta={row['delta']:+.6f} oracle={'call' if row['label'] else 'skip'}"
        )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
