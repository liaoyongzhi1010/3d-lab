"""Add automatic ROI boxes to an E-201 qualitative comparison panel.

The input panel has five 256x256 columns and six rows:
GT, Gen3R, Ours, disocclusion mask, baseline error, and ours error.
For each column, this script locates the disoccluded window where Gen3R and
Ours differ most, then draws the same yellow ROI box on the image and error rows.
This makes qualitative improvements easier to inspect without hand-picking boxes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def integral_sum(values: np.ndarray, y: int, x: int, size: int) -> float:
    integral = np.pad(values.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    y1, x1 = y + size, x + size
    return float(integral[y1, x1] - integral[y, x1] - integral[y1, x] + integral[y, x])


def select_roi(
    baseline: np.ndarray,
    ours: np.ndarray,
    mask_cell: np.ndarray,
    window: int,
    stride: int,
) -> tuple[int, int, int, int] | None:
    disocc = (
        (mask_cell[..., 0] > 150) & (mask_cell[..., 1] < 80) & (mask_cell[..., 2] < 80)
    )
    if disocc.sum() < 20:
        return None

    difference = np.abs(baseline.astype(np.float32) - ours.astype(np.float32)).mean(-1)
    score = difference * disocc
    height, width = score.shape
    window = min(window, height, width)

    best: tuple[float, int, int] | None = None
    for y in range(0, height - window + 1, stride):
        for x in range(0, width - window + 1, stride):
            mask_count = disocc[y : y + window, x : x + window].sum()
            if mask_count < 0.08 * window * window:
                continue
            value = integral_sum(score, y, x, window)
            if best is None or value > best[0]:
                best = (value, y, x)

    if best is None:
        ys, xs = np.where(disocc)
        center_y = int(np.median(ys))
        center_x = int(np.median(xs))
        y = max(0, min(height - window, center_y - window // 2))
        x = max(0, min(width - window, center_x - window // 2))
    else:
        _, y, x = best
    return x, y, x + window - 1, y + window - 1


def draw_roi(
    image: Image.Image,
    box: tuple[int, int, int, int],
    offset_x: int,
    offset_y: int,
    width: int,
) -> None:
    draw = ImageDraw.Draw(image)
    translated = tuple(
        value + (offset_x if index % 2 == 0 else offset_y)
        for index, value in enumerate(box)
    )
    for inset in range(width):
        draw.rectangle(
            (
                translated[0] - inset,
                translated[1] - inset,
                translated[2] + inset,
                translated[3] + inset,
            ),
            outline=(255, 220, 0),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cell", type=int, default=256)
    parser.add_argument("--gap", type=int, default=4)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--window", type=int, default=84)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    panel = Image.open(args.input).convert("RGB")
    pixels = np.asarray(panel)
    expected_width = args.columns * args.cell + (args.columns - 1) * args.gap
    expected_height = 6 * args.cell + 5 * args.gap
    if panel.size != (expected_width, expected_height):
        raise ValueError(
            f"Unexpected panel size {panel.size}; expected "
            f"{(expected_width, expected_height)}"
        )

    output = panel.copy()
    boxed_columns = 0
    for column in range(args.columns):
        x0 = column * (args.cell + args.gap)
        baseline_y = args.cell + args.gap
        ours_y = 2 * (args.cell + args.gap)
        mask_y = 3 * (args.cell + args.gap)

        baseline = pixels[baseline_y : baseline_y + args.cell, x0 : x0 + args.cell]
        ours = pixels[ours_y : ours_y + args.cell, x0 : x0 + args.cell]
        mask = pixels[mask_y : mask_y + args.cell, x0 : x0 + args.cell]
        roi = select_roi(baseline, ours, mask, args.window, args.stride)
        if roi is None:
            continue

        for row in (0, 1, 2):
            draw_roi(
                output,
                roi,
                x0,
                row * (args.cell + args.gap),
                width=3,
            )
        for row in (4, 5):
            draw_roi(
                output,
                roi,
                x0,
                row * (args.cell + args.gap),
                width=2,
            )
        boxed_columns += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    print(
        f"saved {args.output} with ROI boxes in {boxed_columns}/{args.columns} columns"
    )


if __name__ == "__main__":
    main()
