"""Build the Paper 1 teaser from released qualitative panels only.

The release does not contain raw input or Flash3D images for the selected cases.
This script therefore composes the strongest honest comparison available:
GT, Gen3R, and the injected candidate for three successes and one injected
failure that motivates a future fallback. It never claims a routing decision or
synthesizes or substitutes missing method images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

YELLOW = (255, 214, 0)
DARK = (22, 24, 30)
GREEN = (30, 130, 76)
RED = (180, 45, 45)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def success_rows(path: Path) -> list[Image.Image]:
    image = Image.open(path).convert("RGB")
    header = 32
    row_height = (image.height - header - 8) // 3
    return [
        image.crop(
            (
                0,
                header + i * (row_height + 4),
                image.width,
                header + i * (row_height + 4) + row_height,
            )
        )
        for i in range(3)
    ]


def failure_rows(path: Path) -> list[Image.Image]:
    image = Image.open(path).convert("RGB")
    columns = 6
    rows = 6
    cell_width = image.width // columns
    cell_height = image.height // rows
    # The fourth target has the released yellow ROI and is readable when enlarged.
    column = 3
    return [
        image.crop(
            (
                column * cell_width,
                i * cell_height,
                (column + 1) * cell_width,
                (i + 1) * cell_height,
            )
        )
        for i in range(3)
    ]


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_width, target_height = size
    scale = min(target_width / image.width, target_height / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, "white")
    canvas.paste(
        resized,
        ((target_width - resized.width) // 2, (target_height - resized.height) // 2),
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "paper" / "figs",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    fig_dir = args.fig_dir
    output = args.out or fig_dir / "fig_teaser_aligned.png"

    cases = [
        (
            "Window recovery",
            "+12.58 dB local",
            success_rows(fig_dir / "obvious_cases" / "01_window_recovery.png"),
            False,
        ),
        (
            "Doorway recovery",
            "+12.53 dB local",
            success_rows(fig_dir / "obvious_cases" / "02_doorway_recovery.png"),
            False,
        ),
        (
            "Sofa/window recovery",
            "+5.54 dB local",
            success_rows(fig_dir / "obvious_cases" / "03_sofa_window_recovery.png"),
            False,
        ),
        (
            "Injected failure",
            "Future fallback needed",
            failure_rows(fig_dir / "panel_failure_boxed.png"),
            True,
        ),
    ]

    label_width, cell_width, cell_height = 155, 390, 245
    header_height, footer_height, gap = 82, 66, 8
    width = label_width + len(cases) * cell_width + (len(cases) - 1) * gap
    height = header_height + 3 * cell_height + 2 * gap + footer_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, header_height), fill=DARK)

    for index, (title, metric, rows, rejected) in enumerate(cases):
        x = label_width + index * (cell_width + gap)
        accent = RED if rejected else GREEN
        draw.rectangle((x, 0, x + cell_width, 7), fill=accent)
        draw.text((x + 10, 18), title, fill="white", font=font(22, True))
        draw.text(
            (x + 10, 48),
            metric,
            fill=YELLOW if rejected else (180, 235, 200),
            font=font(18, True),
        )
        for row_index, image in enumerate(rows):
            y = header_height + row_index * (cell_height + gap)
            canvas.paste(fit(image, (cell_width, cell_height)), (x, y))
            draw.rectangle(
                (x, y, x + cell_width - 1, y + cell_height - 1),
                outline=(180, 180, 180),
                width=1,
            )

    labels = ["Ground truth", "Gen3R baseline", "Injected candidate"]
    for index, label in enumerate(labels):
        y = header_height + index * (cell_height + gap)
        draw.text((8, y + cell_height // 2 - 12), label, fill=DARK, font=font(19, True))

    footer_y = height - footer_height
    draw.rectangle((0, footer_y, width, height), fill=(242, 244, 247))
    draw.text(
        (label_width + 10, footer_y + 10),
        "Released GT / Gen3R / injected assets only; input and Flash3D unavailable.",
        fill=DARK,
        font=font(17),
    )
    failure_x = label_width + 3 * (cell_width + gap)
    draw.text(
        (failure_x + 10, footer_y + 34),
        "No routing decision; proposed fallback only",
        fill=RED,
        font=font(17, True),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    print(f"saved {output} ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
