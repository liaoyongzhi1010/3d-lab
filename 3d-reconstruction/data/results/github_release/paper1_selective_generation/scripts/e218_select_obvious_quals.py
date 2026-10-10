"""Select visually obvious qualitative examples from full render arrays.

For each disoccluded frame, scan fixed-size windows and rank them by:
- local error reduction relative to GT,
- visible difference between Gen3R and Ours,
- disocclusion coverage,
- GT texture (to avoid flat, uninformative patches).

The script emits individual full-frame + zoom comparison panels and contact
sheets for manual review. It also emits the strongest failure cases, which
illustrate why a future source-observable fallback would be useful; no routing
decision is available for these qualitative examples. Published examples are
mapped back to this ranking in results/E-218_selected_qual_manifest.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Candidate:
    sid: str
    frame: int
    x0: int
    y0: int
    x1: int
    y1: int
    score: float
    psnr_gain: float
    l1_reduction: float
    method_difference: float
    disocc_fraction: float
    texture: float


def normalize(frame: np.ndarray) -> np.ndarray:
    value = np.asarray(frame, dtype=np.float32)
    if value.max() > 1.5:
        value = value / 255.0
    return np.clip(value.transpose(1, 2, 0), 0.0, 1.0)


def integral(values: np.ndarray) -> np.ndarray:
    return np.pad(values.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def rect_sum(ii: np.ndarray, y: int, x: int, size: int) -> float:
    y1, x1 = y + size, x + size
    return float(ii[y1, x1] - ii[y, x1] - ii[y1, x] + ii[y, x])


def texture_map(image: np.ndarray) -> np.ndarray:
    gray = image.mean(-1)
    dx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
    dy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
    return 0.5 * (dx + dy)


def scan_frame(
    sid: str,
    frame_index: int,
    gt: np.ndarray,
    base: np.ndarray,
    ours: np.ndarray,
    disocc: np.ndarray,
    window: int,
    stride: int,
) -> list[Candidate]:
    height, width = gt.shape[:2]
    base_l1 = np.abs(base - gt).mean(-1)
    ours_l1 = np.abs(ours - gt).mean(-1)
    base_mse = ((base - gt) ** 2).mean(-1)
    ours_mse = ((ours - gt) ** 2).mean(-1)
    method_diff = np.abs(base - ours).mean(-1)
    texture = texture_map(gt)

    maps = {
        "mask": integral(disocc.astype(np.float32)),
        "base_l1": integral(base_l1 * disocc),
        "ours_l1": integral(ours_l1 * disocc),
        "base_mse": integral(base_mse * disocc),
        "ours_mse": integral(ours_mse * disocc),
        "difference": integral(method_diff * disocc),
        "texture": integral(texture * disocc),
    }

    candidates: list[Candidate] = []
    for y in range(0, height - window + 1, stride):
        for x in range(0, width - window + 1, stride):
            count = rect_sum(maps["mask"], y, x, window)
            disocc_fraction = count / (window * window)
            if disocc_fraction < 0.12 or count < 512:
                continue

            denominator = max(count, 1.0)
            base_l1_mean = rect_sum(maps["base_l1"], y, x, window) / denominator
            ours_l1_mean = rect_sum(maps["ours_l1"], y, x, window) / denominator
            base_mse_mean = rect_sum(maps["base_mse"], y, x, window) / denominator
            ours_mse_mean = rect_sum(maps["ours_mse"], y, x, window) / denominator
            difference_mean = rect_sum(maps["difference"], y, x, window) / denominator
            texture_mean = rect_sum(maps["texture"], y, x, window) / denominator

            l1_reduction = base_l1_mean - ours_l1_mean
            psnr_gain = 10.0 * math.log10(
                max(base_mse_mean, 1e-10) / max(ours_mse_mean, 1e-10)
            )
            # Scores are in perceptually meaningful 8-bit units. A strong candidate
            # must both change visibly and move toward GT; texture and mask area are
            # mild tie-breakers rather than dominant terms.
            score = (
                255.0 * l1_reduction
                + 0.45 * 255.0 * difference_mean
                + 0.35 * psnr_gain
                + 8.0 * texture_mean
            ) * math.sqrt(disocc_fraction)

            candidates.append(
                Candidate(
                    sid=sid,
                    frame=frame_index,
                    x0=x,
                    y0=y,
                    x1=x + window,
                    y1=y + window,
                    score=float(score),
                    psnr_gain=float(psnr_gain),
                    l1_reduction=float(l1_reduction),
                    method_difference=float(difference_mean),
                    disocc_fraction=float(disocc_fraction),
                    texture=float(texture_mean),
                )
            )
    return candidates


def draw_label(image: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(image)
    width = max(72, 7 * len(text) + 8)
    draw.rectangle((0, 0, width, 18), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))


def image_from_array(array: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))


def make_panel(
    candidate: Candidate,
    gt: np.ndarray,
    base: np.ndarray,
    ours: np.ndarray,
    output: Path,
) -> None:
    box = (candidate.x0, candidate.y0, candidate.x1 - 1, candidate.y1 - 1)
    full_size = 280
    zoom_size = 280
    rows: list[Image.Image] = []
    for name, array in (("GT", gt), ("Gen3R", base), ("Ours", ours)):
        full = image_from_array(array)
        draw = ImageDraw.Draw(full)
        for inset in range(4):
            draw.rectangle(
                (box[0] - inset, box[1] - inset, box[2] + inset, box[3] + inset),
                outline=(255, 220, 0),
            )
        draw_label(full, name)
        full = full.resize((full_size, full_size), Image.Resampling.LANCZOS)

        crop = (
            image_from_array(array)
            .crop(box)
            .resize((zoom_size, zoom_size), Image.Resampling.LANCZOS)
        )
        draw_label(crop, f"{name} zoom")
        row = Image.new("RGB", (full_size + 4 + zoom_size, full_size), "white")
        row.paste(full, (0, 0))
        row.paste(crop, (full_size + 4, 0))
        rows.append(row)

    header_height = 32
    panel = Image.new(
        "RGB", (rows[0].width, header_height + 3 * full_size + 8), "white"
    )
    header = ImageDraw.Draw(panel)
    header.text(
        (4, 7),
        (
            f"{candidate.sid} frame {candidate.frame} | "
            f"local PSNR gain {candidate.psnr_gain:+.2f} dB | "
            f"score {candidate.score:+.2f}"
        ),
        fill=(0, 0, 0),
    )
    for index, row in enumerate(rows):
        panel.paste(row, (0, header_height + index * (full_size + 4)))
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)


def make_contact_sheet(paths: list[Path], output: Path, columns: int = 3) -> None:
    thumbs: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((420, 630), Image.Resampling.LANCZOS)
        thumbs.append(image.copy())
    if not thumbs:
        return
    rows = math.ceil(len(thumbs) / columns)
    cell_width = max(image.width for image in thumbs) + 8
    cell_height = max(image.height for image in thumbs) + 8
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    for index, image in enumerate(thumbs):
        x = (index % columns) * cell_width + 4
        y = (index // columns) * cell_height + 4
        sheet.paste(image, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_dir", required=True)
    parser.add_argument("--data", default="/home/data/gen3r_re10k/re10k")
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=int, default=144)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--top_success", type=int, default=18)
    parser.add_argument("--top_failure", type=int, default=9)
    parser.add_argument("--per_scene", type=int, default=2)
    args = parser.parse_args()

    teacher_dir = Path(args.teacher_dir)
    output_dir = Path(args.out)
    scene_ids = sorted(
        path.name[len("gt_") : -4] for path in teacher_dir.glob("gt_*.npy")
    )
    all_candidates: list[Candidate] = []
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for scene_index, sid in enumerate(scene_ids, start=1):
        gt_array = np.load(teacher_dir / f"gt_{sid}.npy", mmap_mode="r")
        base_array = np.load(teacher_dir / f"baseline_{sid}.npy", mmap_mode="r")
        ours_array = np.load(teacher_dir / f"adaptive2_{sid}.npy", mmap_mode="r")
        visibility = np.load(Path(args.data) / sid / "visibility.npy")
        frame_count = min(
            len(gt_array), len(base_array), len(ours_array), len(visibility)
        )
        for frame_index in range(1, frame_count):
            gt = normalize(gt_array[frame_index])
            base = normalize(base_array[frame_index])
            ours = normalize(ours_array[frame_index])
            mask_image = Image.fromarray(
                ((1.0 - visibility[frame_index]) * 255).astype(np.uint8)
            ).resize((gt.shape[1], gt.shape[0]), Image.Resampling.NEAREST)
            disocc = np.asarray(mask_image) > 127
            if disocc.mean() < 0.01:
                continue
            frame_candidates = scan_frame(
                sid,
                frame_index,
                gt,
                base,
                ours,
                disocc,
                args.window,
                args.stride,
            )
            if frame_candidates:
                all_candidates.append(
                    max(frame_candidates, key=lambda value: value.score)
                )
        print(
            f"[{scene_index}/{len(scene_ids)}] {sid}: cumulative {len(all_candidates)}"
        )

    def diverse_select(
        values: list[Candidate], limit: int, reverse: bool
    ) -> list[Candidate]:
        sorted_values = sorted(values, key=lambda value: value.score, reverse=reverse)
        selected: list[Candidate] = []
        scene_counts: dict[str, int] = {}
        for candidate in sorted_values:
            if scene_counts.get(candidate.sid, 0) >= args.per_scene:
                continue
            selected.append(candidate)
            scene_counts[candidate.sid] = scene_counts.get(candidate.sid, 0) + 1
            if len(selected) >= limit:
                break
        return selected

    successes = diverse_select(
        [candidate for candidate in all_candidates if candidate.psnr_gain > 0.5],
        args.top_success,
        reverse=True,
    )
    failures = diverse_select(
        [candidate for candidate in all_candidates if candidate.psnr_gain < -0.5],
        args.top_failure,
        reverse=False,
    )

    rendered: dict[str, list[Path]] = {"success": [], "failure": []}
    for kind, candidates in (("success", successes), ("failure", failures)):
        for rank, candidate in enumerate(candidates, start=1):
            sid = candidate.sid
            if sid not in arrays:
                arrays[sid] = (
                    np.load(teacher_dir / f"gt_{sid}.npy", mmap_mode="r"),
                    np.load(teacher_dir / f"baseline_{sid}.npy", mmap_mode="r"),
                    np.load(teacher_dir / f"adaptive2_{sid}.npy", mmap_mode="r"),
                )
            gt_array, base_array, ours_array = arrays[sid]
            output = output_dir / kind / f"{rank:02d}_{sid}_f{candidate.frame:02d}.png"
            make_panel(
                candidate,
                normalize(gt_array[candidate.frame]),
                normalize(base_array[candidate.frame]),
                normalize(ours_array[candidate.frame]),
                output,
            )
            rendered[kind].append(output)

    make_contact_sheet(rendered["success"], output_dir / "success_contact.png")
    make_contact_sheet(rendered["failure"], output_dir / "failure_contact.png")
    result = {
        "success": [asdict(candidate) for candidate in successes],
        "failure": [asdict(candidate) for candidate in failures],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ranking.json").write_text(json.dumps(result, indent=2))
    print(
        f"saved {len(successes)} successes and {len(failures)} failures to {output_dir}"
    )


if __name__ == "__main__":
    main()
