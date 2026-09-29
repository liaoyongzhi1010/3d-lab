"""Frozen row, naming, and scene-isolation contracts for MINE evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Set, Tuple


@dataclass(frozen=True)
class MineRow:
    scene: str
    src: int
    tgt5: int
    tgt10: int
    tgt_rand: int


def parse_mine_split(path) -> Tuple[MineRow, ...]:
    rows = []
    for line_number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        parts = raw.split()
        if len(parts) != 5:
            raise ValueError(
                f"malformed MINE split line {line_number}: expected 5 columns"
            )
        scene = parts[0]
        try:
            frames = tuple(int(value) for value in parts[1:])
        except ValueError as error:
            raise ValueError(
                f"malformed MINE split line {line_number}: non-integer frame"
            ) from error
        if not scene or any(frame < 0 for frame in frames):
            raise ValueError(
                f"malformed MINE split line {line_number}: invalid scene or frame"
            )
        rows.append(MineRow(scene, *frames))
    if not rows:
        raise ValueError("malformed MINE split line 1: split is empty")
    return tuple(rows)


def row_export_name(row_index: int, row: MineRow) -> str:
    if row_index < 0:
        raise ValueError("row index must be non-negative")
    return f"{row_index:05d}_{row.scene}_src{row.src}"


def load_scene_ids(path) -> Set[str]:
    return {row.scene for row in parse_mine_split(path)}


def assert_scene_disjoint(splits: Mapping[str, Iterable[str]]) -> None:
    normalized = {name: set(scenes) for name, scenes in splits.items()}
    names = list(normalized)
    overlaps = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            shared = sorted(normalized[left] & normalized[right])
            if shared:
                overlaps.append(f"{left} vs {right}: {', '.join(shared)}")
    if overlaps:
        raise ValueError("scene overlap detected: " + "; ".join(overlaps))
