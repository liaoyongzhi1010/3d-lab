"""Strict render-query API and exact accounting for black-box attacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


class BudgetExceeded(RuntimeError):
    """Raised when a render would exceed the configured query budget."""


@dataclass(frozen=True)
class QueryRecord:
    index: int
    phase: str
    purpose: str
    success: bool
    error: str | None = None


class CountingRenderAPI:
    """Charge exactly one query per attempted single-image, single-pose render."""

    def __init__(self, renderer: Callable, budget: int | None = None):
        if budget is not None and budget < 0:
            raise ValueError("budget must be non-negative")
        self.renderer = renderer
        self.budget = budget
        self.records: list[QueryRecord] = []

    @property
    def queries(self) -> int:
        return len(self.records)

    @property
    def remaining(self) -> int | None:
        return None if self.budget is None else self.budget - self.queries

    def can_afford(self, count: int) -> bool:
        return self.remaining is None or count <= self.remaining

    def render(
        self, image: torch.Tensor, pose: torch.Tensor, *, phase: str, purpose: str
    ) -> torch.Tensor:
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(f"Expected BCHW image with B=1, got {tuple(image.shape)}")
        if pose.shape != (4, 4):
            raise ValueError(
                f"Expected one pose with shape (4, 4), got {tuple(pose.shape)}"
            )
        if not self.can_afford(1):
            raise BudgetExceeded(f"Query budget {self.budget} exhausted")

        index = self.queries + 1
        try:
            result = self.renderer(image, pose)
        except Exception as exc:
            self.records.append(QueryRecord(index, phase, purpose, False, repr(exc)))
            raise
        self.records.append(QueryRecord(index, phase, purpose, True))
        return result

    def clean_references(
        self, image: torch.Tensor, novel_poses: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        source = self.render(
            image,
            torch.eye(4, device=image.device),
            phase="reference",
            purpose="source",
        )
        novel = [
            self.render(image, pose, phase="reference", purpose="novel")
            for pose in novel_poses
        ]
        return source, novel
