"""Confidence/reliability router with calibration and shuffled/constant controls.

The router is a small trainable MLP producing per-primitive reliability in [0,1].
The charter (spec boundary conditions) demands that reliability be:
  * finite and bounded in [0,1],
  * non-degenerate (not all-zero / all-one / constant),
  * calibrated against correctness — and strictly better than a shuffled control.

``calibration_error`` is a simple expected-calibration-style L1 between predicted
reliability and binary correctness; a shuffled label assignment must not do better.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ReliabilityRouter(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 16, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        with torch.no_grad():
            for layer in (self.fc1, self.fc2):
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=gen) * 0.1)
                layer.bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(features))
        return torch.sigmoid(self.fc2(x))


def detect_degenerate(reliability: torch.Tensor, *, tol: float = 1e-4) -> bool:
    r = reliability.detach().reshape(-1)
    if r.numel() == 0:
        return True
    if float(r.max()) <= tol:
        return True
    if float(r.min()) >= 1.0 - tol:
        return True
    if float(r.std()) <= tol:
        return True
    return False


def calibration_error(reliability: torch.Tensor, correct: torch.Tensor) -> float:
    r = reliability.detach().reshape(-1)
    c = correct.detach().reshape(-1).float()
    if r.numel() != c.numel() or r.numel() == 0:
        raise ValueError("reliability and correctness must be same non-zero length")
    return float((r - c).abs().mean())


def shuffled_control(
    reliability: torch.Tensor, correct: torch.Tensor, *, seed: int = 0
) -> float:
    r = reliability.detach().reshape(-1)
    c = correct.detach().reshape(-1).float()
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(c.numel(), generator=gen)
    return calibration_error(r, c[perm])
