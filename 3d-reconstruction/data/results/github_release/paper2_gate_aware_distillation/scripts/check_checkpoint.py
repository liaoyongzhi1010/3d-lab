#!/usr/bin/env python3
"""Validate the released TinyStudent checkpoint and architecture."""

import hashlib
from pathlib import Path

import torch
from torch import nn

EXPECTED_PARAMETERS = 46_371
EXPECTED_SHA256 = "1aa66b23bf6d30fe24a55aa7966ff398238a8102ea89efb213b916a375af76f2"
EXPECTED_OUTPUT_SHAPE = (1, 3, 64, 64)


class TinyStudent(nn.Module):
    def __init__(self, in_ch=8, hidden=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    checkpoint = Path(__file__).resolve().parents[1] / "checkpoints" / "student.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Released checkpoint not found: {checkpoint}. "
            "Restore checkpoints/student.pt before running this validation."
        )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"SHA256 mismatch: expected {EXPECTED_SHA256}, got {digest}")

    model = TinyStudent().eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"parameter count mismatch: expected {EXPECTED_PARAMETERS:,}, got {parameter_count:,}"
        )

    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)

    with torch.inference_mode():
        output = model(torch.zeros(1, 8, 64, 64))
    if tuple(output.shape) != EXPECTED_OUTPUT_SHAPE:
        raise RuntimeError(
            f"output shape mismatch: expected {EXPECTED_OUTPUT_SHAPE}, got {tuple(output.shape)}"
        )

    print(f"Parameters: {parameter_count:,}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"SHA256: {digest}")
    print("CHECKPOINT PASS")


if __name__ == "__main__":
    main()
