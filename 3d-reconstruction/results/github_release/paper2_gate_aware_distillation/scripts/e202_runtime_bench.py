"""E-202: runtime benchmark — fast student vs slow generative teacher.

Measures the per-scene wall-clock of the distilled feed-forward student (a single
CNN forward over all target frames) and compares it to the Gen3R teacher's
diffusion cost (measured from logs: ~247 s per scene for a 30-step denoise of the
clip). Produces the conditional quality--speed table for Paper 2.
"""

import time
import argparse
import numpy as np
import torch
import torch.nn as nn


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=48, help="target frames per scene")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument(
        "--teacher_sec", type=float, default=247.0, help="measured teacher sec/scene"
    )
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = TinyStudent().to(dev).eval()
    x = torch.randn(1, 8, args.size, args.size, device=dev)

    # warmup
    with torch.no_grad():
        for _ in range(10):
            net(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.reps):
            net(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    per_frame = (time.time() - t0) / args.reps
    per_scene = per_frame * args.frames

    print(f"device={dev}")
    print(f"student per-frame forward: {per_frame * 1000:.2f} ms")
    print(
        f"student per-scene ({args.frames} frames): {per_scene * 1000:.1f} ms = {per_scene:.3f} s"
    )
    print(f"teacher per-scene (measured): {args.teacher_sec:.1f} s")
    print(f"SPEEDUP: {args.teacher_sec / per_scene:.0f}x")


if __name__ == "__main__":
    main()
