"""Structured non-PGD attacks for feed-forward 3DGS/NVS systems (e.g. Flash3D).

Model-agnostic: operates on Flash3D-style batch dicts or image tensors.
Does NOT touch model weights or target RGB.

Two attacks:
  1. CP-Drift  (apply_camera_preprocess_drift): camera/preprocess metadata attack.
  2. Depth-Cue Trap (make_depth_cue_trap): deterministic monocular geometry-cue attack.

Neither is an iterative pixel-gradient (PGD/FGSM) attack.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CameraDriftConfig:
    """Camera/preprocess drift attack parameters.

    focal_scale changes ray directions globally; pp_shift_px emulates a crop/pad or
    principal-point bookkeeping error; yaw/pitch/roll apply a coherent target-pose relabel.
    """

    focal_scale: float = 1.0
    pp_shift_px: tuple[float, float] = (0.0, 0.0)
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    translation_scale: float = 1.0


@dataclass(frozen=True)
class DepthCueTrapConfig:
    """Natural-looking geometry cue trap parameters.

    Combines a soft diagonal cast shadow, repeated vertical epipolar stripes, and
    boundary halo cues. Deterministic and amplitude-bounded (L-infinity <= strength).
    """

    strength: float = 0.08
    stripe_period: int = 12
    shadow_width: int = 21
    edge_blur: int = 5
    stripe_weight: float = 0.45
    shadow_weight: float = 0.35
    halo_weight: float = 0.20


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    return value


def _rotation_matrix_xyz(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = torch.tensor(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], device=device, dtype=dtype
    )
    rx = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], device=device, dtype=dtype
    )
    rz = torch.tensor(
        [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
    )
    return rz @ ry @ rx


def _apply_k_drift(K: torch.Tensor, cfg: CameraDriftConfig) -> torch.Tensor:
    out = K.clone()
    out[..., 0, 0] = out[..., 0, 0] * cfg.focal_scale
    out[..., 1, 1] = out[..., 1, 1] * cfg.focal_scale
    out[..., 0, 2] = out[..., 0, 2] + cfg.pp_shift_px[0]
    out[..., 1, 2] = out[..., 1, 2] + cfg.pp_shift_px[1]
    return out


def _apply_pose_drift(T_c2w: torch.Tensor, cfg: CameraDriftConfig) -> torch.Tensor:
    out = T_c2w.clone()
    R = _rotation_matrix_xyz(
        cfg.yaw_deg,
        cfg.pitch_deg,
        cfg.roll_deg,
        device=out.device,
        dtype=out.dtype,
    )
    if T_c2w.ndim == 2:
        out[:3, :3] = out[:3, :3] @ R
        out[:3, 3] = out[:3, 3] * cfg.translation_scale
    elif T_c2w.ndim == 3:
        out[:, :3, :3] = out[:, :3, :3] @ R
        out[:, :3, 3] = out[:, :3, 3] * cfg.translation_scale
    else:
        raise ValueError(f"Expected pose tensor with ndim 2 or 3, got {T_c2w.ndim}")
    return out


def apply_camera_preprocess_drift(
    inputs: dict[Any, Any], cfg: CameraDriftConfig
) -> dict[Any, Any]:
    """Return a metadata-attacked COPY of a Flash3D-style input batch.

    RGB tensors are copied but unchanged. Intrinsics (K_src/K_tgt) are coherently
    drifted. Target poses (T_c2w with frame != 0) are relabelled by a small rigid
    drift; matching T_w2c entries are recomputed as the inverse.
    """

    attacked = {key: _clone_value(value) for key, value in inputs.items()}
    for key, value in list(attacked.items()):
        if (
            isinstance(key, tuple)
            and key[0] in {"K_src", "K_tgt"}
            and isinstance(value, torch.Tensor)
        ):
            attacked[key] = _apply_k_drift(value, cfg)

    source_frame = 0
    for key, value in list(attacked.items()):
        if not (
            isinstance(key, tuple)
            and key[0] == "T_c2w"
            and isinstance(value, torch.Tensor)
        ):
            continue
        frame_id = key[1]
        if frame_id == source_frame:
            continue
        drifted = _apply_pose_drift(value, cfg)
        attacked[key] = drifted
        inv_key = ("T_w2c", frame_id)
        if inv_key in attacked and isinstance(attacked[inv_key], torch.Tensor):
            attacked[inv_key] = torch.linalg.inv(drifted.float()).to(
                dtype=drifted.dtype
            )
    return attacked


def _box_blur(x: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return x
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    _, c, _, _ = x.shape
    weight = torch.ones((c, 1, kernel, kernel), device=x.device, dtype=x.dtype)
    weight = weight / float(kernel * kernel)
    return F.conv2d(x, weight, padding=pad, groups=c)


def make_depth_cue_trap(
    image: torch.Tensor, cfg: DepthCueTrapConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a deterministic structured source-image attack.

    Returns `(attacked_image, delta)`. `delta` is bounded by `cfg.strength` in
    L-infinity but is not optimized with gradients. The pattern injects false
    monocular depth cues: repeated vertical correspondence traps, a soft cast-shadow
    diagonal, and halos around image edges.
    """

    if image.ndim != 4:
        raise ValueError(f"Expected BCHW image tensor, got shape {tuple(image.shape)}")
    if cfg.strength < 0:
        raise ValueError("strength must be non-negative")
    b, c, h, w = image.shape
    device, dtype = image.device, image.dtype
    yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1)

    period = max(int(cfg.stripe_period), 2)
    stripe_phase = torch.arange(w, device=device, dtype=dtype).view(1, 1, 1, w)
    stripes = torch.sin(2.0 * math.pi * stripe_phase / float(period))
    stripes = stripes.expand(b, 1, h, w)

    xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w)
    width = max(float(cfg.shadow_width), 1.0) / float(max(h, w)) * 4.0
    diagonal = xx + 0.65 * yy
    shadow = torch.exp(-(diagonal**2) / max(width * width, 1e-6)).expand(b, 1, h, w)
    shadow = -shadow

    gray = image.mean(dim=1, keepdim=True)
    gx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    gy = F.pad(gy, (0, 0, 0, 1))
    edge = (gx.abs() + gy.abs()).clamp(0, 1)
    halo = _box_blur(edge, cfg.edge_blur)
    halo = halo - halo.mean(dim=(2, 3), keepdim=True)

    pattern = (
        cfg.stripe_weight * stripes
        + cfg.shadow_weight * shadow
        + cfg.halo_weight * halo
    )
    pattern = pattern / pattern.abs().amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    delta = (cfg.strength * pattern).expand(b, c, h, w)
    attacked = (image + delta).clamp(0.0, 1.0)
    return attacked, attacked - image
