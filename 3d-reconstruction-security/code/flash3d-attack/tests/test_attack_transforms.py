"""Unit tests for the standalone flash3d-attack transforms.

Run on server (Flash3D venv):
  cd /root/projects/flash3d && source .venv/bin/activate
  cd /root/flash3d-attack && python -m pytest tests/test_attack_transforms.py -q
"""

import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attacks.attack_transforms import (
    CameraDriftConfig,
    DepthCueTrapConfig,
    apply_camera_preprocess_drift,
    make_depth_cue_trap,
)


def _sample_inputs():
    rgb = torch.linspace(0, 1, 3 * 8 * 10, dtype=torch.float32).reshape(1, 3, 8, 10)
    K0 = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    K0[:, 0, 0] = 100.0
    K0[:, 1, 1] = 120.0
    K0[:, 0, 2] = 5.0
    K0[:, 1, 2] = 4.0
    T = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    T[:, 0, 3] = 0.2
    return {
        ("color", 0, 0): rgb.clone(),
        ("K_src", 0): K0.clone(),
        ("K_tgt", 0): K0.clone(),
        ("K_tgt", 1): K0.clone(),
        ("T_c2w", 0): torch.eye(4, dtype=torch.float32).unsqueeze(0),
        ("T_w2c", 0): torch.eye(4, dtype=torch.float32).unsqueeze(0),
        ("T_c2w", 1): T.clone(),
        ("T_w2c", 1): torch.linalg.inv(T),
    }


def test_camera_preprocess_drift_changes_camera_metadata_not_rgb():
    inputs = _sample_inputs()
    attacked = apply_camera_preprocess_drift(
        inputs,
        CameraDriftConfig(focal_scale=1.04, pp_shift_px=(2.0, -1.5), yaw_deg=1.0),
    )

    assert torch.equal(attacked[("color", 0, 0)], inputs[("color", 0, 0)])
    assert attacked[("K_src", 0)][0, 0, 0] == pytest.approx(104.0)
    assert attacked[("K_src", 0)][0, 1, 1] == pytest.approx(124.8)
    assert attacked[("K_src", 0)][0, 0, 2] == pytest.approx(7.0)
    assert attacked[("K_src", 0)][0, 1, 2] == pytest.approx(2.5)
    assert attacked[("K_tgt", 1)][0, 0, 0] == pytest.approx(104.0)
    assert not torch.equal(attacked[("T_c2w", 1)], inputs[("T_c2w", 1)])


def test_camera_preprocess_drift_does_not_mutate_original_inputs():
    inputs = _sample_inputs()
    original_k = inputs[("K_src", 0)].clone()
    original_pose = inputs[("T_c2w", 1)].clone()

    _ = apply_camera_preprocess_drift(
        inputs,
        CameraDriftConfig(focal_scale=0.97, pp_shift_px=(-3.0, 4.0), yaw_deg=-0.5),
    )

    assert torch.equal(inputs[("K_src", 0)], original_k)
    assert torch.equal(inputs[("T_c2w", 1)], original_pose)


def test_depth_cue_trap_is_structured_bounded_and_deterministic():
    image = torch.full((1, 3, 32, 48), 0.5, dtype=torch.float32)
    cfg = DepthCueTrapConfig(
        strength=0.12, stripe_period=8, shadow_width=9, edge_blur=3
    )

    attacked_a, delta_a = make_depth_cue_trap(image, cfg)
    attacked_b, delta_b = make_depth_cue_trap(image, cfg)

    assert attacked_a.shape == image.shape
    assert delta_a.shape == image.shape
    assert torch.equal(attacked_a, attacked_b)
    assert torch.equal(delta_a, delta_b)
    assert attacked_a.min().item() >= 0.0
    assert attacked_a.max().item() <= 1.0
    assert delta_a.abs().max().item() <= cfg.strength + 1e-6
    assert delta_a.abs().sum().item() > 0.0


def test_depth_cue_trap_preserves_batch_device_and_dtype():
    image = torch.zeros((2, 3, 16, 16), dtype=torch.float64)
    attacked, delta = make_depth_cue_trap(image, DepthCueTrapConfig(strength=0.05))

    assert attacked.dtype == torch.float64
    assert delta.dtype == torch.float64
    assert attacked.device == image.device
    assert delta.device == image.device


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
