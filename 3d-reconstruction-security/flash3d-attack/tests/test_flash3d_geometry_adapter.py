"""Flash3D geometry-adapter tests.

Two tiers:
  * CPU-safe: verify the adapter DECLARES the exact Flash3D conventions (padding, K sizes,
    principal points, znear/zfar, SH degree, offset-after-scale, shift +0.5) as importable
    constants/config. These run anywhere and pin the contract from the flash3d-gaussian-render
    skill so a regression is caught even without the model.
  * SERVER-ONLY (require CUDA + Flash3D repo): renderer-parity and gradient tests that load
    the frozen model and compare the adapter's GeometryState-driven render against Flash3D's
    native forward(). Marked `flash3d` so they are skipped on CPU-only machines.

Run CPU tier locally:  python3 -m pytest tests/test_flash3d_geometry_adapter.py -q
Run full on server:
  cd /root/flash3d-attack && \
  /root/projects/flash3d/.venv/bin/python -m pytest tests/test_flash3d_geometry_adapter.py -q -m ''
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attacks.flash3d_geometry_adapter import (
    FLASH3D_RE10K_CONVENTIONS,
    Flash3DConventions,
)


def _flash3d_available():
    if os.environ.get("FLASH3D_TEST", "0") != "1":
        return False
    try:
        import torch  # noqa: F401

        if not torch.cuda.is_available():
            return False
        sys.path.insert(0, "/root/projects/flash3d")
        import models.model  # noqa: F401

        return True
    except Exception:
        return False


requires_flash3d = pytest.mark.skipif(
    not _flash3d_available(),
    reason="requires CUDA + Flash3D repo (set FLASH3D_TEST=1 on server)",
)


def test_conventions_match_re10k_exactly():
    c = FLASH3D_RE10K_CONVENTIONS
    assert isinstance(c, Flash3DConventions)
    # Unpadded output resolution.
    assert (c.height, c.width) == (256, 384)
    # pad_border_aug=32 -> padded splatter 320 x 448.
    assert c.pad_border_aug == 32
    assert (c.padded_height, c.padded_width) == (320, 448)
    # znear/zfar for RealEstate10K.
    assert c.znear == pytest.approx(0.01)
    assert c.zfar == pytest.approx(100.0)
    # SH degree 1, 2 gaussian layers.
    assert c.max_sh_degree == 1
    assert c.gaussians_per_pixel == 2
    # forward half-pixel shift -> +0.5.
    assert c.shift_rays_half_pixel == pytest.approx(0.5)
    # offset is NOT scaled by depth (scaled_offset=false) -> applied AFTER depth scaling.
    assert c.scaled_offset is False


def test_padded_principal_point_is_center_of_padded_grid():
    c = FLASH3D_RE10K_CONVENTIONS
    # inv_K_src for unprojection uses the PADDED intrinsics: principal point (224,160).
    px, py = c.padded_principal_point()
    assert px == pytest.approx(192.0 + 32.0)  # 224
    assert py == pytest.approx(128.0 + 32.0)  # 160


def test_unpadded_principal_point_is_center_of_output():
    c = FLASH3D_RE10K_CONVENTIONS
    px, py = c.unpadded_principal_point()
    assert px == pytest.approx(192.0)
    assert py == pytest.approx(128.0)


def test_extract_clean_scaled_eval_poses_is_importable():
    from attacks.flash3d_geometry_adapter import extract_clean_scaled_eval_poses

    assert callable(extract_clean_scaled_eval_poses)


@requires_flash3d
def test_validate_renderer_parity_matches_native_forward():
    """Adapter's GeometryState render must match Flash3D forward() within tolerance.

    Source view ~36 dB, nearby novel view ~28 dB per the flash3d-gaussian-render skill;
    anything ~11-18 dB means a projection/padding/coords bug. Hard gate.
    """
    from attacks.flash3d_geometry_adapter import Flash3DGeometryAdapter

    adapter = Flash3DGeometryAdapter.from_default_checkpoint()
    report = adapter.validate_renderer_parity(max_scenes=1)
    assert report["source_psnr"] >= 35.0
    assert report["novel_psnr"] >= 24.0
    assert report["max_abs_diff"] < 0.2


@requires_flash3d
def test_adapter_geometry_state_preserves_primitive_ids_and_grad():
    import torch

    from attacks.flash3d_geometry_adapter import Flash3DGeometryAdapter

    adapter = Flash3DGeometryAdapter.from_default_checkpoint()
    image = adapter.load_example_source()
    image = image.clone().requires_grad_(True)
    state = adapter.build_geometry_state(image)
    N = state.means.shape[0]
    assert len(state.primitive_ids) == N
    # ids are (layer, y, x) with 2 layers over the unpadded 256x384 grid.
    layers = set(i[0] for i in state.primitive_ids)
    assert layers == {0, 1}
    # gradient flows from means back to the input image.
    state.means.sum().backward()
    assert image.grad is not None and image.grad.abs().sum().item() > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
