"""Correspondence-aware explicit-3D geometry metrics for white-box 3DGS attacks.

The unit of geometry here is a *primitive* (one Gaussian) with a STABLE id ``(layer, y, x)``
so the same primitive can be tracked between a clean scene and an adversarial scene even
after its 3D position changes. Everything is measured in a COMMON target-camera frame after
removing the global depth scale, because a feed-forward monocular model is only defined up to
a global scale: a pure 2x depth scaling is not geometric damage and must align to ~0.

Conventions (shared with the Flash3D adapter, see flash3d-gaussian-render skill):
  * primitive means live in SOURCE-camera coordinates, shape ``[N, 3]`` (x, y, z), z>0 in front;
  * a target view is reached by a 4x4 target-from-source rigid transform ``T_target_from_source``;
  * ``project_points`` uses a pinhole ``K`` with (u, v) = (fx X/Z + cx, fy Y/Z + cy).

This module has NO Flash3D / CUDA dependency and is fully CPU-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations  # noqa: F401 (kept for potential future use)

import torch

PrimitiveId = tuple


def project_points(
    points_cam: torch.Tensor, K: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project camera-space points ``[N,3]`` through pinhole ``K`` ``[3,3]``.

    Returns ``(pixels[N,2], depth[N])`` where depth is the camera-space z. Points with
    ``z <= 0`` keep their (meaningless) projected coordinates; callers filter them via depth.
    """
    if points_cam.ndim != 2 or points_cam.shape[1] != 3:
        raise ValueError(f"points must be [N,3], got {tuple(points_cam.shape)}")
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    z = points_cam[:, 2]
    eps = 1e-8
    z_safe = torch.where(z.abs() < eps, torch.full_like(z, eps), z)
    u = fx * points_cam[:, 0] / z_safe + cx
    v = fy * points_cam[:, 1] / z_safe + cy
    pixels = torch.stack([u, v], dim=1)
    return pixels, z


def transform_points(
    points_cam: torch.Tensor, T_target_from_source: torch.Tensor
) -> torch.Tensor:
    """Apply a 4x4 target-from-source rigid transform to source-camera points ``[N,3]``."""
    if T_target_from_source.shape != (4, 4):
        raise ValueError(f"T must be [4,4], got {tuple(T_target_from_source.shape)}")
    R = T_target_from_source[:3, :3]
    t = T_target_from_source[:3, 3]
    return points_cam @ R.T + t


def align_scale_median(adv_depth: torch.Tensor, clean_depth: torch.Tensor) -> float:
    """Global scale s minimizing per-point ratio, via the median of adv/clean."""
    _check_same_len(adv_depth, clean_depth)
    ratio = adv_depth / clean_depth.clamp_min(1e-8)
    return float(ratio.median())


def align_scale_least_squares(
    adv_depth: torch.Tensor, clean_depth: torch.Tensor
) -> float:
    """Global scale s minimizing ``|| adv - s*clean ||^2`` (closed form)."""
    _check_same_len(adv_depth, clean_depth)
    num = (adv_depth * clean_depth).sum()
    den = (clean_depth * clean_depth).sum().clamp_min(1e-12)
    return float(num / den)


def _check_same_len(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"length mismatch: {a.shape[0]} vs {b.shape[0]}")


def _match_by_id(
    adv_pts: torch.Tensor,
    clean_pts: torch.Tensor,
    adv_ids: list,
    clean_ids: list,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (adv_matched, clean_matched) aligned by shared primitive id, in clean order."""
    adv_index = {pid: i for i, pid in enumerate(adv_ids)}
    rows_adv = []
    rows_clean = []
    for j, pid in enumerate(clean_ids):
        if pid in adv_index:
            rows_adv.append(adv_index[pid])
            rows_clean.append(j)
    if not rows_clean:
        raise ValueError("no shared primitive ids between clean and adv")
    a = adv_pts[torch.tensor(rows_adv, dtype=torch.long)]
    c = clean_pts[torch.tensor(rows_clean, dtype=torch.long)]
    return a, c


def aligned_point_displacement(
    adv_pts: torch.Tensor,
    clean_pts: torch.Tensor,
    adv_ids: list,
    clean_ids: list,
) -> float:
    """Mean 3D displacement after removing the global depth scale (least squares).

    A pure global scale aligns to ~0; only local (non-uniform) deformation survives.
    """
    a, c = _match_by_id(adv_pts, clean_pts, adv_ids, clean_ids)
    s = align_scale_least_squares(a[:, 2], c[:, 2])
    a_aligned = a / max(s, 1e-8)
    return float((a_aligned - c).norm(dim=1).mean())


def reprojection_displacement(
    adv_pts: torch.Tensor,
    clean_pts: torch.Tensor,
    K: torch.Tensor,
    adv_ids: list,
    clean_ids: list,
) -> float:
    """Mean pixel displacement of matched primitives after global-scale alignment."""
    a, c = _match_by_id(adv_pts, clean_pts, adv_ids, clean_ids)
    s = align_scale_least_squares(a[:, 2], c[:, 2])
    a_aligned = a / max(s, 1e-8)
    pa, _ = project_points(a_aligned, K)
    pc, _ = project_points(c, K)
    return float((pa - pc).norm(dim=1).mean())


@dataclass
class VisibilityResult:
    """Which primitive id is the visible (nearest, in-frame, in-front) one per target pixel."""

    visible_ids: set = field(default_factory=set)
    pixel_to_id: dict = field(default_factory=dict)
    depth_by_id: dict = field(default_factory=dict)


def raster_depth_visibility(
    points_cam: torch.Tensor,
    K: torch.Tensor,
    hw: tuple,
    ids: list,
) -> VisibilityResult:
    """Nearest-depth z-buffer visibility filtering.

    A primitive is a visibility candidate iff it is in front of the camera (z>0) and projects
    inside the ``[H,W]`` frame. When several land on the same integer pixel, only the nearest
    (smallest z) is visible.
    """
    H, W = hw
    pixels, depth = project_points(points_cam, K)
    px = torch.round(pixels[:, 0]).long()
    py = torch.round(pixels[:, 1]).long()
    best_depth: dict = {}
    best_id: dict = {}
    for i, pid in enumerate(ids):
        z = float(depth[i])
        if z <= 0:
            continue
        u = int(px[i])
        v = int(py[i])
        if u < 0 or u >= W or v < 0 or v >= H:
            continue
        key = (u, v)
        if key not in best_depth or z < best_depth[key]:
            best_depth[key] = z
            best_id[key] = pid
    result = VisibilityResult()
    for key, pid in best_id.items():
        result.pixel_to_id[key] = pid
        result.visible_ids.add(pid)
        result.depth_by_id[pid] = best_depth[key]
    return result


def mutual_visibility_correspondences(
    clean_vis: VisibilityResult, adv_vis: VisibilityResult
) -> set:
    """Primitive ids visible in BOTH the clean and adversarial rasterizations."""
    return clean_vis.visible_ids & adv_vis.visible_ids


@dataclass(frozen=True)
class OrderReversalConfig:
    """Predicate thresholds for a valid near/far order reversal (relative depth)."""

    clean_margin_rel: float = 0.05
    adv_reverse_rel: float = 0.02
    max_pairs: int = 8192


@dataclass(frozen=True)
class OrderReversalResult:
    n_eligible_pairs: int
    n_reversed: int
    rate: float | None


@dataclass
class GeometryState:
    """A snapshot of explicit-3D primitives in SOURCE-camera coordinates.

    Attributes:
        means: ``[N,3]`` primitive centers in source-camera coords (z>0 in front).
        primitive_ids: length-N list of stable ids ``(layer, y, x)``.
        K_src: ``[3,3]`` source-camera intrinsics used for source-frame visibility.
    """

    means: torch.Tensor
    primitive_ids: list
    K_src: torch.Tensor

    def __post_init__(self):
        if self.means.ndim != 2 or self.means.shape[1] != 3:
            raise ValueError(f"means must be [N,3], got {tuple(self.means.shape)}")
        if len(self.primitive_ids) != self.means.shape[0]:
            raise ValueError("primitive_ids length must equal number of means")

    def target_points(self, T_target_from_source: torch.Tensor) -> torch.Tensor:
        return transform_points(self.means, T_target_from_source)


def order_reversal_rate(
    clean_state: GeometryState,
    adv_state: GeometryState,
    T_target_from_source: torch.Tensor,
    cfg: OrderReversalConfig,
) -> OrderReversalResult:
    """Fraction of frozen clean-eligible near/far pairs that flip in the adversarial scene.

    Rules (all enforced):
      * eligibility is frozen from the CLEAN scene: a pair is eligible iff the two primitives
        are clean-mutually-present and their relative clean depth gap ``>= clean_margin_rel``;
      * a pair FLIPS iff (a) both primitives still exist (present in adv), and (b) the sign of
        the depth ordering reverses with adv relative gap ``>= adv_reverse_rel``;
      * same-direction motion that preserves order is NOT a flip;
      * a pair whose primitive disappeared in adv is excluded from the numerator but the
        eligible denominator stays frozen (so disappearance cannot inflate the rate);
      * an empty eligible set returns ``rate=None`` (explicit failure, not zero damage).
    """
    clean_t = clean_state.target_points(T_target_from_source)
    adv_t = adv_state.target_points(T_target_from_source)

    clean_depth = {
        pid: float(clean_t[i, 2]) for i, pid in enumerate(clean_state.primitive_ids)
    }
    adv_depth = {
        pid: float(adv_t[i, 2]) for i, pid in enumerate(adv_state.primitive_ids)
    }

    # Clean-present = in front of the camera (z>0).
    clean_ids = [pid for pid, z in clean_depth.items() if z > 0]

    n_clean = len(clean_ids)
    if n_clean < 2:
        return OrderReversalResult(0, 0, None)

    max_sample = min(cfg.max_pairs * 10, n_clean * (n_clean - 1) // 2, 500000)
    import random as _rand

    rng = _rand.Random(42)
    eligible = []
    for _ in range(max_sample):
        a, b = rng.sample(clean_ids, 2)
        za, zb = clean_depth[a], clean_depth[b]
        denom = max(abs(za), abs(zb), 1e-8)
        if abs(za - zb) / denom >= cfg.clean_margin_rel:
            eligible.append((a, b))
        if len(eligible) >= cfg.max_pairs:
            break

    n_eligible = len(eligible)
    if n_eligible == 0:
        return OrderReversalResult(0, 0, None)

    n_reversed = 0
    for a, b in eligible:
        za, zb = clean_depth[a], clean_depth[b]
        # adv must still have both primitives in front of the camera.
        zav = adv_depth.get(a)
        zbv = adv_depth.get(b)
        if zav is None or zbv is None or zav <= 0 or zbv <= 0:
            continue
        clean_sign = 1.0 if za < zb else -1.0  # who is nearer in clean
        adv_sign = 1.0 if zav < zbv else -1.0
        if adv_sign == clean_sign:
            continue  # same order -> not a flip (covers same-direction motion)
        denom = max(abs(zav), abs(zbv), 1e-8)
        if abs(zav - zbv) / denom >= cfg.adv_reverse_rel:
            n_reversed += 1

    return OrderReversalResult(n_eligible, n_reversed, n_reversed / n_eligible)


def local_explosion_rate(
    adv_pts: torch.Tensor,
    clean_pts: torch.Tensor,
    adv_ids: list,
    clean_ids: list,
    lo: float = 0.5,
    hi: float = 2.0,
    min_radial: float = 0.01,
) -> dict:
    """Fraction of primitives whose post-alignment radial ratio is outside ``[lo, hi]``.

    After removing the global depth scale (least-squares), compute the radial distance from
    the optical axis (sqrt(x^2 + y^2)) for each matched primitive. A primitive is "exploded"
    if ``adv_radial / clean_radial`` falls outside ``[lo, hi]``. Primitives with clean radial
    < ``min_radial`` are excluded from the denominator (ratio is ill-defined on-axis).

    Returns a dict with keys ``n_valid``, ``n_exploded``, ``rate``.
    """
    a, c = _match_by_id(adv_pts, clean_pts, adv_ids, clean_ids)
    s = align_scale_least_squares(a[:, 2], c[:, 2])
    a_aligned = a / max(s, 1e-8)
    clean_radial = c[:, :2].norm(dim=1)
    adv_radial = a_aligned[:, :2].norm(dim=1)
    valid_mask = clean_radial >= min_radial
    n_valid = int(valid_mask.sum().item())
    if n_valid == 0:
        return {"n_valid": 0, "n_exploded": 0, "rate": None}
    ratio = adv_radial[valid_mask] / clean_radial[valid_mask].clamp_min(1e-8)
    exploded = ((ratio < lo) | (ratio > hi)).sum().item()
    return {"n_valid": n_valid, "n_exploded": int(exploded), "rate": exploded / n_valid}
