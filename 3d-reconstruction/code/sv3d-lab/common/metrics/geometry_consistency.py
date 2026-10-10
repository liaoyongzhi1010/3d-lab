"""Common-coordinate geometry metrics with mutual visibility and scale alignment."""

import numpy as np


def backproject_depth(depth, intrinsics):
    depth = np.asarray(depth, dtype=float)
    intrinsics = np.asarray(intrinsics, dtype=float)
    if depth.ndim != 2 or intrinsics.shape != (3, 3):
        raise ValueError("depth must be [H,W] and intrinsics must be [3,3]")
    height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
    rays = pixels @ np.linalg.inv(intrinsics).T
    return rays * depth[..., None]


def transform_points(points, transform):
    points = np.asarray(points, dtype=float)
    transform = np.asarray(transform, dtype=float)
    if points.shape[-1] != 3 or transform.shape != (4, 4):
        raise ValueError("points must end in 3 coordinates and transform must be [4,4]")
    homogeneous = np.concatenate((points, np.ones(points.shape[:-1] + (1,))), axis=-1)
    return (homogeneous @ transform.T)[..., :3]


def project_points(points, intrinsics):
    points = np.asarray(points, dtype=float)
    intrinsics = np.asarray(intrinsics, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be [N,3]")
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 0)
    projected = points @ intrinsics.T
    uv = np.full((len(points), 2), np.nan)
    uv[valid] = projected[valid, :2] / projected[valid, 2:3]
    return uv, valid


def sample_depth_nearest(depth, uv):
    depth = np.asarray(depth, dtype=float)
    uv = np.asarray(uv, dtype=float)
    height, width = depth.shape
    finite = np.isfinite(uv).all(axis=1)
    rounded = np.zeros_like(uv, dtype=int)
    rounded[finite] = np.floor(uv[finite] + 0.5).astype(int)
    valid = (
        finite
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    values = np.full(len(uv), np.nan)
    values[valid] = depth[rounded[valid, 1], rounded[valid, 0]]
    valid &= np.isfinite(values) & (values > 0)
    return values, valid


def bidirectional_correspondences(
    source_depth,
    target_depth,
    source_intrinsics,
    target_intrinsics,
    target_from_source,
    *,
    depth_tolerance=1e-3,
    pixel_tolerance=0.51,
):
    source_depth = np.asarray(source_depth, dtype=float)
    target_depth = np.asarray(target_depth, dtype=float)
    source_points = backproject_depth(source_depth, source_intrinsics).reshape(-1, 3)
    target_points = transform_points(source_points, target_from_source)
    target_uv, front = project_points(target_points, target_intrinsics)
    sampled_target_depth, in_target = sample_depth_nearest(target_depth, target_uv)
    depth_valid = np.abs(
        sampled_target_depth - target_points[:, 2]
    ) <= depth_tolerance * np.maximum(1.0, sampled_target_depth)
    candidate = front & in_target & depth_valid

    sampled_target_points = backproject_depth(target_depth, target_intrinsics)
    rounded_target = np.zeros_like(target_uv, dtype=int)
    rounded_target[candidate] = np.floor(target_uv[candidate] + 0.5).astype(int)
    selected_target_points = np.zeros_like(source_points)
    selected_target_points[candidate] = sampled_target_points[
        rounded_target[candidate, 1], rounded_target[candidate, 0]
    ]
    source_from_target = np.linalg.inv(np.asarray(target_from_source, dtype=float))
    roundtrip_points = transform_points(selected_target_points, source_from_target)
    roundtrip_uv, roundtrip_front = project_points(roundtrip_points, source_intrinsics)
    height, width = source_depth.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    source_uv = np.stack((xx, yy), axis=-1).reshape(-1, 2).astype(float)
    mutual = (
        candidate
        & roundtrip_front
        & (np.linalg.norm(roundtrip_uv - source_uv, axis=1) <= pixel_tolerance)
    )
    indices = np.flatnonzero(mutual)
    return {
        "count": len(indices),
        "source_indices": indices,
        "source_points": source_points[indices],
        "source_uv": source_uv[indices],
        "target_points": target_points[indices],
        "target_uv": target_uv[indices],
    }


def estimate_scale(predicted, reference, *, method="median"):
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    valid = (
        np.isfinite(predicted) & np.isfinite(reference) & (np.abs(predicted) > 1e-12)
    )
    if not valid.any():
        raise ValueError("empty valid set for scale alignment")
    if method == "median":
        return float(np.median(reference[valid] / predicted[valid]))
    if method in ("least_squares", "ls"):
        denominator = np.dot(predicted[valid], predicted[valid])
        if denominator <= 0:
            raise ValueError("empty valid set for scale alignment")
        return float(np.dot(predicted[valid], reference[valid]) / denominator)
    raise ValueError(f"unknown scale method: {method}")


def aligned_depth_error(predicted, reference, *, method="median"):
    predicted = np.asarray(predicted, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if predicted.shape != reference.shape:
        raise ValueError("depth arrays must have matching shapes")
    valid = (
        np.isfinite(predicted)
        & np.isfinite(reference)
        & (predicted > 0)
        & (reference > 0)
    )
    if not valid.any():
        raise ValueError("empty valid set for aligned depth error")
    scale = estimate_scale(predicted[valid], reference[valid], method=method)
    errors = np.abs(scale * predicted[valid] - reference[valid])
    return {
        "mae": float(errors.mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "scale": scale,
        "count": int(valid.sum()),
    }


def point_displacement(predicted, reference, *, method="least_squares"):
    predicted = np.asarray(predicted, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if (
        predicted.shape != reference.shape
        or predicted.ndim != 2
        or predicted.shape[1] != 3
    ):
        raise ValueError("point arrays must have matching [N,3] shapes")
    valid = np.isfinite(predicted).all(axis=1) & np.isfinite(reference).all(axis=1)
    if not valid.any():
        raise ValueError("empty valid set for point displacement")
    scale = estimate_scale(predicted[valid], reference[valid], method=method)
    distances = np.linalg.norm(scale * predicted[valid] - reference[valid], axis=1)
    return {
        "mean": float(distances.mean()),
        "rmse": float(np.sqrt(np.mean(distances**2))),
        "scale": scale,
        "count": int(valid.sum()),
    }


def reprojection_displacement(predicted_uv, reference_uv, valid=None):
    predicted_uv = np.asarray(predicted_uv, dtype=float)
    reference_uv = np.asarray(reference_uv, dtype=float)
    if (
        predicted_uv.shape != reference_uv.shape
        or predicted_uv.ndim != 2
        or predicted_uv.shape[1] != 2
    ):
        raise ValueError("reprojection arrays must have matching [N,2] shapes")
    finite = np.isfinite(predicted_uv).all(axis=1) & np.isfinite(reference_uv).all(
        axis=1
    )
    if valid is not None:
        finite &= np.asarray(valid, dtype=bool)
    if not finite.any():
        raise ValueError("empty valid set for reprojection displacement")
    distances = np.linalg.norm(predicted_uv[finite] - reference_uv[finite], axis=1)
    return {"mean": float(distances.mean()), "count": int(finite.sum())}


def construct_order_pairs(reference_depth, *, margin):
    reference_depth = np.asarray(reference_depth, dtype=float).reshape(-1)
    pairs = []
    for left in range(len(reference_depth)):
        for right in range(left + 1, len(reference_depth)):
            difference = reference_depth[right] - reference_depth[left]
            if abs(difference) >= margin:
                pairs.append((left, right) if difference > 0 else (right, left))
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def reversal_rate(reference_depth, candidate_depth, pairs, *, margin):
    reference_depth = np.asarray(reference_depth, dtype=float).reshape(-1)
    candidate_depth = np.asarray(candidate_depth, dtype=float).reshape(-1)
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    if len(pairs) == 0:
        raise ValueError("empty valid set for order reversal")
    near = pairs[:, 0]
    far = pairs[:, 1]
    reference_valid = reference_depth[far] - reference_depth[near] >= margin
    finite = np.isfinite(candidate_depth[near]) & np.isfinite(candidate_depth[far])
    valid = reference_valid & finite
    if not valid.any():
        raise ValueError("empty valid set for order reversal")
    reversals = candidate_depth[near[valid]] - candidate_depth[far[valid]] >= margin
    count = int(valid.sum())
    reversed_count = int(reversals.sum())
    return {"rate": reversed_count / count, "reversals": reversed_count, "count": count}
