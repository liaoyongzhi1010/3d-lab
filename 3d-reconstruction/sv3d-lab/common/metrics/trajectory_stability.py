"""Similarity-aligned camera trajectory stability metrics."""

import numpy as np


def _similarity_alignment(source, target):
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = target_zero.T @ source_zero / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(source_zero**2, axis=1))
    if variance <= 0:
        raise ValueError("trajectory positions must not all coincide")
    scale = np.sum(singular_values * np.diag(correction)) / variance
    translation = target_center - scale * (rotation @ source_center)
    return scale, rotation, translation


def _rotation_angle_degrees(rotation):
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def evaluate_trajectory(estimated, reference):
    estimated = np.asarray(estimated, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if (
        estimated.shape != reference.shape
        or estimated.ndim != 3
        or estimated.shape[1:] != (4, 4)
    ):
        raise ValueError(
            "estimated and reference poses must have matching [N,4,4] shape"
        )
    if len(estimated) < 3:
        raise ValueError("trajectory evaluation requires at least three poses")
    scale, alignment_rotation, translation = _similarity_alignment(
        estimated[:, :3, 3], reference[:, :3, 3]
    )
    aligned_positions = (
        scale * (alignment_rotation @ estimated[:, :3, 3].T).T + translation
    )
    position_errors = np.linalg.norm(aligned_positions - reference[:, :3, 3], axis=1)
    rotation_errors = np.array(
        [
            _rotation_angle_degrees(
                reference[index, :3, :3].T
                @ alignment_rotation
                @ estimated[index, :3, :3]
            )
            for index in range(len(estimated))
        ]
    )
    return {
        "ate_rmse": float(np.sqrt(np.mean(position_errors**2))),
        "count": len(estimated),
        "rotation_mean_deg": float(rotation_errors.mean()),
        "rotation_rmse_deg": float(np.sqrt(np.mean(rotation_errors**2))),
        "similarity_scale": float(scale),
    }
