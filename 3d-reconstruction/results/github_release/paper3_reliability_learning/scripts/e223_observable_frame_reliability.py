"""E-223: RGB disagreement + oracle-visibility diagnostic.

The predictor combines baseline/Flash3D RGB and camera poses with
``visibility.npy``. VGGT derives that mask from the complete target clip, so it
crosses the single-image inference boundary. Ground truth and teacher renders
are used only for labels and policy evaluation. Evaluation is grouped
leave-one-scene-out over 21 scenes selected by availability in the high-gain
``teacher_npy`` directory; this population is not aligned with E-145b/E-224.
"""

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image


EVIDENCE_TIER = "RGB disagreement + oracle-visibility diagnostic"
VISIBILITY_PROVENANCE = (
    "visibility.npy is derived by VGGT from the complete target clip and is not "
    "available at single-image inference"
)
SELECTION_SOURCE = "/home/data/E-161_gen3r_highgain/teacher_npy"


FEATURE_NAMES = [
    "disagree_inv_mean",
    "disagree_inv_std",
    "disagree_inv_q25",
    "disagree_inv_q50",
    "disagree_inv_q75",
    "disagree_inv_q90",
    "disagree_vis_mean",
    "disagree_vis_std",
    "disagree_vis_q25",
    "disagree_vis_q50",
    "disagree_vis_q75",
    "disagree_vis_q90",
    "base_grad_inv_mean",
    "base_grad_inv_std",
    "base_grad_vis_mean",
    "base_grad_vis_std",
    "f3d_grad_inv_mean",
    "f3d_grad_inv_std",
    "f3d_grad_vis_mean",
    "f3d_grad_vis_std",
    "disocc_frac",
    "cam_translation",
    "cam_rotation_deg",
    "disagree_inv_minus_vis",
    "disagree_inv_vis_ratio",
    "disagree_inv_x_disocc",
    "cam_translation_x_disocc",
    "cam_rotation_x_disocc",
]


def resize_visibility(mask, height, width):
    if mask.shape == (height, width):
        return mask > 0.5
    image = Image.fromarray((mask > 0.5).astype(np.uint8) * 255)
    image = image.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def gradient_magnitude(rgb):
    gray = rgb.mean(axis=0)
    grad_y, grad_x = np.gradient(gray)
    return np.sqrt(grad_x * grad_x + grad_y * grad_y)


def distribution_stats(values):
    quantiles = np.quantile(values, [0.25, 0.5, 0.75, 0.9])
    return [float(values.mean()), float(values.std()), *map(float, quantiles)]


def mean_std(values):
    return float(values.mean()), float(values.std())


def rotation_angle_deg(reference, target):
    relative = reference[:3, :3].T @ target[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def diagnostic_features(baseline, f3d, visibility, camera, source_camera):
    height, width = baseline.shape[-2:]
    visible = resize_visibility(visibility, height, width)
    invisible = ~visible
    disagreement = np.abs(f3d - baseline).mean(axis=0)
    base_grad = gradient_magnitude(baseline)
    f3d_grad = gradient_magnitude(f3d)

    inv_disagreement = distribution_stats(disagreement[invisible])
    vis_disagreement = distribution_stats(disagreement[visible])
    base_inv = mean_std(base_grad[invisible])
    base_vis = mean_std(base_grad[visible])
    f3d_inv = mean_std(f3d_grad[invisible])
    f3d_vis = mean_std(f3d_grad[visible])
    disocc_frac = float(invisible.mean())
    translation = float(np.linalg.norm(camera[:3, 3] - source_camera[:3, 3]))
    rotation = rotation_angle_deg(source_camera, camera)
    inv_mean = inv_disagreement[0]
    vis_mean = vis_disagreement[0]

    values = [
        *inv_disagreement,
        *vis_disagreement,
        *base_inv,
        *base_vis,
        *f3d_inv,
        *f3d_vis,
        disocc_frac,
        translation,
        rotation,
        inv_mean - vis_mean,
        inv_mean / max(vis_mean, 1e-6),
        inv_mean * disocc_frac,
        translation * disocc_frac,
        rotation * disocc_frac,
    ]
    return dict(zip(FEATURE_NAMES, values))


def masked_psnr(prediction, target, mask):
    squared_error = ((prediction - target) ** 2).mean(axis=0)
    mse = float(squared_error[mask].mean())
    return float(10.0 * np.log10(1.0 / max(mse, 1e-10)))


def build_frame_rows(
    scene_id,
    baseline,
    f3d,
    visibility,
    cameras,
    gt,
    teacher,
    min_mask_pixels=100,
):
    frame_count = min(
        len(baseline), len(f3d), len(visibility), len(cameras), len(gt), len(teacher)
    )
    rows = []
    for frame_index in range(1, frame_count):
        height, width = baseline[frame_index].shape[-2:]
        visible = resize_visibility(visibility[frame_index], height, width)
        invisible = ~visible
        if visible.sum() < min_mask_pixels or invisible.sum() < min_mask_pixels:
            continue
        features = diagnostic_features(
            baseline[frame_index],
            f3d[frame_index],
            visibility[frame_index],
            cameras[frame_index],
            cameras[0],
        )
        baseline_psnr = masked_psnr(baseline[frame_index], gt[frame_index], invisible)
        teacher_psnr = masked_psnr(teacher[frame_index], gt[frame_index], invisible)
        gain = teacher_psnr - baseline_psnr
        rows.append(
            {
                "scene_id": scene_id,
                "frame_index": frame_index,
                **features,
                "baseline_invisible_psnr": baseline_psnr,
                "teacher_invisible_psnr": teacher_psnr,
                "teacher_gain_db": gain,
                "label": int(gain > 0.1),
            }
        )
    return rows


def leave_one_scene_out(scene_ids):
    scene_ids = np.asarray(scene_ids)
    for held_out in sorted(set(scene_ids.tolist())):
        test_idx = np.flatnonzero(scene_ids == held_out)
        train_idx = np.flatnonzero(scene_ids != held_out)
        yield held_out, train_idx, test_idx


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def fit_logistic(features, labels, steps=2000, learning_rate=0.15, l2=0.05):
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    normalized = (features - mean) / scale
    design = np.column_stack([normalized, np.ones(len(normalized))])
    weights = np.zeros(design.shape[1])
    penalty = np.r_[np.ones(design.shape[1] - 1), 0.0]
    for _ in range(steps):
        probability = sigmoid(design @ weights)
        gradient = design.T @ (probability - labels) / len(labels)
        gradient += l2 * penalty * weights
        weights -= learning_rate * gradient
    return mean, scale, weights


def predict_logistic(model, features):
    mean, scale, weights = model
    normalized = (features - mean) / scale
    design = np.column_stack([normalized, np.ones(len(normalized))])
    return sigmoid(design @ weights)


def grouped_predictions(features, labels, scene_ids):
    probability = np.zeros(len(labels), dtype=float)
    folds = []
    for held_out, train_idx, test_idx in leave_one_scene_out(scene_ids):
        if set(scene_ids[train_idx]).intersection(scene_ids[test_idx]):
            raise AssertionError("scene leakage in grouped fold")
        model = fit_logistic(features[train_idx], labels[train_idx])
        probability[test_idx] = predict_logistic(model, features[test_idx])
        folds.append(
            {
                "held_out_scene": held_out,
                "train_scenes": int(len(set(scene_ids[train_idx]))),
                "train_frames": int(len(train_idx)),
                "test_frames": int(len(test_idx)),
            }
        )
    return probability, folds


def roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    if not len(positive_scores) or not len(negative_scores):
        return None
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    return float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())


def pr_auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positives)


def policy_stats(gains, decisions):
    decisions = np.asarray(decisions, dtype=bool)
    realized = np.where(decisions, gains, 0.0)
    calls = int(decisions.sum())
    return {
        "threshold": None,
        "mean_gain_db": float(realized.mean()),
        "worst_gain_db": float(realized.min()),
        "teacher_calls": calls,
        "calls_saved": int(len(gains) - calls),
        "calls_saved_fraction": float(1.0 - calls / len(gains)),
    }


def evaluate(rows, probabilities, folds):
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    gains = np.asarray([row["teacher_gain_db"] for row in rows], dtype=float)
    default_decisions = probabilities >= 0.5
    frontier = []
    for threshold in np.linspace(0.05, 0.95, 19):
        stats = policy_stats(gains, probabilities >= threshold)
        stats["threshold"] = float(round(threshold, 2))
        frontier.append(stats)

    always = policy_stats(gains, np.ones(len(rows), dtype=bool))
    oracle_positive = policy_stats(gains, gains > 0.0)
    oracle_label = policy_stats(gains, gains > 0.1)
    default_policy = policy_stats(gains, default_decisions)
    default_policy["threshold"] = 0.5
    return {
        "status": "DONE_WITH_CONCERNS",
        "concern": "Evaluation uses 21 scenes selected from the available high-gain teacher_npy directory; it is not population-aligned and generalization is not established.",
        "protocol": "grouped leave-one-scene-out CV",
        "evidence_tier": EVIDENCE_TIER,
        "selection": "selected, not population-aligned",
        "selection_source": SELECTION_SOURCE,
        "visibility_provenance": VISIBILITY_PROVENANCE,
        "label_definition": "teacher invisible PSNR - baseline invisible PSNR > 0.1 dB",
        "predictor_data_policy": "RGB and camera inputs are combined with oracle visibility from the complete target clip; GT and teacher arrays are label/evaluation-only.",
        "feature_names": FEATURE_NAMES,
        "scene_count": int(len(set(row["scene_id"] for row in rows))),
        "frame_count": int(len(rows)),
        "positive_frame_count": int(labels.sum()),
        "negative_frame_count": int(len(labels) - labels.sum()),
        "positive_fraction": float(labels.mean()),
        "scene_frame_counts": {
            scene_id: int(sum(row["scene_id"] == scene_id for row in rows))
            for scene_id in sorted(set(row["scene_id"] for row in rows))
        },
        "folds": folds,
        "metrics": {
            "accuracy_at_0.5": float((default_decisions == labels).mean()),
            "roc_auc": roc_auc(labels, probabilities),
            "average_precision": pr_auc(labels, probabilities),
        },
        "policies": {
            "always_teacher": always,
            "diagnostic_model_at_0.5": default_policy,
            "oracle_positive_gain": oracle_positive,
            "oracle_label_threshold": oracle_label,
        },
        "threshold_frontier": frontier,
    }


def normalized_array(path):
    array = np.load(path).astype(np.float32)
    if array.max() > 1.5:
        array /= 255.0
    return array


def load_cameras(path):
    with open(path) as handle:
        transforms = json.load(handle)
    return np.asarray(
        [frame["transform_matrix"] for frame in transforms["frames"]], dtype=float
    )


def discover_scene_ids(teacher_dir, evidence_dir, data_root):
    scene_ids = []
    for path in sorted(glob.glob(os.path.join(teacher_dir, "gt_*.npy"))):
        scene_id = os.path.basename(path)[3:-4]
        required = [
            os.path.join(teacher_dir, f"baseline_{scene_id}.npy"),
            os.path.join(teacher_dir, f"adaptive2_{scene_id}.npy"),
            os.path.join(evidence_dir, f"f3d_{scene_id}.npy"),
            os.path.join(data_root, scene_id, "visibility.npy"),
            os.path.join(data_root, scene_id, "transforms.json"),
        ]
        if all(os.path.isfile(item) for item in required):
            scene_ids.append(scene_id)
    return scene_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher-dir", default="/home/data/E-161_gen3r_highgain/teacher_npy"
    )
    parser.add_argument("--evidence-dir", default="/home/data/E-160_highgain_evidence")
    parser.add_argument("--data-root", default="/home/data/gen3r_re10k/re10k")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    scene_ids = discover_scene_ids(args.teacher_dir, args.evidence_dir, args.data_root)
    rows = []
    for scene_id in scene_ids:
        scene_dir = os.path.join(args.data_root, scene_id)
        scene_rows = build_frame_rows(
            scene_id,
            normalized_array(
                os.path.join(args.teacher_dir, f"baseline_{scene_id}.npy")
            ),
            normalized_array(os.path.join(args.evidence_dir, f"f3d_{scene_id}.npy")),
            np.load(os.path.join(scene_dir, "visibility.npy")),
            load_cameras(os.path.join(scene_dir, "transforms.json")),
            normalized_array(os.path.join(args.teacher_dir, f"gt_{scene_id}.npy")),
            normalized_array(
                os.path.join(args.teacher_dir, f"adaptive2_{scene_id}.npy")
            ),
        )
        rows.extend(scene_rows)
        print(f"{scene_id}: {len(scene_rows)} frames", flush=True)

    if len(scene_ids) < 2 or not rows:
        raise RuntimeError("need at least two scenes with valid frames")
    features = np.asarray([[row[name] for name in FEATURE_NAMES] for row in rows])
    labels = np.asarray([row["label"] for row in rows], dtype=float)
    groups = np.asarray([row["scene_id"] for row in rows])
    probabilities, folds = grouped_predictions(features, labels, groups)
    result = evaluate(rows, probabilities, folds)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["metrics"], indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
