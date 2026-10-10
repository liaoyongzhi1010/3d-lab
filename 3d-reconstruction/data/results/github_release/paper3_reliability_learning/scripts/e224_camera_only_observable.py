"""Reproduce the full-frame camera-only observable reliability baseline.

Only camera translation and rotation are predictor inputs. Ground-truth-derived
quality fields and teacher gain are used exclusively for labels and evaluation.
"""

import argparse
import json
from pathlib import Path

import numpy as np


FEATURE_NAMES = ("cam_trans", "cam_rot_deg")
FORBIDDEN_PREDICTOR_FIELDS = {
    "base_inv",
    "teacher_inv",
    "delta",
    "label",
    "f3d_vis",
    "f3d_inv",
    "vis_gap",
    "vis_frac",
    "disocc_frac",
    "f3d_vi_ratio",
    "f3d_vi_prod",
}


def feature_matrix(rows):
    if FORBIDDEN_PREDICTOR_FIELDS.intersection(FEATURE_NAMES):
        raise AssertionError("predictor feature whitelist contains evaluation fields")
    return np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows])


def leave_one_scene_out(scene_ids):
    scene_ids = np.asarray(scene_ids)
    for held_out in sorted(set(scene_ids.tolist())):
        test_idx = np.flatnonzero(scene_ids == held_out)
        train_idx = np.flatnonzero(scene_ids != held_out)
        if set(scene_ids[train_idx]).intersection(scene_ids[test_idx]):
            raise AssertionError("scene leakage in grouped fold")
        yield held_out, train_idx, test_idx


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def fit_logistic(features, labels, learning_rate=0.2, steps=5000, l2=0.02):
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1e-3, 1.0, scale)
    normalized = (features - mean) / scale
    design = np.column_stack([normalized, np.ones(len(normalized))])
    weights = np.zeros(design.shape[1])
    penalty = np.r_[np.ones(design.shape[1] - 1), 0.0]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(steps):
            probability = sigmoid(design @ weights)
            gradient = design.T @ (probability - labels) / len(labels)
            gradient += l2 * penalty * weights
            weights -= learning_rate * gradient
            weights = np.clip(weights, -20.0, 20.0)
    return mean, scale, weights


def predict_logistic(model, features):
    mean, scale, weights = model
    normalized = (features - mean) / scale
    design = np.column_stack([normalized, np.ones(len(normalized))])
    return sigmoid(design @ weights)


def grouped_predictions(features, labels, scene_ids):
    probabilities = np.zeros(len(labels), dtype=float)
    folds = []
    for held_out, train_idx, test_idx in leave_one_scene_out(scene_ids):
        model = fit_logistic(features[train_idx], labels[train_idx])
        probabilities[test_idx] = predict_logistic(model, features[test_idx])
        folds.append(
            {
                "held_out_scene": held_out,
                "train_scenes": int(len(set(scene_ids[train_idx]))),
                "train_frames": int(len(train_idx)),
                "test_frames": int(len(test_idx)),
            }
        )
    return probabilities, folds


def roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    comparisons = positive[:, None] - negative[None, :]
    return float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())


def pr_auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    true_positives = np.cumsum(ranked)
    false_positives = np.cumsum(1 - ranked)
    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall = true_positives / max(ranked.sum(), 1)
    return float(np.trapezoid(precision, recall))


def policy_stats(gains, decisions, threshold):
    decisions = np.asarray(decisions, dtype=bool)
    realized = np.where(decisions, gains, 0.0)
    calls = int(decisions.sum())
    return {
        "threshold": float(threshold),
        "mean_gain_db": float(realized.mean()),
        "worst_gain_db": float(realized.min()),
        "teacher_calls": calls,
        "calls_saved": int(len(gains) - calls),
        "calls_saved_fraction": float(1.0 - calls / len(gains)),
    }


def run(rows):
    features = feature_matrix(rows)
    labels = np.asarray([row["label"] for row in rows], dtype=float)
    gains = np.asarray([row["delta"] for row in rows], dtype=float)
    scene_ids = np.asarray([row["sid"] for row in rows])
    probabilities, folds = grouped_predictions(features, labels, scene_ids)
    default_decisions = probabilities > 0.5
    thresholds = np.linspace(0.2, 0.95, 31)
    return {
        "status": "DONE",
        "protocol": "grouped leave-one-scene-out CV",
        "evidence_tier": "genuinely observable camera-only baseline",
        "label_definition": "teacher invisible PSNR - baseline invisible PSNR > 0.1 dB",
        "feature_names": list(FEATURE_NAMES),
        "excluded_predictor_fields": sorted(FORBIDDEN_PREDICTOR_FIELDS),
        "frame_count": int(len(rows)),
        "scene_count": int(len(set(scene_ids.tolist()))),
        "positive_frame_count": int(labels.sum()),
        "negative_frame_count": int(len(labels) - labels.sum()),
        "folds": folds,
        "metrics": {
            "accuracy": float((default_decisions == labels).mean()),
            "roc_auc": roc_auc(labels, probabilities),
            "pr_auc_trapezoidal": pr_auc(labels, probabilities),
        },
        "policy_at_0.5": policy_stats(gains, default_decisions, 0.5),
        "threshold_frontier": [
            policy_stats(gains, probabilities > threshold, threshold)
            for threshold in thresholds
        ],
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=root / "results/E-149_frames_b5678_aug.json"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.data.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    result = run(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(
        json.dumps(
            {"metrics": result["metrics"], "policy_at_0.5": result["policy_at_0.5"]},
            indent=2,
        )
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
