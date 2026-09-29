"""Phase 4: does DEPLOYABLE VGGT evidence lift the reliability AUC above the
camera-only 0.650 floor, toward the GT-quality-probe 0.947 ceiling?

Same protocol as Paper3 (grouped leave-one-scene-out logistic regression, ROC-AUC).
Merges per-scene VGGT source evidence (E-300) into per-frame rows (E-149), then
compares feature sets:

  1. camera-only            : cam_trans, cam_rot_deg                  (deployable floor)
  2. vggt-only              : VGGT source conf stats                  (deployable)
  3. camera + vggt          : union of 1 and 2                        (deployable, KEY)
  4. gt-quality-probe       : f3d_vis,f3d_inv,vis_frac,vis_gap        (NON-deployable ceiling)

Label: teacher_inv - base_inv > 0.1 dB (identical to Paper3).
"""

import json
import argparse
from pathlib import Path
import numpy as np

CAM_FEATS = ["cam_trans", "cam_rot_deg"]
VGGT_FEATS = [
    "conf_mean",
    "conf_std",
    "conf_p10",
    "conf_p50",
    "conf_p90",
    "lowconf_frac",
    "pt_conf_mean",
    "pt_conf_std",
    "pt_conf_p10",
    "pt_conf_p50",
    "pt_conf_p90",
    "pt_lowconf_frac",
]
GT_FEATS = ["f3d_vis", "f3d_inv", "vis_frac", "vis_gap"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def fit_logistic(X, y, lr=0.2, steps=5000, l2=0.02):
    mean = X.mean(0)
    scale = X.std(0)
    scale = np.where(scale < 1e-3, 1.0, scale)
    Xn = (X - mean) / scale
    design = np.column_stack([Xn, np.ones(len(Xn))])
    w = np.zeros(design.shape[1])
    penalty = np.r_[np.ones(design.shape[1] - 1), 0.0]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(steps):
            p = sigmoid(design @ w)
            grad = design.T @ (p - y) / len(y) + l2 * penalty * w
            w -= lr * grad
            w = np.clip(w, -20.0, 20.0)
    return mean, scale, w


def predict(model, X):
    mean, scale, w = model
    Xn = (X - mean) / scale
    return sigmoid(np.column_stack([Xn, np.ones(len(Xn))]) @ w)


def grouped_predictions(X, y, sids):
    prob = np.zeros(len(y))
    for s in sorted(set(sids)):
        te = np.flatnonzero(sids == s)
        tr = np.flatnonzero(sids != s)
        model = fit_logistic(X[tr], y[tr])
        prob[te] = predict(model, X[te])
    return prob


def roc_auc(y, s):
    y = np.asarray(y, int)
    pos = s[y == 1]
    neg = s[y == 0]
    comp = pos[:, None] - neg[None, :]
    return float((comp > 0).mean() + 0.5 * (comp == 0).mean())


def build_matrix(rows, cols):
    return np.asarray(
        [
            [
                float(np.nan_to_num(r.get(c, 0.0), nan=0.0, posinf=60.0, neginf=-60.0))
                for c in cols
            ]
            for r in rows
        ]
    )


def evaluate(rows, cols, name):
    X = build_matrix(rows, cols)
    y = np.asarray([r["label"] for r in rows], float)
    sids = np.asarray([r["sid"] for r in rows])
    prob = grouped_predictions(X, y, sids)
    auc = roc_auc(y, prob)
    acc = float(((prob > 0.5).astype(int) == y).mean())
    return {
        "name": name,
        "n_feats": len(cols),
        "roc_auc": auc,
        "accuracy": acc,
        "feats": cols,
    }


def main():
    root = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path(".")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--frames", default=str(root / "results/E-149_frames_b5678_aug.json")
    )
    ap.add_argument("--vggt", default=str(root / "results/E-300_vggt_evidence.json"))
    ap.add_argument("--out", default=str(root / "results/E-301_vggt_reliability.json"))
    args = ap.parse_args()

    rows = json.load(open(args.frames))
    vggt = json.load(open(args.vggt))

    # merge scene-level VGGT evidence into each frame row
    merged = []
    miss = 0
    for r in rows:
        ev = vggt.get(r["sid"])
        if ev is None:
            miss += 1
            continue
        rr = dict(r)
        rr.update(ev)
        merged.append(rr)
    print(
        f"frames={len(merged)} (dropped {miss} without vggt) scenes={len(set(x['sid'] for x in merged))}"
    )

    results = []
    results.append(evaluate(merged, CAM_FEATS, "1_camera_only(floor)"))
    results.append(evaluate(merged, VGGT_FEATS, "2_vggt_only(deployable)"))
    results.append(
        evaluate(merged, CAM_FEATS + VGGT_FEATS, "3_camera+vggt(deployable,KEY)")
    )
    results.append(evaluate(merged, GT_FEATS, "4_gt_quality_probe(ceiling)"))
    results.append(
        evaluate(merged, GT_FEATS + VGGT_FEATS + CAM_FEATS, "5_all(ceiling+deployable)")
    )

    print("\n=== ROC-AUC comparison (grouped leave-one-scene-out) ===")
    for r in results:
        print(
            f"  {r['name']:38s} AUC={r['roc_auc']:.4f}  acc={r['accuracy']:.3f}  ({r['n_feats']} feats)"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "protocol": "grouped leave-one-scene-out logistic, ROC-AUC",
            "label": "teacher_inv - base_inv > 0.1 dB",
            "n_frames": len(merged),
            "n_scenes": len(set(x["sid"] for x in merged)),
            "results": results,
        },
        open(args.out, "w"),
        indent=2,
    )
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
