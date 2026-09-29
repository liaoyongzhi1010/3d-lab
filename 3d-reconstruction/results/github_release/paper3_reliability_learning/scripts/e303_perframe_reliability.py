"""Phase 4 v2: does PER-FRAME deployable VGGT evidence lift AUC above the
camera-only 0.650 floor? Same Paper3 grouped-LOSO protocol.

Merges E-302 per-frame evidence (keyed by global row index) into E-149 rows.
"""

import json
import argparse
from pathlib import Path
import numpy as np

CAM_FEATS = ["cam_trans", "cam_rot_deg"]
VGGT_PF_FEATS = [
    "disocc_frac_v",
    "src_conf_in_disocc",
    "src_conf_in_vis",
    "conf_gap_v",
    "src_conf_disocc_p10",
    "g_conf_p10",
    "g_conf_p50",
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
        prob[te] = predict(fit_logistic(X[tr], y[tr]), X[te])
    return prob


def roc_auc(y, s):
    y = np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    comp = pos[:, None] - neg[None, :]
    return float((comp > 0).mean() + 0.5 * (comp == 0).mean())


def build(rows, cols):
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
    X = build(rows, cols)
    y = np.asarray([r["label"] for r in rows], float)
    sids = np.asarray([r["sid"] for r in rows])
    prob = grouped_predictions(X, y, sids)
    return {
        "name": name,
        "n_feats": len(cols),
        "roc_auc": roc_auc(y, prob),
        "accuracy": float(((prob > 0.5).astype(int) == y).mean()),
        "feats": cols,
    }


def single_feature_auc(rows, feat):
    """Univariate AUC of one raw feature (sign-agnostic)."""
    y = np.asarray([r["label"] for r in rows], int)
    s = np.asarray([float(np.nan_to_num(r.get(feat, 0.0))) for r in rows])
    a = roc_auc(y, s)
    return max(a, 1 - a)


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--frames", default=str(root / "results/E-149_frames_b5678_aug.json")
    )
    ap.add_argument("--pf", default=str(root / "results/E-302_perframe_evidence.json"))
    ap.add_argument(
        "--out", default=str(root / "results/E-303_perframe_reliability.json")
    )
    args = ap.parse_args()

    rows = json.load(open(args.frames))
    pf = json.load(open(args.pf))

    merged = []
    for gi, r in enumerate(rows):
        ev = pf.get(str(gi))
        if ev is None:
            continue
        rr = dict(r)
        rr.update(ev)
        merged.append(rr)
    print(f"merged frames={len(merged)} scenes={len(set(x['sid'] for x in merged))}")

    # univariate screen of the new features
    print("\n=== univariate AUC of per-frame VGGT features ===")
    for f in VGGT_PF_FEATS + CAM_FEATS:
        print(f"  {f:24s} {single_feature_auc(merged, f):.4f}")

    results = [
        evaluate(merged, CAM_FEATS, "1_camera_only(floor)"),
        evaluate(merged, VGGT_PF_FEATS, "2_vggt_perframe(deployable)"),
        evaluate(merged, CAM_FEATS + VGGT_PF_FEATS, "3_camera+vggt(deployable,KEY)"),
        evaluate(merged, GT_FEATS, "4_gt_quality_probe(ceiling)"),
        evaluate(merged, ["conf_gap_v", "disocc_frac_v"], "2b_vggt_selected(2f)"),
        evaluate(
            merged, CAM_FEATS + ["conf_gap_v", "disocc_frac_v"], "3b_cam+vggt_sel(KEY)"
        ),
    ]
    print("\n=== ROC-AUC (grouped leave-one-scene-out) ===")
    for r in results:
        print(
            f"  {r['name']:34s} AUC={r['roc_auc']:.4f}  acc={r['accuracy']:.3f}  ({r['n_feats']}f)"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "protocol": "grouped leave-one-scene-out logistic, ROC-AUC",
            "label": "teacher_inv - base_inv > 0.1 dB",
            "n_frames": len(merged),
            "n_scenes": len(set(x["sid"] for x in merged)),
            "univariate_auc": {
                f: single_feature_auc(merged, f) for f in VGGT_PF_FEATS + CAM_FEATS
            },
            "results": results,
        },
        open(args.out, "w"),
        indent=2,
    )
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
