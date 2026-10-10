"""E-040: scene-level quality probe diagnostic for Paper 3.

The PSNR inputs are computed against GT and test whether teacher usefulness is
diagnostically separable; they are not inference-time predictors. At scene level,
``vis_gap`` is ``f3d_vis - base_vis``. The label is
``adaptive2_inv - baseline_inv > 0.1``. Evaluation uses leave-one-scene-out
cross-validation and compares the fitted diagnostic with fixed and oracle rules.
"""

import json
import argparse
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fit_logreg(X, y, lr=0.1, steps=3000, l2=0.05):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mean = X.mean(0)
    std = X.std(0) + 1e-6
    Xn = (X - mean) / std
    Xb = np.concatenate([Xn, np.ones((len(Xn), 1))], axis=1)
    w = np.zeros(Xb.shape[1])
    for _ in range(steps):
        p = sigmoid(Xb @ w)
        grad = Xb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0]
        w -= lr * grad
    return {"w": w, "mean": mean, "std": std}


def predict(model, X):
    X = np.asarray(X, dtype=float)
    Xn = (X - model["mean"]) / model["std"]
    Xb = np.concatenate([Xn, np.ones((len(Xn), 1))], axis=1)
    return sigmoid(Xb @ model["w"])


def feats(r):
    return [r["vis_gap"], r["f3d_vis"], r["base_vis"], r["vis_frac"] or 0.9]


def eval_policy(rows, decisions, name):
    deltas = []
    for r, use in zip(rows, decisions):
        deltas.append(r["delta"] if use else 0.0)
    deltas = np.asarray(deltas, dtype=float)
    return {
        "name": name,
        "mean": float(deltas.mean()),
        "worst": float(deltas.min()),
        "win": int((deltas > 0.01).sum()),
        "inject": int(np.asarray(decisions).sum()),
        "n": len(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/data/E-032_router_dataset.json")
    ap.add_argument("--out", default="/home/data/E-040_reliability_net.json")
    args = ap.parse_args()
    rows = json.load(open(args.data))
    rows = [r for r in rows if r.get("vis_gap") is not None]

    probs = []
    preds = []
    for i, r in enumerate(rows):
        train = [x for j, x in enumerate(rows) if j != i]
        model = fit_logreg([feats(x) for x in train], [x["label"] for x in train])
        p = float(predict(model, [feats(r)])[0])
        probs.append(p)
        preds.append(p > 0.5)

    y = np.array([r["label"] for r in rows], dtype=int)
    pred_arr = np.array(preds, dtype=int)
    acc = float((pred_arr == y).mean())
    tp = int(((pred_arr == 1) & (y == 1)).sum())
    fp = int(((pred_arr == 1) & (y == 0)).sum())
    fn = int(((pred_arr == 0) & (y == 1)).sum())
    tn = int(((pred_arr == 0) & (y == 0)).sum())

    rule = [r["vis_gap"] > 0 for r in rows]
    always = [True for _ in rows]
    oracle = [r["delta"] > 0 for r in rows]

    evals = [
        eval_policy(rows, always, "always"),
        eval_policy(rows, rule, "rule_vis_gap>0"),
        eval_policy(rows, preds, "ReliabilityNet_LOOCV"),
        eval_policy(rows, oracle, "oracle"),
    ]

    print(f"rows={len(rows)} acc={acc:.3f} TP={tp} FP={fp} FN={fn} TN={tn}")
    print("scene predictions:")
    for r, p, pr in sorted(zip(rows, probs, preds), key=lambda z: z[0]["delta"]):
        print(
            f"{r['sid'][:8]} delta={r['delta']:+.2f} label={r['label']} vis_gap={r['vis_gap']:+.2f} p={p:.3f} pred={int(pr)}"
        )
    print("\npolicy eval:")
    for e in evals:
        print(
            f"{e['name']:20s} mean={e['mean']:+.3f} worst={e['worst']:+.3f} win={e['win']}/{e['n']} inject={e['inject']}/{e['n']}"
        )

    out = {
        "acc": acc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "evals": evals,
        "rows": [
            {**r, "prob": p, "pred": int(pr)} for r, p, pr in zip(rows, probs, preds)
        ],
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
