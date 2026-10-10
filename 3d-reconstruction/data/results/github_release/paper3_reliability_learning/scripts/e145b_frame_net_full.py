"""E-145b: frame-level quality-probe diagnostic with grouped CV.

The released columns f3d_vis, f3d_inv, vis_gap, and their interactions are
PSNR-based quality probes derived using ground truth. They are not available at
inference. Camera motion is observable, but mixing it with these probes does not
make this experiment operational. The reported routing utility is therefore a
diagnostic upper bound, not an operational scheduler result.

Label: teacher_inv - base_inv > 0.1 dB. Protocol: grouped leave-one-scene-out
cross-validation, with all frames from each held-out scene excluded from fitting.
"""

import json
import argparse
import numpy as np

BASE_FEATS = ["f3d_vis", "f3d_inv", "vis_frac", "vis_gap"]
EXT_FEATS = ["cam_trans", "cam_rot_deg", "disocc_frac", "f3d_vi_ratio", "f3d_vi_prod"]


def sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1 / (1 + np.exp(-x))


def fit_logreg(X, y, lr=0.2, steps=5000, l2=0.02):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mean = X.mean(0)
    std = X.std(0)
    std = np.where(std < 1e-3, 1.0, std)
    Xn = (X - mean) / std
    Xb = np.concatenate([Xn, np.ones((len(Xn), 1))], axis=1)
    w = np.zeros(Xb.shape[1])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(steps):
            p = sigmoid(Xb @ w)
            grad = Xb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0]
            w -= lr * grad
            w = np.clip(w, -20.0, 20.0)
    return {"w": w, "mean": mean, "std": std}


def predict(model, X):
    X = np.asarray(X, float)
    Xn = (X - model["mean"]) / model["std"]
    Xb = np.concatenate([Xn, np.ones((len(Xn), 1))], axis=1)
    return sigmoid(Xb @ model["w"])


def feats(r, cols):
    out = []
    for c in cols:
        v = r.get(c, 0.0)
        v = float(np.nan_to_num(v, nan=0.0, posinf=60.0, neginf=-60.0))
        out.append(float(np.clip(v, -1e4, 1e4)))
    return out


def group_cv(rows, cols, thr=0.5):
    scenes = sorted(set(r["sid"] for r in rows))
    prob = np.zeros(len(rows))
    idx_by_scene = {s: [i for i, r in enumerate(rows) if r["sid"] == s] for s in scenes}
    for s in scenes:
        te = idx_by_scene[s]
        tr = [i for i in range(len(rows)) if rows[i]["sid"] != s]
        model = fit_logreg(
            [feats(rows[i], cols) for i in tr], [rows[i]["label"] for i in tr]
        )
        p = predict(model, [feats(rows[i], cols) for i in te])
        for j, i in enumerate(te):
            prob[i] = p[j]
    return prob


def roc_auc(y, s):
    y = np.asarray(y)
    order = np.argsort(-s)
    y = y[order]
    P = y.sum()
    N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = tps / P
    fpr = fps / N
    return float(np.trapezoid(tpr, fpr))


def pr_auc(y, s):
    y = np.asarray(y)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(y.sum(), 1)
    return float(np.trapezoid(prec, rec))


def policy_stats(rows, dec):
    ds = np.array([r["delta"] if d else 0.0 for r, d in zip(rows, dec)], float)
    return float(ds.mean()), float(ds.min()), int((ds > 0.01).sum()), int(np.sum(dec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext", action="store_true", help="use extended features")
    args = ap.parse_args()
    rows = json.load(open(args.data))
    rows = [
        r for r in rows if r.get("vis_gap") is not None and r.get("delta") is not None
    ]
    cols = BASE_FEATS + (EXT_FEATS if args.ext else [])
    have = [c for c in cols if c in rows[0]]
    cols = have
    scenes = sorted(set(r["sid"] for r in rows))
    print(f"frames={len(rows)} scenes={len(scenes)} feats={cols}")

    prob = group_cv(rows, cols)
    y = np.array([r["label"] for r in rows], int)
    preds = prob > 0.5
    acc = float((preds.astype(int) == y).mean())
    auc = roc_auc(y, prob)
    pra = pr_auc(y, prob)
    print(f"acc={acc:.3f} ROC-AUC={auc:.3f} PR-AUC={pra:.3f}")

    n = len(rows)
    rows_pos = int(y.sum())
    print(
        f"\noperating frontier (threshold -> mean, worst, win, inject, teacher-calls-saved):"
    )
    best = None
    for t in np.linspace(0.2, 0.95, 31):
        dec = prob > t
        m, w, win, inj = policy_stats(rows, dec)
        saved = 100.0 * (1 - inj / n)
        tag = ""
        if w >= -1.0 and (best is None or m > best[1]):
            best = (t, m, w, inj)
            tag = " <-risk-averse*"
        if abs(t - 0.5) < 0.02 or abs(t - 0.75) < 0.02 or tag:
            print(
                f"  thr={t:.2f} mean={m:+.3f} worst={w:+.3f} win={win}/{n} inject={inj}/{n} saved={saved:.0f}%{tag}"
            )

    # baselines
    always = [True] * n
    rule = [r["vis_gap"] > 0 for r in rows]
    oracle = [r["delta"] > 0 for r in rows]
    for name, dec in [
        ("always", always),
        ("rule vis_gap>0", rule),
        ("FrameNet@0.5", preds),
        ("oracle", oracle),
    ]:
        m, w, win, inj = policy_stats(rows, dec)
        print(
            f"[{name:16s}] mean={m:+.3f} worst={w:+.3f} win={win}/{n} inject={inj}/{n} saved={100 * (1 - inj / n):.0f}%"
        )

    ig = np.array([r["f3d_inv"] - r["base_inv"] for r in rows])
    dl = np.array([r["delta"] for r in rows])
    mech = float(np.corrcoef(ig, dl)[0, 1])
    print(f"\n[MECHANISM] corr(f3d_inv - base_inv, delta) = {mech:+.3f}")

    json.dump(
        {
            "acc": acc,
            "roc_auc": auc,
            "pr_auc": pra,
            "mechanism": mech,
            "feats": cols,
            "n": n,
            "pos": rows_pos,
            "risk_averse": best,
        },
        open(args.out, "w"),
        indent=2,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
