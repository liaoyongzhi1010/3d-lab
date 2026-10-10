"""E-200: publication-grade figures for the three papers (matplotlib, local).

Reads the expanded scene-level dataset (E-142 combined) and the augmented
frame-level dataset (E-149) and renders conference-style figures as PDF+PNG.

Figures:
  Paper 1 (Selective Geometry-Guided Generation):
    fig1_mechanism    scatter: invisible-gap vs actual injection benefit (color=difficulty)
    fig2_difficulty   bars: mean invisible delta by disocclusion-difficulty bucket
    fig3_gate         oracle diagnostic: always/GT-proxy/oracle mean & worst
  Paper 3 (Learning Disocclusion Reliability):
    fig4_frontier     operating frontier: mean vs threshold + worst-case, marked points
    fig5_roc_pr       ROC and PR curves with AUC
    fig6_saved        compute saved (%) vs mean gain trade-off

Usage:
  python e200_paper_figures.py --scene results/E-142_combined_N169.json \
    --out paper/figs
Add ``--frame`` only when regenerating the separate Paper 3 figures.
"""

import os
import json
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)


def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1 / (1 + np.exp(-x))


def fit_logreg(X, y, lr=0.2, steps=5000, l2=0.02):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mean = X.mean(0)
    std = X.std(0)
    std = np.where(std < 1e-3, 1.0, std)
    Xb = np.concatenate([(X - mean) / std, np.ones((len(X), 1))], 1)
    w = np.zeros(Xb.shape[1])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(steps):
            p = sigmoid(Xb @ w)
            w -= lr * (Xb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0])
            w = np.clip(w, -20, 20)
    return {"w": w, "mean": mean, "std": std}


def predict(m, X):
    X = np.asarray(X, float)
    Xb = np.concatenate([(X - m["mean"]) / m["std"], np.ones((len(X), 1))], 1)
    return sigmoid(Xb @ m["w"])


# ---------- Paper 1 figures ----------
def paper1_figs(scene_path, out):
    rows = json.load(open(scene_path))
    d = np.array([r["teacher_inv"] - r["base_inv"] for r in rows])
    invgap = np.array([r["f3d_inv"] - r["base_inv"] for r in rows])
    base_inv = np.array([r["base_inv"] for r in rows])
    N = len(rows)

    # fig1 mechanism
    cc = np.corrcoef(invgap, d)[0, 1]
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    sc = ax.scatter(
        invgap, d, c=base_inv, cmap="viridis", s=48, edgecolor="k", linewidth=0.3
    )
    cb = fig.colorbar(sc)
    cb.set_label("Gen3R baseline invisible PSNR (dB)")
    ax.set_xlabel(r"Flash3D$_{inv}$ $-$ Gen3R$_{inv}$  (invisible-region gap, dB)")
    ax.set_ylabel(r"Actual injection benefit $\Delta$ (dB)")
    ax.set_title(f"Mechanism: gap explains benefit  (r={cc:.3f}, N={N})")
    save(fig, out, "fig1_mechanism")

    # fig2 difficulty buckets
    buckets = {"hard\n(<12dB)": [], "mid\n(12-20dB)": [], "easy\n(>=20dB)": []}
    for r in rows:
        dd = r["teacher_inv"] - r["base_inv"]
        if r["base_inv"] < 12:
            buckets["hard\n(<12dB)"].append(dd)
        elif r["base_inv"] < 20:
            buckets["mid\n(12-20dB)"].append(dd)
        else:
            buckets["easy\n(>=20dB)"].append(dd)
    names = list(buckets)
    means = [np.mean(buckets[n]) for n in names]
    ns = [len(buckets[n]) for n in names]
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    colors = ["tab:green" if v > 0 else "tab:red" for v in means]
    bars = ax.bar(names, means, color=colors, edgecolor="k")
    ax.axhline(0, color="k", lw=0.8)
    for b, n, mn in zip(bars, ns, means):
        ax.text(
            b.get_x() + b.get_width() / 2,
            mn + (0.12 if mn > 0 else -0.3),
            f"n={n}\n{mn:+.2f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylabel(r"Mean invisible-region $\Delta$ (dB)")
    ax.set_title(f"Injection benefit vs disocclusion difficulty (N={N})")
    save(fig, out, "fig2_difficulty")

    # fig3 gate comparison (mean + worst)
    def gate(fn):
        dd = np.array(
            [(r["teacher_inv"] - r["base_inv"]) if fn(r) else 0.0 for r in rows]
        )
        return dd.mean(), dd.min()

    policies = [
        ("always", lambda r: True),
        ("GT proxy\nvis_gap>0", lambda r: r["vis_gap"] > 0),
        ("quality-gap\noracle >0", lambda r: r["f3d_inv"] > r["base_inv"]),
        ("quality-gap\noracle >5", lambda r: (r["f3d_inv"] - r["base_inv"]) > 5),
    ]
    labels = [p[0] for p in policies]
    means = [gate(p[1])[0] for p in policies]
    worst = [gate(p[1])[1] for p in policies]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.bar(x - 0.2, means, 0.4, label="mean $\\Delta$", color="tab:blue", edgecolor="k")
    ax.bar(
        x + 0.2,
        worst,
        0.4,
        label="worst-case $\\Delta$",
        color="tab:red",
        edgecolor="k",
    )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Invisible-region $\Delta$ (dB)")
    ax.set_title(f"Oracle diagnostic: mean vs worst-case (N={N})")
    ax.legend()
    save(fig, out, "fig3_gate")
    return {"mechanism_r": float(cc), "N": N}


# ---------- Paper 3 figures ----------
FEATS = ["f3d_vis", "f3d_inv", "vis_frac", "vis_gap"]


def group_cv(rows, cols):
    scenes = sorted(set(r["sid"] for r in rows))
    prob = np.zeros(len(rows))
    idx = {s: [i for i, r in enumerate(rows) if r["sid"] == s] for s in scenes}

    def fv(r):
        return [
            float(np.nan_to_num(r.get(c, 0.0), posinf=60, neginf=-60)) for c in cols
        ]

    for s in scenes:
        te = idx[s]
        tr = [i for i in range(len(rows)) if rows[i]["sid"] != s]
        m = fit_logreg([fv(rows[i]) for i in tr], [rows[i]["label"] for i in tr])
        p = predict(m, [fv(rows[i]) for i in te])
        for j, i in enumerate(te):
            prob[i] = p[j]
    return prob


def roc_pr(y, s):
    y = np.asarray(y)
    order = np.argsort(-s)
    y = y[order]
    P = y.sum()
    Nn = len(y) - P
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / max(P, 1)
    fpr = fp / max(Nn, 1)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(P, 1)
    return fpr, tpr, rec, prec, float(np.trapz(tpr, fpr)), float(np.trapz(prec, rec))


def paper3_figs(frame_path, out):
    rows = json.load(open(frame_path))
    rows = [r for r in rows if r.get("vis_gap") is not None]
    N = len(rows)
    prob = group_cv(rows, FEATS)
    y = np.array([r["label"] for r in rows], int)
    delta = np.array([r["delta"] for r in rows])

    # fig4 operating frontier
    ths = np.linspace(0.15, 0.95, 40)
    means, worsts, saved = [], [], []
    for t in ths:
        dec = prob > t
        dd = np.where(dec, delta, 0.0)
        means.append(dd.mean())
        worsts.append(dd.min())
        saved.append(100 * (1 - dec.sum() / N))
    always_m = delta.mean()
    rule = np.array([r["vis_gap"] > 0 for r in rows])
    rule_m = np.where(rule, delta, 0.0).mean()
    oracle_m = np.where(delta > 0, delta, 0.0).mean()
    fig, ax1 = plt.subplots(figsize=(6.2, 4.3))
    ax1.plot(ths, means, "o-", color="tab:blue", label="ReliabilityNet mean $\\Delta$")
    ax1.axhline(always_m, color="gray", ls="--", label=f"always ({always_m:+.2f})")
    ax1.axhline(rule_m, color="tab:orange", ls="--", label=f"rule ({rule_m:+.2f})")
    ax1.axhline(oracle_m, color="tab:green", ls=":", label=f"oracle ({oracle_m:+.2f})")
    ax1.set_xlabel("decision threshold")
    ax1.set_ylabel(r"mean invisible $\Delta$ (dB)")
    ax2 = ax1.twinx()
    ax2.plot(ths, worsts, "s--", color="tab:red", alpha=0.6, label="worst-case")
    ax2.set_ylabel("worst-case $\\Delta$ (dB)", color="tab:red")
    ax2.grid(False)
    ax1.set_title(f"Reliability operating frontier (frames={N})")
    ax1.legend(loc="lower left", fontsize=9)
    save(fig, out, "fig4_frontier")

    # fig5 ROC + PR
    fpr, tpr, rec, prec, auc, pra = roc_pr(y, prob)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4.2))
    a.plot(fpr, tpr, color="tab:blue", lw=2)
    a.plot([0, 1], [0, 1], "k--", lw=0.8)
    a.set_xlabel("FPR")
    a.set_ylabel("TPR")
    a.set_title(f"ROC (AUC={auc:.3f})")
    b.plot(rec, prec, color="tab:green", lw=2)
    b.axhline(y.mean(), color="k", ls="--", lw=0.8, label=f"prior={y.mean():.2f}")
    b.set_xlabel("Recall")
    b.set_ylabel("Precision")
    b.set_title(f"PR (AUC={pra:.3f})")
    b.legend()
    save(fig, out, "fig5_roc_pr")

    # fig6 compute saved vs gain
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    ax.plot(saved, means, "o-", color="tab:purple")
    for t, s, m in zip(ths, saved, means):
        if abs(t - 0.5) < 0.02 or abs(t - 0.75) < 0.02:
            ax.annotate(
                f"thr={t:.2f}",
                (s, m),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )
    ax.axhline(
        oracle_m, color="tab:green", ls=":", label=f"oracle mean ({oracle_m:+.2f})"
    )
    ax.set_xlabel("teacher calls saved (%)")
    ax.set_ylabel(r"mean invisible $\Delta$ (dB)")
    ax.set_title(f"Compute saved vs quality (frames={N})")
    ax.legend()
    save(fig, out, "fig6_saved")
    return {"roc_auc": auc, "pr_auc": pra, "frames": N}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--frame")
    ap.add_argument("--out", default="results/paper_figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    s1 = paper1_figs(args.scene, args.out)
    print("Paper1:", s1)
    if args.frame:
        s3 = paper3_figs(args.frame, args.out)
        print("Paper3:", s3)
    print(f"[SAVED] figures (pdf+png) to {args.out}")


if __name__ == "__main__":
    main()
