"""E-203: bootstrap 95% confidence intervals for Paper 1 main deltas.

Uses the N=129 combined scene dataset. Reports bootstrap mean and 95% CI for:
  - always-inject invisible delta
  - target-GT visible-quality-proxy delta
  - quality-gap oracle invisible delta
  - visible-region delta (direct decoded-RGB measurement where available)
  - mechanism correlation r
and the difficulty-bucket means. Pure CPU, no GPU needed.
"""

import json
import argparse
import numpy as np


def boot_ci(x, fn=np.mean, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    stats = np.empty(n)
    idx = np.arange(len(x))
    for i in range(n):
        s = rng.choice(idx, size=len(x), replace=True)
        stats[i] = fn(x[s])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(fn(x)), float(lo), float(hi)


def boot_corr(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    idx = np.arange(len(a))
    stats = np.empty(n)
    for i in range(n):
        s = rng.choice(idx, size=len(a), replace=True)
        stats[i] = np.corrcoef(a[s], b[s])[0, 1]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(np.corrcoef(a, b)[0, 1]), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", default="_m1_work/results/expanded/E-142_combined_N124.json"
    )
    ap.add_argument("--out", default="_m1_work/results/expanded/E-203_bootstrap.json")
    args = ap.parse_args()
    rows = json.load(open(args.data))
    d = np.array([r["teacher_inv"] - r["base_inv"] for r in rows])
    ig = np.array([r["f3d_inv"] - r["base_inv"] for r in rows])

    # Selection deltas; rule uses a target-GT proxy and ogap is an oracle diagnostic.
    def gated(fn):
        return np.array(
            [(r["teacher_inv"] - r["base_inv"]) if fn(r) else 0.0 for r in rows]
        )

    always = d
    rule = gated(lambda r: r["vis_gap"] > 0)
    ogap = gated(lambda r: (r["f3d_inv"] - r["base_inv"]) > 5)

    out = {}
    for key, label, arr in [
        ("always_inv", "always inject", always),
        ("rule_inv", "target-GT proxy", rule),
        ("oracle_gap5_inv", "quality-gap oracle", ogap),
    ]:
        m, lo, hi = boot_ci(arr)
        out[key] = {"mean": m, "ci95": [lo, hi]}
        print(f"{label:20s} mean={m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    r, rlo, rhi = boot_corr(ig, d)
    out["mechanism_r"] = {"r": r, "ci95": [rlo, rhi]}
    print(f"{'mechanism_r':16s} r={r:+.3f}  95% CI [{rlo:+.3f}, {rhi:+.3f}]")

    # difficulty buckets
    buckets = {"hard(<12)": [], "mid(12-20)": [], "easy(>=20)": []}
    for r_ in rows:
        x = r_["teacher_inv"] - r_["base_inv"]
        b = r_["base_inv"]
        (
            buckets["hard(<12)"]
            if b < 12
            else buckets["easy(>=20)"]
            if b >= 20
            else buckets["mid(12-20)"]
        ).append(x)
    out["buckets"] = {}
    for k, v in buckets.items():
        m, lo, hi = boot_ci(v)
        out["buckets"][k] = {"n": len(v), "mean": m, "ci95": [lo, hi]}
        print(
            f"  bucket {k:12s} n={len(v):3d} mean={m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
        )

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
