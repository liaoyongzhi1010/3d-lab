"""E-216: gate-threshold sensitivity ablation (Paper 1).

Oracle diagnostic of sensitivity to the target-GT quality-gap oracle threshold.
We sweep the gap threshold tau in
    inject scene iff (f3d_inv - base_inv) > tau
over the N=166 scene set and report mean invisible-region delta, worst case, and
inject rate. Shows (a) a broad plateau of good thresholds (method is not
knife-edge sensitive), (b) the mean/worst trade-off, and (c) that our default
tau=5 sits near the mean-optimal while keeping worst-case near zero.

Pure CPU, uses E-142_combined_N169.json.
"""

import json
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", default="_m1_work/results/expanded/E-142_combined_N169.json"
    )
    ap.add_argument(
        "--out", default="_m1_work/results/expanded/E-216_gate_sensitivity.json"
    )
    args = ap.parse_args()
    rows = json.load(open(args.data))
    N = len(rows)
    gain = np.array([r["teacher_inv"] - r["base_inv"] for r in rows])
    gap = np.array([r["f3d_inv"] - r["base_inv"] for r in rows])

    out = {"N": N, "sweep": []}
    print(f"=== E-216 quality-gap oracle threshold sensitivity (N={N}) ===")
    print(f"{'tau':>5s} {'mean':>8s} {'worst':>8s} {'inject':>8s} {'inj%':>6s}")
    best = None
    for tau in np.arange(-2, 12.1, 0.5):
        dec = gap > tau
        applied = np.where(dec, gain, 0.0)
        m, w, inj = float(applied.mean()), float(applied.min()), int(dec.sum())
        out["sweep"].append(
            {
                "tau": float(tau),
                "mean": m,
                "worst": w,
                "inject": inj,
                "inj_rate": inj / N,
            }
        )
        if best is None or m > best[1]:
            best = (float(tau), m, w, inj)
        if abs(tau - round(tau)) < 1e-6:
            print(f"{tau:5.1f} {m:+8.3f} {w:+8.3f} {inj:8d} {100 * inj / N:5.0f}%")

    # plateau: thresholds within 0.1 dB of best mean
    means = np.array([s["mean"] for s in out["sweep"]])
    taus = np.array([s["tau"] for s in out["sweep"]])
    plateau = taus[means >= best[1] - 0.1]
    out["best_tau"] = best[0]
    out["best_mean"] = best[1]
    out["plateau_range"] = [float(plateau.min()), float(plateau.max())]
    # worst-case near zero: smallest tau with worst >= -0.5
    safe = [s for s in out["sweep"] if s["worst"] >= -0.5]
    out["safe_tau_min"] = min((s["tau"] for s in safe), default=None)
    print(f"\nbest mean {best[1]:+.3f} dB at tau={best[0]:.1f}")
    print(
        f"plateau (within 0.1 dB of best): tau in [{plateau.min():.1f}, {plateau.max():.1f}]"
    )
    print(
        f"oracle diagnostic tau=5.0 mean={means[np.argmin(np.abs(taus - 5))]:+.3f} dB "
        f"(within {best[1] - means[np.argmin(np.abs(taus - 5))]:.3f} dB of best)"
    )
    print(f"worst-case >= -0.5 dB once tau >= {out['safe_tau_min']}")
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
