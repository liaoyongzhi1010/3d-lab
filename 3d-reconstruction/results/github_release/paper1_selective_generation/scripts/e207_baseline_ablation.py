"""E-207: external-baseline ablation table for Paper 1.

Honest positioning: Gen3R and Flash3D are our two *components*, so they appear
as ablation rows (component-alone), NOT as competitors. The table answers:
"given the two feed-forward reconstructors we build on, how does the injected
candidate compare with either component alone and naive fusion?"

Rows (all reported on invisible + visible regions, N scenes, bootstrap 95% CI):
  - Gen3R-alone            (base_inv / base_vis)          -- our diffusion backbone
  - Flash3D-alone          (f3d_inv  / f3d_vis)           -- our per-pixel GS reconstructor
  - Best-of-components      per-scene max(base, f3d)      -- oracle component selector
  - Always-inject          teacher applied every scene    -- naive fusion (no gate)
  - Target-GT visible proxy (vis_gap>0)                    -- oracle diagnostic
  - Quality-gap oracle (gap>5) selects Ours or Gen3R using target GT
  - Benefit oracle (upper bound) inject only if delta>0

Pure CPU. No GPU. Uses E-142_combined_N*.json.
"""

import json
import argparse
import numpy as np


def boot_ci(x, fn=np.mean, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    idx = np.arange(len(x))
    stats = np.empty(n)
    for i in range(n):
        stats[i] = fn(x[rng.choice(idx, size=len(x), replace=True)])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(fn(x)), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", default="_m1_work/results/expanded/E-142_combined_N124.json"
    )
    ap.add_argument(
        "--out", default="_m1_work/results/expanded/E-207_baseline_ablation.json"
    )
    args = ap.parse_args()
    rows = json.load(open(args.data))
    N = len(rows)

    base_inv = np.array([r["base_inv"] for r in rows])
    f3d_inv = np.array([r["f3d_inv"] for r in rows])
    teach_inv = np.array([r["teacher_inv"] for r in rows])
    base_vis = np.array([r["base_vis"] for r in rows])
    f3d_vis = np.array([r["f3d_vis"] for r in rows])
    # No decoded selected-output visible metric is stored at N=166. The output JSON
    # assigns baseline visible PSNR by the latent-mask construction assumption.
    vis_gap = np.array([r["vis_gap"] for r in rows])

    # per-scene absolute invisible PSNR under each policy (teacher applied or base kept)
    def policy_inv(mask):
        return np.where(mask, teach_inv, base_inv)

    always = teach_inv
    rule = policy_inv(vis_gap > 0)
    ogap = policy_inv((f3d_inv - base_inv) > 5)
    oracle = policy_inv((teach_inv - base_inv) > 0)
    best_comp_inv = np.maximum(base_inv, f3d_inv)

    # invisible-region absolute PSNR rows
    inv_rows = {
        "Gen3R-alone (single-view baseline)": base_inv,
        "Flash3D evidence (injection-source ref.)": f3d_inv,
        "Always-inject (naive fusion)": always,
        "Target-GT visible proxy (oracle diagnostic)": rule,
        "Quality-gap oracle (gap>5; selects Ours)": ogap,
        "Benefit oracle (upper bound)": oracle,
    }
    vis_rows = {
        "Gen3R-alone (direct measurement)": base_vis,
        "Selected output (assigned baseline by construction; not measured)": base_vis,
    }

    out = {
        "N": N,
        "invisible_psnr": {},
        "visible_psnr": {},
        "visible_metric_semantics": (
            "Selected-output values are assigned from Gen3R by the latent-mask "
            "construction assumption, not independently measured after decoding. "
            "The direct 16-scene run measured +0.016 dB visible change."
        ),
        "deltas_vs_gen3r": {},
    }
    print(f"=== E-207 external-baseline ablation  (N={N}) ===")
    print("\n-- invisible-region PSNR (absolute) --")
    for k, v in inv_rows.items():
        m, lo, hi = boot_ci(v)
        out["invisible_psnr"][k] = {"mean": m, "ci95": [lo, hi]}
        print(f"  {k:36s} {m:6.3f}  95% CI [{lo:6.3f}, {hi:6.3f}]")

    print("\n-- invisible-region DELTA vs Gen3R-alone --")
    for k, v in inv_rows.items():
        d = v - base_inv
        m, lo, hi = boot_ci(d)
        out["deltas_vs_gen3r"][k] = {"mean": m, "ci95": [lo, hi]}
        print(f"  {k:36s} {m:+6.3f}  95% CI [{lo:+6.3f}, {hi:+6.3f}]")

    print(
        "\n-- visible-region PSNR (construction assignment; selected output not measured) --"
    )
    for k, v in vis_rows.items():
        m, lo, hi = boot_ci(v)
        out["visible_psnr"][k] = {"mean": m, "ci95": [lo, hi]}
        print(f"  {k:36s} {m:6.3f}  95% CI [{lo:6.3f}, {hi:6.3f}]")

    # Win/tie counts for quality-gap oracle selection over Gen3R on invisible.
    d_ours = ogap - base_inv
    wins = int((d_ours > 0.05).sum())
    ties = int((np.abs(d_ours) <= 0.05).sum())
    losses = int((d_ours < -0.05).sum())
    out["quality_gap_oracle_vs_gen3r_counts"] = {
        "win": wins,
        "tie": ties,
        "loss": losses,
    }
    print(
        f"\nQuality-gap oracle (gap>5; selects Ours) vs Gen3R-alone invisible: "
        f"win={wins} tie={ties} loss={losses}"
    )

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
