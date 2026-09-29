"""E-209: STANDARD full-image metrics (no region split), for comparison with
published methods. PSNR is exactly reconstructable in the MSE domain from the
per-region PSNR + visible fraction we already logged:

    mse = vis_frac * 10^(-vis_psnr/10) + (1-vis_frac) * 10^(-inv_psnr/10)
    full_psnr = -10 log10(mse)          (max pixel value = 1)

This is mathematically exact (PSNR aggregates linearly in the MSE domain),
NOT an approximation. We report standard full-image PSNR for:
  - Gen3R-alone  (single-view generative baseline)
  - Flash3D      (feed-forward geometry expert; note: evidence is scene-folder
                  constructed, an optimistic reference — flagged honestly)
  - Ours         (constructed aggregate: visible assigned from base, invisible=teacher)
  - Quality-gap oracle (target-GT selection between Ours and Gen3R)

Pure CPU. Uses E-142_combined_N169.json (N=166).
"""

import json
import argparse
import numpy as np


def mse_from_psnr(p):
    return 10.0 ** (-np.asarray(p, float) / 10.0)


def full_psnr(vis_psnr, inv_psnr, vis_frac):
    vf = np.asarray(vis_frac, float)
    m = vf * mse_from_psnr(vis_psnr) + (1 - vf) * mse_from_psnr(inv_psnr)
    return -10.0 * np.log10(np.clip(m, 1e-10, None))


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    idx = np.arange(len(x))
    s = np.array([x[rng.choice(idx, len(x), True)].mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", default="_m1_work/results/expanded/E-142_combined_N169.json"
    )
    ap.add_argument(
        "--out", default="_m1_work/results/expanded/E-209_fullimage_psnr.json"
    )
    args = ap.parse_args()
    rows = json.load(open(args.data))
    N = len(rows)

    vf = np.array([r["vis_frac"] for r in rows])
    gen3r = full_psnr([r["base_vis"] for r in rows], [r["base_inv"] for r in rows], vf)
    flash = full_psnr([r["f3d_vis"] for r in rows], [r["f3d_inv"] for r in rows], vf)
    # Constructed aggregate only: assign base_vis because no decoded N=166
    # selected-output visible metric was stored; this is not an RGB measurement.
    ours = full_psnr(
        [r["base_vis"] for r in rows], [r["teacher_inv"] for r in rows], vf
    )
    # Quality-gap oracle: target GT selects Ours where f3d_inv-base_inv > 5.
    ours_gate = np.array(
        [
            full_psnr(
                [r["base_vis"]],
                [
                    r["teacher_inv"]
                    if (r["f3d_inv"] - r["base_inv"]) > 5
                    else r["base_inv"]
                ],
                [r["vis_frac"]],
            )[0]
            for r in rows
        ]
    )

    out = {
        "N": N,
        "mean_vis_frac": float(vf.mean()),
        "metric": "constructed full-image PSNR from stored region metrics",
        "visible_metric_semantics": (
            "Injected/selected visible PSNR is assigned from Gen3R by construction, "
            "not independently measured after decoding; direct N=16 visible change "
            "was +0.016 dB."
        ),
    }
    print(
        f"=== E-209 standard full-image PSNR (N={N}, mean visible frac={vf.mean():.3f}) ===\n"
    )
    for name, arr in [
        ("Gen3R-alone (baseline)", gen3r),
        ("Flash3D (evidence ref.*)", flash),
        ("Ours (always inject)", ours),
        ("Quality-gap oracle (selects Ours)", ours_gate),
    ]:
        m, lo, hi = boot_ci(arr)
        out[name] = {"mean": m, "ci95": [lo, hi]}
        print(f"  {name:28s} {m:6.3f} dB  95% CI [{lo:.3f}, {hi:.3f}]")

    dg = ours_gate - gen3r
    m, lo, hi = boot_ci(dg)
    out["quality_gap_oracle_minus_gen3r"] = {
        "mean": m,
        "ci95": [lo, hi],
        "win": int((dg > 0.01).sum()),
        "n": N,
    }
    print(
        f"\n  Quality-gap oracle - Gen3R full-image: {m:+.3f} dB  "
        f"95% CI [{lo:+.3f}, {hi:+.3f}]  win={int((dg > 0.01).sum())}/{N}"
    )
    print(
        "\n  *Flash3D evidence is scene-folder constructed (sees near-target frames),"
    )
    print(
        "   so its full-image PSNR is an optimistic reference, not a fair single-view baseline."
    )
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
