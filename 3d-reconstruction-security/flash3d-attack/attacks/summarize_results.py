"""Aggregate all DPC result JSONs into paper-ready tables (pure stdlib, no torch)."""

import json
import os
import glob

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)


def _load(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _delta_row(d, view):
    c = d.get(f"clean_{view}")
    a = d.get(f"attacked_{view}")
    if not c or not a:
        return None
    return {
        "clean_psnr": c["psnr"],
        "att_psnr": a["psnr"],
        "d_psnr": a["psnr"] - c["psnr"],
        "clean_lpips": c["lpips"],
        "att_lpips": a["lpips"],
        "d_lpips": a["lpips"] - c["lpips"],
    }


def print_main(name, label):
    d = _load(name)
    if not d:
        print(f"[{label}] MISSING ({name})")
        return
    print(f"\n=== {label} (n={d.get('n_scenes', '?')}) ===")
    print(f"{'view':10s} {'clean':>8s} {'attack':>8s} {'dPSNR':>8s} {'dLPIPS':>8s}")
    for v in ["src", "tgt5", "tgt10", "tgt_rand"]:
        r = _delta_row(d, v)
        if r:
            print(
                f"{v:10s} {r['clean_psnr']:8.2f} {r['att_psnr']:8.2f} {r['d_psnr']:+8.2f} {r['d_lpips']:+8.3f}"
            )


def print_baseline_compare():
    methods = [
        ("main_random_n100.json", "random"),
        ("main_naivepgd_n100.json", "naive-PGD"),
        ("main_dpc_n100.json", "DPC (ours)"),
    ]
    print("\n=== Baseline comparison (n=100, eps=8/255) ===")
    print(
        f"{'method':12s} {'srcD':>7s} {'tgt5D':>7s} {'tgt10D':>7s} {'trandD':>7s} {'novelAvg':>9s} {'gap(nov-src)':>13s}"
    )
    for name, label in methods:
        d = _load(name)
        if not d:
            print(f"{label:12s} MISSING")
            continue
        src = _delta_row(d, "src")["d_psnr"]
        novel = [_delta_row(d, v)["d_psnr"] for v in ["tgt5", "tgt10", "tgt_rand"]]
        navg = sum(novel) / len(novel)
        gap = abs(navg) - abs(src)
        print(
            f"{label:12s} {src:+7.1f} {novel[0]:+7.1f} {novel[1]:+7.1f} {novel[2]:+7.1f} {navg:+9.1f} {gap:+13.1f}"
        )


def print_defense():
    d = _load("defense_n20.json") or _load("defense_n40.json")
    if not d:
        print("\n=== Defense: MISSING ===")
        return
    print(f"\n=== Defense detector (n={d.get('n_scenes', '?')}) ===")
    print(
        f"AUC={d.get('detector_auc')}  clean_mean={d.get('clean_score_mean'):.4e}  attacked_mean={d.get('attacked_score_mean'):.4e}"
    )


def print_depth_transfer():
    d = _load("depth_transfer_n40.json")
    if not d:
        print("\n=== Depth-transfer (cross-arch): MISSING ===")
        return
    print(
        f"\n=== Depth-transfer via shared UniDepth prior (n={d.get('n_scenes', '?')}) ==="
    )
    print(f"depth AbsRel (attacked vs clean) = {d.get('depth_absrel'):.3f}")
    print(
        f"pixels with >10% depth shift     = {d.get('depth_delta_gt10pct') * 100:.0f}%"
    )
    print(f"source RGB PSNR (attacked vs clean) = {d.get('src_rgb_psnr'):.1f} dB")


def print_blackbox():
    d = _load("blackbox_n5.json") or _load("blackbox_n20.json")
    if not d:
        print("\n=== Black-box: MISSING ===")
        return
    print_main(os.path.basename("blackbox"), "Black-box DPC")
    print(f"\n=== Black-box DPC (n={d.get('n_scenes', '?')}) ===")
    for v in ["src", "tgt5", "tgt10", "tgt_rand"]:
        r = _delta_row(d, v)
        if r:
            print(f"{v:10s} dPSNR={r['d_psnr']:+.2f}")


if __name__ == "__main__":
    print("#" * 60)
    print("DPC RESULTS SUMMARY")
    print("#" * 60)
    print_main("main_dpc_n100.json", "MAIN: DPC eps=8/255")
    print_baseline_compare()
    print("\n=== Epsilon ablation ===")
    for name, lab in [
        ("abl_eps4_n50.json", "eps=4/255"),
        ("main_dpc_n100.json", "eps=8/255"),
        ("dpc_eps16_lam3.json", "eps=16/255"),
    ]:
        d = _load(name)
        if d:
            src = _delta_row(d, "src")["d_psnr"]
            t = [_delta_row(d, v)["d_psnr"] for v in ["tgt5", "tgt10", "tgt_rand"]]
            print(
                f"{lab:12s} srcD={src:+.1f} tgt5D={t[0]:+.1f} tgt10D={t[1]:+.1f} trandD={t[2]:+.1f}"
            )
    print_depth_transfer()
    print_blackbox()
    print_defense()
