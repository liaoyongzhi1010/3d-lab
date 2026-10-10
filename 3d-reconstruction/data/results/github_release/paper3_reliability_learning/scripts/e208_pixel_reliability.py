"""E-208: selected-population patch-level quality probe diagnostic.

This historical experiment evaluates within-frame patch discrimination on 21
selected high-gain scenes. It is not population-aligned with the full frame
experiments on 74 scenes, so its metrics are a diagnostic boundary rather than a
controlled granularity or deployment comparison.

Data (all on server, 21 selected high-gain scenes, per-frame npy):
  gt_*, baseline_*, adaptive2_*  in E-161_gen3r_highgain/teacher_npy
  f3d_* (Flash3D evidence)        in E-160_highgain_evidence
  visibility.npy [F,70,70]        in the scene folder

Per (scene, frame, patch) on a GRID x GRID tiling, restricted to patches that
overlap the disocclusion region (visible fraction < VIS_THR):
  candidate RGB-derived quality probe inputs:
     f3d_base_l1     mean |f3d - baseline| in patch   (evidence disagreement)
     f3d_base_l1_std std of the above
     base_grad       mean gradient magnitude of baseline (texture/edge proxy)
     disocc_frac     fraction of disoccluded pixels in patch
     cx, cy          normalized patch center coords
     f3d_energy      mean |f3d| activity
  label:
     1 if teacher reduces MSE-to-GT vs baseline in the patch's disocc pixels.

The RGB disagreement terms can be inference-time inputs only if both baseline
and Flash3D renders are available; visibility is an evaluation artifact in this
release and is not established as an operational input. Grouped CV holds out scenes.
Reported policy gains use GT-derived labels and errors and remain diagnostic.
Pure CPU/numpy; run on the server where the arrays live.
"""

import os
import glob
import json
import argparse
import numpy as np

GRID = 8
VIS_THR = 0.9  # patch counts as disocclusion patch if disocc_frac > 1-VIS_THR


def load(scene_id, teacher_dir, evid_dir):
    def L(p):
        return np.load(p).astype(np.float32)

    g = L(os.path.join(teacher_dir, f"gt_{scene_id}.npy"))
    b = L(os.path.join(teacher_dir, f"baseline_{scene_id}.npy"))
    a = L(os.path.join(teacher_dir, f"adaptive2_{scene_id}.npy"))
    f = L(os.path.join(evid_dir, f"f3d_{scene_id}.npy"))
    for arr in (g, b, a, f):
        if arr.max() > 1.5:
            arr /= 255.0
    return g, b, a, f


def resize_mask(mask, H, W):
    from PIL import Image

    im = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (W, H), Image.Resampling.NEAREST
    )
    return (np.asarray(im).astype(np.float32) / 255.0) > 0.5


def grad_mag(img_gray):
    gy, gx = np.gradient(img_gray)
    return np.sqrt(gx * gx + gy * gy)


def build_rows(scene_id, data_root, teacher_dir, evid_dir):
    g, b, a, f = load(scene_id, teacher_dir, evid_dir)
    vis = np.load(os.path.join(data_root, scene_id, "visibility.npy")).astype(
        np.float32
    )
    F = min(len(g), len(b), len(a), len(f), len(vis))
    _, _, H, W = g.shape
    ph, pw = H // GRID, W // GRID
    rows = []
    for i in range(1, F):
        disocc = ~resize_mask(vis[i], H, W)  # True where disoccluded
        if disocc.sum() < 100:
            continue
        gi, bi, ai, fi = g[i], b[i], a[i], f[i]
        base_gray = bi.mean(0)
        gmag = grad_mag(base_gray)
        l1 = np.abs(fi - bi).mean(0)  # [H,W] evidence disagreement
        fen = np.abs(fi).mean(0)
        base_err = ((bi - gi) ** 2).mean(0)  # [H,W]
        adap_err = ((ai - gi) ** 2).mean(0)
        for r in range(GRID):
            for c in range(GRID):
                ys, xs = r * ph, c * pw
                ye, xe = ys + ph, xs + pw
                dm = disocc[ys:ye, xs:xe]
                dfrac = float(dm.mean())
                if dfrac <= (1 - VIS_THR):
                    continue
                # metrics restricted to disocc pixels in patch
                be = base_err[ys:ye, xs:xe][dm].mean()
                ae = adap_err[ys:ye, xs:xe][dm].mean()
                rows.append(
                    {
                        "sid": scene_id,
                        "frame": i,
                        "f3d_base_l1": float(l1[ys:ye, xs:xe].mean()),
                        "f3d_base_l1_std": float(l1[ys:ye, xs:xe].std()),
                        "base_grad": float(gmag[ys:ye, xs:xe].mean()),
                        "disocc_frac": dfrac,
                        "cx": (c + 0.5) / GRID,
                        "cy": (r + 0.5) / GRID,
                        "f3d_energy": float(fen[ys:ye, xs:xe].mean()),
                        "base_mse": float(be),
                        "adap_mse": float(ae),
                        "label": int(ae < be),  # teacher improves this patch
                    }
                )
    return rows


FEATS = [
    "f3d_base_l1",
    "f3d_base_l1_std",
    "base_grad",
    "disocc_frac",
    "cx",
    "cy",
    "f3d_energy",
]


def standardize(X):
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    return (X - mu) / sd, mu, sd


def fit_logreg(X, y, iters=400, lr=0.3, l2=1e-3):
    n, d = X.shape
    w = np.zeros(d)
    bset = 0.0
    for _ in range(iters):
        z = X @ w + bset
        p = 1 / (1 + np.exp(-z))
        gw = X.T @ (p - y) / n + l2 * w
        gb = (p - y).mean()
        w -= lr * gw
        bset -= lr * gb
    return w, bset


def predict(X, w, b):
    return 1 / (1 + np.exp(-(X @ w + b)))


def roc_auc(y, p):
    order = np.argsort(-p)
    y = y[order]
    P = y.sum()
    N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / P
    fpr = fp / N
    return float(np.trapezoid(tpr, fpr))


def pr_auc(y, p):
    order = np.argsort(-p)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (y.sum() + 1e-9)
    return float(np.trapezoid(prec, rec))


def psnr_gain(rows, decide):
    # mean per-patch PSNR gain in disocc pixels when we inject only where decide=1
    gains = []
    for r, d in zip(rows, decide):
        chosen_mse = r["adap_mse"] if d else r["base_mse"]
        g = 10 * np.log10(1.0 / max(chosen_mse, 1e-10)) - 10 * np.log10(
            1.0 / max(r["base_mse"], 1e-10)
        )
        gains.append(g)
    return float(np.mean(gains))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/home/data/gen3r_re10k/re10k")
    ap.add_argument(
        "--teacher_dir", default="/home/data/E-161_gen3r_highgain/teacher_npy"
    )
    ap.add_argument("--evid_dir", default="/home/data/E-160_highgain_evidence")
    ap.add_argument("--out", default="/home/data/E-208_pixel_reliability.json")
    args = ap.parse_args()

    scene_ids = sorted(
        os.path.basename(p)[3:-4]
        for p in glob.glob(os.path.join(args.teacher_dir, "gt_*.npy"))
    )
    all_rows = []
    for sid in scene_ids:
        try:
            rows = build_rows(sid, args.data_root, args.teacher_dir, args.evid_dir)
            all_rows.extend(rows)
            print(f"{sid[:20]} patches={len(rows)}")
        except Exception as e:
            print(f"{sid[:20]} ERR {e}")
    n = len(all_rows)
    pos = sum(r["label"] for r in all_rows)
    scenes = sorted(set(r["sid"] for r in all_rows))
    print(
        f"\ntotal patches={n} scenes={len(scenes)} positive={pos} ({100 * pos / n:.1f}%)"
    )

    X = np.array([[r[f] for f in FEATS] for r in all_rows], float)
    y = np.array([r["label"] for r in all_rows], int)

    # grouped CV by scene
    prob = np.zeros(n)
    for s in scenes:
        te = np.array([r["sid"] == s for r in all_rows])
        tr = ~te
        Xs, mu, sd = standardize(X[tr])
        w, b = fit_logreg(Xs, y[tr].astype(float))
        prob[te] = predict((X[te] - mu) / sd, w, b)

    preds = prob > 0.5
    acc = float((preds.astype(int) == y).mean())
    auc = roc_auc(y, prob)
    pra = pr_auc(y, prob)
    print(f"\nPATCH-LEVEL grouped-CV: acc={acc:.3f} ROC-AUC={auc:.3f} PR-AUC={pra:.3f}")

    always = np.ones(n, bool)
    oracle = y.astype(bool)
    print("\nselective-injection (patch-level disocc PSNR gain):")
    print(f"  always inject   {psnr_gain(all_rows, always):+.3f} dB")
    for t in [0.5, 0.6, 0.7]:
        dec = prob > t
        print(
            f"  PatchNet@{t:.1f}    {psnr_gain(all_rows, dec):+.3f} dB  inject={int(dec.sum())}/{n} ({100 * dec.mean():.0f}%)"
        )
    print(
        f"  oracle          {psnr_gain(all_rows, oracle):+.3f} dB  inject={int(oracle.sum())}/{n}"
    )

    out = {
        "n_patches": n,
        "scenes": len(scenes),
        "positive": pos,
        "grid": GRID,
        "vis_thr": VIS_THR,
        "acc": acc,
        "roc_auc": auc,
        "pr_auc": pra,
        "gain_always": psnr_gain(all_rows, always),
        "gain_patchnet@0.5": psnr_gain(all_rows, prob > 0.5),
        "gain_oracle": psnr_gain(all_rows, oracle),
        "feats": FEATS,
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
