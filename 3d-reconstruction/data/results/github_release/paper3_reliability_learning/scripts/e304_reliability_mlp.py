"""V3: train a feed-forward reliability predictor (small MLP) for single-view
generative-prior reliability, comparing deployable feature sets against the
camera-only floor (0.650) and the GT-quality-probe ceiling (0.947).

This is our TRAINING CONTRIBUTION: a learned reliability gate that decides when
to trust the generative prior, using only deployable single-view evidence
(VGGT confidence + camera motion + Flash3D statistics). Same label & grouped
protocol as Paper3, so results are directly comparable.

Protocol: grouped K-fold CV by scene (no scene leakage). Metric: ROC-AUC pooled
over out-of-fold predictions. MLP with early stopping on an inner grouped split.
"""

import json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)
DEVICE = "cpu"

CAM = ["cam_trans", "cam_rot_deg"]
VGGT = [
    "disocc_frac_v",
    "src_conf_in_disocc",
    "src_conf_in_vis",
    "conf_gap_v",
    "src_conf_disocc_p10",
    "g_conf_p10",
    "g_conf_p50",
]
F3D = ["f3d_vis", "f3d_inv", "vis_frac", "vis_gap"]  # GT-derived (ceiling ref)


class MLP(nn.Module):
    def __init__(self, d_in, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def roc_auc(y, s):
    y = np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    c = pos[:, None] - neg[None, :]
    return float((c > 0).mean() + 0.5 * (c == 0).mean())


def train_one(Xtr, ytr, Xva, yva, epochs=300, lr=1e-3, wd=1e-3):
    net = MLP(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    best_auc, best_state, patience, bad = -1, None, 40, 0
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        loss = lossf(net(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
        if ep % 5 == 0:
            net.eval()
            with torch.no_grad():
                va = torch.sigmoid(net(Xva_t)).numpy()
            a = roc_auc(yva, va)
            if a > best_auc:
                best_auc, best_state, bad = (
                    a,
                    {k: v.clone() for k, v in net.state_dict().items()},
                    0,
                )
            else:
                bad += 1
                if bad > patience:
                    break
    if best_state:
        net.load_state_dict(best_state)
    return net


def grouped_kfold(sids, k=5, seed=0):
    scenes = sorted(set(sids))
    rng = np.random.RandomState(seed)
    rng.shuffle(scenes)
    folds = [scenes[i::k] for i in range(k)]
    return folds


def evaluate_featureset(rows, cols, name, k=5):
    sids = np.array([r["sid"] for r in rows])
    y = np.array([float(r["label"]) for r in rows])
    X = np.array(
        [
            [
                float(np.nan_to_num(r.get(c, 0.0), nan=0.0, posinf=60, neginf=-60))
                for c in cols
            ]
            for r in rows
        ]
    )
    folds = grouped_kfold(sids, k)
    oof = np.zeros(len(y))
    for i in range(k):
        test_scenes = set(folds[i])
        val_scenes = set(folds[(i + 1) % k])
        te = np.array([s in test_scenes for s in sids])
        va = np.array([s in val_scenes for s in sids])
        tr = ~te & ~va
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-6, 1, sd)
        Xn = (X - mu) / sd
        net = train_one(Xn[tr], y[tr], Xn[va], y[va])
        net.eval()
        with torch.no_grad():
            oof[te] = torch.sigmoid(
                net(torch.tensor(Xn[te], dtype=torch.float32))
            ).numpy()
    auc = roc_auc(y, oof)
    acc = float(((oof > 0.5).astype(int) == y).mean())
    return {"name": name, "n_feats": len(cols), "roc_auc": auc, "accuracy": acc}


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--frames", default=str(root / "results/E-149_frames_b5678_aug.json")
    )
    ap.add_argument("--pf", default=str(root / "results/E-302_perframe_evidence.json"))
    ap.add_argument("--out", default=str(root / "results/E-304_reliability_mlp.json"))
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
    print(f"frames={len(merged)} scenes={len(set(x['sid'] for x in merged))}")

    results = []
    for cols, name in [
        (CAM, "1_camera_only(floor,MLP)"),
        (VGGT, "2_vggt_only(deployable,MLP)"),
        (CAM + VGGT, "3_camera+vggt(deployable,MLP,KEY)"),
        (F3D, "4_gt_probe(ceiling,MLP)"),
        (CAM + VGGT + F3D, "5_all(MLP)"),
    ]:
        r = evaluate_featureset(merged, cols, name)
        print(
            f"  {r['name']:36s} AUC={r['roc_auc']:.4f} acc={r['accuracy']:.3f} ({r['n_feats']}f)"
        )
        results.append(r)

    json.dump(
        {
            "protocol": "grouped 5-fold CV, MLP, ROC-AUC pooled OOF",
            "label": "teacher_inv-base_inv>0.1dB",
            "n_frames": len(merged),
            "results": results,
        },
        open(args.out, "w"),
        indent=2,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
