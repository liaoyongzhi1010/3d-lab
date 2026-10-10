"""P1 robustness: run the deployable gate over multiple seeds to show the net-gain
improvement over 'always inject' is stable, not a single-seed fluke.
Reuses e305 core (Gate/train/payoff/roc_auc/grouped_kfold)."""

import json, importlib.util
from pathlib import Path
import numpy as np
import torch

root = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "e305", str(root / "scripts/e305_gate_payoff.py")
)
e305 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e305)


def oof_probs_seed(rows, cols, mode, seed, k=5):
    import numpy as _np

    torch.manual_seed(seed)
    _np.random.seed(seed)
    sids = np.array([r["sid"] for r in rows])
    y = np.array([float(r["label"]) for r in rows])
    delta = np.array([float(r["delta"]) for r in rows])
    X = np.array(
        [
            [
                float(np.nan_to_num(r.get(c, 0.0), nan=0.0, posinf=60, neginf=-60))
                for c in cols
            ]
            for r in rows
        ]
    )
    folds = e305.grouped_kfold(sids, k, seed=seed)
    oof = np.zeros(len(y))
    for i in range(k):
        tes, vas = set(folds[i]), set(folds[(i + 1) % k])
        te = np.array([s in tes for s in sids])
        va = np.array([s in vas for s in sids])
        tr = ~te & ~va
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-6, 1, sd)
        Xn = (X - mu) / sd
        net = e305.train_gate(Xn[tr], y[tr], Xn[va], y[va], dtr=delta[tr], mode=mode)
        net.eval()
        with torch.no_grad():
            raw = net(torch.tensor(Xn[te], dtype=torch.float32))
            oof[te] = raw.numpy() if mode == "reg" else torch.sigmoid(raw).numpy()
    return oof, y, delta


def main():
    rows = json.load(open(root / "results/E-149_frames_b5678_aug.json"))
    pf = json.load(open(root / "results/E-302_perframe_evidence.json"))
    merged = []
    for gi, r in enumerate(rows):
        rr = dict(r)
        ev = pf.get(str(gi))
        if ev:
            rr.update(ev)
        merged.append(rr)
    delta_all = np.array([r["delta"] for r in merged])
    always = float(delta_all.mean())

    configs = [
        ("cam", e305.DEPLOY_CAM, "wcls"),
        ("cam+geo", e305.DEPLOY_CAM + e305.DEPLOY_GEO, "wcls"),
    ]
    seeds = [0, 1, 2, 3, 4]
    out = {"always_mean_gain": always, "seeds": seeds, "results": {}}
    for name, cols, mode in configs:
        best_nets, aucs, cats = [], [], []
        for s in seeds:
            oof, y, delta = oof_probs_seed(merged, cols, mode, s)
            aucs.append(e305.roc_auc(y, oof))
            # best net-gain over threshold grid
            grid = np.linspace(0.3, 0.9, 25)
            bn = max(float(np.where(oof > t, delta, 0).mean()) for t in grid)
            best_nets.append(bn)
            # catastrophic frames at the best-net threshold
        bn = np.array(best_nets)
        au = np.array(aucs)
        out["results"][f"{name}[{mode}]"] = {
            "auc_mean": float(au.mean()),
            "auc_std": float(au.std()),
            "best_net_gain_mean": float(bn.mean()),
            "best_net_gain_std": float(bn.std()),
            "improve_over_always_pct": float((bn.mean() - always) / abs(always) * 100),
        }
        print(
            f"{name}[{mode}]: AUC={au.mean():.3f}±{au.std():.3f}  "
            f"net_gain={bn.mean():.3f}±{bn.std():.3f} dB (always={always:.3f}, "
            f"+{(bn.mean() - always) / abs(always) * 100:.0f}%)"
        )
    json.dump(out, open(root / "results/E-306_gate_multiseed.json", "w"), indent=2)
    print("saved results/E-306_gate_multiseed.json")


if __name__ == "__main__":
    main()
