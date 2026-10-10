"""Paper1 P1-2: train a deployable RISK-AVERSE gate for single-view generative
prior injection. Unlike V1/V3 (which only reported AUC), the gate is evaluated
by its ACTUAL PAYOFF: net dB gain, worst-case dB, and #catastrophic frames,
against always/never/oracle policies. This is Paper1's positive training
contribution: even an imperfect gate can secure large net gain by cutting
catastrophic injections.

Deployable features (NO GT-derived quality probes):
  - cam_trans, cam_rot_deg            (camera motion, known at inference)
  - disocc_frac                       (target disocclusion fraction from feed-forward visibility)
  - VGGT source evidence (E-302)      (single-view geometric confidence)

Label: delta = teacher_inv - base_inv; positive = injection helps.
Protocol: grouped 5-fold CV by scene. Gate = MLP -> P(helpful). Decision by a
risk-averse threshold tau (inject only if P>tau). Report payoff frontier.
"""

import json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)
DEVICE = "cpu"

DEPLOY_CAM = ["cam_trans", "cam_rot_deg"]
DEPLOY_GEO = ["disocc_frac"]  # feed-forward visibility fraction (deployable)
DEPLOY_VGGT = [
    "disocc_frac_v",
    "src_conf_in_disocc",
    "src_conf_in_vis",
    "conf_gap_v",
    "src_conf_disocc_p10",
    "g_conf_p10",
    "g_conf_p50",
]


class Gate(nn.Module):
    def __init__(self, d, h=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1)
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


def grouped_kfold(sids, k=5, seed=0):
    scenes = sorted(set(sids))
    rng = np.random.RandomState(seed)
    rng.shuffle(scenes)
    return [scenes[i::k] for i in range(k)]


def train_gate(Xtr, ytr, Xva, yva, epochs=400, dtr=None, mode="cls"):
    """mode: 'cls' = BCE on helpful label; 'reg' = regress delta (risk-sensitive:
    predict expected gain, gate by predicted gain>0 -> naturally avoids large losses);
    'wcls' = |delta|-weighted BCE (penalize misclassifying high-magnitude frames)."""
    net = Gate(Xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-3)
    Xtr_t, ytr_t, Xva_t = map(
        lambda a: torch.tensor(a, dtype=torch.float32), (Xtr, ytr, Xva)
    )
    if mode == "reg":
        dtr_t = torch.tensor(dtr, dtype=torch.float32)
        lossf = nn.SmoothL1Loss()
    elif mode == "wcls":
        w = torch.tensor(np.clip(np.abs(dtr), 0.1, 10.0), dtype=torch.float32)
        lossf = nn.BCEWithLogitsLoss(reduction="none")
    else:
        lossf = nn.BCEWithLogitsLoss()
    best, bstate, bad = -1, None, 0
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        out = net(Xtr_t)
        if mode == "reg":
            loss = lossf(out, dtr_t)
        elif mode == "wcls":
            loss = (lossf(out, ytr_t) * w).mean()
        else:
            loss = lossf(out, ytr_t)
        loss.backward()
        opt.step()
        if ep % 5 == 0:
            net.eval()
            with torch.no_grad():
                sc = (
                    net(Xva_t).numpy()
                    if mode == "reg"
                    else torch.sigmoid(net(Xva_t)).numpy()
                )
            a = roc_auc(yva, sc)
            if a > best:
                best, bstate, bad = (
                    a,
                    {k: v.clone() for k, v in net.state_dict().items()},
                    0,
                )
            else:
                bad += 1
                if bad > 50:
                    break
    if bstate:
        net.load_state_dict(bstate)
    return net


def oof_probs(rows, cols, k=5, mode="cls"):
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
    folds = grouped_kfold(sids, k)
    oof = np.zeros(len(y))
    for i in range(k):
        tes, vas = set(folds[i]), set(folds[(i + 1) % k])
        te = np.array([s in tes for s in sids])
        va = np.array([s in vas for s in sids])
        tr = ~te & ~va
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-6, 1, sd)
        Xn = (X - mu) / sd
        net = train_gate(Xn[tr], y[tr], Xn[va], y[va], dtr=delta[tr], mode=mode)
        net.eval()
        with torch.no_grad():
            raw = net(torch.tensor(Xn[te], dtype=torch.float32))
            oof[te] = raw.numpy() if mode == "reg" else torch.sigmoid(raw).numpy()
    return oof, y


def payoff(delta, decisions):
    realized = np.where(decisions, delta, 0.0)
    return {
        "mean_gain": float(realized.mean()),
        "worst": float(realized.min()),
        "n_inject": int(decisions.sum()),
        "inject_frac": float(decisions.mean()),
        "n_catastrophic(<-3)": int((realized < -3).sum()),
    }


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--frames", default=str(root / "results/E-149_frames_b5678_aug.json")
    )
    ap.add_argument("--pf", default=str(root / "results/E-302_perframe_evidence.json"))
    ap.add_argument("--out", default=str(root / "results/E-305_gate_payoff.json"))
    args = ap.parse_args()

    rows = json.load(open(args.frames))
    pf = json.load(open(args.pf))
    merged = []
    for gi, r in enumerate(rows):
        ev = pf.get(str(gi))
        rr = dict(r)
        if ev:
            rr.update(ev)
        merged.append(rr)
    delta = np.array([r["delta"] for r in merged])
    print(f"frames={len(merged)} scenes={len(set(r['sid'] for r in merged))}")

    # reference policies
    ref = {
        "always": payoff(delta, np.ones(len(delta), bool)),
        "never": {
            "mean_gain": 0.0,
            "worst": 0.0,
            "n_inject": 0,
            "inject_frac": 0.0,
            "n_catastrophic(<-3)": 0,
        },
        "oracle(delta>0)": payoff(delta, delta > 0),
    }
    print("\n=== reference policies ===")
    for k, v in ref.items():
        print(
            f"  {k:16s} mean={v['mean_gain']:+.3f} worst={v['worst']:+.3f} inj={v['inject_frac']:.2f} cat={v['n_catastrophic(<-3)']}"
        )

    feature_sets = {
        "cam": DEPLOY_CAM,
        "cam+geo": DEPLOY_CAM + DEPLOY_GEO,
        "cam+geo+vggt": DEPLOY_CAM + DEPLOY_GEO + DEPLOY_VGGT,
    }
    out = {"reference": ref, "gates": {}}
    for name, cols in feature_sets.items():
        for mode in ("cls", "wcls", "reg"):
            oof, y = oof_probs(merged, cols, mode=mode)
            auc = roc_auc(y, oof)
            key = f"{name}[{mode}]"
            thr_grid = (
                np.linspace(-0.5, 3.0, 30)
                if mode == "reg"
                else np.linspace(0.3, 0.9, 25)
            )
            best_ra, best_net = None, None
            for tau in thr_grid:
                pf_stat = payoff(delta, oof > tau)
                if pf_stat["worst"] >= -1.0 and (
                    best_ra is None or pf_stat["mean_gain"] > best_ra["mean_gain"]
                ):
                    best_ra = {"tau": float(tau), **pf_stat}
                if best_net is None or pf_stat["mean_gain"] > best_net["mean_gain"]:
                    best_net = {"tau": float(tau), **pf_stat}
            out["gates"][key] = {
                "n_feats": len(cols),
                "roc_auc": auc,
                "best_net_gain": best_net,
                "risk_averse_best": best_ra,
            }
            print(f"\n=== gate [{key}] AUC={auc:.4f} ===")
            print(
                f"  best-net: tau={best_net['tau']:.2f} mean={best_net['mean_gain']:+.3f} worst={best_net['worst']:+.3f} inj={best_net['inject_frac']:.2f} cat={best_net['n_catastrophic(<-3)']}"
            )
            if best_ra:
                print(
                    f"  risk-averse(worst>=-1): tau={best_ra['tau']:.2f} mean={best_ra['mean_gain']:+.3f} worst={best_ra['worst']:+.3f} inj={best_ra['inject_frac']:.2f} cat={best_ra['n_catastrophic(<-3)']}"
                )
            else:
                print("  risk-averse(worst>=-1): none reached")

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
