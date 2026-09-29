"""E-031b: image-level selective disocclusion student trained on FULL per-frame
teacher renders (npy), scaling up the Paper-2 MVP (E-030) beyond 5 keyframes.

Data (produced by Gen3R runner with --save_teacher_npy):
  teacher_dir/adaptive2_<sid>.npy  [F,3,H,W]  teacher (Paper-1 selective output)
  teacher_dir/gt_<sid>.npy         [F,3,H,W]  ground truth
Flash3D evidence:
  f3d_dir/f3d_<sid>.npy            [F,3,560,560]
Visibility (computed offline from the full target clip):
  data/<sid>/visibility.npy        [F,70,70]   (sid keeps train_/test_ prefix)

Student input per frame (image plane at --size):
  concat[f3d_rgb(3), baseline_rgb(3), vis(1), inv(1)] = 8 ch
  where baseline_rgb is the required Gen3R baseline render. Missing baseline data
  is an error: ground truth is supervision only and is never substituted as input.

This is an oracle-curated, oracle-visibility upper-bound experiment. It does not
provide a deployable single-view selector or an online visibility estimator.

Loss:
  L = L1(out, teacher)|inv  + w_gt*L1(out, gt)|inv + w_id*L1(out, base)|vis
  oracle-selected: invisible loss weighted by offline frame label from --frame_dataset.

This is the scaled, submission-oriented version of the validated MVP. Geometry
is NOT modified here (that is the separately-tested, negative Gaussian-color
result); this student operates at the image level by design and is compared
against the Gaussian-color adapter as an ablation.
"""

import os, json, glob, argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyStudent(nn.Module):
    def __init__(self, in_ch=8, hidden=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def to_t(arr):
    x = torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))
    return x


def rs(img, size):
    return F.interpolate(
        img[None], size=(size, size), mode="bilinear", align_corners=False
    )[0]


def rs_nn(img, size):
    return F.interpolate(img[None], size=(size, size), mode="nearest")[0]


def psnr_mask(pred, gt, mask):
    m = mask.expand_as(pred)
    denom = m.sum().clamp(min=1.0)
    mse = (((pred - gt) ** 2) * m).sum() / denom
    return float(10 * torch.log10(1.0 / mse.clamp(min=1e-10)))


def load_scene(sid, teacher_dir, f3d_dir, data, size):
    tp = os.path.join(teacher_dir, f"adaptive2_{sid}.npy")
    gp = os.path.join(teacher_dir, f"gt_{sid}.npy")
    bp = os.path.join(teacher_dir, f"baseline_{sid}.npy")
    fp = os.path.join(f3d_dir, f"f3d_{sid}.npy")
    vp = os.path.join(data, sid, "visibility.npy")
    required = {
        "teacher render": tp,
        "ground truth supervision": gp,
        "Gen3R baseline render": bp,
        "Flash3D evidence": fp,
        "offline full-clip visibility mask": vp,
    }
    missing = [
        (label, path) for label, path in required.items() if not os.path.exists(path)
    ]
    if missing:
        details = "\n".join(f"  - {label}: {path}" for label, path in missing)
        raise FileNotFoundError(
            f"Missing required files for scene '{sid}':\n{details}\n"
            "Generate all teacher, baseline, evidence, and oracle-visibility arrays "
            "before running E-031b; ground truth is never used as a baseline fallback."
        )
    teacher = np.load(tp)
    gt = np.load(gp)
    f3d = np.load(fp)
    base = np.load(bp)
    vis = np.load(vp).astype(np.float32)
    F_ = min(len(teacher), len(gt), len(f3d), len(vis))
    samples = []
    for k in range(1, F_):
        t = rs(
            to_t(teacher[k]).clamp(0, 1)
            if teacher[k].max() <= 1.5
            else to_t(teacher[k] / 255.0),
            size,
        )
        g = rs(to_t(gt[k]) if gt[k].max() <= 1.5 else to_t(gt[k] / 255.0), size)
        ff = to_t(f3d[k])
        ff = (ff / 255.0) if ff.max() > 1.5 else ff
        ff = rs(ff.clamp(0, 1), size)
        bb = to_t(base[k])
        bb = (bb / 255.0) if bb.max() > 1.5 else bb
        bb = rs(bb.clamp(0, 1), size)
        v = rs_nn(to_t(vis[k])[None], size)
        inv = 1.0 - v
        if v.sum() < 50 or inv.sum() < 50:
            continue
        inp = torch.cat([ff, bb, v, inv], dim=0)
        samples.append(
            {
                "sid": sid,
                "k": k,
                "inp": inp,
                "f3d": ff,
                "base": bb,
                "teacher": t,
                "gt": g,
                "vis": v,
                "inv": inv,
            }
        )
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir", required=True)
    ap.add_argument("--f3d_dir", required=True)
    ap.add_argument("--data", default="/home/data/gen3r_re10k/re10k")
    ap.add_argument("--frame_dataset", default=None)
    ap.add_argument(
        "--holdout", default="", help="comma sids (with prefix) for holdout"
    )
    ap.add_argument("--out", default="/home/data/E-031b_student")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--w_gt", type=float, default=0.2)
    ap.add_argument("--w_id", type=float, default=2.0)
    ap.add_argument("--gate_aware", action="store_true")
    ap.add_argument(
        "--base_mode",
        default="gen3r",
        choices=["gen3r", "f3d"],
        help="disocclusion basis the student edits: gen3r baseline render (distillation) or f3d evidence",
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    sids = sorted(
        os.path.basename(p)[len("adaptive2_") : -4]
        for p in glob.glob(os.path.join(args.teacher_dir, "adaptive2_*.npy"))
    )
    hold = set(x for x in args.holdout.split(",") if x)
    gate = {}
    if args.frame_dataset and os.path.exists(args.frame_dataset):
        for r in json.load(open(args.frame_dataset)):
            gate[(r["sid"], r["frame_idx"])] = r["label"]

    all_samples = {}
    for sid in sids:
        s = load_scene(sid, args.teacher_dir, args.f3d_dir, args.data, args.size)
        if s:
            all_samples[sid] = s

    def is_holdout(sid):
        bare = sid.replace("test_", "").replace("train_", "")
        for h in hold:
            hb = h.replace("test_", "").replace("train_", "")
            if sid == h or bare == hb or bare.startswith(hb) or sid.startswith(h):
                return True
        return False

    train_sids = [s for s in all_samples if not is_holdout(s)]
    hold_sids = [s for s in all_samples if s not in train_sids]
    train = [x for s in train_sids for x in all_samples[s]]
    holdout = [x for s in hold_sids for x in all_samples[s]]
    print(
        f"scenes={len(all_samples)} train_frames={len(train)} holdout_frames={len(holdout)}",
        flush=True,
    )
    if not train:
        print("no training data yet", flush=True)
        return

    for x in train + holdout:
        for kk in ["inp", "f3d", "base", "teacher", "gt", "vis", "inv"]:
            x[kk] = x[kk].to(dev)

    net = TinyStudent().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def basis(x):
        return x["base"] if args.base_mode == "gen3r" else x["f3d"]

    for ep in range(args.epochs):
        tot = 0.0
        for x in train:
            resid = net(x["inp"][None])[0]
            out = x["vis"] * x["base"] + x["inv"] * (basis(x) + resid).clamp(0, 1)
            w = 1.0
            if args.gate_aware:
                w = float(gate.get((x["sid"], x["k"]), 1))
            loss = (
                w * (torch.abs(out - x["teacher"]) * x["inv"]).mean()
                + args.w_gt * (torch.abs(out - x["gt"]) * x["inv"]).mean()
                + args.w_id * (torch.abs(out - x["base"]) * x["vis"]).mean()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        if ep % 50 == 0 or ep == args.epochs - 1:
            print(f"ep{ep:03d} loss={tot / max(1, len(train)):.5f}", flush=True)
    torch.save(net.state_dict(), os.path.join(args.out, "student.pt"))

    def ev(name, data):
        if not data:
            print(f"[{name}] empty", flush=True)
            return []
        rows = []
        with torch.no_grad():
            for x in data:
                resid = net(x["inp"][None])[0]
                out = x["vis"] * x["base"] + x["inv"] * (basis(x) + resid).clamp(0, 1)
                rows.append(
                    {
                        "sid": x["sid"],
                        "k": x["k"],
                        "base_inv": psnr_mask(x["base"], x["gt"], x["inv"]),
                        "f3d_inv": psnr_mask(x["f3d"], x["gt"], x["inv"]),
                        "stud_inv": psnr_mask(out, x["gt"], x["inv"]),
                        "teach_inv": psnr_mask(x["teacher"], x["gt"], x["inv"]),
                        "stud_vis": psnr_mask(out, x["gt"], x["vis"]),
                        "base_vis": psnr_mask(x["base"], x["gt"], x["vis"]),
                    }
                )
        di = np.mean([r["stud_inv"] - r["base_inv"] for r in rows])
        dv = np.mean([r["stud_vis"] - r["base_vis"] for r in rows])
        df = np.mean([r["stud_inv"] - r["f3d_inv"] for r in rows])
        print(
            f"[{name}] n={len(rows)} stud-vs-baseline invΔ={di:+.3f} "
            f"stud-vs-f3d invΔ={df:+.3f} visΔ={dv:+.3f} "
            f"base_inv={np.mean([r['base_inv'] for r in rows]):.2f} "
            f"stud_inv={np.mean([r['stud_inv'] for r in rows]):.2f} "
            f"teach_inv={np.mean([r['teach_inv'] for r in rows]):.2f}",
            flush=True,
        )
        return rows

    out = {
        "train": ev("train", train),
        "holdout": ev("holdout", holdout),
        "config": {k: str(v) for k, v in vars(args).items()},
    }
    json.dump(out, open(os.path.join(args.out, "results.json"), "w"), indent=2)
    print("E-031b DONE", flush=True)


if __name__ == "__main__":
    main()
