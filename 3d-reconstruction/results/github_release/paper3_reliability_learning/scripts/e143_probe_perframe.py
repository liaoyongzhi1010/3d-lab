"""E-143 per-frame probe: dump per-frame Flash3D visible/invisible PSNR + vis_frac.

Same math as e019_probe_anysplit but keeps per-frame arrays instead of averaging.
Used to build frame-level features for Paper 3 ReliabilityNet.
"""

import os, json, glob, argparse
import numpy as np
from PIL import Image

RES = 560


def scene_dir(data, sid):
    if sid.startswith("train_") or sid.startswith("test_"):
        return os.path.join(data, sid)
    return os.path.join(data, f"test_{sid}")


def load_gt(scene, n):
    tj = json.load(open(os.path.join(scene, "transforms.json")))
    imgs = []
    for f in tj["frames"][:n]:
        im = Image.open(os.path.join(scene, f["file_path"])).convert("RGB")
        W, H = im.size
        s = RES / min(W, H)
        im = im.resize((round(W * s), round(H * s)))
        W2, H2 = im.size
        left, top = (W2 - RES) // 2, (H2 - RES) // 2
        im = im.crop((left, top, left + RES, top + RES))
        imgs.append(np.asarray(im).astype(np.float32) / 255.0)
    return np.stack(imgs).transpose(0, 3, 1, 2)


def resize_mask(mask, hw):
    if mask.shape == hw:
        return mask
    im = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (hw[1], hw[0]), Image.Resampling.NEAREST
    )
    return np.asarray(im).astype(np.float32) / 255.0


def psnr(pred, gt, mask):
    m = resize_mask(mask, pred.shape[1:]) > 0.5
    if m.sum() < 100:
        return None
    mse = (((pred - gt) ** 2) * m[None]).sum() / (m.sum() * 3)
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/data/gen3r_re10k/re10k")
    ap.add_argument("--f3d_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_frames", type=int, default=49)
    args = ap.parse_args()
    rows = []
    for fp in sorted(glob.glob(os.path.join(args.f3d_dir, "f3d_*.npy"))):
        sid = os.path.basename(fp)[4:-4]
        sc = scene_dir(args.data, sid)
        if not os.path.isdir(sc):
            print("skip no scene", sid)
            continue
        arr = np.load(fp).astype(np.float32)
        if arr.ndim != 4 or arr.shape[1] != 3:
            print("skip bad shape", sid, arr.shape)
            continue
        if arr.max() > 1.5:
            arr = arr / 255.0
        gt = load_gt(sc, args.n_frames)
        vis = np.load(os.path.join(sc, "visibility.npy"))[: args.n_frames].astype(
            np.float32
        )
        F = min(len(arr), len(gt), len(vis))
        frames = []
        for i in range(1, F):
            vp = psnr(arr[i], gt[i], vis[i])
            ip = psnr(arr[i], gt[i], 1 - vis[i])
            vf = float(vis[i].mean())
            frames.append(
                {
                    "frame": i,
                    "f3d_vis_psnr": vp,
                    "f3d_inv_psnr": ip,
                    "vis_frac": vf,
                }
            )
        rows.append({"scene": sid, "frames": frames})
        print(sid[:16], "F", len(frames))
    json.dump(rows, open(args.out, "w"), indent=2)
    print("saved", args.out, "scenes", len(rows))


if __name__ == "__main__":
    main()
