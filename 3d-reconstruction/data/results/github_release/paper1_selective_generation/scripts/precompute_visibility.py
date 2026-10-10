"""E-008 pre-step: precompute per-frame VISIBILITY masks (visible-from-input-frame)
for each converted RE10K scene, at DiT latent resolution (70x70), cached to disk.

Definition: run VGGT on the clip -> self-consistent depth + camera poses (no scale
mismatch). Backproject frame-0 pixels to world, project into every target frame i,
splat coverage -> visibility_i in [0,1] at 70x70. frame0 ~ all-visible; larger camera
motion -> less visible (disocclusion grows). Saved as <scene>/visibility.npy [F,70,70].

These masks drive E-008's asymmetric (visible=faithful / invisible=generate) loss.

IMPORTANT: VGGT consumes the complete target clip, including target RGB frames.
The resulting masks cross the single-image inference boundary and are oracle
visibility for offline diagnostics. They must not be described as source-only or
deployable inference inputs.
"""

import argparse
import glob
import json
import os
import sys

LEGACY_GEN3R_ROOT = "/root/projects/Gen3R"
LEGACY_VGGT_ROOT = "/root/projects/vggt"
UPSTREAM_SETUP = (
    "Install Gen3R following https://github.com/JaceyHuang/Gen3R and VGGT following "
    "https://github.com/facebookresearch/vggt, then pass --gen3r_root/--vggt_root "
    "or set GEN3R_ROOT/VGGT_ROOT."
)
DEVICE = "cuda:0"
LAT = 70  # DiT latent spatial size for 560px


def configure_runtime(args):
    missing = [
        f"GEN3R_ROOT={args.gen3r_root}" if not os.path.isdir(args.gen3r_root) else None,
        f"VGGT_ROOT={args.vggt_root}" if not os.path.isdir(args.vggt_root) else None,
    ]
    missing = [item for item in missing if item]
    if missing:
        raise RuntimeError(
            f"Missing upstream repositories: {', '.join(missing)}. {UPSTREAM_SETUP}"
        )

    sys.path[:0] = [args.gen3r_root, args.vggt_root]
    global np, torch, Image
    try:
        import numpy as np
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            f"Missing runtime dependency {exc.name!r}. {UPSTREAM_SETUP}"
        ) from exc


def load_frames(scene, n, res=560):
    tj = json.load(open(os.path.join(scene, "transforms.json")))
    fr = tj["frames"][:n]
    imgs = []
    for f in fr:
        im = Image.open(os.path.join(scene, f["file_path"])).convert("RGB")
        # aspect-preserving resize + center crop to res (match Gen3R)
        W, H = im.size
        s = res / min(W, H)
        im = im.resize((round(W * s), round(H * s)))
        W2, H2 = im.size
        left, top = (W2 - res) // 2, (H2 - res) // 2
        im = im.crop((left, top, left + res, top + res))
        imgs.append(np.asarray(im).astype(np.float32) / 255.0)
    return torch.tensor(np.stack(imgs)).permute(0, 3, 1, 2)  # [F,3,res,res]


def _run_vggt(vggt, x):
    """Return per-frame depth [F,H,W], extrinsics w2c [F,4,4], intrinsics [F,3,3]."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    agg_all, ps_idx = vggt.aggregator(x)
    pose_enc = vggt.camera_head(agg_all)[-1]
    H, W = x.shape[-2:]
    extr, intr = pose_encoding_to_extri_intri(pose_enc, (H, W))  # [1,F,3,4], [1,F,3,3]
    depth, _ = vggt.depth_head(agg_all, x, ps_idx)  # [1,F,H,W,1]
    return depth[0, ..., 0], extr[0], intr[0]  # [F,H,W],[F,3,4],[F,3,3]


def visibility_from_frame0(vggt, frames, res=560, lat=LAT):
    x = frames[None].to(DEVICE)
    depth, extr, intr = _run_vggt(
        vggt, x
    )  # depth [F,H,W]; extr w2c [F,3,4]; intr [F,3,3]
    F, H, W = depth.shape
    # backproject frame-0 pixels to world
    d0 = depth[0]
    ys, xs = torch.meshgrid(
        torch.arange(H, device=DEVICE).float(),
        torch.arange(W, device=DEVICE).float(),
        indexing="ij",
    )
    K0 = intr[0]
    fx, fy, cx, cy = K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]
    cam0 = torch.stack([(xs - cx) / fx * d0, (ys - cy) / fy * d0, d0], -1).reshape(
        -1, 3
    )  # [HW,3] cam0
    R0 = extr[0, :3, :3]
    t0 = extr[0, :3, 3]  # w2c
    world = (
        cam0 - t0
    ) @ R0  # cam0->world (R0^T (X - t)); R0 rows orthonormal -> @R0 == R0^T applied
    vis = []
    for i in range(F):
        Ri = extr[i, :3, :3]
        ti = extr[i, :3, 3]
        Ki = intr[i]
        pc = world @ Ri.T + ti  # world->cam i
        z = pc[:, 2]
        u = pc[:, 0] / z.clamp(min=1e-6) * Ki[0, 0] + Ki[0, 2]
        v = pc[:, 1] / z.clamp(min=1e-6) * Ki[1, 1] + Ki[1, 2]
        inb = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        # splat coverage at latent res
        ui = (u / W * lat).clamp(0, lat - 1).long()[inb]
        vi = (v / H * lat).clamp(0, lat - 1).long()[inb]
        cov = torch.zeros(lat, lat, device=DEVICE)
        cov[vi, ui] = 1.0
        vis.append(cov)
    return torch.stack(vis).cpu().numpy()  # [F,lat,lat]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gen3r_root",
        default=os.environ.get("GEN3R_ROOT", LEGACY_GEN3R_ROOT),
        help="Gen3R checkout (default: GEN3R_ROOT or /root/projects/Gen3R)",
    )
    ap.add_argument(
        "--vggt_root",
        default=os.environ.get("VGGT_ROOT", LEGACY_VGGT_ROOT),
        help="VGGT checkout (default: VGGT_ROOT or /root/projects/vggt)",
    )
    ap.add_argument("--data", default="/home/data/gen3r_re10k/re10k")
    ap.add_argument("--n_frames", type=int, default=49)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    configure_runtime(args)

    try:
        from vggt.models.vggt import VGGT
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import VGGT runtime dependency {exc.name!r}. {UPSTREAM_SETUP}"
        ) from exc

    vggt = VGGT.from_pretrained("facebook/VGGT-1B").to(DEVICE).eval()
    for p in vggt.parameters():
        p.requires_grad = False

    scenes = sorted(
        glob.glob(os.path.join(args.data, "train_*"))
        + glob.glob(os.path.join(args.data, "test_*"))
    )
    scenes = [s for s in scenes if os.path.isdir(s)]
    if args.limit:
        scenes = scenes[: args.limit]
    print(f"{len(scenes)} scenes", flush=True)
    done = 0
    for sc in scenes:
        outp = os.path.join(sc, "visibility.npy")
        if os.path.exists(outp):
            done += 1
            continue
        try:
            frames = load_frames(sc, args.n_frames)
            if frames.shape[0] < 8:
                continue
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                vis = visibility_from_frame0(vggt, frames)
            np.save(outp, vis.astype(np.float32))
            done += 1
            if done % 10 == 0:
                fr0 = float(vis[0].mean())
                frl = float(vis[-1].mean())
                print(
                    f"  {done} scenes; vis frame0={fr0:.2f} last={frl:.2f}", flush=True
                )
        except Exception as e:
            print(f"  skip {os.path.basename(sc)}: {e}", flush=True)
    print(f"DONE {done} scenes", flush=True)


if __name__ == "__main__":
    main()
