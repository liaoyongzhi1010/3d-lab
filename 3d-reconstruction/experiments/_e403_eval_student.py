"""V2-4: evaluate a trained hole-Gaussian student on held-out teacher-cache samples.

Metric is computed against GROUND TRUTH (gt_rgb), in the hole (invisible) region:
  - PSNR_hole  (higher better)
  - LPIPS_hole (lower better; on the composed image restricted by hole bbox)
Also reports metrics vs teacher for reference.

This is the fair comparison for baseline (use_unc=0) vs uncertainty-guided
(use_unc=1) students: does OracleGS-style uncertainty weighting make the
feed-forward completion closer to REAL geometry (not just imitate teacher)?

Run (flash3d venv):
  python _e403_eval_student.py --ckpt <student_final.pt> --test <test_files.txt> \
      --out <metrics.json>
"""

import sys, os, json, argparse
from pathlib import Path

sys.path.insert(0, "/root/projects/flash3d")
os.chdir("/root/projects/flash3d")

import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage as ndi
import lpips as lpips_lib

# reuse network + rasteriser helpers from the training script
import importlib.util

spec = importlib.util.spec_from_file_location(
    "e402", "/root/projects/flash3d/_e402_student_unc.py"
)
e402 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e402)

DEVICE = "cuda:0"


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 10.0 * np.log10(1.0 / (mse + 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", type=int, default=96)
    args = ap.parse_args()

    files = [l.strip() for l in open(args.test) if l.strip()]
    print(f"[eval] {len(files)} held-out samples", flush=True)

    net = e402.HoleHead(base=args.base).to(DEVICE).eval()
    net.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
    lpips_fn = lpips_lib.LPIPS(net="vgg").to(DEVICE).eval()

    rows = []
    with torch.no_grad():
        for f in files:
            d = np.load(f, allow_pickle=True)
            base_rgb = (
                torch.from_numpy(d["base_rgb"]).float().permute(2, 0, 1).to(DEVICE)
                / 255.0
            )
            teacher_rgb = (
                torch.from_numpy(d["teacher_rgb"]).float().permute(2, 0, 1).to(DEVICE)
                / 255.0
            )
            gt_rgb = (
                torch.from_numpy(d["gt_rgb"]).float().permute(2, 0, 1).to(DEVICE)
                / 255.0
            )
            hole_mask = torch.from_numpy(d["hole"]).bool().to(DEVICE)
            depth_tgt = torch.from_numpy(d["depth_tgt"]).float().to(DEVICE)
            K = torch.from_numpy(d["K_tgt"]).float().to(DEVICE)
            H, W = hole_mask.shape
            if hole_mask.sum() < 16:
                continue

            valid = (~hole_mask).cpu().numpy()
            _, (iy, ix) = ndi.distance_transform_edt(
                ~valid, return_distances=True, return_indices=True
            )
            depth_filled = torch.tensor(
                depth_tgt.cpu().numpy()[iy, ix], device=DEVICE, dtype=torch.float32
            )
            cond = torch.cat(
                [base_rgb, hole_mask.float()[None], depth_filled[None]], 0
            )[None]
            gbuf = net(cond)[0]
            hys, hxs = torch.where(hole_mask)
            anchor = depth_filled[hys, hxs].clamp(min=1e-3)
            hole_pc, z, rgb_pred = e402.make_hole_gaussians(gbuf, hys, hxs, anchor, K)
            cam = e402.identity_cam(H, W, K)
            composed, hole_render, hole_alpha = e402.render_hole(
                hole_pc, base_rgb, hole_mask, cam, H, W
            )

            hm = hole_mask.float()[None]

            # PSNR in hole vs GT and vs teacher
            def masked_psnr(pred, ref):
                diff2 = ((pred - ref) ** 2 * hm).sum() / (hm.sum() * 3 + 1e-9)
                return 10.0 * np.log10(1.0 / (float(diff2) + 1e-10))

            p_gt = masked_psnr(composed, gt_rgb)
            p_tea = masked_psnr(composed, teacher_rgb)
            # baseline (no completion) hole PSNR vs GT, for delta
            p_base_gt = masked_psnr(base_rgb, gt_rgb)
            lp_gt = float(lpips_fn(composed[None] * 2 - 1, gt_rgb[None] * 2 - 1).mean())
            rows.append(
                {
                    "scene": str(d["scene"]),
                    "psnr_hole_gt": p_gt,
                    "psnr_hole_teacher": p_tea,
                    "psnr_hole_base_gt": p_base_gt,
                    "delta_vs_base": p_gt - p_base_gt,
                    "lpips_gt": lp_gt,
                    "hole_frac": float(hole_mask.float().mean()),
                }
            )

    arr = lambda k: np.array([r[k] for r in rows], float)
    summary = {
        "n": len(rows),
        "psnr_hole_gt_mean": float(arr("psnr_hole_gt").mean()),
        "psnr_hole_teacher_mean": float(arr("psnr_hole_teacher").mean()),
        "psnr_hole_base_gt_mean": float(arr("psnr_hole_base_gt").mean()),
        "delta_vs_base_mean": float(arr("delta_vs_base").mean()),
        "lpips_gt_mean": float(arr("lpips_gt").mean()),
    }
    print(json.dumps(summary, indent=2))
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
