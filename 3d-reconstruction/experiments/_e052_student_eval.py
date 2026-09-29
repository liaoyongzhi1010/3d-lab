"""E-052 student evaluation on wide700 (feed-forward, no SD, no per-scene opt).

Mirrors E-051's exact hole-mask + invisible-crop protocol so hidden-region FID/KID are
directly comparable to the already-computed Flash3D / Ours-GAN / CATSplat numbers.

Pipeline per wide700 scene (target frame 3 = max gap):
  Flash3D forward -> base gaussians + base_render + alpha + depth
  detect hole (alpha<0.5 cleaned, same as E-051)
  student(base_render, hole_mask, depth_filled) -> hole G-buffer
  back-project + composite hole gaussians onto base -> render (Flash3D rasterizer)
  masked-blend into hole region -> Ours-student render
  metrics: visible/invisible masked PSNR, full LPIPS, invisible-crop FID/KID
  dumps: ours_student_crop/<scene>.png  and  sidebyside5/<scene>.png [GT|F3D|GAN?|Ours-student]

Run (cd /root/projects/flash3d && source .venv/bin/activate && CUDA_HOME=/usr/local/cuda-11.8):
  python _e052_student_eval.py \
    +experiment=layered_re10k +dataset.crop_border=true \
    dataset.data_path=data/RealEstate10K \
    dataset.test_split_path=splits/re10k_mine_filtered/test_files_wide700.txt \
    model.depth.version=v1 \
    ++student.ckpt=outputs/e052_student/student_final.pt \
    ++student.out_dir=/home/data/E-052_student_eval \
    ++student.n_scenes=572 ++student.min_invis=0.03
"""

import os, sys, json
from pathlib import Path

sys.path.insert(0, "/root/projects/flash3d")

import hydra
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from einops import rearrange
from PIL import Image
import scipy.ndimage as ndi
import lpips as lpips_lib

from models.model import GaussianPredictor, to_device
from datasets.util import create_datasets
from misc.util import add_source_frame_id
from models.decoder.gauss_util import focal2fov, getProjectionMatrix, render_predicted

DEVICE = "cuda:0"


class HoleGBufUNet(nn.Module):
    def __init__(self, in_ch=5, out_ch=12, base=64):
        super().__init__()

        def cbr(i, o, s=1):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.SiLU()
            )

        self.e0 = cbr(in_ch, base)
        self.e1 = cbr(base, base * 2, 2)
        self.e2 = cbr(base * 2, base * 4, 2)
        self.e3 = cbr(base * 4, base * 8, 2)
        self.mid = cbr(base * 8, base * 8)
        self.d2 = cbr(base * 8 + base * 8, base * 4)
        self.d1 = cbr(base * 4 + base * 4, base * 2)
        self.d0 = cbr(base * 2 + base * 2, base)
        self.final = cbr(base + base, base)
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        m = self.mid(e3)

        def up_to(t, ref):
            return F.interpolate(
                t, size=ref.shape[-2:], mode="bilinear", align_corners=False
            )

        d2 = self.d2(torch.cat([up_to(m, e3), e3], 1))
        d1 = self.d1(torch.cat([up_to(d2, e2), e2], 1))
        d0 = self.d0(torch.cat([up_to(d1, e1), e1], 1))
        f = self.final(torch.cat([up_to(d0, e0), e0], 1))
        return self.head(f)


def build_base_point_clouds(model, outputs):
    gpp = model.cfg.model.gaussians_per_pixel
    return {
        "xyz_src": outputs["gauss_means"].float(),
        "opacity": outputs["gauss_opacity"],
        "scaling": outputs["gauss_scaling"],
        "rotation": outputs["gauss_rotation"],
        "features_dc": outputs["gauss_features_dc"],
        "features_rest": outputs.get("gauss_features_rest", None),
        "gpp": gpp,
    }


def per_view_pc_and_cam(model, inputs, outputs, base_pcs, frame_id):
    cfg = model.cfg
    B, _, H, W = inputs["color", 0, 0].shape
    gpp = base_pcs["gpp"]
    if frame_id == 0:
        T = torch.eye(4, device=DEVICE).unsqueeze(0).repeat(B, 1, 1)
    else:
        T = outputs[("cam_T_cam", 0, frame_id)].float()
    pos = base_pcs["xyz_src"]
    pc = {
        "xyz": rearrange(pos[:, :3, :], "(b n) c l -> b (n l) c", n=gpp),
        "opacity": rearrange(base_pcs["opacity"], "(b n) c h w -> b (n h w) c", n=gpp),
        "scaling": rearrange(base_pcs["scaling"], "(b n) c h w -> b (n h w) c", n=gpp),
        "rotation": rearrange(
            base_pcs["rotation"], "(b n) c h w -> b (n h w) c", n=gpp
        ),
        "features_dc": rearrange(
            base_pcs["features_dc"], "(b n) c h w -> b (n h w) 1 c", n=gpp
        ),
    }
    if cfg.model.max_sh_degree > 0 and base_pcs["features_rest"] is not None:
        pc["features_rest"] = rearrange(
            base_pcs["features_rest"], "(b n) (sh c) h w -> b (n h w) sh c", c=3, n=gpp
        )
    b = 0
    K_tgt = inputs[("K_tgt", frame_id)]
    focals_pixels = torch.diag(K_tgt[b])[:2]
    fovY = focal2fov(focals_pixels[1].item(), H)
    fovX = focal2fov(focals_pixels[0].item(), W)
    proj_mtrx = getProjectionMatrix(
        cfg.dataset.znear, cfg.dataset.zfar, fovX, fovY, pX=0, pY=0
    ).to(DEVICE)
    world_view_transform = T[b].transpose(0, 1).float()
    camera_center = (
        -world_view_transform[3, :3] @ world_view_transform[:3, :3].transpose(0, 1)
    ).float()
    proj_mtrx = proj_mtrx.transpose(0, 1).float()
    full_proj_transform = (world_view_transform @ proj_mtrx).float()
    pc = {k: v[b].contiguous().float() for k, v in pc.items()}
    cam = dict(
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
        proj_mtrx=proj_mtrx,
        camera_center=camera_center,
        fovX=fovX,
        fovY=fovY,
        H=H,
        W=W,
    )
    return pc, cam


def student_render(
    net, base_render, hole_mask, base_depth, base_pc, cam, K_tgt, T_src_from_tgt
):
    """Feed-forward: predict hole G-buffer, composite, render."""
    H, W = hole_mask.shape
    valid = (~hole_mask).cpu().numpy()
    _, (iy, ix) = ndi.distance_transform_edt(
        ~valid, return_distances=True, return_indices=True
    )
    depth_filled = torch.tensor(
        base_depth.cpu().numpy()[iy, ix], device=DEVICE, dtype=torch.float32
    )[None]
    inp = torch.cat([base_render, hole_mask.float()[None], depth_filled], dim=0)[None]
    pred = net(inp)[0]

    ys, xs = torch.where(hole_mask)
    if ys.numel() == 0:
        return base_render
    z = pred[0, ys, xs].clamp(min=1e-3)
    sh_dc = pred[1:4, ys, xs].T
    logscale = pred[4:7, ys, xs].T
    opa_logit = pred[7, ys, xs][:, None]
    rot_raw = pred[8:12, ys, xs].T
    fx, fy = K_tgt[0, 0], K_tgt[1, 1]
    cx, cy = K_tgt[0, 2], K_tgt[1, 2]
    x = (xs.float() + 0.5 - cx) / fx * z
    y = (ys.float() + 0.5 - cy) / fy * z
    pts_tgt = torch.stack([x, y, z, torch.ones_like(z)], dim=-1)
    pts_src = (T_src_from_tgt @ pts_tgt.T).T[:, :3]
    M = pts_src.shape[0]
    hole_pc = {
        "xyz": pts_src,
        "opacity": torch.sigmoid(opa_logit).clamp(1e-4, 1 - 1e-4),
        "scaling": torch.exp(logscale.clamp(-8, 2)),
        "rotation": F.normalize(rot_raw, dim=-1),
        "features_dc": sh_dc[:, None, :],
        "features_rest": torch.zeros(M, 3, 3, device=DEVICE),
    }
    merged = {k: torch.cat([base_pc[k], hole_pc[k]], 0).contiguous() for k in hole_pc}
    from types import SimpleNamespace

    rcfg = SimpleNamespace(model=SimpleNamespace(renderer_w_pose=True, max_sh_degree=1))
    bg = torch.zeros(3, device=DEVICE)
    out = render_predicted(
        rcfg,
        merged,
        cam["world_view_transform"],
        cam["full_proj_transform"],
        cam["proj_mtrx"],
        cam["camera_center"],
        (cam["fovX"], cam["fovY"]),
        (H, W),
        bg,
        1,
    )
    return out["render"].clamp(0, 1)


def clean_mask(invis_np, open_iter=1, close_iter=3, min_area_frac=0.01):
    m = invis_np > 0.5
    m = ndi.binary_opening(m, iterations=open_iter)
    m = ndi.binary_closing(m, iterations=close_iter)
    lbl, n = ndi.label(m)
    if n == 0:
        return np.zeros_like(invis_np, dtype=np.float32)
    H, W = m.shape
    keep = np.zeros_like(m)
    for i in range(1, n + 1):
        comp = lbl == i
        if comp.sum() >= min_area_frac * H * W:
            keep |= comp
    return keep.astype(np.float32)


def bbox_of_mask(m):
    ys, xs = np.where(m > 0.5)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def masked_psnr(pred, gt, mask):
    mse = (((pred - gt) ** 2) * mask).sum() / (mask.sum() * 3 + 1e-8)
    if mse <= 1e-10:
        return None
    return float(-10 * torch.log10(mse))


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    scfg = cfg.get("student", {})
    ckpt = scfg.get("ckpt", "outputs/e052_student/student_final.pt")
    n_scenes = int(scfg.get("n_scenes", 572))
    min_invis = float(scfg.get("min_invis", 0.03))
    crop_pad = int(scfg.get("crop_pad", 8))
    out_dir = Path(scfg.get("out_dir", "/home/data/E-052_student_eval"))
    for sub in ["gt_crop", "f3d_crop", "ours_crop", "sidebyside"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    cfg.data_loader.batch_size = 1
    cfg.data_loader.num_workers = 4

    print("[1] Flash3D backbone...", flush=True)
    model = GaussianPredictor(cfg).to(DEVICE)
    model.load_model(model.checkpoint_dir(), ckpt_ids=0)
    model.set_eval()

    print(f"[2] student head from {ckpt} ...", flush=True)
    net = HoleGBufUNet(in_ch=5, out_ch=12, base=64).to(DEVICE).eval()
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    for p in net.parameters():
        p.requires_grad = False

    lpips_fn = lpips_lib.LPIPS(net="vgg").to(DEVICE).eval()
    print("[3] dataset (wide700)...", flush=True)
    _, dataloader = create_datasets(cfg, split="test")

    tgt = 3
    n_done = n_ok = 0
    manifest = []
    acc = {
        k: []
        for k in [
            "vis_psnr_f3d",
            "vis_psnr_ours",
            "invis_psnr_f3d",
            "invis_psnr_ours",
            "full_lpips_f3d",
            "full_lpips_ours",
        ]
    }
    for inputs in dataloader:
        if n_done >= n_scenes:
            break
        n_done += 1
        try:
            frame_tag = inputs[("frame_id", 0)][0]
            scene = frame_tag.split("+")[1] if "+" in frame_tag else f"{n_done:05d}"
        except Exception:
            scene = f"{n_done:05d}"
        with torch.no_grad():
            to_device(inputs, DEVICE)
            inputs["target_frame_ids"] = [1, 2, 3]
            try:
                outputs = model(inputs)
                base_pcs = build_base_point_clouds(model, outputs)
                pred = outputs[("color_gauss", tgt, 0)][0].clamp(0, 1)
                gt = inputs[("color", tgt, 0)][0].clamp(0, 1)
                alpha = outputs[("alpha_gauss", tgt, 0)][0, 0]
                depth_tgt = outputs[("depth_gauss", tgt, 0)][0, 0]
            except Exception as e:
                print(f"  skip {n_done} ({scene}): {e}", flush=True)
                continue
        invis_raw = (alpha < 0.5).float().cpu().numpy()
        if invis_raw.mean() < min_invis:
            continue
        invis = clean_mask(invis_raw)
        if invis.mean() < min_invis:
            continue
        H, W = pred.shape[1], pred.shape[2]
        invis_t = torch.from_numpy(invis).to(DEVICE)
        hole_mask = invis_t.bool()

        K_tgt = inputs[("K_tgt", tgt)][0]
        T_src_from_tgt = outputs[("cam_T_cam", tgt, 0)][0].float()
        base_pc, cam = per_view_pc_and_cam(model, inputs, outputs, base_pcs, tgt)
        with torch.no_grad():
            ours_full = student_render(
                net, pred, hole_mask, depth_tgt, base_pc, cam, K_tgt, T_src_from_tgt
            )
        # masked blend: keep Flash3D visible region, student only in hole
        comp = (pred * (1 - invis_t[None]) + ours_full * invis_t[None]).clamp(0, 1)

        vis_m = 1.0 - invis_t[None]
        acc["vis_psnr_f3d"].append(masked_psnr(pred, gt, vis_m))
        acc["vis_psnr_ours"].append(masked_psnr(comp, gt, vis_m))
        acc["invis_psnr_f3d"].append(masked_psnr(pred, gt, invis_t[None]))
        acc["invis_psnr_ours"].append(masked_psnr(comp, gt, invis_t[None]))
        with torch.no_grad():
            acc["full_lpips_f3d"].append(
                lpips_fn(pred[None] * 2 - 1, gt[None] * 2 - 1).item()
            )
            acc["full_lpips_ours"].append(
                lpips_fn(comp[None] * 2 - 1, gt[None] * 2 - 1).item()
            )

        bb = bbox_of_mask(invis)
        if bb is not None:
            y0, y1, x0, x1 = bb
            y0 = max(0, y0 - crop_pad)
            x0 = max(0, x0 - crop_pad)
            y1 = min(H - 1, y1 + crop_pad)
            x1 = min(W - 1, x1 + crop_pad)
            if (y1 - y0) >= 16 and (x1 - x0) >= 16:

                def save_crop(t, folder, name):
                    c = t[:, y0:y1, x0:x1].cpu()
                    Image.fromarray(
                        (c.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                    ).save(out_dir / folder / name)

                nm = f"{scene}.png"
                save_crop(gt, "gt_crop", nm)
                save_crop(pred, "f3d_crop", nm)
                save_crop(comp, "ours_crop", nm)

        def to_np(t):
            return (
                (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            )

        strip = np.concatenate([to_np(gt), to_np(pred), to_np(comp)], axis=1)
        Image.fromarray(strip).save(
            out_dir / "sidebyside" / f"{scene}_h{invis.mean():.3f}.png"
        )

        manifest.append({"scene": scene, "invis": float(invis.mean())})
        n_ok += 1
        if n_ok % 25 == 0:
            print(f"  [{n_ok}] scene {n_done} invis={invis.mean():.3f}", flush=True)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    def mean(x):
        x = [v for v in x if v is not None]
        return float(np.mean(x)) if x else float("nan")

    m = {k: mean(v) for k, v in acc.items()}

    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    def load_imgs(folder):
        fs = sorted(Path(folder).glob("*.png"))
        arr = []
        for f in fs:
            im = Image.open(f).convert("RGB").resize((299, 299), Image.BICUBIC)
            arr.append(torch.from_numpy(np.array(im)).permute(2, 0, 1))
        return torch.stack(arr).to(torch.uint8) if arr else None

    gt_imgs = load_imgs(out_dir / "gt_crop")
    f3d_imgs = load_imgs(out_dir / "f3d_crop")
    ours_imgs = load_imgs(out_dir / "ours_crop")
    fid_res = {}
    if gt_imgs is not None and gt_imgs.shape[0] >= 10:
        subset = min(50, gt_imgs.shape[0])
        for tag, imgs in [("f3d", f3d_imgs), ("ours", ours_imgs)]:
            fid = FrechetInceptionDistance(feature=2048, normalize=False)
            fid.update(gt_imgs, real=True)
            fid.update(imgs, real=False)
            fid_res[f"fid_{tag}"] = float(fid.compute())
            kid = KernelInceptionDistance(subset_size=subset, normalize=False)
            kid.update(gt_imgs, real=True)
            kid.update(imgs, real=False)
            kmean, _ = kid.compute()
            fid_res[f"kid_{tag}"] = float(kmean)

    summary = {"n_ok": n_ok, "n_done": n_done, **m, **fid_res}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("=" * 64)
    print(f"E-052 student wide700 ({n_ok} scenes with holes):")
    print(
        f"  VISIBLE PSNR  F3D {m['vis_psnr_f3d']:.3f} | Ours {m['vis_psnr_ours']:.3f}"
    )
    print(
        f"  INVIS   PSNR  F3D {m['invis_psnr_f3d']:.3f} | Ours {m['invis_psnr_ours']:.3f}"
    )
    print(
        f"  FULL    LPIPS F3D {m['full_lpips_f3d']:.4f} | Ours {m['full_lpips_ours']:.4f}"
    )
    if fid_res:
        print(
            f"  INVIS crop FID F3D {fid_res['fid_f3d']:.2f} | Ours {fid_res['fid_ours']:.2f}"
        )
        print(
            f"  INVIS crop KID F3D {fid_res['kid_f3d']:.4f} | Ours {fid_res['kid_ours']:.4f}"
        )
    print("=" * 64)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
