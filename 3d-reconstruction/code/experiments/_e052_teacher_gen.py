"""E-052 teacher generator (Flash3D backbone).

Offline teacher = E-050 per-scene SD-anchor + multiview sdfit (VIOLATES the feed-forward
constraint, used ONLY to make labels). It emits, per (train scene, large-gap target frame):
  - student INPUT  : base_render[3], hole_mask[1], depth_filled[1]   (all target frame)
  - student TARGET : hole G-buffer at hole pixels
        z(1) [teacher gaussian depth in TARGET cam], sh_dc(3), logscale(3), opa_logit(1), rot(4)
  - teacher_render[3]  (sharp completion, render-space distill target)
  - base_pc + camera matrices  (so student training renders WITHOUT re-running Flash3D)
  - geometry K_tgt, T_src_from_tgt  (deterministic back-projection, shared at inference)

WHY self-supervised teacher (no target GT): the SD anchor is a (near-)deterministic function
of the visible input, so a feed-forward student can reproduce it sharply. Regressing the true
(unknowable-from-one-view) GT would re-trigger mean-regression blur (E-006/E-040). Eval scenes
(wide700) are disjoint from these TRAIN scenes -> no leakage.

Run (cd /root/projects/flash3d && source .venv/bin/activate && export CUDA_HOME=/usr/local/cuda-11.8):
  python _e052_teacher_gen.py --n 8 --fit_iters 60 --out /home/data/E-052_teacher_smoke2   # smoke
  python _e052_teacher_gen.py --n 4000 --max_keep 1600 --fit_iters 150 --out /home/data/E-052_teacher_cache
"""

import os, sys, math, json, argparse, importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/root/projects/flash3d")
os.chdir("/root/projects/flash3d")

from omegaconf import OmegaConf
from einops import rearrange
from models.model import GaussianPredictor, to_device
from datasets.util import create_datasets
from misc.util import add_source_frame_id
from models.decoder.gauss_util import focal2fov, getProjectionMatrix, render_predicted
from torch.utils.data._utils.collate import default_collate

DEVICE = "cuda:0"

spec = importlib.util.spec_from_file_location(
    "e050", "/root/projects/flash3d/_e050_flash3d_sdfit.py"
)
E50 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E50)


def per_view_pc_and_cam(model, inputs, outputs, base_pcs, frame_id):
    """Replicate render_view's per-view pc dict (source coords) + camera matrices."""
    cfg = model.cfg
    B, _, H, W = inputs["color", 0, 0].shape
    gpp = base_pcs["gpp"]
    device = DEVICE
    if frame_id == 0:
        T = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    else:
        T = outputs[("cam_T_cam", 0, frame_id)].float()
    pos = base_pcs["xyz_src"]  # use_gt_poses: gaussians stay in source coords
    point_clouds = {
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
        point_clouds["features_rest"] = rearrange(
            base_pcs["features_rest"], "(b n) (sh c) h w -> b (n h w) sh c", c=3, n=gpp
        )
    b = 0
    K_tgt = inputs[("K_tgt", frame_id)]
    focals_pixels = torch.diag(K_tgt[b])[:2]
    fovY = focal2fov(focals_pixels[1].item(), H)
    fovX = focal2fov(focals_pixels[0].item(), W)
    proj_mtrx = getProjectionMatrix(
        cfg.dataset.znear, cfg.dataset.zfar, fovX, fovY, pX=0, pY=0
    ).to(device)
    world_view_transform = T[b].transpose(0, 1).float()
    camera_center = (
        -world_view_transform[3, :3] @ world_view_transform[:3, :3].transpose(0, 1)
    ).float()
    proj_mtrx = proj_mtrx.transpose(0, 1).float()
    full_proj_transform = (world_view_transform @ proj_mtrx).float()
    pc = {k: v[b].contiguous().float() for k, v in point_clouds.items()}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/data/E-052_teacher_cache")
    ap.add_argument(
        "--cfg_path", default="/root/projects/flash3d/eval_official/.hydra/config.yaml"
    )
    ap.add_argument("--n", type=int, default=4000, help="train indices to scan")
    ap.add_argument("--max_keep", type=int, default=1600)
    ap.add_argument("--dilation", type=int, default=20)
    ap.add_argument("--fit_iters", type=int, default=150)
    ap.add_argument("--hole_thr", type=float, default=0.95)
    ap.add_argument("--min_hole_frac", type=float, default=0.06)
    ap.add_argument("--w_anchor", type=float, default=1.0)
    ap.add_argument("--w_mv", type=float, default=0.5)
    ap.add_argument("--w_reg", type=float, default=0.01)
    ap.add_argument("--min_baseline", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump_vis", type=int, default=24)
    args = ap.parse_args()

    out = args.out
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    os.makedirs(os.path.join(out, "vis"), exist_ok=True)

    import cv2
    import lpips as _lpips
    from diffusers import StableDiffusionInpaintPipeline

    cfg = OmegaConf.load(args.cfg_path)
    cfg.data_loader.batch_size = 1
    cfg.data_loader.num_workers = 0
    cfg.dataset.from_tar = False
    cfg.dataset.copy_to_local = False
    cfg.dataset.frame_sampling_method = "two_forward_one_back"
    cfg.dataset.dilation = args.dilation
    cfg.dataset.max_dilation = args.dilation
    cfg.dataset.color_aug = False

    model = GaussianPredictor(cfg).to(DEVICE)
    model.load_model("checkpoints", ckpt_ids=0)
    model.set_eval()

    lpips_fn = _lpips.LPIPS(net="vgg").to(DEVICE).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    dataset, _ = create_datasets(cfg, split="train")
    rng = np.random.RandomState(args.seed)
    order = rng.permutation(len(dataset)).tolist()[: args.n]
    novel_fids = [-1, 1, 2]

    def masked_psnr(pred, gt, m):
        if m.sum() < 10:
            return None
        mse = (((pred - gt) ** 2).mean(0)[m]).mean().item()
        return -10 * math.log10(max(mse, 1e-10))

    manifest = []
    kept = 0
    dumped = 0
    for pos, idx in enumerate(order):
        if kept >= args.max_keep:
            break
        try:
            inputs = default_collate([dataset[idx]])
        except Exception as e:
            continue
        to_device(inputs, DEVICE)
        inputs["target_frame_ids"] = novel_fids
        with torch.no_grad():
            outputs = model(inputs)
            base_pcs = E50.build_base_point_clouds(model, outputs)

        # pick the novel frame with the largest cleaned hole
        best = None
        for fid in novel_fids:
            if ("alpha_gauss", fid, 0) not in outputs:
                continue
            alpha = outputs[("alpha_gauss", fid, 0)][0, 0]
            hm = E50.detect_holes(alpha, args.hole_thr, args.min_hole_frac)
            hf = float(hm.float().mean())
            if hf < args.min_hole_frac:
                continue
            if best is None or hf > best[2]:
                best = (fid, hm, hf)
        if best is None:
            continue
        fid, hole_mask, hole_frac = best

        with torch.no_grad():
            base_render = outputs[("color_gauss", fid, 0)][0].clamp(0, 1)
            depth_tgt = outputs[("depth_gauss", fid, 0)][0, 0]
            gt = inputs[("color", fid, 0)][0].clamp(0, 1)
        K_tgt = inputs[("K_tgt", fid)][0]
        T_src_from_tgt = outputs[("cam_T_cam", fid, 0)][0].float()

        with torch.no_grad():
            anchor = E50.sd_inpaint_target(pipe, base_render, hole_mask).detach()

        hole_params = E50.create_hole_gaussians_src(
            hole_mask, depth_tgt, anchor, K_tgt, T_src_from_tgt
        )
        if hole_params is None:
            continue

        # boundary-propagated init depth (target frame) = student input geometry prior
        valid = (~hole_mask).cpu().numpy()
        from scipy import ndimage

        _, (iy, ix) = ndimage.distance_transform_edt(
            ~valid, return_distances=True, return_indices=True
        )
        depth_filled = torch.tensor(
            depth_tgt.detach().cpu().numpy()[iy, ix],
            device=DEVICE,
            dtype=depth_tgt.dtype,
        )

        # leak-safe-for-student multiview aux frames: OTHER novel frames + source(0),
        # baseline-gated. (train scene, no held-out; this only grounds geometry.)
        cand = [0] + [f for f in novel_fids if f != fid]
        aux_views = []
        aux_fids = []
        for fj in cand:
            T_rel = inputs[("T_w2c", fj)][0].float() @ inputs[("T_c2w", fid)][0].float()
            if T_rel[:3, 3].norm().item() < args.min_baseline:
                continue
            with torch.no_grad():
                aux_gt = inputs[("color", fj, 0)][0].clamp(0, 1)
                aux_alpha = outputs[("alpha_gauss", fj, 0)][0, 0]
            aux_valid = (aux_alpha >= args.hole_thr).float()[None]
            aux_views.append((fj, aux_gt, aux_valid))
            aux_fids.append(fj)
        if aux_fids:
            E50.multiview_color_init(
                hole_params, inputs, aux_fids, hole_params["sh_dc"].data
            )

        opt = torch.optim.Adam(
            [
                {"params": [hole_params["xyz_src"]], "lr": 1e-3},
                {"params": [hole_params["logscale"]], "lr": 1e-2},
                {"params": [hole_params["opacity_logit"]], "lr": 1e-2},
                {"params": [hole_params["rot_raw"]], "lr": 1e-3},
                {"params": [hole_params["sh_dc"]], "lr": 1e-2},
            ]
        )
        anchor_hole = anchor.detach()
        hm3 = hole_mask[None].float()
        for it in range(args.fit_iters):
            hole = E50.assemble_hole(hole_params, cfg)
            render = E50.render_view(model, inputs, outputs, base_pcs, fid, hole=hole)
            l1 = ((render - anchor_hole).abs() * hm3).sum() / (hm3.sum() * 3 + 1e-6)
            lp = lpips_fn(
                (render * hm3)[None] * 2 - 1, (anchor_hole * hm3)[None] * 2 - 1
            ).mean()
            loss = args.w_anchor * (l1 + 0.5 * lp)
            if aux_views and args.w_mv > 0:
                mv = 0.0
                for fj, aux_gt, aux_valid in aux_views:
                    r_j = E50.render_view(
                        model, inputs, outputs, base_pcs, fj, hole=hole
                    )
                    mv = mv + ((r_j - aux_gt).abs() * aux_valid).sum() / (
                        aux_valid.sum() * 3 + 1e-6
                    )
                loss = loss + args.w_mv * mv / max(len(aux_views), 1)
            if args.w_reg > 0:
                scale_pen = torch.exp(hole_params["logscale"].clamp(-8, 2)).mean()
                opa = torch.sigmoid(hole_params["opacity_logit"])
                loss = loss + args.w_reg * (scale_pen + (opa * (1 - opa)).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        with torch.no_grad():
            hole = E50.assemble_hole(hole_params, cfg)
            teacher_render = E50.render_view(
                model, inputs, outputs, base_pcs, fid, hole=hole
            ).clamp(0, 1)
            # naive-lift render for reference (no optimization)
            naive_hole = E50.assemble_hole(
                E50.create_hole_gaussians_src(
                    hole_mask, depth_tgt, anchor, K_tgt, T_src_from_tgt
                ),
                cfg,
            )
            naive_render = E50.render_view(
                model, inputs, outputs, base_pcs, fid, hole=naive_hole
            ).clamp(0, 1)

        # ---- scatter optimized teacher params into a target-frame G-buffer ----
        ys, xs = torch.where(hole_mask)
        H, W = hole_mask.shape
        # teacher gaussian depth in TARGET camera coords
        xyz_src = hole_params["xyz_src"].detach()  # [M,3]
        T_tgt_from_src = torch.inverse(T_src_from_tgt)
        pts_h = torch.cat([xyz_src, torch.ones(xyz_src.shape[0], 1, device=DEVICE)], -1)
        z_tgt = (T_tgt_from_src @ pts_h.T).T[:, 2].clamp(min=1e-3)  # [M]

        gbuf_z = torch.zeros(1, H, W, device=DEVICE)
        gbuf_shdc = torch.zeros(3, H, W, device=DEVICE)
        gbuf_logscale = torch.zeros(3, H, W, device=DEVICE)
        gbuf_opa = torch.zeros(1, H, W, device=DEVICE)
        gbuf_rot = torch.zeros(4, H, W, device=DEVICE)
        gbuf_z[0, ys, xs] = z_tgt
        gbuf_shdc[:, ys, xs] = hole_params["sh_dc"].detach().T
        gbuf_logscale[:, ys, xs] = hole_params["logscale"].detach().T
        gbuf_opa[0, ys, xs] = hole_params["opacity_logit"].detach().squeeze(-1)
        gbuf_rot[:, ys, xs] = F.normalize(hole_params["rot_raw"].detach(), dim=-1).T

        base_pc, cam = per_view_pc_and_cam(model, inputs, outputs, base_pcs, fid)

        p_base = masked_psnr(base_render, gt, hole_mask.bool())
        p_naive = masked_psnr(naive_render, gt, hole_mask.bool())
        p_teacher = masked_psnr(teacher_render, gt, hole_mask.bool())

        f16 = lambda t: t.detach().half().cpu()
        name = f"t{kept:05d}_idx{idx:06d}_f{fid}_h{hole_frac:.3f}"
        torch.save(
            {
                "base_render": f16(base_render),
                "hole_mask": hole_mask.detach().bool().cpu(),
                "depth_filled": f16(depth_filled[None]),
                "base_depth": f16(depth_tgt[None]),
                "gbuf_z": f16(gbuf_z),
                "gbuf_shdc": f16(gbuf_shdc),
                "gbuf_logscale": f16(gbuf_logscale),
                "gbuf_opa": f16(gbuf_opa),
                "gbuf_rot": f16(gbuf_rot),
                "teacher_render": f16(teacher_render),
                "base_pc": {k: f16(v) for k, v in base_pc.items()},
                "cam": {
                    k: (f16(v) if torch.is_tensor(v) else v) for k, v in cam.items()
                },
                "K_tgt": K_tgt.detach().float().cpu(),
                "T_src_from_tgt": T_src_from_tgt.detach().float().cpu(),
                "meta": dict(
                    idx=int(idx),
                    fid=int(fid),
                    hole_frac=hole_frac,
                    n_hole=int(ys.numel()),
                    n_aux=len(aux_fids),
                    psnr_base=p_base,
                    psnr_naive=p_naive,
                    psnr_teacher=p_teacher,
                    frame_id=inputs[("frame_id", 0)][0],
                ),
            },
            os.path.join(out, "labels", name + ".pt"),
        )
        manifest.append(
            dict(
                name=name,
                idx=int(idx),
                fid=int(fid),
                hole_frac=hole_frac,
                psnr_base=p_base,
                psnr_naive=p_naive,
                psnr_teacher=p_teacher,
            )
        )
        if dumped < args.dump_vis:

            def tn(x):
                return (
                    x.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255
                ).astype(np.uint8)

            strip = np.concatenate(
                [
                    tn(base_render),
                    tn(anchor),
                    tn(naive_render),
                    tn(teacher_render),
                    tn(gt),
                ],
                axis=1,
            )
            cv2.imwrite(os.path.join(out, "vis", name + ".jpg"), strip[:, :, ::-1])
            dumped += 1
        kept += 1
        print(
            f"[{pos}/{len(order)}] SAVE {name} h={hole_frac:.3f} base={p_base:.2f} "
            f"naive={p_naive:.2f} teacher={p_teacher:.2f} nG={ys.numel()} aux={len(aux_fids)}",
            flush=True,
        )

    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=2)
    print(f"DONE kept={kept} scanned={pos + 1} -> {out}", flush=True)


if __name__ == "__main__":
    main()
