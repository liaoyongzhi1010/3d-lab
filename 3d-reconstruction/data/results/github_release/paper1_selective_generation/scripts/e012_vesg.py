"""E-012: Visible-Evidence guidance for invisible-region generation (Gen3R).

Zero-training, offline oracle-visibility diagnostic. Tests goal.md's CENTRAL claim:
"visible region as evidence constrains/drives invisible-region generation." Three arms, same pipeline,
same seed, same pair set (single fair comparison):

  1. baseline          : Gen3R as-is (E-010/E-011 recap: invis LPIPS ~0.31, 6x visible).
  2. adain             : cheap zero-train method. AdaIN match invisible-region latent
                         channel stats -> visible-region stats each denoising step.
                         E-011 found KID~=0 (already distribution-matched) => PREDICTED
                         near no-op. This arm is the falsifiable check of that prediction.
  3. repaint (ORACLE)  : the CEILING of the whole thesis. Pin the VISIBLE region RGB
                         latents to GT-encoded values (re-noised to each step's level,
                         RePaint-style), let the INVISIBLE region generate freely,
                         conditioned on perfect visible evidence through attention.
                         If even this oracle can't lower invisible LPIPS -> the visible->
                         invisible conditioning mechanism in Gen3R is impotent -> Tier-3
                         pivot signal. If it can -> mechanism real -> a trained adapter
                         is worth building.

Latent: [B=1, C=16, f, 70, W=140] where W = 70(rgb) | 70(geo). Guidance touches RGB half.
Scheduler: FlowMatchEulerDiscreteScheduler, x_t = (1-sigma)*x0 + sigma*noise, sigma 1->0.
Metrics: per-region PSNR + oracle-substitution LPIPS(vgg) vs GT (re10k-eval-alignment).
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
RES = 560
LATENT_W = 70  # RGB half width in latent; geo half is the other 70 -> total 140
SEED = 42


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
    global np, torch, F, Image
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            f"Missing runtime dependency {exc.name!r}. {UPSTREAM_SETUP}"
        ) from exc


def safe_config(args):
    """JSON-serializable view of args: drop private (_) and non-serializable values."""
    out = {}
    for k, v in vars(args).items():
        if k.startswith("_"):
            continue
        if isinstance(v, (str, int, float, bool, type(None), list, dict)):
            out[k] = v
    return out


def load_scene(scene, n, res=RES):
    tj = json.load(open(os.path.join(scene, "transforms.json")))
    fr = tj["frames"][:n]
    imgs, Ks, c2ws = [], [], []
    for f in fr:
        im = Image.open(os.path.join(scene, f["file_path"])).convert("RGB")
        W, H = im.size
        s = res / min(W, H)
        im = im.resize((round(W * s), round(H * s)))
        W2, H2 = im.size
        left, top = (W2 - res) // 2, (H2 - res) // 2
        im = im.crop((left, top, left + res, top + res))
        imgs.append(np.asarray(im).astype(np.float32) / 255.0)
        Ks.append(
            torch.tensor(
                [
                    [f["fl_x"] * s, 0, f["cx"] * s - left],
                    [0, f["fl_y"] * s, f["cy"] * s - top],
                    [0, 0, 1.0],
                ]
            )
        )
        c2ws.append(torch.tensor(f["transform_matrix"], dtype=torch.float32))
    frames = torch.tensor(np.stack(imgs)).permute(0, 3, 1, 2)
    return frames, torch.stack(Ks), torch.stack(c2ws)


def build_visibility_mask_latent(vis_full, num_frames, latent_h, latent_w):
    """Downsample per-frame visibility to latent spatial+temporal size.
    vis_full: [F,H,W] float, 1=visible. Returns [f,latent_h,latent_w], f=(num_frames-1)//4+1.
    Temporal 4:1 grouping is approximate (VGGT/Wan temporal compression); acceptable for
    a region mask. area-mode downsample so a latent cell is 'visible' only if mostly visible.
    """
    Fn = vis_full.shape[0]
    f = (num_frames - 1) // 4 + 1
    groups = []
    for i in range(f):
        start = i * 4
        end = min(start + 4, Fn)
        groups.append(vis_full[start:end].mean(dim=0))  # [H,W]
    vis_frames = torch.stack(groups)  # [f,H,W]
    vis_latent = F.interpolate(
        vis_frames.unsqueeze(1), size=(latent_h, latent_w), mode="area"
    ).squeeze(1)  # [f,latent_h,latent_w]
    return vis_latent


def encode_gt_rgb_latent(pipe, gt_frames, dtype):
    """Encode GT frames [F,3,H,W] in [0,1] to the RGB-half clean latent [1,16,f,70,70]."""
    from einops import rearrange

    x = gt_frames[None].to(DEVICE, dtype)  # [1,F,3,H,W]
    x = rearrange((x * 2 - 1).clamp(-1, 1), "b f c h w -> b c f h w")
    with torch.no_grad():
        dist = pipe.wan_vae.encode(x).latent_dist
        for attr in ("mode", "mean"):
            if hasattr(dist, attr):
                v = getattr(dist, attr)
                x0 = v() if callable(v) else v
                x0 = x0.detach()
                break
        else:
            x0 = dist.sample().detach()
    torch.cuda.empty_cache()
    return x0  # [1,16,f,70,70]


def make_adain_callback(vis_latent, alpha=1.0):
    """Match invisible-region latent channel stats -> visible-region stats (AdaIN)."""
    vis = vis_latent.to(DEVICE)
    visible = (vis > 0.5).float()
    invisible = (vis <= 0.5).float()

    def callback(pipe, step_idx, timestep, kw):
        latents = kw["latents"]
        rgb = latents[..., :LATENT_W]
        geo = latents[..., LATENT_W:]
        f_lat = rgb.shape[2]
        vm = visible[:f_lat][None, None]  # [1,1,f,H,W]
        im = invisible[:f_lat][None, None]
        n_vis, n_invis = vm.sum(), im.sum()
        if n_vis < 10 or n_invis < 10:
            return kw
        ve = vm.expand_as(rgb).to(rgb.dtype)
        ie = im.expand_as(rgb).to(rgb.dtype)
        vmean = (rgb * ve).sum(dim=(0, 2, 3, 4)) / n_vis.clamp(min=1)  # [C]
        vstd = (
            (
                ((rgb - vmean[None, :, None, None, None]) ** 2 * ve).sum(
                    dim=(0, 2, 3, 4)
                )
                / n_vis.clamp(min=1)
            )
            .add(1e-8)
            .sqrt()
        )
        imean = (rgb * ie).sum(dim=(0, 2, 3, 4)) / n_invis.clamp(min=1)
        istd = (
            (
                ((rgb - imean[None, :, None, None, None]) ** 2 * ie).sum(
                    dim=(0, 2, 3, 4)
                )
                / n_invis.clamp(min=1)
            )
            .add(1e-8)
            .sqrt()
        )
        tgt_mean = imean + (vmean - imean) * alpha
        tgt_std = istd + (vstd - istd) * alpha
        b = lambda z: z[None, :, None, None, None]
        normed = (rgb - b(imean)) / b(istd) * b(tgt_std) + b(tgt_mean)
        rgb_new = rgb * (1 - ie) + normed * ie
        kw["latents"] = torch.cat([rgb_new, geo], dim=-1)
        return kw

    return callback


def make_repaint_callback(x0_rgb, vis_latent, generator):
    """ORACLE: pin VISIBLE region RGB latents to GT (re-noised to step level).
    Invisible region left free -> generated conditioned on perfect visible evidence.
    """
    vis = vis_latent.to(DEVICE)
    visible = (vis > 0.5).float()  # [f,H,W]

    def callback(pipe, step_idx, timestep, kw):
        latents = kw["latents"]
        rgb = latents[..., :LATENT_W]
        geo = latents[..., LATENT_W:]
        f_lat = rgb.shape[2]
        vm = visible[:f_lat][None, None].to(rgb.dtype)  # [1,1,f,H,W]
        sigmas = pipe.scheduler.sigmas
        sig = float(sigmas[step_idx + 1]) if step_idx + 1 < len(sigmas) else 0.0
        x0 = x0_rgb[:, :, :f_lat].to(rgb.dtype)
        noise = torch.randn(
            x0.shape, generator=generator, device=DEVICE, dtype=rgb.dtype
        )
        x_known = (1.0 - sig) * x0 + sig * noise  # forward-noise GT to this level
        rgb_new = vm * x_known + (1.0 - vm) * rgb
        kw["latents"] = torch.cat([rgb_new, geo], dim=-1)
        return kw

    return callback


def make_flash3d_callback(x0_rgb_f3d, vis_latent, generator):
    """REALISTIC evidence: identical to repaint oracle, but the VISIBLE region is pinned
    to FLASH3D-rendered content (available at inference) instead of GT. Measures how much
    of the oracle's invisible-region gain a realistically-obtainable evidence source
    recovers -> decides whether the visible-evidence -> invisible-generation adapter (B)
    is worth building.
    """
    return make_repaint_callback(x0_rgb_f3d, vis_latent, generator)


def make_asym_callback(x0_rgb_f3d, vis_latent, generator, alpha=0.5):
    """ASYMMETRIC injection (our method): the VISIBLE region is LEFT UNTOUCHED (Gen3R's
    own generation is already good there), and Flash3D geometric evidence is SOFTLY blended
    into the INVISIBLE region only (guidance, not hard replacement).

    Fixes E-013's flash3d-arm failures:
      - flash3d arm pinned VISIBLE to Flash3D -> hurt visible region (Flash3D blurrier).
      - here the masked visible latent stays at baseline; decoded RGB may still mix spatially.
      - invisible gets a soft prior toward Flash3D geometry (strength alpha), not a hard pin.

        rgb_new[visible]   = rgb (unchanged)
        rgb_new[invisible] = (1-alpha)*rgb + alpha*x_known_f3d
    """
    vis = vis_latent.to(DEVICE)
    visible = (vis > 0.5).float()  # [f,H,W]

    def callback(pipe, step_idx, timestep, kw):
        latents = kw["latents"]
        rgb = latents[..., :LATENT_W]
        geo = latents[..., LATENT_W:]
        f_lat = rgb.shape[2]
        vm = visible[:f_lat][None, None].to(rgb.dtype)  # [1,1,f,H,W] 1=visible
        inv = 1.0 - vm
        sigmas = pipe.scheduler.sigmas
        sig = float(sigmas[step_idx + 1]) if step_idx + 1 < len(sigmas) else 0.0
        x0 = x0_rgb_f3d[:, :, :f_lat].to(rgb.dtype)
        noise = torch.randn(
            x0.shape, generator=generator, device=DEVICE, dtype=rgb.dtype
        )
        x_known = (
            1.0 - sig
        ) * x0 + sig * noise  # forward-noise F3D evidence to this level
        # visible: untouched; invisible: soft blend toward evidence
        rgb_inv = (1.0 - alpha) * rgb + alpha * x_known
        rgb_new = vm * rgb + inv * rgb_inv
        kw["latents"] = torch.cat([rgb_new, geo], dim=-1)
        return kw

    return callback


def make_adaptive_callback(x0_rgb_f3d, vis_latent, generator, alpha=0.5, tau=1.0):
    """ADAPTIVE asymmetric injection (our full method): like asym, but the injection
    strength in the invisible region is modulated per-pixel by a CONFIDENCE that Flash3D
    evidence is locally trustworthy.

    Confidence signal (zero GT, inference-time): agreement between the Flash3D evidence
    latent x0 and the model's current denoised latent rgb. Where they grossly disagree
    (e.g. Flash3D smears a moving foreground object -> huge mismatch), Flash3D is likely
    wrong -> suppress injection. Where they roughly agree (static geometry like shelves)
    -> trust and inject.

        conf = exp(-|x0 - rgb| / tau)   (per-pixel, mean over channels), in (0,1]
        alpha_eff = alpha * conf
        rgb_new[invisible] = (1-alpha_eff)*rgb + alpha_eff*x_known
        rgb_new[visible]   = rgb (unchanged)

    This fixes the d590-type failure (dynamic foreground) while preserving edaf-type gains.
    """
    vis = vis_latent.to(DEVICE)
    visible = (vis > 0.5).float()

    def callback(pipe, step_idx, timestep, kw):
        latents = kw["latents"]
        rgb = latents[..., :LATENT_W]
        geo = latents[..., LATENT_W:]
        f_lat = rgb.shape[2]
        vm = visible[:f_lat][None, None].to(rgb.dtype)
        inv = 1.0 - vm
        sigmas = pipe.scheduler.sigmas
        sig = float(sigmas[step_idx + 1]) if step_idx + 1 < len(sigmas) else 0.0
        x0 = x0_rgb_f3d[:, :, :f_lat].to(rgb.dtype)
        noise = torch.randn(
            x0.shape, generator=generator, device=DEVICE, dtype=rgb.dtype
        )
        x_known = (1.0 - sig) * x0 + sig * noise
        # per-pixel confidence from Flash3D<->model agreement (channel-mean abs diff)
        disagree = (x0 - rgb).abs().mean(dim=1, keepdim=True)  # [B,1,f,H,W]
        conf = torch.exp(-disagree / tau)  # (0,1], high where they agree
        alpha_eff = alpha * conf
        rgb_inv = (1.0 - alpha_eff) * rgb + alpha_eff * x_known
        rgb_new = vm * rgb + inv * rgb_inv
        kw["latents"] = torch.cat([rgb_new, geo], dim=-1)
        return kw

    return callback


def make_adaptive2_callback(
    x0_rgb_f3d, vis_latent, generator, conf_map, alpha=0.5, conf_power=1.0
):
    """ADAPTIVE-v2 (fixed confidence): same injection as asym, but with a PRECOMPUTED
    per-pixel confidence map (computed ONCE in CLEAN latent space = agreement between
    Flash3D evidence and a Gen3R-baseline pass), instead of the noisy-latent agreement
    that failed in E-016. conf_map: [1,1,f,H,W] in (0,1]. alpha_eff = alpha * conf^k.

    conf_power (k): steepens the gate. k>1 pushes low-confidence regions toward 0
    faster, which protects scenes where the Gen3R baseline is already good and any
    Flash3D disagreement is Flash3D being wrong (E-018 249fd -6.88dB failure mode).
    """
    vis = vis_latent.to(DEVICE)
    visible = (vis > 0.5).float()
    conf = conf_map.to(DEVICE)
    if conf_power != 1.0:
        conf = conf.float().pow(conf_power).to(conf_map.dtype)

    def callback(pipe, step_idx, timestep, kw):
        latents = kw["latents"]
        rgb = latents[..., :LATENT_W]
        geo = latents[..., LATENT_W:]
        f_lat = rgb.shape[2]
        vm = visible[:f_lat][None, None].to(rgb.dtype)
        inv = 1.0 - vm
        cf = conf[:, :, :f_lat].to(rgb.dtype)
        sigmas = pipe.scheduler.sigmas
        sig = float(sigmas[step_idx + 1]) if step_idx + 1 < len(sigmas) else 0.0
        x0 = x0_rgb_f3d[:, :, :f_lat].to(rgb.dtype)
        noise = torch.randn(
            x0.shape, generator=generator, device=DEVICE, dtype=rgb.dtype
        )
        x_known = (1.0 - sig) * x0 + sig * noise
        alpha_eff = alpha * cf
        rgb_inv = (1.0 - alpha_eff) * rgb + alpha_eff * x_known
        rgb_new = vm * rgb + inv * rgb_inv
        kw["latents"] = torch.cat([rgb_new, geo], dim=-1)
        return kw

    return callback


def run_pipe(pipe, sc, args, callback):
    from gen3r.utils.data_utils import compute_rays, preprocess_poses

    frames, Ks, c2ws = load_scene(sc, args.n_frames)
    F_ = frames.shape[0]
    gt = frames.to(DEVICE)
    ctrl = gt[:1][None].to(DEVICE, torch.bfloat16)
    c2wp = preprocess_poses(c2ws.to(DEVICE))[None]
    rays_o, rays_d = compute_rays(c2wp[0], Ks.to(DEVICE), h=RES, w=RES, device=DEVICE)
    pluck = torch.cat([torch.cross(rays_o, rays_d, dim=1), rays_d], dim=1)[None].to(
        DEVICE, torch.bfloat16
    )
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    with torch.no_grad():
        out = pipe(
            prompt="A video walkthrough of an indoor real estate scene.",
            control_cameras=pluck,
            control_images=ctrl,
            num_frames=F_,
            negative_prompt="bad detailed",
            height=RES,
            width=RES,
            num_inference_steps=args.steps,
            guidance_scale=5,
            return_dict=True,
            min_max_depth_mask=True,
            generator=g,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    r = out.rgbs[0]
    r = (
        r.float().to(DEVICE)
        if isinstance(r, torch.Tensor)
        else torch.as_tensor(np.asarray(r)).float().to(DEVICE)
    )
    if r.max() > 1.5:
        r = r / 255.0
    return r.permute(0, 3, 1, 2), gt  # pred [F,3,H,W], gt [F,3,H,W]


def score(pred, gt, vis_full, lp):
    """Per-region PSNR + oracle-substitution LPIPS(vgg). Returns per-frame-averaged."""
    F_ = pred.shape[0]
    vp, ip, vl, il = [], [], [], []
    for i in range(1, F_):
        v = F.interpolate(vis_full[i][None, None], size=(RES, RES), mode="nearest")[
            0, 0
        ]
        vm = v > 0.5
        im = v <= 0.5
        if vm.sum() < 100 or im.sum() < 100:
            continue
        g_ = gt[i]
        p_ = pred[i].clamp(0, 1)
        mse_v = ((g_ - p_) ** 2 * vm.float()).sum() / vm.sum()
        vp.append(float(10 * torch.log10(1.0 / mse_v.clamp(min=1e-10))))
        mse_i = ((g_ - p_) ** 2 * im.float()).sum() / im.sum()
        ip.append(float(10 * torch.log10(1.0 / mse_i.clamp(min=1e-10))))
        cg_i = g_.clone()
        cg_i[:, im] = p_[:, im]  # only invisible differs from GT
        il.append(float(lp(cg_i[None] * 2 - 1, g_[None] * 2 - 1).item()))
        cg_v = g_.clone()
        cg_v[:, vm] = p_[:, vm]  # only visible differs from GT
        vl.append(float(lp(cg_v[None] * 2 - 1, g_[None] * 2 - 1).item()))
    return {
        "vis_psnr": float(np.mean(vp)) if vp else None,
        "invis_psnr": float(np.mean(ip)) if ip else None,
        "vis_lpips": float(np.mean(vl)) if vl else None,
        "invis_lpips": float(np.mean(il)) if il else None,
        "n_frames": len(ip),
        "per_frame_vis_psnr": vp,
        "per_frame_invis_psnr": ip,
    }


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
    ap.add_argument(
        "--ckpt",
        default=None,
        help="Gen3R checkpoint path (default: <gen3r_root>/checkpoints)",
    )
    ap.add_argument("--data", default="/home/data/gen3r_re10k/re10k")
    ap.add_argument("--n_scenes", type=int, default=5)
    ap.add_argument("--n_frames", type=int, default=49)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default="/home/data/E-012_vesg")
    ap.add_argument("--adain_alpha", type=float, default=1.0)
    ap.add_argument(
        "--f3d_dir",
        default="/home/data/E-013_f3d_evidence",
        help="dir with f3d_<scene>.npy [49,3,560,560] for the flash3d arm",
    )
    ap.add_argument(
        "--scene_names",
        nargs="*",
        default=None,
        help="explicit scene ids (with or without test_ prefix) to run",
    )
    ap.add_argument(
        "--asym_alpha",
        type=float,
        default=0.5,
        help="blend strength of Flash3D evidence in invisible region (asym arm)",
    )
    ap.add_argument(
        "--save_frames",
        action="store_true",
        help="save per-arm rendered videos + key frames for qualitative/consistency analysis",
    )
    ap.add_argument(
        "--adaptive_tau",
        type=float,
        default=1.0,
        help="temperature for adaptive-arm confidence exp(-|x0-rgb|/tau)",
    )
    ap.add_argument(
        "--conf_power",
        type=float,
        default=3.0,
        help="steepness exponent k for adaptive3 arm: alpha_eff = alpha * conf^k "
        "(k>1 protects scenes with already-good baseline from Flash3D-wrong injection)",
    )
    ap.add_argument(
        "--save_teacher_npy",
        action="store_true",
        help="dump full per-frame render npy per arm (teacher targets for Paper-2 student)",
    )
    ap.add_argument(
        "--arms",
        nargs="+",
        default=["baseline", "adain", "repaint"],
        help="subset of {baseline, adain, repaint, flash3d}",
    )
    args = ap.parse_args()
    configure_runtime(args)
    if args.ckpt is None:
        args.ckpt = os.path.join(args.gen3r_root, "checkpoints")
    if "learned" in args.arms:
        ap.error(
            "learned arm is unavailable in this release: its model source, "
            "checkpoint, and validated results are not released. Use adaptive2 "
            "to reproduce the reported injected output."
        )

    try:
        import lpips as _lpips
        from gen3r.pipeline import Gen3RPipeline
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import Gen3R runtime dependency {exc.name!r}. {UPSTREAM_SETUP}"
        ) from exc

    os.makedirs(args.out, exist_ok=True)
    lp = _lpips.LPIPS(net="vgg").to(DEVICE).eval()
    pipe = Gen3RPipeline.from_pretrained(args.ckpt).to(DEVICE).to(torch.bfloat16)
    pipe.transformer.eval()

    scenes = sorted(
        glob.glob(os.path.join(args.data, "test_*"))
        + glob.glob(os.path.join(args.data, "train_*"))
    )
    scenes = [s for s in scenes if os.path.isdir(s)]
    if args.scene_names:
        want = set(args.scene_names)
        scenes = [
            s
            for s in scenes
            if os.path.basename(s).replace("test_", "") in want
            or os.path.basename(s) in want
        ]
    else:
        scenes = scenes[: args.n_scenes]
    print(
        f"E-012: {len(scenes)} scenes, steps={args.steps}, arms={args.arms}", flush=True
    )

    f_lat = (args.n_frames - 1) // 4 + 1
    results = {a: {} for a in args.arms}
    for si, sc in enumerate(scenes):
        visp = os.path.join(sc, "visibility.npy")
        if not os.path.exists(visp):
            continue
        name = os.path.basename(sc)
        vis_full = torch.tensor(np.load(visp))[: args.n_frames].float().to(DEVICE)
        vis_lat = build_visibility_mask_latent(
            vis_full.cpu(), args.n_frames, LATENT_W, LATENT_W
        )
        print(f"\n[{si + 1}/{len(scenes)}] {name}", flush=True)

        x0_rgb = None
        if "repaint" in args.arms:
            frames, _, _ = load_scene(sc, args.n_frames)
            x0_rgb = encode_gt_rgb_latent(pipe, frames, torch.bfloat16)

        x0_rgb_f3d = None
        if any(
            a in args.arms
            for a in (
                "flash3d",
                "asym",
                "adaptive",
                "adaptive2",
                "adaptive3",
            )
        ):
            f3dp = os.path.join(args.f3d_dir, f"f3d_{name.replace('test_', '')}.npy")
            if not os.path.exists(f3dp):
                f3dp = os.path.join(args.f3d_dir, f"f3d_{name}.npy")
            if os.path.exists(f3dp):
                f3d_arr = np.load(f3dp)[: args.n_frames]  # [F,3,560,560] in [0,1]
                f3d_frames = torch.tensor(f3d_arr, dtype=torch.float32)
                if f3d_frames.shape[-1] != RES:
                    f3d_frames = F.interpolate(
                        f3d_frames,
                        size=(RES, RES),
                        mode="bilinear",
                        align_corners=False,
                    )
                x0_rgb_f3d = encode_gt_rgb_latent(pipe, f3d_frames, torch.bfloat16)
            else:
                print(
                    f"  [flash3d] MISSING {f3dp}, skipping arm for {name}", flush=True
                )

        # adaptive2: precompute a per-pixel confidence map in CLEAN latent space =
        # agreement between Flash3D evidence and a Gen3R-baseline pass. Runs baseline once.
        conf_map = None
        if (
            any(a in args.arms for a in ("adaptive2", "adaptive3"))
            and x0_rgb_f3d is not None
        ):
            pred_base, _ = run_pipe(pipe, sc, args, None)  # clean Gen3R baseline
            base_lat = encode_gt_rgb_latent(
                pipe, pred_base.to(torch.float32), torch.bfloat16
            )
            disagree = (
                (x0_rgb_f3d.float() - base_lat.float()).abs().mean(dim=1, keepdim=True)
            )
            conf_map = torch.exp(-disagree / args.adaptive_tau).to(
                torch.bfloat16
            )  # [1,1,f,H,W]
            print(
                f"  [adaptive2] conf mean={float(conf_map.mean()):.3f} min={float(conf_map.min()):.3f}",
                flush=True,
            )
            del pred_base, base_lat
            torch.cuda.empty_cache()

        for arm in args.arms:
            cb = None
            if arm == "adain":
                cb = make_adain_callback(vis_lat, alpha=args.adain_alpha)
            elif arm == "repaint":
                gcb = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
                cb = make_repaint_callback(x0_rgb, vis_lat, gcb)
            elif arm == "flash3d":
                if x0_rgb_f3d is None:
                    continue
                gcb = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
                cb = make_flash3d_callback(x0_rgb_f3d, vis_lat, gcb)
            elif arm == "asym":
                if x0_rgb_f3d is None:
                    continue
                gcb = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
                cb = make_asym_callback(x0_rgb_f3d, vis_lat, gcb, alpha=args.asym_alpha)
            elif arm == "adaptive":
                if x0_rgb_f3d is None:
                    continue
                gcb = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
                cb = make_adaptive_callback(
                    x0_rgb_f3d,
                    vis_lat,
                    gcb,
                    alpha=args.asym_alpha,
                    tau=args.adaptive_tau,
                )
            elif arm == "adaptive2":
                if x0_rgb_f3d is None or conf_map is None:
                    continue
                gcb = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
                cb = make_adaptive2_callback(
                    x0_rgb_f3d, vis_lat, gcb, conf_map, alpha=args.asym_alpha
                )
            elif arm == "adaptive3":
                if x0_rgb_f3d is None or conf_map is None:
                    continue
                gcb = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
                cb = make_adaptive2_callback(
                    x0_rgb_f3d,
                    vis_lat,
                    gcb,
                    conf_map,
                    alpha=args.asym_alpha,
                    conf_power=args.conf_power,
                )
            pred, gt = run_pipe(pipe, sc, args, cb)
            m = score(pred, gt, vis_full, lp)
            results[arm][name] = m
            if getattr(args, "save_teacher_npy", False):
                try:
                    tdir = os.path.join(args.out, "teacher_npy")
                    os.makedirs(tdir, exist_ok=True)
                    pv_full = pred.clamp(0, 1).cpu().numpy().astype(np.float32)
                    sid_clean = name.replace("test_", "")
                    np.save(os.path.join(tdir, f"{arm}_{sid_clean}.npy"), pv_full)
                    if arm == args.arms[0]:
                        gtv_full = gt.clamp(0, 1).cpu().numpy().astype(np.float32)
                        np.save(os.path.join(tdir, f"gt_{sid_clean}.npy"), gtv_full)
                except Exception as _e:
                    print(f"  [warn] teacher_npy save failed: {_e}", flush=True)
            try:
                ckpt = os.path.join(args.out, "e012_vesg_partial.json")
                json.dump(
                    {"per_scene": results, "config": safe_config(args)},
                    open(ckpt, "w"),
                    indent=2,
                )
            except Exception as _e:
                print(f"  [warn] checkpoint save failed: {_e}", flush=True)
            if args.save_frames:
                import imageio.v2 as _iio

                fdir = os.path.join(args.out, "frames", name)
                os.makedirs(fdir, exist_ok=True)
                pv = pred.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
                vid = [(pv[i] * 255).astype(np.uint8) for i in range(pv.shape[0])]
                _iio.mimwrite(
                    os.path.join(fdir, f"{arm}.mp4"),
                    vid,
                    fps=8,
                    quality=8,
                    macro_block_size=1,
                )
                for k in [
                    0,
                    args.n_frames // 4,
                    args.n_frames // 2,
                    3 * args.n_frames // 4,
                    args.n_frames - 1,
                ]:
                    if k < pv.shape[0]:
                        _iio.imwrite(
                            os.path.join(fdir, f"{arm}_f{k:03d}.png"),
                            (pv[k] * 255).astype(np.uint8),
                        )
                if arm == args.arms[0]:  # save GT once
                    gtv = gt.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
                    for k in [
                        0,
                        args.n_frames // 4,
                        args.n_frames // 2,
                        3 * args.n_frames // 4,
                        args.n_frames - 1,
                    ]:
                        if k < gtv.shape[0]:
                            _iio.imwrite(
                                os.path.join(fdir, f"gt_f{k:03d}.png"),
                                (gtv[k] * 255).astype(np.uint8),
                            )
            del pred, gt
            torch.cuda.empty_cache()

            def _f(x, nd):
                return f"{x:.{nd}f}" if x is not None else "n/a"

            print(
                f"  {arm:9s} invis_lpips={_f(m['invis_lpips'], 4)} "
                f"invis_psnr={_f(m['invis_psnr'], 2)} vis_lpips={_f(m['vis_lpips'], 4)}",
                flush=True,
            )

    def agg(d):
        def m(k):
            vs = [v[k] for v in d.values() if v.get(k) is not None]
            return (
                (round(float(np.mean(vs)), 4), round(float(np.std(vs)), 4))
                if vs
                else None
            )

        return {
            "invis_lpips": m("invis_lpips"),
            "invis_psnr": m("invis_psnr"),
            "vis_lpips": m("vis_lpips"),
            "vis_psnr": m("vis_psnr"),
            "n_scenes": len(d),
        }

    summary = {a: agg(results[a]) for a in args.arms}
    if "baseline" in summary and summary["baseline"]["invis_lpips"]:
        bl = summary["baseline"]["invis_lpips"][0]
        for a in args.arms:
            if a != "baseline" and summary[a]["invis_lpips"]:
                summary[a]["delta_invis_lpips_vs_baseline"] = round(
                    summary[a]["invis_lpips"][0] - bl, 4
                )
    out = {"summary": summary, "per_scene": results, "config": safe_config(args)}
    path = os.path.join(args.out, "e012_vesg.json")
    json.dump(out, open(path, "w"), indent=2)
    print("\n[RESULT SUMMARY]")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved {path}")
    print("E-012 DONE", flush=True)


if __name__ == "__main__":
    main()
