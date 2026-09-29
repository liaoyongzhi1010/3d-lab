"""E-064 student: feed-forward hole-Gaussian head distilled from a SD-inpaint teacher.

WHY this breaks the blur that E-063 mean-only could not:
  E-063 regressed hole colour toward GT_tgt, which is TRULY occluded / unpredictable from
  the input view -> the net hedges over all plausible contents -> blur (E-005 single-GT
  trap; no amount of LPIPS fixes an unlearnable target). The SD-inpaint teacher fill is
  DERIVED FROM VISIBLE CONTEXT -> it IS a deterministic learnable function of the input ->
  the student CAN regress to it -> SHARP. This converts the ill-posed hole into a
  well-posed regression (the E-022 3D-GT insight, now with a 2D generative teacher).

3D-NATIVE (rule 5, NOT 2D-lift): the student predicts a per-hole-pixel G-buffer (depth,
colour, scale, opacity, quat); hole pixels are back-projected to 3D via predicted depth
and rendered through the REAL 3DGS rasteriser (render_predicted) at the target camera.
  - Teacher-colour loss at tgt (identity reprojection -> depth-independent) = appearance.
  - Depth-anchor loss (hole depth continues the EDT-filled Flash3D boundary depth) +
    depth-smoothness = geometry -> coherent 3D, no flying splats / speckle.
Visible gaussians are FROZEN (Flash3D) -> visible-region parity preserved.

INFERENCE: single image + target pose -> Flash3D base + student hole gaussians. Zero SD,
zero optimisation. Explicit 3DGS. All hard constraints satisfied.

Trains ONLY from the teacher cache (no dataloader, no Flash3D forward, no frame replay):
  each npz has teacher_rgb, hole, base_rgb, depth_tgt, K_tgt (see _e064_teacher_cache.py).

Run (cd /root/projects/flash3d && source .venv/bin/activate, CUDA 11.8):
  python _e064_student_train.py ++st.cache=/home/data/E-064_teacher_cache \
    ++st.out=/home/data/E-064_student ++st.iters=6000
"""

import os, sys, math, json, glob
from pathlib import Path

sys.path.insert(0, "/root/projects/flash3d")
os.chdir("/root/projects/flash3d")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as ndi
import lpips as lpips_lib
from types import SimpleNamespace
from models.decoder.gauss_util import getProjectionMatrix, focal2fov, render_predicted

DEVICE = "cuda:0"
RENDER_CFG = SimpleNamespace(
    model=SimpleNamespace(renderer_w_pose=True, max_sh_degree=1)
)


def cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.SiLU())


class Encoder(nn.Module):
    def __init__(self, cond_ch=5, base=96):
        super().__init__()
        self.e0 = cbr(cond_ch, base)
        self.e1 = cbr(base, base * 2, 2)
        self.e2 = cbr(base * 2, base * 4, 2)
        self.e3 = cbr(base * 4, base * 4, 2)

    def forward(self, c):
        e0 = self.e0(c)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        return e0, e1, e2, e3


class HoleHead(nn.Module):
    """U-Net decoder -> per-pixel G-buffer (12ch): depth(1) sh_dc(3) logscale(3)
    opacity(1) quat(4). Deterministic regression to the (learnable) teacher fill."""

    def __init__(self, base=96, x_ch=12):
        super().__init__()
        self.enc = Encoder(5, base)
        self.mid = cbr(base * 4, base * 4)
        self.d2 = cbr(base * 4 + base * 4, base * 4)
        self.d1 = cbr(base * 4 + base * 4, base * 2)
        self.d0 = cbr(base * 2 + base * 2, base)
        self.head = nn.Conv2d(base, x_ch, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # opacity (ch7) starts HIGH so hole gaussians actually paint (v1 bug: opacity
        # collapsed to ~0.01 -> hole rendered gray). sigmoid(2.2)=0.90.
        with torch.no_grad():
            self.head.bias[7] = 2.2
            # log-scale (ch4:7): start near exp(-4)=0.018 (a few px), avoid degenerate size
            self.head.bias[4:7] = -4.0

    def forward(self, cond):
        e0, e1, e2, e3 = self.enc(cond)
        m = self.mid(e3)
        up = lambda t_, r: F.interpolate(
            t_, size=r.shape[-2:], mode="bilinear", align_corners=False
        )
        d2 = self.d2(torch.cat([up(m, e3), e3], 1))
        d1 = self.d1(torch.cat([up(d2, e2), e2], 1))
        d0 = self.d0(torch.cat([up(d1, e1), e1], 1))
        f = up(d0, e0)
        return self.head(f)


def make_hole_gaussians(gbuf, hole_ys, hole_xs, depth_anchor, K):
    """gbuf(12,H,W) -> explicit 3D gaussians for the hole pixels (in tgt-cam coords).
    depth = softplus(raw)+anchor so it stays near the Flash3D boundary depth.
    COLOUR FIX (v4): channels 1:4 are a DIRECT RGB prediction (sigmoid -> [0,1]) that we
    convert to the SH-DC coefficient. Regressing RGB is a standard easy image task; letting
    a zero-init head hit a specific RGB THROUGH the inverse-SH+rasteriser collapsed to gray
    (v1-v3 bug: sh_dc std ~0.01). Now colour is well-posed image regression."""
    SH_C0 = 0.28209479177387814
    z = F.softplus(gbuf[0, hole_ys, hole_xs]) + depth_anchor  # (M,)
    rgb = torch.sigmoid(gbuf[1:4, hole_ys, hole_xs].T)  # (M,3) direct RGB in [0,1]
    sh_dc = (rgb - 0.5) / SH_C0  # convert to DC so rasteriser outputs rgb
    # SCALE FIX (v6/v7): a bloated splat (v5 scale 0.67 world @ depth 5.5) overlaps its
    # neighbours and AVERAGES colour -> blur (post-raster std 0.01). But TOO small leaves
    # inter-splat GAPS -> black bg shows -> partial coverage (v6 alpha 0.56, corner gray).
    # Sweet spot: ~2-3px footprint reliably tiles the hole yet stays sharp. base=2px, head
    # modulates x0.7..x2.0.
    fx_ = K[0, 0]
    px_scale = (2.0 * z / fx_).clamp(min=1e-4)[:, None]  # ~2px world size at depth z
    scale_mult = torch.exp(
        gbuf[4:7, hole_ys, hole_xs].T.clamp(-0.35, 0.7)
    )  # 0.7x..2.0x
    scaling = (px_scale * scale_mult).clamp(1e-4, 0.15)
    opa_logit = gbuf[7, hole_ys, hole_xs][:, None]
    rot_raw = gbuf[8:12, hole_ys, hole_xs].T
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (hole_xs.float() + 0.5 - cx) / fx * z
    y = (hole_ys.float() + 0.5 - cy) / fy * z
    xyz = torch.stack([x, y, z], -1)  # tgt-cam coords (tgt is our render camera)
    M = xyz.shape[0]
    return (
        {
            "xyz": xyz,
            "opacity": torch.sigmoid(opa_logit).clamp(1e-4, 1 - 1e-4),
            "scaling": scaling,
            "rotation": F.normalize(rot_raw, dim=-1),
            "features_dc": sh_dc[:, None, :],
            "features_rest": torch.zeros(M, 3, 3, device=DEVICE),
        },
        z,
        rgb,
    )


def identity_cam(H, W, K, znear=0.1, zfar=1000.0):
    """Render camera = the tgt camera itself (gaussians already in tgt-cam coords).
    world_view = identity -> we look straight down the tgt optical axis."""
    fovX = focal2fov(K[0, 0].item(), W)
    fovY = focal2fov(K[1, 1].item(), H)
    proj = getProjectionMatrix(znear, zfar, fovX, fovY, pX=0, pY=0).to(DEVICE)
    wvt = torch.eye(4, device=DEVICE)
    proj_t = proj.transpose(0, 1).float()
    fpt = (wvt @ proj_t).float()
    cc = torch.zeros(3, device=DEVICE)
    return dict(
        world_view_transform=wvt,
        full_proj_transform=fpt,
        proj_mtrx=proj_t,
        camera_center=cc,
        fovX=fovX,
        fovY=fovY,
    )


def render_hole(hole_pc, base_rgb, hole_mask, cam, H, W):
    """Render hole gaussians at tgt cam, then composite over the frozen base_rgb in
    the hole region only (visible region kept exactly = Flash3D parity). Also returns
    the hole gaussians' accumulated alpha (for a coverage loss -> fill, no black bg)."""
    bg = torch.zeros(3, device=DEVICE)
    out = render_predicted(
        RENDER_CFG,
        hole_pc,
        cam["world_view_transform"],
        cam["full_proj_transform"],
        cam["proj_mtrx"],
        cam["camera_center"],
        (cam["fovX"], cam["fovY"]),
        (H, W),
        bg,
        1,
    )
    hole_render = out["render"].clamp(0, 1)  # (3,H,W)
    hole_alpha = out["rendered_alpha"]  # (1,H,W) accumulated opacity
    m = hole_mask.float()[None]
    composed = base_rgb * (1 - m) + hole_render * m
    return composed, hole_render, hole_alpha


def load_cache(cache_dir):
    files = sorted(glob.glob(str(Path(cache_dir) / "*.npz")))
    return files


def main():
    import argparse

    # hydra-free: read ++st.* style flags manually
    args = {
        a.split("=")[0].lstrip("+"): a.split("=")[1] for a in sys.argv[1:] if "=" in a
    }
    seed = int(args.get("st.seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cache_dir = args.get("st.cache", "/home/data/E-064_teacher_cache")
    out = Path(args.get("st.out", "/home/data/E-064_student"))
    out.mkdir(parents=True, exist_ok=True)
    iters = int(args.get("st.iters", 6000))
    lr = float(args.get("st.lr", 2e-4))
    base_ch = int(args.get("st.base", 96))
    w_l1 = float(args.get("st.w_l1", 1.0))
    w_lpips = float(args.get("st.w_lpips", 1.0))
    w_cover = float(args.get("st.w_cover", 0.5))
    w_dsmooth = float(args.get("st.w_dsmooth", 0.05))
    log_every = int(args.get("st.log_every", 100))
    save_every = int(args.get("st.save_every", 1000))
    dilate = int(args.get("st.dilate", 6))
    # V2-3: OracleGS-style uncertainty-weighted distillation
    use_unc = int(args.get("st.use_unc", 0))
    unc_dir = args.get("st.unc_dir", "/home/data/E-401_vggt_unc")
    unc_gamma = float(args.get("st.unc_gamma", 1.0))  # U := U**gamma sharpening
    unc_mode = args.get(
        "st.unc_mode", "vggt"
    )  # "vggt" (deployable) | "oracle" (upper bound)
    unc_tau = float(args.get("st.unc_tau", 0.2))  # temperature for oracle exp(-err/tau)
    print(
        f"[unc] use_unc={use_unc} mode={unc_mode} dir={unc_dir} gamma={unc_gamma} tau={unc_tau}",
        flush=True,
    )
    # Paper2 P2-2: gate-aware distillation -- only train on samples where the
    # teacher is reliable (teacher hole-PSNR gain over base vs GT >= gate_thr).
    gate_aware = int(args.get("st.gate_aware", 0))
    gate_thr = float(args.get("st.gate_thr", 0.5))
    gate_qfile = args.get("st.gate_qfile", "/home/data/E-410_teacher_quality.json")

    files = load_cache(cache_dir)
    print(f"[cache] {len(files)} teacher samples in {cache_dir}", flush=True)
    assert len(files) >= 8, "teacher cache too small"
    # V2-4: deterministic train/test split by filename hash (held-out for fair eval)
    holdout = float(args.get("st.holdout", 0.0))
    if holdout > 0:
        import hashlib

        def _istest(fp):
            h = int(hashlib.md5(Path(fp).stem.encode()).hexdigest(), 16)
            return (h % 100) < int(holdout * 100)

        test_files = [f for f in files if _istest(f)]
        files = [f for f in files if not _istest(f)]
        print(
            f"[split] train={len(files)} test={len(test_files)} (holdout={holdout})",
            flush=True,
        )
        with open(out / "test_files.txt", "w") as fh:
            fh.write("\n".join(test_files))

    # Paper2 P2-2: gate-aware distillation -- filter TRAIN set to reliable teacher
    # samples only (test set untouched for fair comparison).
    if gate_aware:
        import json as _json

        q = _json.load(open(gate_qfile))
        before = len(files)
        files = [f for f in files if q.get(Path(f).name, -99) >= gate_thr]
        print(
            f"[gate-aware] train {before} -> {len(files)} (teacher_gain>={gate_thr})",
            flush=True,
        )
        assert len(files) >= 8, "gate-aware filter left too few samples"

    net = HoleHead(base=base_ch).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    lpips_fn = lpips_lib.LPIPS(net="vgg").to(DEVICE).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False
    print(
        f"[net] HoleHead {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M",
        flush=True,
    )

    def sample():
        f = files[np.random.randint(len(files))]
        d = np.load(f, allow_pickle=True)
        base = (
            torch.from_numpy(d["base_rgb"]).float().permute(2, 0, 1).to(DEVICE) / 255.0
        )
        teacher = (
            torch.from_numpy(d["teacher_rgb"]).float().permute(2, 0, 1).to(DEVICE)
            / 255.0
        )
        hole = torch.from_numpy(d["hole"]).bool().to(DEVICE)
        depth = torch.from_numpy(d["depth_tgt"]).float().to(DEVICE)
        K = torch.from_numpy(d["K_tgt"]).float().to(DEVICE)
        # per-pixel reliability map U in [0,1]
        unc = None
        if use_unc:
            if unc_mode == "oracle":
                # UPPER BOUND oracle: reliability = how close teacher is to GT
                # (exp(-err/tau)); high where teacher completion matches real geometry.
                gt = (
                    torch.from_numpy(d["gt_rgb"]).float().permute(2, 0, 1).to(DEVICE)
                    / 255.0
                )
                err = (teacher - gt).abs().mean(0)  # (H,W)
                U = torch.exp(-err / max(unc_tau, 1e-3)).clamp(0, 1)
                unc = U**unc_gamma
            else:  # vggt (deployable, precomputed)
                up = os.path.join(unc_dir, Path(f).stem + ".npy")
                if os.path.exists(up):
                    U = np.load(up).astype(np.float32)
                    U = np.clip(U, 0.0, 1.0) ** unc_gamma
                    unc = torch.from_numpy(U).to(DEVICE)  # (H,W)
        return base, teacher, hole, depth, K, unc

    step = 0
    next_save = save_every
    while step < iters:
        base_rgb, teacher_rgb, hole_mask, depth_tgt, K, unc = sample()
        H, W = hole_mask.shape

        # EDT-filled depth so hole pixels get a boundary-continued depth anchor
        valid = (~hole_mask).cpu().numpy()
        _, (iy, ix) = ndi.distance_transform_edt(
            ~valid, return_distances=True, return_indices=True
        )
        depth_filled = torch.tensor(
            depth_tgt.cpu().numpy()[iy, ix], device=DEVICE, dtype=torch.float32
        )

        cond = torch.cat([base_rgb, hole_mask.float()[None], depth_filled[None]], 0)[
            None
        ]
        gbuf = net(cond)[0]  # (12,H,W)

        hys, hxs = torch.where(hole_mask)
        if hys.numel() < 16:
            step += 1
            continue
        anchor = depth_filled[hys, hxs].clamp(min=1e-3)
        hole_pc, z, rgb_pred = make_hole_gaussians(gbuf, hys, hxs, anchor, K)
        cam = identity_cam(H, W, K)
        composed, hole_render, hole_alpha = render_hole(
            hole_pc, base_rgb, hole_mask, cam, H, W
        )

        # hole-focused region (teacher only trustworthy in/near the hole)
        hole_dil = torch.from_numpy(
            ndi.binary_dilation(hole_mask.cpu().numpy(), iterations=dilate)
        ).to(DEVICE)
        m = hole_dil.float()[None]
        hm = hole_mask.float()[None]

        # (0) DIRECT colour supervision (v4 fix): the KILLER bug across v1-v3 was that colour
        # was only supervised THROUGH the rasteriser -> diffuse/weak gradient -> the head
        # output near-constant gray (pre-raster RGB std 0.016 vs teacher 0.127). Supervise the
        # per-hole-pixel predicted RGB DIRECTLY against teacher (identity: hole pixel (x,y)'s
        # gaussian colour should equal teacher[x,y]). Strong, direct gradient to colour.
        teacher_hole = teacher_rgb[:, hys, hxs].T  # (M,3)
        # V2-3: OracleGS-style uncertainty weighting. U high = oracle trusts the
        # teacher proposal here -> full supervision; U low = likely hallucination
        # -> down-weight so the student does NOT imitate unreliable teacher pixels.
        if unc is not None:
            u_pt = unc[hys, hxs].clamp(0, 1)[:, None]  # (M,1) per hole-pixel weight
            u_map = unc[None].clamp(0, 1)  # (1,H,W)
            u_scalar = float(unc[hole_mask].mean().clamp(1e-3, 1).item())
        else:
            u_pt = torch.ones((hys.numel(), 1), device=DEVICE)
            u_map = torch.ones((1, H, W), device=DEVICE)
            u_scalar = 1.0

        l1_color = ((rgb_pred - teacher_hole).abs() * u_pt).sum() / (
            u_pt.sum() * 3 + 1e-6
        )

        # (1) teacher colour THROUGH the render, per-pixel weighted by U (eq.6 ⊙U)
        l1_hole = ((hole_render - teacher_rgb).abs() * hm * u_map).sum() / (
            (hm * u_map).sum() * 3 + 1e-6
        )
        l1_ring = ((composed - teacher_rgb).abs() * m * u_map).sum() / (
            (m * u_map).sum() * 3 + 1e-6
        )
        l1 = 2.0 * l1_color + l1_hole + 0.5 * l1_ring
        # LPIPS modulated by scalar mean-uncertainty (eq.6 Ū·L_LPIPS)
        lp = (
            u_scalar
            * lpips_fn(composed[None] * 2 - 1, teacher_rgb[None] * 2 - 1).mean()
        )

        # (2) COVERAGE (v1 bug fix): the hole gaussians' accumulated alpha must ->1
        # INSIDE the hole, so they actually paint (v1 collapsed opacity to ~0.01 ->
        # gray hole at eval). This gives opacity a direct, strong gradient.
        cover = (F.relu(1.0 - hole_alpha) * hm).sum() / (hm.sum() + 1e-6)

        # (3) geometry: hole depth stays near boundary-continued anchor + local smoothness
        d_anchor = ((z - anchor).abs()).mean()
        # smoothness on predicted depth map inside hole
        dmap = F.softplus(gbuf[0:1]) + depth_filled[None]
        dx = (dmap[:, :, 1:] - dmap[:, :, :-1]).abs().mean()
        dy = (dmap[:, 1:, :] - dmap[:, :-1, :]).abs().mean()
        d_smooth = dx + dy

        loss = (
            w_l1 * l1
            + w_lpips * lp
            + w_cover * cover
            + w_dsmooth * (d_anchor + 0.1 * d_smooth)
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1

        if step % log_every == 0:
            print(
                f"[step {step}/{iters}] loss={loss.item():.4f} l1h={l1_hole.item():.4f} "
                f"lpips={lp.item():.4f} cover={cover.item():.4f} d_anchor={d_anchor.item():.4f} "
                f"nhole={hys.numel()} lr={sched.get_last_lr()[0]:.6f}",
                flush=True,
            )
        if step >= next_save or step == iters:
            next_save += save_every
            torch.save(net.state_dict(), out / f"student_step{step:06d}.pt")
            print(f"  saved student_step{step:06d}.pt", flush=True)

    torch.save(net.state_dict(), out / "student_final.pt")
    print(f"DONE -> {out}/student_final.pt", flush=True)


if __name__ == "__main__":
    main()
