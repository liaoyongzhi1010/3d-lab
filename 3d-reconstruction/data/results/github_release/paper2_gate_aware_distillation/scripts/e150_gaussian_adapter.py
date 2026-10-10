"""E-150: Color-only Gaussian Adapter for Paper 2 (disocclusion prior distillation).

Upgrades the Paper-2 student from an image-plane residual (E-030) to a
representation-level adapter that edits Flash3D's per-pixel Gaussian COLOR
(features_dc) and re-renders, while FREEZING geometry (xyz / opacity / scaling /
rotation). This keeps the fast feed-forward 3DGS representation and only distills
the teacher's disocclusion appearance into the color field.

Design (aligned with the project's hard rules):
  * Do NOT touch visible regions: identity loss ties the adapted render to the
    baseline Flash3D render on visible pixels.
  * Only intervene where it survives downstream: we edit features_dc (the color
    that the splatter actually renders), not a late 2D residual.
  * Oracle-selected: disocclusion loss is applied only on frames labeled as
    teacher-helpful (label==1) using offline teacher/GT outcomes.

Per (scene, target frame):
  1. Run frozen Flash3D -> Gaussian outputs (means/opacity/scaling/rotation/
     features_dc[/rest]).
  2. adapter CNN takes [features_dc, base_render, vis_mask] and predicts a color
     residual on the source image plane; features_dc' = features_dc + residual.
  3. render_predicted with features_dc' to the target camera -> adapted render.
  4. losses:
       L_dis  = L1(adapted, teacher) on invisible      (oracle-selected: label==1 only)
       L_gt   = L1(adapted, gt)      on invisible       (small weight)
       L_id   = L1(adapted, base_render) on visible      (identity / no-harm)
  5. baseline geometry stays frozen; only adapter weights train.

Target sources:
  --teacher_dir : dir with teacher renders. Supports either
      <scene>/adaptive2_f%03d.png  (E-021 panel layout), or
      teacher_<scene>.npy [F,3,H,W]  (full per-frame dump).

Run on server (flash3d env):
  python e150_gaussian_adapter.py \
    --scenes test_249fd0890d439aa9 test_5a15212752d1659a ... \
    --teacher_dir /home/data/E-021_panels/frames \
    --out /home/data/E-150_gaussian_adapter
"""

import os, sys, json, argparse, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from einops import rearrange


def load_flash3d_dependencies():
    flash3d_root = os.environ.get("FLASH3D_ROOT", "/root/projects/flash3d")
    if not os.path.isdir(flash3d_root):
        raise RuntimeError(
            "Flash3D is required for training but was not found at "
            f"{flash3d_root!r}. Clone the official repository from "
            "https://github.com/eldar/flash3d, set FLASH3D_ROOT to its checkout, "
            "and include that checkout in PYTHONPATH."
        )
    sys.path.insert(0, flash3d_root)
    try:
        from omegaconf import OmegaConf
        from models.model import GaussianPredictor, to_device
        from models.decoder.gauss_util import (
            focal2fov,
            getProjectionMatrix,
            render_predicted,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Flash3D or one of its dependencies could not be imported. Install "
            "the official repository from https://github.com/eldar/flash3d, set "
            "FLASH3D_ROOT to its checkout, and include that checkout in PYTHONPATH. "
            f"Import failed: {exc}"
        ) from None
    return (
        flash3d_root,
        OmegaConf,
        GaussianPredictor,
        to_device,
        focal2fov,
        getProjectionMatrix,
        render_predicted,
    )


DEVICE = "cuda:0"
NF = 49
RES = 560


# ----- geometry / camera helpers (copied from _render_gen3r_scene_evidence.py) -----
def rescale_K_for_crop(K, W, H, target=560):
    scale = target / min(W, H)
    newW, newH = round(W * scale), round(H * scale)
    K = K.copy()
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] *= scale
    K[1, 2] *= scale
    K[0, 2] -= (newW - target) / 2.0
    K[1, 2] -= (newH - target) / 2.0
    return K


def build_K_norm(frame):
    W, H = frame["w"], frame["h"]
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = frame["fl_x"] / W
    K[1, 1] = frame["fl_y"] / H
    K[0, 2] = frame["cx"] / W
    K[1, 2] = frame["cy"] / H
    return K


def make_frame_inputs(scene_dir, frame, frame_name, cfg):
    im = Image.open(os.path.join(scene_dir, frame["file_path"])).convert("RGB")
    im = im.resize((cfg.dataset.width, cfg.dataset.height), Image.LANCZOS)
    arr = np.asarray(im).astype(np.float32) / 255.0
    color = torch.from_numpy(arr).permute(2, 0, 1)[None]
    pad = int(cfg.dataset.pad_border_aug)
    color_aug = (
        torch.nn.functional.pad(color, (pad, pad, pad, pad), mode="replicate")
        if pad
        else color
    )
    K = build_K_norm(frame)
    K_tgt = K.copy()
    K_tgt[0, :] *= cfg.dataset.width
    K_tgt[1, :] *= cfg.dataset.height
    K_src = K.copy()
    K_src[0, 0] *= cfg.dataset.width
    K_src[1, 1] *= cfg.dataset.height
    K_src[0, 2] *= cfg.dataset.width + cfg.dataset.pad_border_aug * 2
    K_src[1, 2] *= cfg.dataset.height + cfg.dataset.pad_border_aug * 2
    inv_K = np.linalg.pinv(K_src)
    T_c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
    return {
        ("K_tgt", frame_name): torch.from_numpy(K_tgt)[None],
        ("K_src", frame_name): torch.from_numpy(K_src)[None],
        ("inv_K_src", frame_name): torch.from_numpy(inv_K)[None],
        ("color", frame_name, 0): color,
        ("color_aug", frame_name, 0): color_aug,
        ("T_c2w", frame_name): torch.from_numpy(T_c2w)[None],
        ("T_w2c", frame_name): torch.from_numpy(np.linalg.inv(T_c2w))[None],
    }


def build_inputs(scene_dir, frames, tgt, cfg):
    inputs = {"target_frame_ids": [1]}
    inputs.update(make_frame_inputs(scene_dir, frames[0], 0, cfg))
    inputs.update(make_frame_inputs(scene_dir, frames[tgt], 1, cfg))
    return inputs


def render_with_features(model, outputs, features_dc, frame_id, K560, HW=(RES, RES)):
    """Differentiable render using a (possibly adapted) features_dc tensor.

    features_dc: [(b n), c, h, w] same layout as outputs['gauss_features_dc'].
    Geometry tensors are used as-is (frozen).
    """
    cfg = model.cfg
    gpp = model.cfg.model.gaussians_per_pixel
    H, W = HW
    if frame_id == 0:
        T = torch.eye(4, device=DEVICE).unsqueeze(0)
    else:
        T = outputs[("cam_T_cam", 0, 1)].float()
    pos_input = outputs["gauss_means"].float()
    if cfg.train.use_gt_poses:
        pos = pos_input
    else:
        P = rearrange(
            T[:, :3, :][:, None, ...].repeat(1, gpp, 1, 1), "b n ... -> (b n) ..."
        )
        pos = torch.matmul(P, pos_input)
    pc = {
        "xyz": rearrange(pos[:, :3, :], "(b n) c l -> b (n l) c", n=gpp),
        "opacity": rearrange(
            outputs["gauss_opacity"], "(b n) c h w -> b (n h w) c", n=gpp
        ),
        "scaling": rearrange(
            outputs["gauss_scaling"], "(b n) c h w -> b (n h w) c", n=gpp
        ),
        "rotation": rearrange(
            outputs["gauss_rotation"], "(b n) c h w -> b (n h w) c", n=gpp
        ),
        "features_dc": rearrange(features_dc, "(b n) c h w -> b (n h w) 1 c", n=gpp),
    }
    if cfg.model.max_sh_degree > 0 and outputs.get("gauss_features_rest") is not None:
        pc["features_rest"] = rearrange(
            outputs["gauss_features_rest"],
            "(b n) (sh c) h w -> b (n h w) sh c",
            c=3,
            n=gpp,
        )
    b = 0
    fx, fy = float(K560[0, 0]), float(K560[1, 1])
    fovY = focal2fov(fy, H)
    fovX = focal2fov(fx, W)
    proj_mtrx = getProjectionMatrix(
        cfg.dataset.znear, cfg.dataset.zfar, fovX, fovY, pX=0, pY=0
    ).to(DEVICE)
    wvt = T[b].transpose(0, 1).float()
    cc = (-wvt[3, :3] @ wvt[:3, :3].transpose(0, 1)).float()
    proj_mtrx = proj_mtrx.transpose(0, 1).float()
    fpt = (wvt @ proj_mtrx).float()
    pcb = {k: v[b].contiguous().float() for k, v in pc.items()}
    bg = torch.tensor(cfg.model.bg_colour, dtype=torch.float32, device=DEVICE)
    out = render_predicted(
        cfg,
        pcb,
        wvt,
        fpt,
        proj_mtrx,
        cc,
        (fovX, fovY),
        (H, W),
        bg,
        cfg.model.max_sh_degree,
    )
    return out["render"].clamp(0, 1)


class ColorAdapter(nn.Module):
    """Predicts a residual on features_dc from [features_dc, base_render, vis]."""

    def __init__(self, feat_ch, hidden=32):
        super().__init__()
        in_ch = feat_ch + 3 + 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, feat_ch, 3, padding=1),
        )
        # start as identity (zero residual)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat_dc, base_render_src, vis_src):
        x = torch.cat([feat_dc, base_render_src, vis_src], dim=1)
        return self.net(x)


def load_teacher(teacher_dir, scene, k, hw):
    """Return teacher render [3,H,W] in [0,1] or None."""
    npy = os.path.join(teacher_dir, f"teacher_{scene}.npy")
    if os.path.exists(npy):
        arr = np.load(npy)
        if k < len(arr):
            t = torch.from_numpy(arr[k].astype(np.float32))
            if t.max() > 1.5:
                t = t / 255.0
            return F.interpolate(
                t[None], size=hw, mode="bilinear", align_corners=False
            )[0].clamp(0, 1)
        return None
    png = os.path.join(teacher_dir, scene, f"adaptive2_f{k:03d}.png")
    if os.path.exists(png):
        im = Image.open(png).convert("RGB").resize((hw[1], hw[0]), Image.BILINEAR)
        return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).permute(
            2, 0, 1
        )
    return None


def load_vis(scene_dir, k, hw):
    v = np.load(os.path.join(scene_dir, "visibility.npy"))[k].astype(np.float32)
    im = Image.fromarray((v * 255).astype(np.uint8)).resize(
        (hw[1], hw[0]), Image.NEAREST
    )
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)[None]


def load_gt(scene_dir, frame, hw):
    im = Image.open(os.path.join(scene_dir, frame["file_path"])).convert("RGB")
    W, H = im.size
    s = RES / min(W, H)
    im = im.resize((round(W * s), round(H * s)))
    W2, H2 = im.size
    left, top = (W2 - RES) // 2, (H2 - RES) // 2
    im = im.crop((left, top, left + RES, top + RES)).resize(
        (hw[1], hw[0]), Image.BILINEAR
    )
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).permute(2, 0, 1)


def psnr_mask(pred, gt, mask):
    m = mask.expand_as(pred)
    denom = m.sum().clamp(min=1.0)
    mse = (((pred - gt) ** 2) * m).sum() / denom
    return float(10 * torch.log10(1.0 / mse.clamp(min=1e-10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="eval_official/.hydra/config.yaml")
    ap.add_argument("--data_root", default="/home/data/gen3r_re10k/re10k")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--teacher_dir", required=True)
    ap.add_argument(
        "--frame_dataset", default=None, help="E-144 frame json for gate labels"
    )
    ap.add_argument("--holdout", default="", help="comma scene ids for holdout eval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--render_hw", type=int, default=256, help="train render resolution (speed)"
    )
    ap.add_argument("--w_gt", type=float, default=0.2)
    ap.add_argument("--w_id", type=float, default=2.0)
    ap.add_argument("--gate_aware", action="store_true")
    args = ap.parse_args()
    global OmegaConf, GaussianPredictor, to_device
    global focal2fov, getProjectionMatrix, render_predicted
    (
        flash3d_root,
        OmegaConf,
        GaussianPredictor,
        to_device,
        focal2fov,
        getProjectionMatrix,
        render_predicted,
    ) = load_flash3d_dependencies()
    os.chdir(flash3d_root)
    os.makedirs(args.out, exist_ok=True)

    cfg = OmegaConf.load(args.cfg)
    cfg.data_loader.batch_size = 1
    cfg.data_loader.num_workers = 0
    cfg.dataset.scale_pose_by_depth = False
    model = GaussianPredictor(cfg).to(DEVICE)
    model.load_model("checkpoints", ckpt_ids=0)
    model.set_eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # gate labels per (scene, frame_idx)
    gate = {}
    if args.frame_dataset and os.path.exists(args.frame_dataset):
        for r in json.load(open(args.frame_dataset)):
            gate[(r["sid"], r["frame_idx"])] = r["label"]

    hw = (args.render_hw, args.render_hw)
    hold = set(x for x in args.holdout.split(",") if x)
    train_scenes = [
        s
        for s in args.scenes
        if s.replace("test_", "").replace("train_", "") not in hold and s not in hold
    ]
    hold_scenes = [s for s in args.scenes if s not in train_scenes]
    print(
        f"train_scenes={len(train_scenes)} hold_scenes={len(hold_scenes)}", flush=True
    )

    # discover feat channels lazily on first sample
    adapter = None
    opt = None

    def iter_samples(scenes):
        for scene in scenes:
            scene_dir = os.path.join(args.data_root, scene)
            tf = json.load(open(os.path.join(scene_dir, "transforms.json")))
            frames = tf["frames"][:NF]
            f0 = frames[0]
            Kraw = np.eye(3, dtype=np.float32)
            Kraw[0, 0] = f0["fl_x"]
            Kraw[1, 1] = f0["fl_y"]
            Kraw[0, 2] = f0["cx"]
            Kraw[1, 2] = f0["cy"]
            K560 = torch.tensor(
                rescale_K_for_crop(Kraw, f0["w"], f0["h"], 560),
                dtype=torch.float32,
                device=DEVICE,
            )
            sid = scene.replace("test_", "")
            for tgt in range(1, NF):
                yield scene, sid, scene_dir, frames, tgt, K560

    def forward_sample(sample, train=True):
        nonlocal adapter, opt
        scene, sid, scene_dir, frames, tgt, K560 = sample
        teacher = load_teacher(args.teacher_dir, scene, tgt, hw)
        if teacher is None:
            return None
        teacher = teacher.to(DEVICE)
        inputs = build_inputs(scene_dir, frames, tgt, cfg)
        to_device(inputs, DEVICE)
        with torch.no_grad():
            outputs = model(inputs)
            base_dc = outputs["gauss_features_dc"].float()  # [(b n), c, h, w]
            base_render = render_with_features(model, outputs, base_dc, 1, K560, hw)
        vis = load_vis(scene_dir, tgt, hw).to(DEVICE)
        inv = 1.0 - vis
        gt = load_gt(scene_dir, frames[tgt], hw).to(DEVICE)
        if adapter is None:
            adapter = ColorAdapter(base_dc.shape[1]).to(DEVICE)
            opt = torch.optim.Adam(adapter.parameters(), lr=args.lr)
        # adapter inputs live on the SOURCE feature plane (aligned to base_dc).
        # source color (frame 0) resized to feature-plane HxW, broadcast over gpp.
        src_color = inputs[("color", 0, 0)].float()  # [1,3,H0,W0]
        feat_hw = base_dc.shape[-2:]
        src_color = F.interpolate(
            src_color, size=feat_hw, mode="bilinear", align_corners=False
        )
        n_group = base_dc.shape[0]
        src_color = src_color.expand(n_group, -1, -1, -1)
        # a source-plane occupancy proxy = 1 everywhere (source is fully observed)
        ones = torch.ones((n_group, 1, feat_hw[0], feat_hw[1]), device=DEVICE)
        resid = adapter(base_dc, src_color, ones)
        dc_adapted = base_dc + resid
        adapted = render_with_features(model, outputs, dc_adapted, 1, K560, hw)
        # losses
        label = gate.get((sid, tgt), 1)
        use_dis = (not args.gate_aware) or (label == 1)
        l_dis = (
            (torch.abs(adapted - teacher) * inv).mean()
            if use_dis
            else torch.tensor(0.0, device=DEVICE)
        )
        l_gt = (torch.abs(adapted - gt) * inv).mean()
        l_id = (torch.abs(adapted - base_render) * vis).mean()
        loss = l_dis + args.w_gt * l_gt + args.w_id * l_id
        metrics = {
            "base_inv": psnr_mask(base_render, gt, inv),
            "adapt_inv": psnr_mask(adapted, gt, inv),
            "base_vis": psnr_mask(base_render, gt, vis),
            "adapt_vis": psnr_mask(adapted, gt, vis),
            "teach_inv": psnr_mask(teacher, gt, inv),
        }
        return loss, metrics

    train_samples = list(iter_samples(train_scenes))
    print(f"train samples (scene,frame) = {len(train_samples)}", flush=True)

    for ep in range(args.epochs):
        tot = 0.0
        n = 0
        for s in train_samples:
            r = forward_sample(s, train=True)
            if r is None:
                continue
            loss, _ = r
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            n += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep{ep:03d} loss={tot / max(1, n):.5f} n={n}", flush=True)

    torch.save(adapter.state_dict(), os.path.join(args.out, "adapter.pt"))

    def eval_scenes(scenes, name):
        rows = []
        with torch.no_grad():
            for s in iter_samples(scenes):
                r = forward_sample(s, train=False)
                if r is None:
                    continue
                _, m = r
                m["sid"] = s[1]
                m["frame"] = s[4]
                rows.append(m)
        if rows:
            di = np.mean([r["adapt_inv"] - r["base_inv"] for r in rows])
            dv = np.mean([r["adapt_vis"] - r["base_vis"] for r in rows])
            print(
                f"[{name}] n={len(rows)} inv Δ={di:+.3f} vis Δ={dv:+.3f} "
                f"base_inv={np.mean([r['base_inv'] for r in rows]):.2f} "
                f"adapt_inv={np.mean([r['adapt_inv'] for r in rows]):.2f} "
                f"teach_inv={np.mean([r['teach_inv'] for r in rows]):.2f}",
                flush=True,
            )
        return rows

    out = {
        "train": eval_scenes(train_scenes, "train"),
        "holdout": eval_scenes(hold_scenes, "holdout") if hold_scenes else [],
        "config": {k: str(v) for k, v in vars(args).items()},
    }
    json.dump(out, open(os.path.join(args.out, "results.json"), "w"), indent=2)
    print("E-150 DONE", flush=True)


if __name__ == "__main__":
    main()
