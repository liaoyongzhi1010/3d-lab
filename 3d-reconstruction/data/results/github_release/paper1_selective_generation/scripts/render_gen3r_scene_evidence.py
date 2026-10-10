"""Render Flash3D evidence directly from Gen3R scene folders.

Avoids Flash3D's RE10K dataloader so train_* scenes align with
/home/data/gen3r_re10k/re10k/{train,test}_*/images + transforms.json.

This constructs the minimal Flash3D input dict manually from source frame 0 and
one target frame at a time, then renders the predicted Gaussians to Gen3R's
560-crop target cameras.
"""

import os, sys, json, argparse
import numpy as np
import torch
from PIL import Image
from einops import rearrange


def load_flash3d_dependencies():
    flash3d_root = os.environ.get("FLASH3D_ROOT", "/root/projects/flash3d")
    if not os.path.isdir(flash3d_root):
        raise RuntimeError(
            "Flash3D is required for rendering but was not found at "
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


def build_base_point_clouds(model, outputs):
    return {
        "xyz_src": outputs["gauss_means"].float(),
        "opacity": outputs["gauss_opacity"],
        "scaling": outputs["gauss_scaling"],
        "rotation": outputs["gauss_rotation"],
        "features_dc": outputs["gauss_features_dc"],
        "features_rest": outputs.get("gauss_features_rest", None),
        "gpp": model.cfg.model.gaussians_per_pixel,
    }


def render_at(model, outputs, base_pcs, frame_id, K560, HW=(560, 560)):
    cfg = model.cfg
    gpp = base_pcs["gpp"]
    H, W = HW
    if frame_id == 0:
        T = torch.eye(4, device=DEVICE).unsqueeze(0)
    else:
        T = outputs[("cam_T_cam", 0, 1)].float()
    pos_input = base_pcs["xyz_src"]
    if cfg.train.use_gt_poses:
        pos = pos_input
    else:
        P = rearrange(
            T[:, :3, :][:, None, ...].repeat(1, gpp, 1, 1), "b n ... -> (b n) ..."
        )
        pos = torch.matmul(P, pos_input)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="eval_official/.hydra/config.yaml")
    ap.add_argument("--data_root", default="/home/data/gen3r_re10k/re10k")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
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
    for scene in args.scenes:
        scene_dir = os.path.join(args.data_root, scene)
        tf = json.load(open(os.path.join(scene_dir, "transforms.json")))
        frames = tf["frames"][:NF]
        Kraw = np.eye(3, dtype=np.float32)
        f0 = frames[0]
        Kraw[0, 0] = f0["fl_x"]
        Kraw[1, 1] = f0["fl_y"]
        Kraw[0, 2] = f0["cx"]
        Kraw[1, 2] = f0["cy"]
        K560 = torch.tensor(
            rescale_K_for_crop(Kraw, f0["w"], f0["h"], 560),
            dtype=torch.float32,
            device=DEVICE,
        )
        renders = [None] * NF
        for tgt in range(NF):
            inputs = build_inputs(scene_dir, frames, max(tgt, 1), cfg)
            to_device(inputs, DEVICE)
            with torch.no_grad():
                outputs = model(inputs)
                base_pcs = build_base_point_clouds(model, outputs)
                renders[tgt] = (
                    render_at(model, outputs, base_pcs, 0 if tgt == 0 else 1, K560)
                    .cpu()
                    .numpy()
                )
            if (tgt + 1) % 12 == 0:
                print(f"  [{scene}] {tgt + 1}/{NF}", flush=True)
        arr = np.stack(renders).astype(np.float32)
        outp = os.path.join(args.out, f"f3d_{scene}.npy")
        np.save(outp, arr)
        print(f"saved {outp} shape={arr.shape}", flush=True)
    print("GEN3R_SCENE_EVIDENCE_DONE", flush=True)


if __name__ == "__main__":
    main()
