"""Debug: compare my SpatialRefine baseline render vs Flash3D native render_images on ONE scene.
Localizes the src-frame PSNR gap (mine 25.5 vs repro 38.4)."""

import os, sys

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, "/root/sv3d-lab")
sys.path.insert(0, "/root/sv3d-lab/paper_a_explicit3d")
import torch
from model.spatial_refine_model import SpatialRefineModel, SpatialRefineConfig
from train_paper_a import load_re10k, build_cfg, to_device
from evaluation.evaluator import Evaluator


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return float(-10 * torch.log10(mse))


def main():
    device = "cuda"
    split = "/root/projects/flash3d/splits/re10k_mine_filtered/test_files_present.txt"
    cfg = build_cfg([1, 2, 3], split)
    Re10K = load_re10k()
    ds = Re10K(cfg, split="test")
    ds._seq_key_src_idx_pairs = ds._seq_key_src_idx_pairs[:5]
    ds.length = 5
    from torch.utils.data import DataLoader
    from datasets.util import custom_collate

    loader = DataLoader(ds, 1, shuffle=False, num_workers=0, collate_fn=custom_collate)

    model = SpatialRefineModel(cfg, SpatialRefineConfig()).to(device)
    fl = "/root/projects/flash3d/checkpoints/model_re10k_v2.pth"
    model.load_visible_pretrained(fl, device=device)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
    model.eval()

    # native Flash3D model for reference
    native = model.visible  # GaussianPredictor
    ev = Evaluator(crop_border=True).to(device)

    for batch in loader:
        inputs = to_device(batch, device)
        tids = [f for f in [1, 2, 3] if ("color", f, 0) in inputs]
        if not tids:
            continue
        # --- native flash3d render ---
        with torch.no_grad():
            ninputs = {k: v for k, v in inputs.items()}
            ninputs["target_frame_ids"] = tids
            nout = native(ninputs)
        # native src render
        if ("color_gauss", 0, 0) in nout:
            nsrc = nout[("color_gauss", 0, 0)].clamp(0, 1)
            print("native src PSNR:", psnr(nsrc, inputs[("color", 0, 0)]))
        for f in tids:
            if ("color_gauss", f, 0) in nout:
                print(
                    f"native tgt{f} PSNR:",
                    psnr(
                        nout[("color_gauss", f, 0)].clamp(0, 1), inputs[("color", f, 0)]
                    ),
                )
        # --- my render ---
        with torch.no_grad():
            mout = model(inputs, tids)
            msrc = model(inputs, [0])
        if ("render", 0) in msrc:
            print(
                "MINE src PSNR:",
                psnr(msrc[("render", 0)].clamp(0, 1), inputs[("color", 0, 0)]),
            )
        for f in tids:
            print(
                f"MINE tgt{f} PSNR:",
                psnr(mout[("render", f)].clamp(0, 1), inputs[("color", f, 0)]),
            )
        # shapes
        print(
            "native src shape",
            nsrc.shape if ("color_gauss", 0, 0) in nout else None,
            "mine src shape",
            msrc[("render", 0)].shape,
        )
        # === param diagnostics ===
        from einops import rearrange

        g = model._visible_gaussians_raw(inputs, nout)
        print("--- my extracted gauss ---")
        print(
            "scaling mean",
            float(g["scaling"].mean()),
            "opacity mean",
            float(g["opacity"].mean()),
        )
        print(
            "rgb mean",
            g["rgb"].mean(dim=(0, 1)).tolist(),
            "min",
            float(g["rgb"].min()),
            "max",
            float(g["rgb"].max()),
        )
        print("rotation mean", g["rotation"].mean(dim=(0, 1)).tolist())
        fdc = rearrange(
            nout["gauss_features_dc"],
            "(b n) c h w -> b (n h w) c",
            n=cfg.model.gaussians_per_pixel,
        )
        print(
            "native features_dc mean",
            fdc.mean(dim=(0, 1)).tolist(),
            "min",
            float(fdc.min()),
            "max",
            float(fdc.max()),
        )
        print(
            "native has features_rest:",
            "gauss_features_rest" in nout,
            "max_sh_degree",
            cfg.model.max_sh_degree,
        )
        for f in [0] + tids:
            kk = inputs.get(("K_tgt", f), None)
            print(f"K_tgt[{f}]", None if kk is None else kk[0].diagonal().tolist())
        break


if __name__ == "__main__":
    main()
