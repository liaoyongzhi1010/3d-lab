"""Verify official Flash3D weights load into UniDepthSpatial with spatial branch at init.

Checks:
  1. All backbone keys (encoder, models.depth, models.gauss_decoder_*) load from official ckpt.
  2. spatial_branch.* keys are NOT in the official ckpt (stay at random init).
  3. fusion gate gamma == 0 (identity at start -> clean A/B: step0 == Flash3D).
Run on server with the flash3d venv from /root/projects/flash3d.
"""

import sys
import torch

sys.path.insert(0, "/root/projects/flash3d")

from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize_config_dir


def main():
    with initialize_config_dir(
        config_dir="/root/projects/flash3d/configs", version_base=None
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=layered_re10k_spatial",
                "model.depth.version=v1",
            ],
        )
    from models.model import GaussianPredictor

    model = GaussianPredictor(cfg)

    ck = torch.load(
        "/home/data/sv3d-lab/checkpoints/flash3d_official/model_0000000.pth",
        map_location="cpu",
    )
    official = ck["model"]
    model_keys = set(model.state_dict().keys())
    official_keys = set(official.keys())

    spatial_keys = [k for k in model_keys if "spatial_branch" in k]
    backbone_keys = [
        k for k in model_keys if "spatial_branch" not in k and "backproject" not in k
    ]
    loaded = [k for k in backbone_keys if k in official_keys]
    missing = [k for k in backbone_keys if k not in official_keys]

    print(f"model total keys: {len(model_keys)}")
    print(f"spatial_branch keys (should NOT be in official): {len(spatial_keys)}")
    print(
        f"  any spatial key in official? {[k for k in spatial_keys if k in official_keys]}"
    )
    print(
        f"backbone keys: {len(backbone_keys)}, loadable from official: {len(loaded)}, missing: {len(missing)}"
    )
    if missing:
        print("  MISSING (first 10):", missing[:10])

    # actually load
    new_dict = {}
    for k, v in official.items():
        if "backproject_depth" in k:
            if k in model.state_dict():
                new_dict[k] = model.state_dict()[k].clone()
        else:
            new_dict[k] = v.clone()
    result = model.load_state_dict(new_dict, strict=False)
    print(
        f"load_state_dict missing_keys={len(result.missing_keys)} (should = spatial+backproject)"
    )
    print(
        f"load_state_dict unexpected_keys={len(result.unexpected_keys)} (should be 0)"
    )

    gamma = model.models["unidepth_extended"].spatial_branch.fusion.gamma.item()
    print(f"fusion gamma = {gamma} (must be 0.0 for clean A/B)")
    print(
        "VERIFICATION",
        "PASS"
        if (gamma == 0.0 and len(missing) == 0 and len(result.unexpected_keys) == 0)
        else "FAIL",
    )


if __name__ == "__main__":
    main()
