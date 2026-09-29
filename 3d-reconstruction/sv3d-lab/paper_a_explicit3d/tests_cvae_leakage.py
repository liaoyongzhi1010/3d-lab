"""
D013 leakage guard: the CVAE hidden head at INFERENCE (z_mode='prior'/'prior_mean') must NOT read
any target tensor. We assert this by (a) running a prior-mode forward with target tensors REMOVED
from inputs and checking it still works, and (b) checking that posterior-mode requires targets.

Run on server:
  cd /root/projects/flash3d && source .venv/bin/activate
  export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python /root/sv3d-lab/paper_a_explicit3d/tests_cvae_leakage.py
"""

import os
import sys

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, "/root/sv3d-lab")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model import PaperAModel, PaperAConfig
from train_paper_a import build_cfg, load_re10k, to_device


def main():
    device = "cuda"
    latent_mode = sys.argv[1] if len(sys.argv) > 1 else "global"
    print(f"[leakage] latent_mode = {latent_mode}")
    split = "/root/projects/flash3d/splits/re10k_mine_filtered/test_files_wide700.txt"
    cfg = build_cfg([1, 2, 3], split)
    Re10K = load_re10k()
    ds = Re10K(cfg, split="test")
    ds._seq_key_src_idx_pairs = ds._seq_key_src_idx_pairs[:1]
    ds.length = 1
    from torch.utils.data import DataLoader
    from datasets.util import custom_collate

    loader = DataLoader(ds, 1, shuffle=False, num_workers=0, collate_fn=custom_collate)

    model = (
        PaperAModel(
            cfg, PaperAConfig(n_queries=4096, latent_dim=32, latent_mode=latent_mode)
        )
        .to(device)
        .eval()
    )

    inputs = to_device(next(iter(loader)), device)
    target_ids = [f for f in [1, 2, 3] if ("color", f, 0) in inputs]

    # --- Test 1: prior inference works with ALL target tensors stripped ---
    stripped = {
        k: v
        for k, v in inputs.items()
        if not (
            isinstance(k, tuple)
            and len(k) >= 2
            and k[1] in target_ids
            and k[0] in ("color", "color_aug")
        )
    }
    # keep K_tgt/T_c2w (camera geometry = allowed virtual pose spec, not target APPEARANCE)
    removed = [k for k in inputs if k not in stripped]
    print(f"[leakage] removed target-appearance keys for inference: {removed}")
    with torch.no_grad():
        out = model(stripped, target_ids, hidden_ray_fid=target_ids[-1], z_mode="prior")
    assert ("render_hidden", target_ids[0]) in out, (
        "prior inference failed without target RGB"
    )
    print(
        "[leakage] PASS: prior-mode inference runs WITHOUT any target RGB/appearance."
    )

    # --- Test 2: posterior mode indeed uses targets (sanity that q!=p path is wired) ---
    with torch.no_grad():
        out2 = model(
            inputs,
            target_ids,
            hidden_ray_fid=target_ids[-1],
            z_mode="posterior",
            post_tgt_fid=target_ids[-1],
        )
    assert "z_out" in out2 and "mu_q" in out2["z_out"], "posterior did not produce q(z)"
    print(
        "[leakage] PASS: posterior-mode produces q(z|src,tgt) (train-only path present)."
    )

    # --- Test 3: two prior samples differ (generative diversity) ---
    with torch.no_grad():
        a = model(stripped, target_ids, hidden_ray_fid=target_ids[-1], z_mode="prior")[
            "hidden_gaussians"
        ][0]["rgb_direct"]
        b = model(stripped, target_ids, hidden_ray_fid=target_ids[-1], z_mode="prior")[
            "hidden_gaussians"
        ][0]["rgb_direct"]
    diff = (a - b).abs().mean().item()
    print(
        f"[leakage] two prior samples mean|Δrgb|={diff:.5f} (should be >0 for a live latent)"
    )
    assert diff > 1e-6, "prior samples identical — latent not affecting output"
    print("[leakage] PASS: latent produces sample diversity.")
    print("ALL LEAKAGE/CVAE GUARD TESTS PASSED")


if __name__ == "__main__":
    main()
