"""Step A-4: backward + optimizer + 3-view render VRAM smoke test.

Must confirm the full training loop fits in 48GB (red line 44GB).
Tests: forward -> build_scene -> merge -> render 3 targets -> loss -> backward -> optimizer.step.
Reports peak VRAM. GO if peak < 44GB.

Run (Flash3D venv):
  python -m paper_b_generative3d.smoke_dual_layer_vram 2>&1
"""

import torch
import torch.nn.functional as F

from paper_b_generative3d.backbone_flash3d import Flash3DBackbone
from paper_b_generative3d.data_loader_re10k import build_flash3d_cfg, FLASH3D_CKPT
from paper_b_generative3d.model.canonical_dual_layer import CanonicalDualLayer


def main():
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    print("Loading Flash3D backbone (frozen)...", flush=True)
    cfg = build_flash3d_cfg(batch_size=1, num_workers=0, stage="dev")
    backbone = Flash3DBackbone(cfg, FLASH3D_CKPT, device=str(device))
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    print("Building CanonicalDualLayer...", flush=True)
    model = CanonicalDualLayer(
        anchor_backbone=backbone,
        feat_proj_dim=256,
        latent_dim=64,
        n_free=4096,
        n_cross=3,
        n_self=2,
    ).to(device)

    # Only train free generator + feat_proj + latent_enc (anchor frozen)
    trainable = [
        p for n, p in model.named_parameters() if "anchor" not in n and p.requires_grad
    ]
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    n_params = sum(p.numel() for p in trainable)
    print(f"Trainable params: {n_params / 1e6:.1f}M", flush=True)

    # Load one real batch
    from paper_b_generative3d.viz_compare import _create_loader_from_cfg

    _, loader = _create_loader_from_cfg(cfg)
    inputs = next(iter(loader))
    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    H, W = 256, 384
    print("Forward: build_scene...", flush=True)
    scene = model.build_scene(inputs)
    merged = model.merge_scene(scene)
    print(f"  Anchor: {scene['anchor']['xyz'].shape[0]} gaussians", flush=True)
    print(f"  Free: {scene['free']['xyz'].shape[0]} gaussians", flush=True)
    print(f"  Total: {merged['xyz'].shape[0]} gaussians", flush=True)

    # Render 3 target views
    loss_total = torch.tensor(0.0, device=device)
    target_fids = [f for f in [1, 2, 3] if ("color", f, 0) in inputs][:3]
    print(f"Rendering {len(target_fids)} target views...", flush=True)
    for fid in target_fids:
        cam = inputs.get(("cam_T_cam", 0, fid))
        if cam is None:
            continue
        K_tgt = inputs.get(("K_tgt", fid), inputs[("K_src", 0)])
        gt = inputs["color", fid, 0][0]
        render = backbone.render_gaussians(merged, cam[0], K_tgt[0], H, W, batch_idx=0)
        loss_total = loss_total + F.l1_loss(render.clamp(0, 1), gt)

    # KL loss
    mu, logvar = scene["mu"], scene["logvar"]
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
    loss_total = loss_total + 0.001 * kl

    print(f"Loss: {loss_total.item():.4f}", flush=True)
    print("Backward...", flush=True)
    opt.zero_grad()
    loss_total.backward()
    print("Optimizer step...", flush=True)
    opt.step()

    peak = torch.cuda.max_memory_allocated(device) / 1e9
    current = torch.cuda.memory_allocated(device) / 1e9
    print(f"\n=== VRAM SMOKE RESULT ===", flush=True)
    print(f"Peak: {peak:.2f} GB", flush=True)
    print(f"Current: {current:.2f} GB", flush=True)
    print(f"Red line: 44 GB", flush=True)
    print(
        f"VERDICT: {'GO' if peak < 44.0 else 'NO-GO (exceeds 44GB red line)'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
