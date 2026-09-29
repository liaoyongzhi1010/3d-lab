"""Train the single-step diffusion refiner on frozen-Flash3D novel-view renders.

For each training batch:
  1. Frozen Flash3D (official ckpt) renders novel views (color_gauss) from a single source image.
  2. The refiner maps each blurry render -> a sharpened image.
  3. Loss = L2 + w_lpips*LPIPS + w_gram*Gram(VGG) against the GT novel view.
Only the refiner's LoRA (UNet) + VAE decoder are trained. Flash3D is frozen.

Deploy to /root/projects/flash3d/ and run with the flash3d venv.
"""

import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/projects/flash3d")
from hydra import compose, initialize_config_dir
from models.model import GaussianPredictor, to_device
from datasets.util import create_datasets
import lpips as lpips_lib
import torchvision


class GramLoss(torch.nn.Module):
    """Style/Gram loss on VGG16 features (sharpness prior, Difix3D+)."""

    def __init__(self, device):
        super().__init__()
        vgg = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
        ).features
        self.slices = torch.nn.ModuleList()
        idxs = [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3
        prev = 0
        for i in idxs:
            self.slices.append(
                torch.nn.Sequential(*[vgg[j] for j in range(prev, i + 1)])
            )
            prev = i + 1
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.to(device)

    @staticmethod
    def gram(x):
        b, c, h, w = x.shape
        f = x.view(b, c, h * w)
        return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)

    def forward(self, pred, gt):
        pred = (pred - self.mean) / self.std
        gt = (gt - self.mean) / self.std
        loss = 0.0
        x, y = pred, gt
        for s in self.slices:
            x = s(x)
            y = s(y)
            loss = loss + F.l1_loss(self.gram(x), self.gram(y))
        return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--flash3d_ckpt",
        default="/home/data/sv3d-lab/checkpoints/flash3d_official/model_0000000.pth",
    )
    ap.add_argument("--out", default="/home/data/sv3d-lab/runs/refiner")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w_lpips", type=float, default=1.0)
    ap.add_argument("--w_gram", type=float, default=0.5)
    ap.add_argument("--noise_level", type=float, default=0.4)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0")

    # ---- frozen Flash3D ----
    with initialize_config_dir(
        config_dir="/root/projects/flash3d/configs", version_base=None
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=layered_re10k",
                "model.depth.version=v1",
                f"data_loader.batch_size={args.batch_size}",
                "data_loader.num_workers=4",
                "dataset.test_split_path=splits/re10k_mine_filtered/val_files_present.txt",
            ],
        )
    flash3d = GaussianPredictor(cfg)
    flash3d.to(device)
    sd = torch.load(args.flash3d_ckpt, map_location="cpu")["model"]
    new = {
        k: (flash3d.state_dict()[k].clone() if "backproject_depth" in k else v)
        for k, v in sd.items()
    }
    flash3d.load_state_dict(new, strict=False)
    flash3d.set_eval()
    for p in flash3d.parameters():
        p.requires_grad_(False)

    # ---- refiner ----
    from models.encoder.diffusion_refiner import DiffusionRefiner

    refiner = DiffusionRefiner(noise_level=args.noise_level, device=str(device))
    refiner.to(device)

    lpips_fn = lpips_lib.LPIPS(net="vgg").to(device)
    lpips_fn.eval()
    gram_fn = GramLoss(device)

    opt = torch.optim.AdamW(refiner.trainable_parameters(), lr=args.lr)
    n_train = sum(p.numel() for p in refiner.trainable_parameters())
    print(f"refiner trainable params: {n_train / 1e6:.2f}M", flush=True)

    train_dataset, train_loader = create_datasets(cfg, split="train")
    print(f"train items: {len(train_dataset)}", flush=True)

    refiner.train()
    step = 0
    t0 = time.time()
    data_iter = iter(train_loader)
    while step < args.steps:
        try:
            inputs = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            inputs = next(data_iter)
        inputs = to_device(inputs, device)
        inputs["target_frame_ids"] = [1, 2, 3]

        with torch.no_grad():
            outputs = flash3d(inputs)

        # gather (render, gt) over the 3 targets
        renders, gts = [], []
        for fid in [1, 2, 3]:
            if ("color_gauss", fid, 0) in outputs and ("color", fid, 0) in inputs:
                renders.append(outputs[("color_gauss", fid, 0)].clamp(0, 1))
                gts.append(inputs[("color", fid, 0)].clamp(0, 1))
        render = torch.cat(renders, 0)
        gt = torch.cat(gts, 0)

        refined = refiner(render)

        l2 = F.mse_loss(refined, gt)
        lp = lpips_fn(refined * 2 - 1, gt * 2 - 1).mean()
        gr = gram_fn(refined, gt)
        loss = l2 + args.w_lpips * lp + args.w_gram * gr

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.log_every == 0:
            with torch.no_grad():
                psnr = -10 * torch.log10(l2).item()
            dt = time.time() - t0
            print(
                f"step {step} loss {loss.item():.4f} l2 {l2.item():.4f} lpips {lp.item():.4f} "
                f"gram {gr.item():.4f} psnr {psnr:.2f} | {dt / (step + 1):.2f}s/it",
                flush=True,
            )

        if step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.out, f"refiner_{step:06d}.pt")
            torch.save(
                {"refiner": refiner.state_dict(), "step": step, "args": vars(args)},
                ckpt_path,
            )
            print(f"saved {ckpt_path}", flush=True)

        step += 1

    ckpt_path = os.path.join(args.out, f"refiner_{step:06d}.pt")
    torch.save(
        {"refiner": refiner.state_dict(), "step": step, "args": vars(args)}, ckpt_path
    )
    print(f"saved final {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
