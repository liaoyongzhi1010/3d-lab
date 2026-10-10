"""Single-step diffusion refiner for feed-forward single-image 3DGS novel views.

Contribution (Paper A): a lightweight single-step image-to-image diffusion refiner that
sharpens the (blurry) novel views rendered by a FROZEN feed-forward single-image Gaussian
model (Flash3D). Unlike Difix3D+/3DGS-Enhancer which refine PER-SCENE optimised NeRF/3DGS,
we refine the output of a SINGLE-IMAGE FEED-FORWARD reconstructor (one forward pass, no
per-scene optimisation) — a setting no prior diffusion-refiner targets.

Design (evidence-backed by Difix3D+ CVPR'25):
  - Base = Stable Diffusion 1.5 UNet + VAE, made SINGLE-STEP via LCM-LoRA (cached locally).
  - img2img conditioning: encode the Flash3D render I_render to a latent, run ONE LCM step
    at a fixed high noise level, decode. The refiner learns render->clean as a deterministic
    single-step map (not stochastic generation).
  - Trainable: LoRA adapters on the UNet + the VAE DECODER (frozen VAE encoder, frozen text
    encoder with a fixed empty prompt). Small # trainable params -> fast, stable.
  - Loss (Difix3D+): L2 + LPIPS + 0.5 * Gram(VGG) — improves LPIPS/FID AND PSNR in their paper.

This module is self-contained; deploy to the flash3d project and train with train_refiner.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionRefiner(nn.Module):
    def __init__(
        self,
        sd_path="runwayml/stable-diffusion-v1-5",
        lcm_lora_path="latent-consistency/lcm-lora-sdv1-5",
        lora_rank=16,
        noise_level=0.4,
        device="cuda",
        dtype=torch.float32,
    ):
        super().__init__()
        from diffusers import AutoencoderKL, UNet2DConditionModel, LCMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer

        self.noise_level = noise_level
        self.dtype = dtype

        # --- frozen VAE encoder, trainable VAE decoder ---
        self.vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
        self.vae_scale = self.vae.config.scaling_factor  # 0.18215
        for p in self.vae.parameters():
            p.requires_grad_(False)
        # unfreeze decoder
        for p in self.vae.decoder.parameters():
            p.requires_grad_(True)
        for p in self.vae.post_quant_conv.parameters():
            p.requires_grad_(True)

        # --- UNet with LCM-LoRA (single-step) + trainable LoRA adapters ---
        self.unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")
        for p in self.unet.parameters():
            p.requires_grad_(False)
        # load LCM-LoRA for single-step capability, then add trainable LoRA on top
        self.unet.load_attn_procs(
            lcm_lora_path, weight_name="pytorch_lora_weights.safetensors"
        )
        from peft import LoraConfig

        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.unet.add_adapter(lora_cfg)
        # only the newly-added LoRA adapters are trainable
        self.unet_trainable = [p for p in self.unet.parameters() if p.requires_grad]

        # --- frozen text encoder, fixed empty prompt ---
        self.tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            sd_path, subfolder="text_encoder"
        )
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            tok = self.tokenizer(
                "",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            self._empty_embed = self.text_encoder(tok.input_ids)[0]  # [1,77,768]

        self.scheduler = LCMScheduler.from_pretrained(sd_path, subfolder="scheduler")

    def trainable_parameters(self):
        params = list(self.unet_trainable)
        params += [p for p in self.vae.decoder.parameters() if p.requires_grad]
        params += [p for p in self.vae.post_quant_conv.parameters() if p.requires_grad]
        return params

    def encode(self, img):
        # img in [0,1] -> [-1,1]
        x = img * 2 - 1
        with torch.no_grad():
            lat = self.vae.encode(x).latent_dist.mean * self.vae_scale
        return lat

    def decode(self, lat):
        lat = lat / self.vae_scale
        img = self.vae.decode(lat).sample
        return (img.clamp(-1, 1) + 1) / 2  # -> [0,1]

    def forward(self, render):
        """render: [B,3,H,W] in [0,1] (blurry Flash3D novel view). Returns refined [B,3,H,W]."""
        B = render.shape[0]
        device = render.device
        lat = self.encode(render)  # [B,4,H/8,W/8]

        # add a fixed level of noise, then a SINGLE LCM denoise step conditioned on the render latent
        t = int(self.noise_level * (self.scheduler.config.num_train_timesteps - 1))
        t = torch.full((B,), t, device=device, dtype=torch.long)
        noise = torch.randn_like(lat)
        noisy = self.scheduler.add_noise(lat, noise, t)

        emb = self._empty_embed.to(device=device, dtype=lat.dtype).expand(B, -1, -1)
        model_pred = self.unet(noisy, t, encoder_hidden_states=emb).sample
        # LCM single-step: get x0 prediction
        # scheduler.step returns denoised sample; use its prev_sample/pred_original_sample
        out = self.scheduler.step(model_pred, t[0], noisy)
        lat0 = (
            out.pred_original_sample
            if hasattr(out, "pred_original_sample")
            else out.prev_sample
        )
        refined = self.decode(lat0)
        return refined
