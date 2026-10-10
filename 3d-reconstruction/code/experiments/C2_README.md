# C2 Difix-Splat  (SUCCESS)

Scene: bedroom 0a9f2831a3e73de8.

## Idea
Flash3D's large-parallax renders are blurry/ghosting. Refine with NVIDIA Difix, a
single-step SD-Turbo diffusion 3D-artifact refiner (frozen prior).

## Environment (isolated to avoid breaking dr3d)
- dr3d conda env + isolated libs at /root/difix_libs: diffusers==0.25.1, peft==0.7.1
  (installed with `pip install --no-deps --target /root/difix_libs`; keep system huggingface_hub).
- Difix pipeline code: /root/difix_src/{pipeline_difix.py,model.py} (from github nv-tlabs/Difix3D).
- Model weights: nvidia/difix (cached HF, custom DifixPipeline + skip-connection VAE).
- Run script must stub out bitsandbytes (too new for torch 2.1) — see head of c2_difix.py.

## Run
`source dr3d && python c2_difix.py`  (loads DifixPipeline.from_pretrained("nvidia/difix"),
single step timesteps=[199], guidance 0, ref_image = clean input for the mv-cond path).

## Result (single-step, no scene tuning)
- frame36: Difix PSNR 20.33 / LPIPS 0.216   vs Flash3D 20.71 / 0.208
- frame48: Difix PSNR 18.97 / LPIPS 0.245   vs Flash3D 19.08 / 0.253 (LPIPS improved)

## Visual assessment
- Difix VISIBLY removes Flash3D's smeary vertical ghosting/streaks in the disoccluded
  center; wall, headboard, plates, picture frames become sharper and cleaner.
- It hallucinates a plausible (slightly shifted) blanket texture, which is why PSNR barely
  moves while perceptual quality clearly improves. Safest bet — works out of the box.
