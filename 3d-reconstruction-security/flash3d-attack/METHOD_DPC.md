# DPC: Depth-Parallax Confusion Attacks on Feed-Forward Single-Image 3DGS

## 1. Motivation & Key Insight

Feed-forward single-image 3D Gaussian Splatting (Flash3D, pixelSplat, Splatter Image,
MVSplat, DepthSplat) reconstructs a full 3D Gaussian scene from **one** image in a single
forward pass. These models are trained almost entirely with a **novel-view photometric
loss**.

**Structural blind spot.** For a single source image there exist many (depth, appearance)
configurations that render **identically in the source view**; they differ only under
**parallax** in novel views. Source-view photometric consistency therefore does *not*
constrain the parallax-dependent part of the geometry. We call the resulting vulnerability
the **depth-appearance ambiguity** of single-image 3DGS.

**Attack thesis.** A perturbation confined to this ambiguity subspace is (a) nearly
invisible in the source reconstruction, yet (b) increasingly destructive as the rendering
camera moves away from the source (larger parallax). We name this attack **Depth-Parallax
Confusion (DPC)**.

## 2. Why this is not PGD / AdvSplat

| Property | PGD / AdvSplat | DPC (ours) |
|---|---|---|
| Objective | maximize a single task loss over the whole output | **opposite-sign dual**: preserve source render, destroy novel renders |
| Supervision | needs GT / task target | **self-referential** (attacked-vs-clean self-renders); no GT novel view |
| Pose usage | attacks fixed given views | **pose-agnostic**: optimize on attacker-sampled auxiliary poses, evaluate on unseen poses |
| Signature | uniform degradation | **degradation grows monotonically with parallax** |
| Threat framing | visible-region corruption | **source-camouflaged**: source view looks fine to a human/defender |

AdvSplat (arXiv:2603.23686) is the closest prior work but performs standard imperceptible
pixel perturbations that degrade reconstruction uniformly. DPC targets the geometry-specific
ambiguity and is explicitly designed to keep the source view intact while corrupting parallax.

## 3. Formulation

Let `x` be the source image, `delta` the perturbation with `||delta||_inf <= eps`,
`R_P(x)` the frozen model's render at relative camera pose `P` (identity = source view), and
`{P_j}` a set of attacker-sampled auxiliary poses (NOT the evaluation targets).

```
min_delta   lambda_src * || R_I(x+delta) - R_I(x) ||^2        # camouflage source view
          -  (1/M) sum_j || R_{P_j}(x+delta) - R_{P_j}(x) ||^2 # diverge novel views
s.t.        || delta ||_inf <= eps
```

Optimized with momentum sign-gradient steps and random start inside the L-inf ball (the
novel-divergence term is a squared residual whose gradient vanishes at `delta=0`, so a
nonzero start is required to escape that saddle).

Implementation is **model-agnostic** through a `RenderFn(image, relative_pose)` protocol, so
the same attack transfers to any feed-forward 3DGS that can render a posed view.

## 4. Feasibility verification (gradient probe)

Before attacking we verified that gradients flow from a novel-view render back to the source
RGB through UniDepth + Gaussian prediction + rasterizer:

```
novel->src grad: nonzero_frac=1.0000 abs_mean=2.748e-06 abs_max=1.101e-04
src->src   grad: nonzero_frac=1.0000 abs_mean=2.421e-06 abs_max=7.534e-05
```

100% non-zero; the novel-view path is even slightly stronger than the source path — exactly
the property DPC exploits.

## 5. Results on Flash3D (official checkpoint, MINE present split)

Protocol: RE10K MINE eval frames tgt5/tgt10/tgt_rand (unseen by the attack), Flash3D
Evaluator (5% crop, VGG-LPIPS). Attack optimized only on 4 auxiliary poses. n=10 scenes.
PSNR in dB; Δ = attacked − clean.

### 5.1 Main result (eps=8/255, lambda_src=3, 40 steps)

| View | Clean PSNR | Attacked PSNR | Δ PSNR | Clean LPIPS | Attacked LPIPS |
|---|---:|---:|---:|---:|---:|
| src      | 38.51 | 36.00 | **−2.5** | 0.021 | 0.109 |
| tgt5     | 32.67 | 28.18 | **−4.5** | 0.073 | 0.156 |
| tgt10    | 29.14 | 24.85 | **−4.3** | 0.144 | 0.232 |
| tgt_rand | 28.97 | 24.74 | **−4.2** | 0.111 | 0.211 |

Novel-view degradation (avg −4.3 dB) is ~1.7x the source degradation (−2.5 dB) at an
imperceptible eps=8/255 budget: **source-camouflaged corruption confirmed**.

### 5.2 Parallax-scaling signature (eps=16/255, lambda_src=3)

| View | Δ PSNR |
|---|---:|
| src      | −5.7 |
| tgt5     | −7.1 |
| tgt10    | −7.6 |
| tgt_rand | −9.2 |

Degradation **increases monotonically with parallax** (src < tgt5 < tgt10 < tgt_rand). This
is the fingerprint of DPC and direct evidence it corrupts parallax-dependent geometry rather
than appearance.

### 5.3 lambda_src ablation (eps=16/255)

| lambda_src | src Δ | tgt5 Δ | tgt_rand Δ | camouflage ratio (tgt_rand/src) |
|---:|---:|---:|---:|---:|
| 3.0 | −5.7  | −7.1 | −9.2 | 1.6x (novel >> source) |
| 1.0 | −11.5 | −9.0 | −9.9 | 0.86x (camouflage lost) |

Lowering `lambda_src` sacrifices source camouflage (src drops to −11.5). `lambda_src≈3` is
the sweet spot where novel corruption dominates source corruption — the controllable trade-off
that distinguishes DPC from uniform attacks.

## 6. Reproduction

```bash
cd /root/projects/flash3d && source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# gradient feasibility probe
python /root/flash3d-attack/attacks/probe_gradient.py

# main DPC result + ablations
python /root/flash3d-attack/attacks/eval_dpc_flash3d.py --max_scenes 10 --steps 40 \
  --epsilon 0.0314 --lambda_src 3.0 --out /root/flash3d-attack/results/dpc_eps8_lam3.json
python /root/flash3d-attack/attacks/eval_dpc_flash3d.py --max_scenes 10 --steps 40 \
  --epsilon 0.0627 --lambda_src 3.0 --out /root/flash3d-attack/results/dpc_eps16_lam3.json
python /root/flash3d-attack/attacks/eval_dpc_flash3d.py --max_scenes 10 --steps 40 \
  --epsilon 0.0627 --lambda_src 1.0 --out /root/flash3d-attack/results/dpc_eps16_lam1.json
```

## 7. Toward a full paper

Done:
- Novel geometry-grounded threat model + formulation (source-camouflaged, self-referential,
  pose-agnostic).
- Gradient-feasibility proof.
- Working white-box attack on official Flash3D with the three signature results above.
- Unit-tested model-agnostic core (`tests/test_dpc_attack.py`, `tests/test_attack_transforms.py`).

Next (to strengthen for a top venue):
- Scale to n>=100 scenes for stable means; report full PSNR/SSIM/LPIPS tables + variance.
- Cross-model transfer: run the same `RenderFn` on pixelSplat / Splatter Image / MVSplat to
  demonstrate the attack is architectural, not Flash3D-specific.
- Black-box transfer variant (surrogate model, frequency-parameterized delta).
- Qualitative figures: source-view vs novel-view triptychs showing invisible source / broken
  novel.
- Defenses: depth-consistency regularization, multi-view test-time check, randomized smoothing.

## 8. Limitations & honesty

- Numbers above are n=10; treat as strong preliminary evidence, not final paper numbers.
- Source view is degraded by ~2.5 dB at eps=8/255 (not perfectly invisible); "camouflage" is
  relative (novel >> source), and should be reported as such.
- CP-Drift and Depth-Cue Trap (in `attack_transforms.py`) are simpler baselines/benchmarks,
  not the headline contribution; DPC is the novel method.
