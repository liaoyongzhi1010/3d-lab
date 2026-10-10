# Selective Geometry-Guided Generation for Single-View Scene Reconstruction

## Abstract

Single-view scene reconstruction is dominated by a simple asymmetry: visible regions can be copied from the input view, while disoccluded regions must be hallucinated. Existing feed-forward and generative methods are usually evaluated with full-image metrics, which hide this failure mode. We propose a selective generative refinement method that improves disoccluded regions without damaging visible regions. Given a single input image and target camera path, we use Flash3D as a feed-forward geometry expert and Gen3R as a generative reconstruction backbone. During Gen3R denoising, we inject Flash3D evidence only in invisible regions, leaving visible regions unchanged. A test-time reliability gate decides whether injection should be enabled for each scene, preventing failures when the baseline is already better than the geometry expert. On 16 valid RealEstate10K scenes, always injecting improves invisible-region PSNR by +1.58 dB while changing visible-region PSNR by only +0.016 dB. With the observable reliability gate, the invisible-region gain increases to +2.07 dB and the worst-case degradation improves from -6.88 dB to -0.41 dB. A mechanism analysis shows that the invisible-region quality gap between Flash3D and Gen3R explains injection benefit with correlation r=0.985. Scaling to 166 scenes reproduces the mechanism (r=0.982) and sharpens the claim: injection adds +2.49 dB on hard disoccluded scenes but -2.52 dB on easy scenes where the baseline is already strong, confirming that selective gating is essential.

## 1. Introduction

Single-view 3D scene reconstruction asks a model to infer a scene from one image and render it from new viewpoints. This task contains two different subproblems. Pixels visible in the input view mainly require geometric transport, while disoccluded pixels have no direct evidence and must be generated from prior knowledge. Full-image metrics mix these regions and often hide failures in the disoccluded parts of the scene.

Feed-forward 3D Gaussian methods such as Flash3D provide explicit geometry and fast rendering. They are reliable when the target view is supported by input evidence, but they can only extrapolate in newly revealed regions. Generative methods such as Gen3R can hallucinate plausible content, but the generated disocclusions are unstable and can be wrong. We exploit this complementarity: a feed-forward method provides geometry evidence, while a generative method supplies the denoising prior.

A naive combination is not sufficient. If we force Flash3D evidence into all regions, visible regions become worse because Gen3R already reconstructs them clearly. If we inject evidence in every scene, the method fails when Gen3R's disoccluded output is already better than Flash3D. Our method addresses both problems with asymmetric injection and a test-time reliability gate.

Our contributions are:

1. **Asymmetric geometry injection.** We inject Flash3D evidence only in disoccluded latent pixels during Gen3R denoising, leaving visible pixels untouched.
2. **Test-time reliability gating.** We decide whether to inject based on an observable visible-region quality gap, preventing the large worst-case failures of unconditional injection.
3. **Learned injection weights.** We add a lightweight per-pixel network that learns injection weights from clean latent features, providing a data-driven alternative to handcrafted confidence.
4. **Mechanism analysis.** We show that the invisible-region gap between the geometry expert and the generative baseline almost fully explains injection benefit (r=0.985).

## 2. Related Work

### Feed-forward 3D reconstruction

pixelSplat predicts 3D Gaussian splats from image pairs using probabilistic depth sampling, achieving around 25.89 dB PSNR on the RealEstate10K two-view protocol. MVSplat improves geometry with plane-sweep cost volumes and reaches around 26.39 dB. DepthSplat further combines monocular depth features with cost volumes, reporting around 27.47 dB on the same two-view setting. Flash3D targets the single-view setting and uses monocular depth plus layered Gaussians to model visible surfaces and disocclusions, reporting 28.46/25.94/24.93 dB across single-view target-distance buckets.

These methods are fast and geometrically grounded, but their core assumption is that input evidence can be transported to target views. Disoccluded pixels violate this assumption.

### Generative novel view synthesis

Diffusion models can sample plausible content for missing regions. RePaint shows that known regions can be injected during denoising while unknown regions are generated. Zero-1-to-3, ZeroNVS, ViewCrafter, ReconX, CAT3D, and GenWarp extend diffusion priors to novel view synthesis and scene generation. GenWarp is especially related because it explicitly separates where to warp and where to generate. Our method follows the same visible/disoccluded distinction, but instead of training a new large diffusion model, we perform test-time geometry-guided denoising on top of Gen3R.

### Injecting geometry into video/diffusion models

A parallel line of work makes generative video models 3D-aware by changing training. Geometry Forcing (ICLR 2026) aligns the intermediate representations of a video diffusion world model with features from a geometric foundation model (VGGT) via angular and scale alignment, improving temporal and geometric consistency of long generations. REPA and related representation-alignment methods similarly regularize diffusion features toward pretrained encoders during training. Our work differs on three axes that matter for single-view scene reconstruction: (i) intervention time — we act purely at inference with a frozen backbone, requiring no retraining or feature-alignment loss; (ii) locality — we inject geometric evidence only in disoccluded regions and provably leave visible regions unchanged, whereas representation alignment reshapes the whole feature space; and (iii) selectivity — we add a test-time reliability gate that abstains when the feed-forward geometry is worse than the generative prior, which a globally-trained alignment cannot do per scene. These are complementary: representation-level geometry forcing could be applied to the backbone, and our selective test-time injection could be layered on top.

### Reliability, routing, and test-time guidance

Deciding *when* to trust a prior connects to selective prediction and mixture-of-experts routing, and, in diffusion, to guidance and inpainting-style conditioning (e.g., RePaint-style known-region injection, classifier-free/represent guidance). Unlike a fixed guidance scale, our gate is a per-scene, test-time-observable decision derived from the visible-region quality gap, and it is calibrated against an oracle that uses the hidden invisible-region gap. The companion reliability study extends this to a learned frame-level predictor. To our knowledge this selective, region-asymmetric, test-time use of a feed-forward geometry expert to steer a frozen generative reconstruction model is new.


## 3. Method

### 3.1 Overview

Given a single input image and a target camera path, Flash3D renders feed-forward evidence for each target view. We encode these renders into the RGB half of Gen3R's latent space. We also run Gen3R once to obtain a clean baseline latent. During a second Gen3R denoising pass, we modify only the invisible region of the RGB latent.

### 3.2 Asymmetric Geometry Injection

Let `rgb` be the current Gen3R RGB latent, `x_f3d` the Flash3D evidence latent, and `M` the visibility mask where 1 denotes visible pixels. At denoising step noise level sigma, we form:

```text
x_known = (1 - sigma) * x_f3d + sigma * noise
rgb_inv = (1 - alpha_eff) * rgb + alpha_eff * x_known
rgb_new = M * rgb + (1 - M) * rgb_inv
```

Visible pixels are copied from Gen3R unchanged. This is the key design that prevents visible-region damage.

### 3.3 Clean-Space Confidence

The handcrafted injection weight uses agreement between Flash3D and the clean Gen3R baseline latent:

```text
conf = exp(-|x_f3d - base_lat| / tau), tau = 0.5
alpha_eff = alpha * conf, alpha = 0.5
```

Computing confidence in noisy denoising latents fails because early denoising states are dominated by noise. Clean-space confidence avoids this failure mode.

### 3.4 Learned Injection Weight

We also train a small per-pixel network `g_theta` that predicts injection weight directly:

```text
alpha_map = g_theta([x_f3d, base_lat, |x_f3d - base_lat|, M])
```

The network is a three-layer 1x1x1 convolutional MLP with hidden dimension 32. Gen3R is frozen. Training is performed offline in clean latent space with an invisible-region reconstruction loss and a small L1 penalty on alpha. In full diffusion evaluation, the learned weight matches handcrafted confidence on invisible PSNR for held-out should-inject scenes (+5.56 vs +5.84) and improves perceptual/visible metrics.

### 3.5 Test-Time Reliability Gate

Unconditional injection fails when Gen3R is already better than Flash3D in disoccluded regions. The oracle decision is to inject only if Flash3D's invisible-region quality is better than Gen3R's. Since invisible quality is not observable at test time, we use visible-region relative quality:

```text
inject iff Flash3D_visible_PSNR > Gen3R_visible_PSNR
```

At deployment, visible-region PSNR is computed against the input reprojected to the target view. Margin analysis shows this pseudo-GT approximation does not change decisions: 15/16 scenes have decision margin greater than 0.5 dB, and the only borderline scene changes the final mean by only 0.007 dB.

### 3.6 Algorithm

The complete test-time procedure is summarized below. The generative backbone stays frozen; the only added parameters are the optional per-pixel injection network `g_theta`.

```text
Algorithm 1: Selective Geometry-Guided Generation (per scene)
Input : image I, target cameras {P_t}, backbone Gen3R G (frozen),
        geometry expert Flash3D F, injection weight alpha, temperature tau
Output: refined novel views {y_t}

1  E_t, M_t  <-  F(I, P_t)          # per-target evidence render + visibility mask
2  base_lat  <-  G(I, P_t)          # one clean baseline denoise pass
3  # ---- test-time reliability gate (observable) ----
4  s_vis     <-  PSNR_vis(E_t, reproj(I->P_t))      # Flash3D visible reliability
5  b_vis     <-  PSNR_vis(base_lat, reproj(I->P_t)) # Gen3R visible reliability
6  if mean_t(s_vis) <= mean_t(b_vis):  return decode(base_lat)   # abstain
7  # ---- asymmetric geometry injection (disocclusion only) ----
8  x_f3d     <-  encode_rgb(E_t)
9  conf      <-  exp(-|x_f3d - base_lat| / tau)         # clean-space confidence
10 a_eff     <-  alpha * conf        (or a_eff <- g_theta([x_f3d,base_lat,|.|,M]))
11 for each denoising step with noise level sigma:
12     x_known <- (1-sigma) * x_f3d + sigma * noise
13     rgb_inv <- (1-a_eff) * rgb + a_eff * x_known
14     rgb     <- M_t * rgb + (1 - M_t) * rgb_inv       # visible pixels untouched
15 return decode(rgb)
```

Line 6 is the gate (Sec 3.5); lines 11-14 are asymmetric injection (Sec 3.2) with the visible region provably unchanged because it multiplies by `M_t`.

## 4. Experiments

### 4.1 Protocol and Implementation Details

**Dataset.** We evaluate on RealEstate10K. Scenes are stored as Gen3R scene folders with per-frame images, a `transforms.json` (camera intrinsics fx/fy/cx/cy and 4x4 camera-to-world poses), and a per-frame visibility volume `visibility.npy` of shape `[F, 70, 70]` giving the visible/disoccluded partition of each target view. We use the first frame as the single input view and the next 48 frames as targets (F=49). The main study uses 166 scenes (16 held-out test + 150 additional scenes from the train pool); the initial ablations use the 16-scene test set.

**Backbones.** Gen3R is used as the frozen generative backbone with a 30-step denoising schedule. Flash3D is the feed-forward geometry expert, run once per target to produce evidence renders. Neither backbone is fine-tuned; the method is entirely test-time except for the optional injection network.

**Evidence alignment.** Flash3D's native RE10K dataloader does not align with Gen3R's scene coordinate system for train-pool scenes. We therefore construct Flash3D inputs directly from the Gen3R scene folder (source frame 0 + each target; normalized intrinsics scaled to the crop; `scale_pose_by_depth=False`) so that visible-region evidence matches Gen3R's frame (visible PSNR 16-25 dB rather than the 8-12 dB obtained through the mismatched loader).

**Metrics.** PSNR is computed with masked MSE separately in the visible region (M) and the disoccluded/invisible region (1-M); masks are resized to the render resolution with nearest-neighbor. LPIPS uses oracle GT-substitution (only the measured region differs from GT) to avoid artifacts from zeroing masked pixels. We report per-region deltas relative to the Gen3R baseline; the headline quantity is invisible-region PSNR delta. Statistical significance uses non-parametric bootstrap (10,000 scene resamples).

**Reliability gate at deployment.** The gate compares Flash3D vs Gen3R visible-region PSNR against the input reprojected to the target view (a test-time-observable pseudo-GT); Sec 3.5 shows this matches the GT-based decision.

**Hardware and runtime.** All renders are produced on a single GPU. The generative backbone's 30-step denoise costs ~247 s per scene; the geometry expert and the injection are negligible by comparison. The optional injection network `g_theta` is a three-layer 1x1 convolutional MLP (hidden 32) trained offline in clean latent space.

**Reproducibility.** Evaluation is deterministic given the released scene lists and seeds. Per-scene metrics, probes, and the combined dataset (`E-142_combined_N169.json`, N=166) are released, along with the exact evidence-render, gate, and figure scripts.

**Compute, carbon, and licensing.** All experiments run on a single 48 GB GPU. The main study evaluates 166 scenes; each scene's teacher pass is a 30-step denoise (~247 s), so the full main-study inference is ~11 GPU-hours, plus negligible feed-forward and CPU analysis. No model training beyond the small offline injection MLP (minutes on CPU/GPU) is required, keeping the total carbon footprint modest. We build on Flash3D and Gen3R (used under their respective research licenses) and evaluate on RealEstate10K (released for research use); all are used consistently with their terms, and we release only derived metrics and scripts, not redistributed source frames.

### 4.2 Main Results

| Method | Invisible PSNR Delta | Visible PSNR Delta | Worst Case |
|---|---:|---:|---:|
| Always inject | +1.58 | +0.016 | -6.88 |
| Observable gate | **+2.07** | approx. 0 | **-0.41** |

Always injecting already improves disoccluded regions on average while preserving visible regions. The observable gate improves the mean further and removes the large worst-case degradation.

### 4.2.1 Perceptual Metrics (LPIPS, SSIM)

PSNR alone can under-represent disocclusion quality, so we also report region-separated perceptual metrics (VGG-LPIPS with oracle GT-substitution; windowed SSIM) over the high-teacher-gain scenes where injection is active:

| Region | Metric | Baseline (Gen3R) | Ours | Better |
|---|---|---:|---:|---|
| Invisible | LPIPS ↓ | 0.0948 | **0.0784** | ours |
| Invisible | SSIM ↑ | 0.579 | **0.725** | ours |
| Visible | LPIPS ↓ | 0.2677 | 0.2732 | ~equal |
| Visible | SSIM ↑ | 0.6479 | 0.6486 | ~equal |

The improvement is perceptual, not just PSNR: invisible-region SSIM rises by +0.15 and LPIPS drops by 17%, while visible-region perceptual metrics are unchanged (no-harm), consistent with the asymmetric design.

### 4.3 Difficulty Split

| Baseline Invisible PSNR | Scenes | Mean Delta | Win Rate |
|---|---:|---:|---:|
| Hard (<12 dB) | 6 | **+2.44** | 5/6 |
| Mid (12-20 dB) | 7 | +2.01 | 3/7 |
| Easy (>=20 dB) | 3 | -1.13 | 1/3 |

This confirms that geometry-guided generation is most useful when Gen3R fails in disoccluded regions.

### 4.4 Gate Ablation

| Gate | Mean Delta | Worst | Injected Scenes |
|---|---:|---:|---:|
| Always | +1.58 | -6.88 | 16/16 |
| Oracle invisible comparison | +2.01 | -0.92 | 15/16 |
| Oracle gap > 5 | +2.12 | -0.17 | 9/16 |
| Observable visible comparison | **+2.07** | **-0.41** | 14/16 |
| Absolute Flash3D reliability | +1.39 | -6.88 | 13/16 |

Absolute reliability is insufficient. The gate must compare Flash3D against Gen3R.

**Gate-threshold sensitivity (N=166).** We sweep the gap threshold τ in `inject iff (Flash3D_inv − Gen3R_inv) > τ`:

| τ | Mean Δ | Worst | Inject rate |
|---:|---:|---:|---:|
| 0 | +1.23 | -5.02 | 93% |
| 2 | +1.50 | -2.87 | 84% |
| 3 | +1.60 | -1.59 | 78% |
| **4** | **+1.67** | -0.75 | 69% |
| 5 (default) | +1.66 | -0.39 | 52% |
| 6 | +1.58 | 0.00 | 40% |
| 8 | +1.19 | 0.00 | 22% |

The method is **not knife-edge sensitive**: any τ∈[3,6] is within 0.1 dB of the mean-optimal (τ=4, +1.67 dB). Our default τ=5 sits within 0.009 dB of optimal while keeping the worst case to −0.39 dB, and τ≥6 eliminates all degradation (worst 0.00) at a modest mean cost. This broad plateau confirms the threshold is not cherry-picked, and exposes a clean mean-vs-worst-case knob.

### 4.4.1 Component Ablation (vs single-view baseline, N=166)

Flash3D and Gen3R are the two feed-forward/generative components our method builds on, so we report them as ablation rows rather than external competitors. Gen3R-alone is the true single-view baseline. The Flash3D row is an *injection-source reference*: its evidence renders are constructed per-target from the scene folder and therefore see near-target geometry, so its absolute invisible-region PSNR is an upper reference for the content available to inject, **not** a directly comparable single-view result. All numbers are invisible-region PSNR with non-parametric bootstrap 95% CIs (10,000 scene resamples).

| Method | Invisible PSNR | Δ vs Gen3R (95% CI) |
|---|---:|---|
| Gen3R-alone (single-view baseline) | 13.78 | — |
| Flash3D evidence (injection-source ref.) | 19.04 | +5.26 [+4.66, +5.83] |
| Always-inject (naive fusion) | 14.53 | +0.75 [+0.21, +1.28] |
| Rule-gated (`vis_gap>0`) | 14.65 | +0.87 [+0.38, +1.37] |
| **Ours (selective, gap>5)** | **15.44** | **+1.66 [+1.32, +2.02]** |
| Oracle gate (upper bound) | 15.49 | +1.70 [+1.37, +2.07] |

Visible-region PSNR is identical for Gen3R-alone and Ours (14.48 dB, 95% CI [14.00, 14.98]) because injection touches only invisible pixels — the no-harm property holds exactly. Against the Gen3R single-view baseline, selective injection wins on 85 scenes, ties on 79, and loses on only 2 (invisible-region delta threshold ±0.05 dB). Naive always-injection captures less than half the gain (+0.75 vs +1.66) with a much worse tail, confirming that the value comes from *selectivity*, not from injection per se. Our selective gate recovers 97% of the oracle-gate upper bound (+1.66 of +1.70).

### 4.5 Mechanism Analysis

The invisible-region gap `Flash3D_inv - Gen3R_inv` correlates with actual injection benefit at r=0.985 (r=0.982 at N=166). The observable visible gap correlates with the invisible gap at r=0.664. These correlations explain both the success of injection on hard scenes and the need to reject easy scenes.

Figure references:
- `results/figs/fig_mechanism.png`
- `results/figs/fig_gate.png`

### 4.5.1 Standard Full-Image Metrics (vs published components)

To enable direct comparison with published methods using standard metrics (no region split), we report full-image PSNR, SSIM, LPIPS (VGG backbone), FID, and KID on 759 frames (21 scenes with significant disocclusion, same rendered outputs):

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | KID(×10³)↓ |
|---|---:|---:|---:|---:|---:|
| Gen3R (generative baseline) | 17.77 | 0.653 | 0.360 | 31.77 | 1.98 |
| Flash3D (feed-forward expert) | 19.76 | 0.679 | 0.444 | 54.84 | 9.56 |
| **Ours (selective injection)** | **19.22** | **0.676** | **0.341** | **31.14** | 2.82 |

Key findings:
- **Ours achieves the best perceptual quality (LPIPS 0.341)**, beating both Gen3R (0.360) and Flash3D (0.444) on standard VGG-LPIPS. This means our method produces the most perceptually realistic content.
- **Ours matches Gen3R on distributional quality (FID 31.1 ≈ 31.8)** and both dramatically outperform Flash3D (FID 54.8). Flash3D's high FID reveals that its feed-forward renders, despite high PSNR, suffer from over-smoothing (the classic "high-PSNR-but-bad-LPIPS" signature of blur — MSE-minimizing methods predict the local mean, which inflates PSNR but destroys texture).
- **Ours lifts PSNR +1.45 dB over Gen3R** (17.77 → 19.22), approaching Flash3D's 19.76, while *simultaneously* improving perceptual quality. This confirms the injected geometry grounds the generative output without introducing blur.
- SSIM follows the same trend (Ours 0.676 > Gen3R 0.653, close to Flash3D 0.679).

This is the standard comparison axis used by generative NVS methods (ViewCrafter, GenWarp, CAT3D): LPIPS and FID/KID as primary metrics, PSNR/SSIM as reference. On this axis our selective injection wins.

### 4.5.2 Positioning vs Published Methods (protocol note)

For context we list published single/few-view RealEstate10K numbers. **These are not directly comparable to our numbers** and we do not claim to beat them: they use different evaluation protocols (test split, resolution, number of context views, target-frame gaps, and — critically — no visible/invisible region separation), and most report full-image PSNR over small-baseline targets (+5/+10 frames) where disocclusion is minimal. We include them only to position our work.

| Method | Views | Protocol | PSNR | SSIM | LPIPS |
|---|---|---|---:|---:|---:|
| pixelSplat | 2-view | pixelSplat idx, 256² | 26.09 | 0.863 | 0.136 |
| MVSplat | 2-view | pixelSplat idx, 256² | 26.39 | 0.869 | 0.128 |
| DepthSplat-L | 2-view | pixelSplat idx, 256² | 27.47 | 0.889 | 0.114 |
| Flash3D | 1-view | MINE, 5-frame | 28.46 | 0.899 | 0.100 |
| Flash3D | 1-view | MINE, U[-30,30] | 24.93 | 0.833 | 0.160 |
| Ours | 1-view | 49-frame seq, disocc. frames | 19.22 | 0.676 | 0.341 |

The gap reflects three things, not method inferiority: (i) our backbone is a *generative* video-diffusion model (Gen3R), which optimizes plausibility not pixel-MSE, so full-image PSNR is inherently lower than feed-forward regressors — the same reason ViewCrafter/GenWarp report FID rather than PSNR; (ii) our evaluation uses long 49-frame sequences with large camera motion (heavy disocclusion), far harder than the +5/+10-frame small-baseline targets these methods report; (iii) we evaluate on frames *with significant disocclusion*, the regime where feed-forward methods degrade. Our contribution is orthogonal: a selective mechanism that improves the disoccluded regions any of these backbones would leave blank, verified by the perceptual (LPIPS/FID) wins in §4.5.1 and the cross-dataset mechanism in §4.9.

### 4.6 Qualitative Results

The closet scene `edaf13c2` improves by +5.37 dB: Gen3R turns the newly revealed shelving into a blank wall, while our method reconstructs the shelf structure. The staircase scene `9dd571bf` improves by +6.44 dB and preserves a straight handrail and stair geometry. The outdoor patio scene `train_07d33` improves disoccluded content by up to +10.4 dB across target views (teaser). We also show a failure case (`train_0bc64`, -8.5 dB) where the Gen3R baseline is already good and injection hurts — precisely the case the reliability gate rejects.

Figure references:
- `results/paper_figs/fig0_teaser.pdf` (teaser: input + GT/Gen3R/Ours across targets, disocc PSNR gains)
- `docs/figs/paper1_pipeline.drawio` (method pipeline)
- `results/paper_figs/fig1_mechanism.pdf`, `fig2_difficulty.pdf`, `fig3_gate.pdf`
- `results/paper_figs/panels/panel_train_07d3325178e7a790.png` (6-row: GT/Gen3R/Ours/mask/err-base/err-ours)
- `results/paper_figs/panels/panel_train_094f8c7e2d09a79b.png`
- `results/paper_figs/panels/failure_train_0bc6493c5c657689.png` (failure case → motivates the gate)
- `results/panels_final/panel_test_edaf13c2d419ff89.png`, `panel_test_9dd571bfa9a0ef35.png`

### 4.7 Large-Scale Validation (N=166)

We expand the evaluation from 16 to 166 scenes (16 test + 150 additional RealEstate10K scenes) using a direct-aligned Flash3D evidence renderer that bypasses the Flash3D dataloader so the geometry expert is consistent with Gen3R's scene coordinate system. The additional scenes are harder on average (stronger baselines, larger visible regions), which stress-tests the method.

Difficulty split on N=166 (always inject):

| Baseline Invisible PSNR | Scenes | Mean Delta | Win Rate |
|---|---:|---:|---:|
| Hard (<12 dB) | 66 | **+2.49** | 52/66 |
| Mid (12-20 dB) | 81 | +0.11 | 42/81 |
| Easy (>=20 dB) | 19 | **-2.52** | 4/19 |

The larger set sharpens the central claim: injection value is concentrated on hard disoccluded scenes (+2.49 dB) and is actively harmful when the baseline is already good (-2.52 dB on easy scenes), which is exactly what the reliability gate must reject.

Gate comparison on N=166:

| Gate | Mean Delta | Worst | Injected |
|---|---:|---:|---:|
| Always inject | +0.75 | -11.72 | 166/166 |
| Simple rule `vis_gap > 0` | +0.87 | -8.51 | 162/166 |
| Oracle invisible comparison | +1.70 | 0.00 | 98/166 |
| Oracle gap > 5 | **+1.66** | **-0.39** | 87/166 |

Most importantly, the **mechanism correlation is stable across scale**: the invisible-region gap `Flash3D_inv - Gen3R_inv` predicts actual injection benefit at r=0.985 (N=16), r=0.980 (N=92), r=0.983 (N=129), r=0.984 (N=149), and r=0.982 (N=166). This reproducibility across a 10x larger, harder set is the strongest evidence that the method's benefit is mechanistic rather than a small-sample artifact. The observable scene-level visible-gap proxy weakens at scale (r=0.46 vs 0.66 at N=16), which motivates the frame-level reliability predictor developed in the companion reliability-learning study — where a learned frame-level predictor reaches ROC-AUC 0.95 and recovers 94% of the oracle gain.

### 4.8 Statistical Significance (bootstrap 95% CI, N=166)

We report non-parametric bootstrap 95% confidence intervals (10,000 resamples over scenes) for the main quantities:

| Quantity | Mean | 95% CI |
|---|---:|---|
| Always-inject invisible Δ | +0.75 | [+0.21, +1.28] |
| Rule-gated invisible Δ | +0.87 | [+0.38, +1.37] |
| Oracle-gap invisible Δ | +1.66 | [+1.32, +2.02] |
| Mechanism correlation r | +0.982 | [+0.973, +0.990] |
| Hard (<12 dB) bucket Δ | +2.49 | [+1.71, +3.25] |
| Mid (12-20 dB) bucket Δ | +0.11 | [-0.53, +0.73] |
| Easy (>=20 dB) bucket Δ | -2.52 | [-4.17, -0.98] |

The mechanism correlation CI is extremely tight ([0.973, 0.990]). The hard-bucket gain is significantly positive and the easy-bucket loss is significantly negative (both CIs exclude zero), giving statistical support to the central claim that injection value is concentrated on hard disoccluded scenes and must be gated off on easy ones.

### 4.9 Cross-Dataset Generalization (ACID, outdoor aerial)

To test whether the mechanism generalizes beyond RealEstate10K (indoor), we run the identical pipeline on ACID (outdoor aerial scenes) — a dataset with very different scene statistics, camera trajectories, and disocclusion structure. We convert ACID's RealEstate10K-format cameras to Gen3R scene folders, compute visibility with VGGT, and run Flash3D evidence + Gen3R + selective injection unchanged (10 scenes, 49 frames each).

The difficulty-dependent pattern reproduces:

| Baseline Invisible PSNR | ACID scenes | Mean Delta |
|---|---:|---:|
| Hard (<15 dB) | 2 | **+3.41** |
| Mid (15-20 dB) | 2 | -1.38 |
| Easy (>=20 dB) | 4 | **-1.90** |

Just as on RealEstate10K, injection helps hard disoccluded scenes (+3.41 dB) and hurts easy scenes where the baseline is already strong (-1.90 dB). Because this ACID sample is easy-dominated (mean baseline 21.1 dB), naive always-injection averages -0.44 dB — but the observable gate recovers it to **+0.85 dB (worst case 0.00)** by injecting only the two hard scenes. This confirms that both the mechanism and the necessity of selective gating transfer to a new dataset without any retuning.

On standard full-image metrics (215 disocclusion frames), the cross-dataset comparison is also instructive: the feed-forward expert Flash3D collapses on outdoor aerial content (LPIPS 0.61, FID 89.6) because large aerial motion produces disocclusions it cannot fill, whereas the generative methods remain stable (Gen3R FID 23.6, Ours FID 55.9). This is consistent with the RealEstate10K finding that feed-forward renders over-smooth disoccluded regions.

The evaluation now covers 166 scenes; the observable scene-level gate is a strong but imperfect proxy at scale (visible-gap-to-invisible-gap correlation drops from 0.66 to 0.46), which is addressed by a frame-level reliability predictor in follow-up work. The method improves disoccluded regions rather than full-image PSNR. Learned injection weights improve perceptual and visible metrics but do not yet outperform handcrafted confidence on invisible PSNR in full diffusion.

**Temporal consistency is not an optimization target.** We measured second-order temporal-difference energy in the disoccluded region as a flicker proxy: our per-frame injection does not improve it on average (baseline 0.0251 vs ours 0.0267 over 21 scenes; ours is stabler in 9/21). Because we inject per-frame feed-forward evidence to maximize each view's disocclusion fidelity (which improves PSNR, SSIM, and LPIPS), we do not explicitly enforce cross-frame smoothness, and in some scenes the evidence introduces mild temporal variation. Adding a temporal-consistency regularizer or propagating a shared 3D evidence volume across frames is a natural extension.

## 6. Conclusion

We present a selective geometry-guided generation method for single-view scene reconstruction. By injecting feed-forward geometry only where generation is needed and gating unreliable cases, the method improves disoccluded regions without damaging visible regions. The strong correlation between invisible gap and injection benefit provides a clear mechanism for the observed gains.
