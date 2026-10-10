# Depth-Parallax Confusion: Source-Camouflaged Adversarial Attacks on Feed-Forward Single-Image 3D Gaussian Splatting

## Abstract

Feed-forward single-image 3D Gaussian Splatting (3DGS) methods reconstruct an entire 3D scene
from one image in a single forward pass. In Flash3D, we identify a vulnerability arising from
the under-constrained single-view setting: because training supervises novel-view photometric
reconstruction, many (depth, appearance) explanations of a single source image are
photometrically indistinguishable in the source view and diverge only under parallax. We call
this the **depth-appearance ambiguity** and exploit it with **Depth-Parallax Confusion (DPC)**,
a source-constrained attack that reduces source-view change relative to novel-view damage. Unlike prior adversarial attacks on
this family (e.g. AdvSplat), which optimize a single task loss and degrade all views uniformly,
DPC uses an opposite-sign dual objective (preserve source, destroy novel), is self-referential
(needs no ground-truth novel views), and is pose-agnostic (optimized on attacker-sampled
auxiliary poses, evaluated on unseen poses). On 100 Flash3D / RealEstate10K scenes at an
L-infinity budget of eps=8/255, DPC drops novel-view PSNR by 4.8, 5.2, and 5.0 dB at the
three evaluation targets while the source view drops 2.6 dB. The degradation increases with
parallax from the source through the fixed +5 and +10 targets, an association consistent with
parallax-dependent corruption but not by itself a causal proof. We provide white-box and query-only
black-box variants, shared-depth-prior analysis, and an initial detector baseline.

## 1. Introduction

- Feed-forward single-image 3DGS is a fast-growing, deployment-oriented paradigm: one image in,
  a renderable 3D Gaussian scene out, no per-scene optimization.
- Security/robustness of this family is underexplored. The one systematic study, AdvSplat,
  treats it like a standard image model and applies imperceptible pixel perturbations that
  degrade reconstruction uniformly.
- We argue the *interesting and dangerous* attack surface is geometry-specific. Single-image
  reconstruction is ill-posed: the model resolves depth using learned monocular priors, and the
  source view alone cannot disambiguate many 3D explanations. This is a structural blind spot,
  not an incidental network weakness.
- **Contributions:**
  1. We formalize the depth-appearance ambiguity of single-image 3DGS as an attack surface.
  2. We propose DPC, a source-camouflaged, self-referential, pose-agnostic attack with a
     controllable source/novel trade-off.
  3. White-box and query-only black-box (NES over low-frequency DCT) instantiations.
   4. Evidence that DPC corrupts geometry: degradation scales with parallax, and the attack
      substantially changes the shared UniDepth prediction used in the evaluated pipeline.
   5. An initial multi-view-consistency detector baseline and analysis of its limits.


## 2. Related Work

- **Feed-forward 3DGS / single-image NVS:** Flash3D, Splatter Image, pixelSplat, MVSplat,
  DepthSplat; monocular depth priors (UniDepth).
- **Adversarial attacks on 3D/NVS:** AdvSplat (feed-forward 3DGS, pixel perturbations, white-box
  + frequency black-box). Contrast: DPC targets the depth-appearance ambiguity, is
  source-camouflaged, and produces a parallax-scaling signature.
- **Attacks on monocular depth estimation:** patch/PGD/physical attacks (Zhang 2020, Mathew 2020,
  Yamanaka 2020, Cheng ECCV22, ASP 2024, Adversarial Manhole 2024). Contrast: those target 2D
  depth maps; DPC targets the *3D reconstruction* and is scored in novel-view space, exploiting
  parallax rather than a single depth output.
- **Adversarial camouflage / stealthy attacks:** our source-camouflage is a 3D analogue —
  invisible in the observed (source) view, destructive in unobserved (novel) views.

## 3. Method

### 3.1 Threat model

Attacker perturbs only the source image with an L-inf budget eps. Two access levels:
white-box (gradients through the frozen model) and black-box (query-only render access). The
attacker never changes model weights. The defender/user typically inspects the *source-view*
reconstruction; novel views are consumed downstream (AR/VR navigation, 3D asset generation).

### 3.2 Depth-appearance ambiguity

For source image x, the model predicts per-pixel Gaussians whose depth is set by a learned
monocular prior. Let R_P(x) be the render at relative pose P (identity = source). There is a
subspace of input perturbations that leaves R_I(x) nearly unchanged but changes depth/scale/
opacity so that R_P(x) changes for P != I. Training never penalizes this because it optimizes
photometric loss at a finite set of novel views, not the full parallax manifold.

### 3.3 DPC objective

```
min_delta   lambda_src * || R_I(x+delta) - R_I(x) ||^2
          -  (1/M) sum_{j=1..M} || R_{P_j}(x+delta) - R_{P_j}(x) ||^2
s.t.        || delta ||_inf <= eps
```

- **Source-camouflage term** (minimize): keep the source-view render close to clean.
- **Novel-divergence term** (maximize): push attacker-sampled novel-view renders away from clean.
- **Self-referential:** targets are the model's own clean renders, so no ground-truth novel
  views are needed.
- **Pose-agnostic:** {P_j} are attacker-sampled small rigid poses, disjoint from the evaluation
  targets; this prevents overfitting specific eval poses and demonstrates geometric corruption.

Optimized with momentum sign-gradient and random start in the L-inf ball (needed because the
squared novel term has zero gradient at delta=0).

### 3.4 White-box instantiation on Flash3D

Gradients flow from the novel render through the rasterizer, Gaussian parameters, and UniDepth
back to the source RGB (verified: 100% non-zero gradient, novel-path magnitude >= source-path).

### 3.5 Black-box instantiation

Query-only: parameterize delta in a low-frequency DCT basis (few variables, matching the smooth
ambiguity subspace) and estimate the same objective's gradient with NES. Query cost
= iters x (2 x nes_samples + 1) x (1 + M).

## 4. Experiments

### 4.1 Setup

- Model: official Flash3D checkpoint, weights frozen.
- Data/protocol: RealEstate10K, MINE present split, eval frames tgt5/tgt10/tgt_rand (unseen by
  attack), Flash3D Evaluator (5% border crop, VGG-LPIPS). Source-view metrics reported too.
- Attack: 4 auxiliary poses, 40 steps, eps in {8/255, 16/255}, lambda_src swept.

### 4.2 Main result

eps=8/255, lambda_src=3, 40 steps, n=100 scenes:

| View | Clean PSNR | DPC PSNR | Δ PSNR | Clean LPIPS | DPC LPIPS |
|---|---:|---:|---:|---:|---:|
| src      | 37.83 | 35.21 | -2.6 | 0.025 | 0.126 |
| tgt5     | 28.73 | 23.96 | -4.8 | 0.104 | 0.208 |
| tgt10    | 26.44 | 21.28 | -5.2 | 0.139 | 0.259 |
| tgt_rand | 26.35 | 21.38 | -5.0 | 0.151 | 0.268 |

Novel degradation (~5.0 dB avg) is ~1.9x the source degradation (2.6 dB) at eps=8/255.
Degradation increases from the source through the fixed +5 and +10 targets; this pattern is
consistent with parallax-dependent corruption.

### 4.3 DPC vs baselines

Same budget (eps=8/255), same optimization (40 steps, 4 aux poses), n=100. random = uniform
noise at the budget; naive-PGD = maximize novel divergence with lambda_src=0 (no camouflage);
DPC = full objective. "gap" = |novel_avg Δ| − |src Δ| (positive means novel hurt more than
source, i.e. camouflage).

| Method | src Δ | tgt5 Δ | tgt10 Δ | tgt_rand Δ | novel_avg Δ | camouflage (novel−src) |
|---|---:|---:|---:|---:|---:|---:|
| random    | -1.9  | -0.7  | -0.6  | -0.7  | -0.67 | worse on source than novel |
| naive-PGD | -20.7 | -13.1 | -11.3 | -11.5 | -12.0 | -8.7 (destroys source) |
| DPC (ours)| -2.6  | -4.8  | -5.2  | -5.0  | -5.0  | +2.4 (novel >> source) |

Three conclusions: (1) random noise at the same budget is nearly harmless (novel -0.67 dB),
so DPC's effect is from optimization, not perturbation magnitude. (2) naive-PGD without the
camouflage term achieves large novel damage but collapses the source view by 20.7 dB — trivially
detectable by inspecting the source reconstruction. (3) DPC is the ONLY method whose novel-view
degradation exceeds its source-view degradation, i.e. the only source-camouflaged attack. The
camouflage term is therefore a genuine contribution, not a weaker PGD.

### 4.4 Parallax-scaling signature (n=10, eps=16/255)

| View | Δ PSNR |
|---|---:|
| src | -5.7 |
| tgt5 | -7.1 |
| tgt10 | -7.6 |
| tgt_rand | -9.2 |

This small-sample monotonic pattern is consistent with parallax-dependent corruption; it does not isolate geometry as the sole cause.

### 4.5 lambda_src ablation (n=10, eps=16/255)

| lambda_src | src Δ | tgt5 Δ | tgt_rand Δ | camouflage ratio |
|---:|---:|---:|---:|---:|
| 3.0 | -5.7 | -7.1 | -9.2 | 1.6x |
| 1.0 | -11.5 | -9.0 | -9.9 | 0.86x |

### 4.6 Epsilon ablation

DPC (lambda_src=3, 40 steps). Source-camouflage holds at all budgets.

| eps | n | src Δ | tgt5 Δ | tgt10 Δ | tgt_rand Δ | camouflage holds |
|---|---:|---:|---:|---:|---:|---|
| 4/255  | 50  | -1.5 | -2.6 | -3.1 | -2.8 | yes (tgt > src) |
| 8/255  | 100 | -2.6 | -4.8 | -5.2 | -5.0 | yes |
| 16/255 | 10  | -5.7 | -7.1 | -7.6 | -9.2 | yes |

Diminishing returns: doubling eps from 8 to 16 gives +50% more novel damage while the
camouflage ratio stays similar (1.6-1.9x).

### 4.7 Black-box DPC

The query-only variant uses NES over a low-frequency DCT parameterization (Section 3.5). Unit
tests confirm it optimizes the DPC objective and reduces novel-view consistency on a synthetic
renderer (3 unit tests pass). However, each NES query requires a full Flash3D forward pass
(UniDepth + Gaussian decode + rasterize), so 25 iterations x 17 function evals x 5 poses
≈ 2,125 full model forwards per scene (~50 GPU-min on A6000). This makes large-scale Flash3D
evaluation impractical within our compute budget.

For real-world threat: the query-only mode is practical for hosted APIs where inference is free
(e.g. a cloud 3D reconstruction service). The algorithmic validity is demonstrated by synthetic
unit tests; efficiency improvements (caching splatter features, hierarchical NES, low-resolution
proxy) are straightforward engineering that we leave as future work.

### 4.8 Shared depth-prior sensitivity

Flash3D and some other single-image reconstructors use a monocular-depth prior to place
Gaussians. We test whether a DPC perturbation optimized through Flash3D also changes the raw
UniDepth prediction. This is a shared-component analysis, not a second-model end-to-end attack.

We craft DPC perturbations using only Flash3D's novel-view rendering loss, then measure changes
to the raw UniDepth depth prediction (n=40, eps=8/255):

| Metric | Value |
|---|---:|
| depth AbsRel (attacked vs clean) | **0.958** |
| pixels with >10% depth shift | **65%** |
| source RGB PSNR (attacked vs clean) | 32.5 dB |

At source-image PSNR 32.5 dB, the UniDepth prediction changes by about 96% relative error on
average, with two-thirds of pixels moving more than 10%. This shows sensitivity in the evaluated
shared component. It does not establish transfer to another reconstruction system. A prior
CATSplat attempt did not produce an auditable end-to-end result, so no CATSplat claim is made.

### 4.9 Defense: multi-view consistency detector

A training-free detector measures internal geometric consistency: render the reconstruction at
a small pose P and at 2P in the same direction; a correct 3D scene has small second-order
photometric curvature, whereas DPC-corrupted geometry could amplify it. We report detection AUC
separating clean vs attacked inputs (n=20, eps=8/255).

| Detector | AUC | clean score | attacked score |
|---|---:|---:|---:|
| multi-view curvature | 0.555 | 4.84e-2 | 5.00e-2 |

This detector barely exceeds chance (AUC 0.555) on the evaluated 20 scenes and does not
reliably flag DPC. This is a negative result for one training-free consistency baseline, not
evidence against training-free defenses as a class. It motivates broader evaluation and dedicated
defenses such as adversarial training or depth-prior regularization.

### 4.10 Region-separated damage analysis

Where in a novel view does DPC's damage land? Using ONLY the clean source depth and the
source-to-target camera transform (no method-specific outputs), we forward-warp source pixels into
each target image and partition it into three method-agnostic regions: **visible** (received a
warped source pixel), **occluded** (in-frame disocclusion hole), and **beyond-frustum** (outside
the source's coverage hull). We compute per-region clean vs. attacked PSNR (n=80, eps=8/255):

| Region | n (target frames) | Clean PSNR | DPC PSNR | Δ PSNR | Mean fraction |
|---|---:|---:|---:|---:|---:|
| visible (source-observable)   | 240 | 26.6 | 21.1 | −5.5  | 96% |
| occluded (disocclusion holes) | 3   | 17.3 | 12.9 | −4.5  | 1.6% |
| **beyond-frustum (never observed)** | **122** | **20.2** | **9.7** | **−10.6** | **8.3%** |

The beyond-frustum region (completely unobservable from the source camera) is damaged ~2× more
than the visible region (Δ −10.6 vs −5.5 dB). This association is consistent with a geometric
failure mode because corruption concentrates where the source view has no photometric constraint;
it does not by itself establish causality. Note the occluded region is small (1.6% of pixels in RE10K's narrow baselines and
only 3 qualifying target frames), but beyond-frustum is substantial (8.3%, 122 frames) and shows
a clear signal.

## 5. Discussion & Limitations

- Source camouflage is *relative* (novel >> source), not perfect invisibility (~2.6 dB source
  drop at eps=8/255).
- Main tables use n=100 and ablations use n=10-50; no full 3,204-sample MINE attack table is
  included in the current repository.
- Shared UniDepth output changes are measured (AbsRel 0.96), but this is not an end-to-end
  second-model attack. The archived VGGT transfer result is invalid because its script reused the
  first scene context across the scene loop. A corrected cross-model experiment is still required.
- Black-box is validated on synthetic + unit tests; full-Flash3D black-box is compute-bound.

## 6. Conclusion

The evaluated Flash3D pipeline exhibits a depth-appearance ambiguity that DPC exploits with a
source-constrained, pose-agnostic objective. In the current experiments, the reconstruction can
remain comparatively stable at the source while degrading more strongly at novel views. Whether
this behavior generalizes to other single-image 3DGS systems requires corrected cross-model tests.
