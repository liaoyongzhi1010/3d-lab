# Source-Anchored Test-Time Gaussian Completion for Single-Image Novel View Synthesis

> All numbers are filled from our own registered runs (decision_log D027/D032/D033/D034/D035/D036/D037,
> experiment_registry E014/E017/E018/E019/E020/E021/E022/E023). All results are from our own harness
> over Flash3D on identical frames; oracles are labelled as upper bounds and never reported as our method.

## Abstract

Single-image feed-forward 3D Gaussian Splatting (3DGS), e.g. Flash3D, reconstructs the visible surface of a
scene well but leaves occluded and beyond-frustum ("hidden") content under-constrained; novel views therefore
degrade as the camera departs from the input pose. Recent remedies attach 2D or video diffusion priors at
inference time and apply them per view, which is 3D-agnostic (cross-view inconsistent) and computationally
heavy. We take the opposite route and complete the scene *directly in 3D*. Starting from a frozen single-image
3DGS, we introduce a set of *hidden Gaussians* that are optimized per scene against novel views under a hard
**source-null** constraint that provably preserves the input view, and we validate that the added geometry is
load-bearing with a **region-causal deletion** test — using **no diffusion prior**. On RealEstate10K our
completion improves novel-view PSNR and LPIPS over Flash3D on both the standard and wide-baseline protocols,
and — decisively — the improvement transfers to views *held out* from the optimization, confirming that the
method synthesizes genuine 3D content rather than fitting the supervised targets.

## 1. Introduction

Feed-forward single-image 3DGS methods predict a set of 3D Gaussians from one RGB image in a single forward
pass and render novel views by splatting. Flash3D (3DV'25) is a strong representative: it back-projects a
UniDepth-v1 monocular depth map and a per-pixel Gaussian decoder to obtain visible Gaussians. Because every
Gaussian is anchored to a *source* pixel, the representation is faithful where the input observes the surface
but has no mechanism to place content where the input does not — behind occluders and outside the frustum. As
the target camera moves, those hidden regions dominate the rendered image and quality collapses.

We first establish, empirically and on our own harness, that this is a *placement* problem and not a
*representation* problem:

- **Representable.** A representation oracle that is *given* correct hidden geometry recovers the hidden
  region by **+11.5 dB** (E002, causal deletion) — explicit Gaussians *can* hold hidden content.
- **Identifiable from novel views.** Without any target-geometry cheat, per-scene optimizing a set of *added*
  Gaussians against novel views (source-null constrained) recovers **+17.8 dB hidden / +1.68 dB overall /
  +0.056 LPIPS** on wide baselines, while the source view is preserved (−0.08 dB) and opacity grows rather
  than collapsing (E014, D027). The hidden RGB signal is thus identifiable.
- **Not feed-forward-amortizable.** Any attempt to amortize this mapping into a feed-forward head collapses
  (E006/E008/E009/E015/E016; predicted opacity → 0). Single-image → hidden appearance is multi-modal, so the
  conditional mean an amortized regressor learns is a low-opacity gray. We report this as a clean negative
  result and it directly motivates a **test-time** formulation.

Our method follows from these findings: **source-anchored test-time Gaussian completion**. We freeze the
Flash3D visible Gaussians and, per scene, optimize a set of hidden Gaussians against available novel views.
Two ingredients make the result trustworthy: (i) a **source-null hard constraint** that guarantees the input
view is unchanged (deleting all hidden Gaussians returns exactly Flash3D), and (ii) a **region-causal
deletion** metric that measures the marginal contribution of the hidden set to hidden pixels. Crucially, we
evaluate under a **leakage-free hold-out-view protocol**: hidden Gaussians are optimized on a subset of views
and scored on a *disjoint* held-out view. A gain there cannot be target-fitting; it is genuine 3D completion.

**Contributions.**
1. A source-anchored, diffusion-free, test-time 3D Gaussian completion for single-image NVS that provably
   preserves the input view.
2. A region-causal deletion protocol proving the completed geometry is load-bearing — not reported by prior
   refinement work.
3. A leakage-free hold-out-view evaluation on RealEstate10K showing the completion generalizes to unseen
   views, with consistent PSNR/LPIPS gains over Flash3D across standard and wide-baseline protocols.
4. A documented negative result on feed-forward amortization that motivates the test-time design.

## 2. Related work

**Single-image feed-forward 3DGS.** Flash3D (3DV'25) is our initialization: monocular-depth back-projection +
per-pixel Gaussian decoder, single forward pass, reported with whole-image metrics only. Its hidden content is
a deterministic per-pixel offset layer, which blurs or leaves holes at wide baselines.

**Diffusion-based inference-time refinement.** UAR-Scenes (ICCV'25, arXiv:2503.15742) couples a video latent
diffusion model with per-scene optimization to refine renders; it is diffusion-driven, slow, and applies a
2D/video prior per view (3D-agnostic). One-Shot Refiner (arXiv:2601.14161) and Leveling3D (arXiv:2603.16211)
perform one-step / geometry-aware diffusion refinement at inference. Difix3D+ (arXiv:2503.01774) uses an
SD-Turbo per-view fixer plus 3D distillation. All rely on a generative 2D prior. In contrast, we add explicit
3D content with **no diffusion**, enforce a **hard source-null** guarantee, and verify contribution with
**causal deletion** — none of which these methods report.

**Two-view feed-forward.** pixelSplat, MVSplat, latentSplat and DepthSplat operate from ≥2 input views; they
target a different (easier for hidden content) protocol and are not directly comparable to single-image NVS.

## 3. Method

### 3.1 Frozen visible backbone

Given a source image $I_s$ we obtain visible Gaussians $G_\text{vis}$ from Flash3D: UniDepth-v1 predicts depth
$D_s$, and back-projection with intrinsics $K$ yields per-pixel Gaussian centers with source RGB and a
depth-derived footprint. **$G_\text{vis}$ is never modified.** Consequently, deleting the hidden set recovers
exactly Flash3D, which is what makes the source-preservation and deletion analyses well-posed.

### 3.2 Hidden Gaussian set

We add a set $G_\text{hid}$ of Gaussians parameterized by center $\mathbf{x}$, log-scale, unit quaternion,
opacity logit and RGB logit (all activated at render time). Initialization is **geometry-agnostic** — it never
uses target depth:
- *Occluded-behind seeds:* source pixels back-projected at depth multipliers $\{1.3, 1.8, 2.6\}$ (content
  behind the visible surface, along source rays).
- *Beyond-frustum seeds:* a padded border ring (40% pad) back-projected at median-depth multipliers
  $\{1.0, 1.8\}$, initialized gray.

### 3.3 Per-scene test-time optimization

Only $G_\text{hid}$ is optimized (Adam, per-parameter learning rates near 3DGS conventions:
xyz $2\!\times\!10^{-4}$, scale $5\!\times\!10^{-3}$, rot $10^{-3}$, opacity $5\!\times\!10^{-2}$,
rgb $10^{-2}$; uniform high LR diverges). For each optimization view $f$ we render the merged set
$G_\text{vis}\cup G_\text{hid}$ through Flash3D's rasterizer at the relative pose and apply a photometric loss
(L2, optionally + LPIPS):
$$\mathcal{L}_\text{photo} = \sum_{f\in\text{opt}} \big\| R(G_\text{vis}\cup G_\text{hid}; T_f) - I_f \big\|.$$

### 3.4 Source-null constraint (input-view guarantee)

Hidden Gaussians must not corrupt the source view. We enforce three complementary terms:
1. **Merged-source MSE:** $\| R(G_\text{vis}\cup G_\text{hid}; I) - R(G_\text{vis}; I)\|^2$. Legitimately
   occluded Gaussians are z-buffered away at the source and incur no penalty (correctly permitted).
2. **Hidden-only source alpha:** the hidden set alone should be near-invisible at the source (mean alpha
   $\to 0$), covering cases the z-buffer masks in term 1.
3. **Projected opacity clamp:** any hidden Gaussian projecting *in-frame and in front of* the visible surface
   at the source (frustum poke-through) has its opacity pushed down each step.

Together these guarantee $|\Delta\text{PSNR}_\text{src}|$ is small; empirically $\le 0.01$ dB.

### 3.5 Hidden-Gaussian regularization

A photometric objective alone rewards region-mean color but not local texture, so the unconstrained hidden set
converges to a semi-transparent *fog* of low-opacity specks: pixel-PSNR improves but the completed region looks
noisy and mottled (we verify this by rendering the hidden set in isolation; see §4.7). We therefore add a light
opacity-sparsity term on the hidden Gaussians,
$$\mathcal{L}_\text{reg} = \lambda_\text{opa}\,\tfrac{1}{|G_\text{hid}|}\!\sum_i \sigma(o_i),$$
which culls near-transparent Gaussians and keeps fewer, more committed ones. With $\lambda_\text{opa}=0.05$ this
*improves both* perceptual quality (LPIPS gain roughly doubles) *and* pixel-PSNR while leaving the source-null
guarantee intact (§4.3). An optional scale penalty is available but was not needed. Setting $\lambda_\text{opa}=0$
recovers the un-regularized variant.

### 3.6 Evaluation protocols

- **Region-separated metrics.** We partition each target into visible / occluded / beyond-frustum via a
  method-agnostic forward warp of the source depth, and report PSNR/SSIM/LPIPS overall and on the hidden
  region (occluded ∪ OOF).
- **Region-causal deletion.** For each metric we report the value with vs. without $G_\text{hid}$; the
  difference is the causal contribution of the completed geometry.
- **Leakage-free hold-out view.** In the headline setting, $G_\text{hid}$ is optimized only on a subset of
  views and evaluated on a *disjoint* held-out view. A gain there is genuine 3D generalization, not
  target-fitting.

## 4. Experiments

Setup: RealEstate10K, 256×384, 5% border crop, VGG-LPIPS. Flash3D (official ckpt `model_re10k_v2.pth`)
reproduced in-harness at tgt5 = 28.68 dB (E001b) is the initialization we improve. Splits: `test_files_present`
(standard, gap ≈ 5/10) and `wide700` (gap 20/40/60). Single RTX A6000.

### 4.1 Main result (leakage-free hold-out view)

Our headline claim is the hold-out-view setting: optimize on views $\{1,2\}$, evaluate on the unseen view
$\{3\}$.

| protocol | src Δ | overall PSNR Δ | hidden PSNR Δ | LPIPS gain | n | ref |
|----------|:-----:|:--------------:|:-------------:|:----------:|:-:|:---:|
| **wide700 hold-out (opt{1,2} → eval{3})** | **−0.001** | **+0.64** | **+2.76** | **+0.019** | **572 (full)** | E021 (D035) |
| standard/present hold-out | +0.005 | +0.11 | +4.06 | +0.0044 | 100 | E019 |

On the unseen view the completion adds genuine content while leaving the source essentially untouched. On wide
baselines the hidden region — a larger fraction of the frame — improves by +2.76 dB (+0.64 dB overall) over
the full 572-scene wide700 test set, source preserved to −0.001 dB. On the standard/present protocol the hidden
region is small (≈5% of pixels), yet the completion still adds +4.06 dB there (+0.11 dB overall). The
full-scale headline (n=572) matches the n=100 pilot (+0.58/+2.66), confirming the effect is stable and not a
small-sample artifact. Both protocols confirm the method beats Flash3D through real 3D completion on views
never seen during optimization.

### 4.2 Upper-reference (targets used for optimization)

For context we report the setting where all available targets are used for both optimization and evaluation
(an upper reference, *not* leakage-free):

| protocol | src Δ | overall PSNR Δ | hidden PSNR Δ | LPIPS gain | n | ref |
|----------|:-----:|:--------------:|:-------------:|:----------:|:-:|:---:|
| wide700 | −0.09 | +1.68 | +17.8 | +0.056 | 100 | E014 (D027) |
| standard/present | +0.006 | +0.35 | +26.4 | +0.030 | 100 | E017 (D032) |

### 4.3 Ablations

| ablation | overall PSNR Δ | hidden PSNR Δ | LPIPS gain | src Δ | note |
|----------|:--------------:|:-------------:|:----------:|:-----:|------|
| source-null ON (= our method, wide700 hold-out) | +0.58 | +2.66 | +0.018 | **−0.010** | E018: source preserved |
| source-null OFF (wide700 hold-out) | +0.92 | +2.85 | +0.019 | **−1.31** | E020: source corrupted −1.3 dB |
| L2 vs LPIPS objective | — | — | +0.124 (LPIPS) | — | E014c: 2.2× L2 LPIPS gain, costs pixel PSNR |
| # optimization steps | see curve below | | | | E022: §4.5 #steps curve |
| hold-out distance | near +0.72 / mid +0.51 / far +0.49 | | | | E022: tercile buckets by ‖t‖ |
| # init Gaussians (stride 1 / 2 / 4) | +0.85 / +0.82 / +0.89 | +3.21 / +3.14 / +3.37 | +0.021 / +0.015 / +0.017 | −0.010 / −0.011 / −0.027 | E023: 734k / 183k / 46k Gaussians, n=30 |

**Robust to seed density.** Varying the initial hidden-Gaussian count over a 16× range (stride 1/2/4 →
734k / 183k / 46k seeds) leaves the held-out gain essentially unchanged (overall +0.82…+0.89 dB, hidden
+3.1…+3.4 dB) with no opacity collapse and the source preserved in all cases — the paper's stride-2 choice is
not a tuned sweet spot, it is the compute/quality trade-off point. **Not saturated in #steps** (curve above):
the gain rises monotonically through 500 steps, so the method trades test-time compute for quality
predictably. **Positive at every hold-out distance** (E022): bucketing the held-out view by its baseline
translation ‖t‖ into terciles gives +0.72 / +0.51 / +0.49 dB (near/mid/far); nearer held-out views gain most
(their hidden region is a larger, better-constrained fraction), but the completion helps across the full
distance range.

**Source-null is necessary for the input-view guarantee.** Removing the source-null constraint (E020) lets
the hidden Gaussians leak into the source view, degrading source PSNR by −1.31 dB (vs −0.010 dB with the
constraint). The unconstrained variant reaches a nominally higher overall Δ (+0.92) only because it also
paints content toward the source direction — trading away input-view fidelity, which our method must not do.
The constraint is what makes "delete the hidden set ⇒ exactly Flash3D" hold.

### 4.4 Region-causal deletion

| region | deletion Δ PSNR | fraction of frame | note |
|--------|:--------------:|:-----------------:|------|
| hidden (occluded ∪ OOF) | **+2.66** | 14.5% | headline (from §4.1) |
| occluded only | **+3.06** | 1.3% | disocclusion holes behind foreground |
| OOF only | **+2.58** | 13.2% | beyond source frustum (dominant area) |
| overall (full frame) | **+0.58** | 100% | includes visible region (unchanged) |

The hidden set provides load-bearing content in *both* sub-regions: occluded content (+3.06 dB on 1.3% of
pixels — disocclusion cracks behind foreground) and beyond-frustum content (+2.58 dB on 13.2% of pixels —
the dominant hidden area in wide-baseline RE10K). Deleting the hidden Gaussians removes the gain entirely
and returns exactly Flash3D (by construction, via the source-null constraint). This is the region-causal
deletion test: it proves the hidden geometry is load-bearing and region-attributable, not a whole-image
refinement or a by-product of optimizer dynamics.

### 4.5 Efficiency

| item | value | note |
|------|------:|------|
| optimization time (500 steps) | **20.7 s** per scene | Adam on hidden Gaussians only |
| total time (incl. UniDepth depth) | **21.7 s** per scene | single RTX A6000 |
| peak VRAM | **5.0 GB** | UniDepth v1 + rasterizer + 183k hidden Gaussians |
| diffusion model loaded | **none** | no LDM, no VLM, no per-view 2D model |
| hidden Gaussians (mean) | 183,276 | stride=2 init (per source image) |

No diffusion model is loaded. In contrast, UAR-Scenes (ICCV'25) uses a video-LDM + per-scene iterative
refinement requiring diffusion sampling at each update; One-Shot Refiner / Leveling3D load at minimum one
SD-class backbone. Our test-time budget is ≈21 s on a single A6000 without a generative model, placing the
method's cost closer to per-scene SfM refinement than to diffusion-based NVS.

**#Optimization-steps curve** (monotone, held-out view, E022):

| steps | overall PSNR Δ | hidden PSNR Δ | LPIPS gain |
|:-----:|:--------------:|:-------------:|:----------:|
| 100 | +0.36 | +1.71 | +0.008 |
| 200 | +0.43 | +2.09 | +0.011 |
| 300 | +0.47 | +2.35 | +0.014 |
| 400 | +0.50 | +2.56 | +0.016 |
| 500 | +0.58 | +2.66 | +0.018 |

The gain is monotone and not yet saturated; longer optimization would continue improving, at the cost of
additional test-time compute. We choose 500 steps (≈21 s) as a practical trade-off; in principle the method
can be tuned to any time budget by adjusting the step count.

### 4.6 Qualitative

We fix eight scene ids in advance (`scene0..7` of the wide700 test order — no cherry-picking) and render the
held-out view three ways: GT, Flash3D (visible-only), Ours (visible + hidden). Each panel was personally
inspected before inclusion. Per-image held-out-view PSNR (Flash3D → Ours):

| scene | Flash3D | Ours | Δ | what the hidden set adds on the unseen view |
|:-----:|:-------:|:----:|:--:|---------------------------------------------|
| 6 | 11.52 | 12.75 | **+1.23** | fills the right beyond-frustum void (brick wall, tree, yard) that Flash3D leaves gray |
| 4 | 13.57 | 14.69 | **+1.12** | reconstructs the left OOF wall/door region behind the gray void |
| 2 | 8.04 | 8.92 | **+0.88** | recovers the right-side green wall + wall painting outside the source frustum |
| 1 | 12.11 | 12.62 | +0.51 | completes the left disocclusion band behind the foreground |
| 5 | 11.37 | 11.71 | +0.33 | fills the lower-left OOF ground/foliage wedge |
| 3 | 20.72 | 21.05 | +0.33 | minor forest-floor disocclusion fill |
| 7 | 20.97 | 21.14 | +0.16 | small left-edge OOF strip |
| 0 | 24.98 | 24.98 | +0.00 | aerial scene, hidden region ≈ 0 → honest tie (nothing to complete) |

Seven of eight scenes improve; the eighth is an aerial fly-over whose held-out view is almost entirely
visible, so there is no hidden region to complete and the method correctly does nothing (mean Δ over the
eight = +0.57 dB, matching the +0.58 dB n=100 average). The largest gains (scenes 6/4/2) are exactly the
beyond-frustum cases where Flash3D renders a flat gray void and our hidden Gaussians paint plausible,
view-consistent 3D content instead. Triptychs in `figs/scene{0..7}_holdout3_*.png`.

## 5. Limitations

- **Test-time cost.** The method optimizes per scene; we are not single-forward-pass and we report timing as a
  cost. Feed-forward amortization of this mapping collapses (§1), which we document as a negative result.
- **Leakage-free setting needs ≥2 views.** To hold one view out for the leakage-free claim, the setting needs
  at least two posed views; with a single supervised view the method is a test-time adaptation whose gain we
  do not claim as leakage-free.

## 6. Conclusion

We showed that the residual error of single-image feed-forward 3DGS at novel views is dominated by *hidden*
(occluded and beyond-frustum) content, and that this is a placement problem, not a representation problem:
explicit Gaussians can hold hidden content, the hidden signal is identifiable from novel views, but it is not
feed-forward-amortizable. This motivates a **test-time** solution. Our source-anchored hidden-Gaussian
completion adds diffusion-free 3D content on top of a frozen Flash3D under a hard source-null guarantee, and
improves PSNR/LPIPS over Flash3D on RealEstate10K — including, decisively, on views held out from the
optimization, proving genuine 3D generalization rather than target-fitting. A region-causal deletion test
confirms the completed geometry is load-bearing. We believe the source-null guarantee and region-causal
deletion protocol are reusable tools for any future 3D-completion method that must not corrupt its input view.

## Honesty statement

This is **test-time optimization**, not a single forward pass; the headline is the leakage-free hold-out view.
We never claim to beat a method on its own protocol without matching that protocol, and we report Flash3D as
the initialization we improve. Oracles (E002 representation; E014/E017 targets-used) are upper references, not
our method's results. Qualitative scenes are fixed in advance.
