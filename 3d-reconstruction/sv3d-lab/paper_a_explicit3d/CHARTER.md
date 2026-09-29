# Paper A — CHARTER (FROZEN 2026-07-18, evidence-based)

**Status:** FROZEN after Phase 1 (baseline reproduced), Phase 2 (instruments), Phase 3 (oracle passed).
Amendments require a dated decision_log entry with justification (charter §6).

## Title (working)
**Visible-Anchored Cross-View-Consistent Hidden Gaussian Completion for Single-View Novel View
Synthesis** (final title TBD).

## One-sentence contribution
A single-image feed-forward model that predicts, in addition to per-pixel visible Gaussians, a set of
**camera/ray-conditioned hidden Gaussians** for occluded and beyond-frustum regions, trained with
wide-baseline real targets under a **cross-view-consistency + causal-contribution** objective, so that
the hidden geometry is (a) measurably better than the strongest reproducible baselines *in the
region-separated occluded/OOF metrics under wide baselines*, and (b) **provably load-bearing** via
Gaussian-deletion counterfactuals that no competitor reports.

## Evidence base (why this is viable — from our own runs)
- **Flash3D reproduced** (E001b): MINE tgt5/10/rand = 28.68/26.09/25.10 (matches published within
  0.2dB). This is THE primary baseline to beat.
- **Representation Oracle** (E002, n=100 wide-baseline): with correct (oracle) hidden geometry,
  explicit Gaussians reconstruct the disoccluded/OOF region at **24.0 dB vs 12.5 dB visible-only =
  +11.5 dB deletion delta**. The ceiling exists exactly where the charter predicted. The bottleneck
  is therefore *learnability of hidden geometry*, not representation capacity.
- **Gap analysis** (literature_matrix, D001): no competitor (Flash3D/CATSplat/studentSplat/latentSplat)
  reports (1) region-separated occluded/OOF pixel metrics, (2) causal Gaussian-deletion, (3) provably
  cross-view-consistent hidden completion under wide baselines. This is the unclaimed methodological
  space.

## Hard task (inherits research_charter §3, non-negotiable)
- Inference input: single source RGB + source K + canonical/virtual target poses + source-derived
  features only. NO target RGB/depth/latent/pointmap/feature ever in inference.
- Output: unified renderable 3D Gaussians. Deleting hidden Gaussians must cause significant
  attributable change (target: region-separated deletion Δ ≥ 3 dB, per oracle it's achievable to +11).
- No default dependence on large diffusion (that is Paper B's territory).

## Method (to implement — Phase 4)
1. **Backbone**: reuse Flash3D's UniDepth + ResNet encoder path for the VISIBLE Gaussians (fair common
   base; whitelist). G_vis = per-pixel Gaussians from source (as Flash3D).
2. **Hidden predictor (the novel module)**: from source features + a target/virtual ray-map
   conditioning, predict hidden Gaussians in a **ray-conditioned canonical 3D volume** (NOT
   per-source-pixel 2D-offset like Flash3D's padding/offset layers). Candidate: a set of learned
   queries / a coarse 3D grid decoded to Gaussian params (xyz, scale, rot, opacity, color), conditioned
   on where the source frustum ends and where occlusion boundaries are.
3. **Training supervision**: wide-baseline real target frames (our wide split). Losses:
   - photometric (L1+SSIM+LPIPS) on FULL target AND region-separated (occluded/OOF weighted);
   - **cross-view consistency**: the SAME hidden Gaussians must render consistently to MULTIPLE target
     views (share one 3D set, render to ≥2 targets, penalize inconsistency);
   - **causal-contribution regularizer**: encourage hidden Gaussians to be load-bearing where visible
     can't reach (deletion-aware).
4. **Unified rendering**: G_vis ∪ G_hidden in source frame, rendered via the validated
   render_gaussians_relpose convention.

## Evaluation (frozen protocol)
- **Primary metric (novel)**: region-separated occluded + OOF PSNR/SSIM/LPIPS(VGG) under WIDE baselines
  (gaps ~20/40/60), method-agnostic forward-warp masks (common/geometry/visibility.py), GT-substitution
  for masked LPIPS. + **causal deletion Δ** on the hidden region.
- **Secondary**: overall MINE tgt5/10/rand (must NOT lose visible-region quality vs Flash3D 28.68/…;
  CATSplat 29.09 is the stronger visible bar).
- **Baselines**: Flash3D (reproduced, primary), CATSplat (published + reproduce if LLaVA feasible).
  Same views/K/res/crop/LPIPS-VGG/masks. Report official-vs-reproduced-vs-diff.
- **3D/consistency**: multi-view consistency of hidden region, depth/reprojection error.
- **Efficiency**: params, latency, VRAM, GPU-hours, no-diffusion.

## What would FALSIFY / kill Paper A (pre-committed)
- If the learned hidden predictor cannot recover a meaningful fraction of the oracle's +11.5 dB
  headroom on the hidden region at pilot scale (L2) after the pre-registered debugging rounds → the
  learnability barrier (same one that killed legacy va-mfgc teacher-distill) stands → do NOT force it;
  reframe (possibly toward Paper B generative hidden appearance on reliable hidden geometry).
- If beating baselines on hidden region requires losing visible-region quality → not a win.

## Relationship to Paper B (independence)
Paper A = explicit, geometry-first, no-diffusion, deterministic hidden geometry+appearance where
recoverable. Paper B = generative (3D-grounded) hidden APPEARANCE where it is fundamentally multi-modal
(cannot be determined from source). Different contribution, different mechanism, not a split of one model.
