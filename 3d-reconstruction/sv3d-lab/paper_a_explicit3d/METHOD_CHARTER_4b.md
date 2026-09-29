# Paper A — Phase 4b Method Charter (post-D027 pivot)

**Status:** DRAFT, pre-registered before implementation (charter §6). Supersedes the Phase 4
canonical-volume hidden predictor (E003–E009, falsified as amortized) and the D-direction 2D
diffusion refiner (D026, killed for novelty collision). Evidence base = E002 + E014.

## Working title
**Amortized Visible-Anchored Hidden Gaussian Completion for Single-View NVS, distilled from a
per-scene optimization + perceptual teacher (inference = one feed-forward pass, no diffusion).**

## One-sentence contribution
A single-image feed-forward model that predicts, in addition to Flash3D-style per-pixel *visible*
Gaussians, a set of **visible-anchored hidden Gaussians** for occluded + beyond-frustum regions,
trained under a **hard source-null constraint** and a **perceptual/diffusion training-time teacher**
so that (a) hidden geometry is load-bearing (deletion Δ), (b) it does NOT corrupt the source view,
and (c) at inference it is ONE feed-forward pass producing explicit 3DGS — no per-scene optimization,
no diffusion in the loop (unlike UAR-Scenes / One-Shot Refiner / Leveling3D / Difix3D+).

## Evidence base (our own runs — why this is viable AND why prior attempts failed)
- **E002 (representation oracle):** given oracle geometry, hidden region reaches +11.5 dB deletion Δ →
  representation is expressive enough.
- **E014 (D027 optimization oracle, NEW):** per-scene FREE optimization of *added* hidden Gaussians,
  supervised by far targets, NO geometry cheat, source-null constrained → **+17.8 dB hidden-region,
  +1.68 dB overall, +0.056 LPIPS, source −0.08 dB, opacity 0.05→0.11 (NO collapse).** ⟹ the far-target
  signal IS identifiable and load-bearing when fit per-scene.
- **E006/E008/E009 (falsified):** amortized feed-forward hidden prediction (deterministic + global-CVAE
  + spatial-CVAE) all collapsed opacity→0 under L2-to-single-target. ⟹ the barrier is
  MULTI-MODAL AMORTIZATION, not representability or signal identifiability.

## Root-cause-driven design (each choice attacks a specific past failure)
1. **Visible-anchored hidden queries (fixes E006 canonical-collapse).** Hidden Gaussians are decoded
   from features sampled at *visible* structure (occlusion boundaries + frustum-edge pixels), each
   emitting a small number of behind-surface / beyond-edge Gaussians with predicted depth-offset.
   NOT free canonical-volume queries (E006) and NOT a post-hoc branch on encoded_features[-1] (D023/D025,
   the falsified spatial branch). Reuse `PaperAModel` scaffold (backbone wrap + merged/vis/hidden
   render); replace the anchor/head internals.
2. **Hard source-null constraint (fixes E010/E011 source drift).** Same mechanism proven in E014:
   merged-source-render MSE to visible-only-source-render + hidden-only source-alpha penalty +
   projected opacity clamp for frustum-pokethrough. Target: source PSNR change ≤ 0.05 dB.
3. **Perceptual/diffusion TRAINING-TIME teacher (fixes E006/8/9 opacity collapse).** Because hidden
   appearance is multi-modal, L2-to-one-target rewards transparency. Train the amortized head with
   (i) L2 on the visible-covered part, (ii) LPIPS/perceptual on the hidden region, and (iii) optionally
   a diffusion-distillation term: run the per-scene E014 optimizer (or a Difix-style teacher) to produce
   a *pseudo-target hidden render*, and distill it into the feed-forward prediction. Teacher is used
   ONLY at training. Inference = pure feed-forward 3DGS. This is the crux novelty vs all diffusion-refine
   competitors (they keep diffusion at inference).
4. **Cross-view consistency (fixes per-view hallucination).** The SAME hidden Gaussian set must render
   consistently to ≥2 targets (share one 3D set, penalize inter-view disagreement) — forces content
   into 3D, not per-view 2D.

## Inference contract (charter §3, non-negotiable)
Single source RGB + source K + virtual/target poses. NO target RGB/depth/feature/latent at inference.
Output = unified renderable 3DGS. One forward pass. No diffusion, no per-scene optimization. Deleting
hidden Gaussians must return ≈Flash3D (deletion counterfactual).

## Evaluation (frozen protocol; matches charter §8 + re10k-eval-alignment skill)
- **Primary (paper claim, MUST beat Flash3D):** full 620-scene RE10K present-test, Flash3D's own
  evaluate(): src/tgt5/tgt10/tgt_rand PSNR/SSIM/**LPIPS(VGG)**, 5% crop, 256×384. Beat our reproduced
  Flash3D baseline (E001b tgt5 28.68; from-scratch 28.16). Report official-vs-reproduced-vs-ours.
- **Wide-baseline (where the method should shine):** wide700 (gaps 20/40/60) overall + region-separated
  (visible/occluded/OOF via visibility.py) PSNR/SSIM/LPIPS + **causal deletion Δ**.
- **Consistency:** hidden-region multi-view consistency (reprojection / inter-target agreement).
- **Efficiency (VLM-free/lightweight selling point):** params, latency, VRAM, GPU-hours; explicitly
  "no diffusion / no VLM at inference, one forward pass" vs UAR/One-Shot/Leveling3D/CATSplat.
- **FID** on hidden/generated region (since we now claim perceptual quality).

## CRITICAL prior-failure reckoning (E006) — do NOT repeat it
The single biggest risk is repeating E006: a deterministic feed-forward hidden predictor collapsed
opacity→0 on held-out scenes and **LPIPS added at step 3000+ did NOT rescue it** (decision_log:719).
Verified root cause (decision_log:731-739): source→hidden is under-determined/multi-modal, so a
deterministic regressor's loss-optimal solution is to emit NOTHING (gray). E014's per-scene optimizer
avoids this because it fits each scene's own targets (no averaging over the multi-modal posterior).
Therefore the amortized method MUST differ from E006 in ways that specifically defeat mean-collapse —
"just add LPIPS" is already falsified:

1. **Distill from the E014 per-scene optimizer, NOT from raw target pixels.** Per training scene, run the
   fast per-scene hidden optimizer to get a concrete self-consistent G*_hidden; train the head to match
   G*_hidden (params and/or its render). Target is ONE concrete 3D explanation → the predictable
   structure (surface continuation behind occluders, texture extension beyond frustum) becomes a
   deterministic learnable signal, instead of regressing multi-modal raw far-RGB whose conditional mean
   is gray.
2. **Visible-anchoring makes collapse-to-gray impossible by construction.** Like Flash3D's per-pixel
   offset layers (which do NOT collapse because they are pixel-anchored and inherit source color), each
   hidden Gaussian is spawned from a visible pixel/feature and INITIALIZED to that pixel's color + a
   behind/beyond depth offset. NO near-transparent opacity init (old opacity_bias=-2 is the exact knob L2
   exploited to emit nothing — decision_log:735).
3. **Stochastic fallback only if (1)+(2) under-fit:** a small per-region latent sampled at inference (one
   forward pass, no diffusion) so appearance is SAMPLED not averaged. Global/spatial CVAE ALONE collapsed
   (E008/E009), so this is layered on top of distillation+anchoring, never the primary mechanism.

## PILOT GATE (pre-registered, run BEFORE full training) — directly tests the E006 risk
On the held-out scene-disjoint dev set, the amortized head (distillation + visible-anchoring) must, at
pilot scale (≤ few-k steps, ≤ 500 train scenes):
- capture ≥ **25%** of the E014 oracle held-out hidden-region deletion Δ (amortized Δ ≥ 0.25 × oracle Δ
  on the same scenes), hidden opacity NOT decaying <0.02, AND source PSNR change ≤ 0.1 dB.
If MET → scale to full. If it COLLAPSES (deletion≈0, opacity→0 like E006) even with distillation+anchoring
→ do NOT run full training; either (a) add the stochastic head once, or (b) report the honest
amortization-gap negative with E014 as the measured ceiling and narrow Paper A's claim to the
wide-baseline regime. NO fabricated win, NO threshold lowering.

## PASS / FAIL (pre-committed)
- **Method PASS (publishable improvement):** on full 620-test, LPIPS improves ≥ 0.005 AND PSNR does NOT
  drop (≥ −0.1 dB) vs reproduced Flash3D, under identical harness; AND on wide700 hidden region the
  deletion Δ ≥ +1.0 dB (real, load-bearing hidden content); AND source PSNR change ≤ 0.05 dB.
- **Method FAIL / re-examine:** if amortized head again collapses (opacity→0, deletion Δ≈0) even WITH
  the perceptual/diffusion teacher → the multi-modal barrier is not broken by distillation → fall back
  to a narrower claim (wide-baseline-only gain) or report the honest amortization-gap negative result
  with the E014 oracle as evidence of the ceiling. Do NOT fabricate a win.

## Differentiation table (must appear in related work)
| Method | setting | diffusion at inference | per-scene opt | output |
|--------|---------|------------------------|---------------|--------|
| Flash3D | single-view FF | no | no | per-pixel 3DGS (baseline) |
| CATSplat | single-view FF | no (VLM text) | no | in-frustum 3DGS |
| UAR-Scenes (ICCV'25) | single-view | **yes (video LDM, iterative)** | **yes** | refined GS |
| One-Shot Refiner (2601) | FF sparse | **yes (1-step)** | no | refined image |
| Leveling3D (2603) | FF | **yes** | no | refined views→GS |
| **Ours** | single-view FF | **NO** | **NO** | **hidden-completed 3DGS, teacher only at train** |

## Build order
1. Re-run E014 hardened L2 (clean G1) + LPIPS variant → final oracle ceilings (in progress).
2. Converge a from-scratch Flash3D baseline (our 19.5K data) → honest in-harness anchor (reuse D024/D025
   scratch_baseline_ref; may need more steps).
3. Implement visible-anchored hidden head in PaperAModel scaffold; smoke (plumbing + deletion>0 overfit).
4. Train amortized head with source-null + perceptual teacher; L1 overfit → L2 held-out → full.
5. Full 620 eval + wide700 region eval + ablations (teacher on/off, source-null on/off, anchor type,
   cross-view on/off) + qualitative + efficiency.
6. Paper.
