# Phase 3 — Representation Oracle: Pre-Registration

**Status:** PRE-REGISTERED (thresholds fixed BEFORE running). Charter §6, §7.
**Date:** 2026-07-18
**Author:** research lead (autonomous)

> This document fixes the Oracle experiment design and PASS/FAIL thresholds *before* any Oracle
> run, so results cannot be rationalized post-hoc. No threshold here may be lowered after seeing
> results (charter §6). If the Oracle FAILS its thresholds, that is decisive evidence about the
> *achievable ceiling* of explicit-Gaussian hidden completion and forces a charter reframe.

---

## 1. Purpose

Measure the **achievable upper bound** of the Paper A representation (explicit 3D Gaussians for
occluded + out-of-frustum regions) using *oracle* (target-derived) geometry, WITHOUT training any
network. This isolates the question:

> If we had perfect hidden geometry (positions + colors from the target frames themselves),
> how well can a *unified renderable Gaussian scene* reconstruct wide-baseline target views —
> and is the hidden geometry causally load-bearing (deletion test)?

This is NOT a model result (charter §7: oracles are upper bounds only). It tells us:
- (a) the **ceiling** for wide-baseline occluded/OOF PSNR/LPIPS the eventual model could aim at;
- (b) whether the explicit-Gaussian representation itself is expressive enough (if the oracle
  itself can't reach a good ceiling, no learnable model will — this kills the representation, not
  the learning);
- (c) empirical confirmation of Flash3D's weakness in these regions (compare oracle-merged vs
  Flash3D on the SAME wide-baseline split).

## 2. Construction (no training, no optimization of Gaussians)

Given one source frame S and K target frames {T_i} (wide baseline), per test scene:

1. **G_vis (visible Gaussians):** back-project source depth (from UniDepth v1, same as Flash3D) to
   3D using source K + source pose. Color = source pixel RGB. This is the "observed" part.
   (We deliberately reuse Flash3D's depth predictor so the visible part is a fair common base.)

2. **G_hidden (oracle hidden Gaussians):** For each target frame T_i:
   - Get target pseudo-GT 3D points: back-project target depth (oracle: from target frame — this
     is the ORACLE cheat, allowed only because this is a ceiling measurement, never in a model).
   - Transform target points into the shared world frame via target pose.
   - Keep only points that are **occluded or out-of-frustum from the source view** (i.e., NOT
     already covered by G_vis) — determined by projecting into source and checking coverage/depth.
   - Color = target pixel RGB. Confidence filter: drop points with low depth confidence.
   - Scale: pre-registered 1-pixel footprint at that depth (see §4). Opacity: fixed high value.
   - Rotation: identity (isotropic) unless anisotropic pre-registered variant tested.

3. **Unified scene:** merge G_vis ∪ G_hidden into ONE Gaussian set, render to source + all targets.

## 3. Metrics reported (all pre-registered)

- **Reprojection error:** oracle hidden points reprojected to their source target — pixel error.
- **Oracle-only-hidden render** vs GT at target (occluded + OOF regions, GT-substitution masked).
- **Merged render** (G_vis ∪ G_hidden) vs GT at each target: overall + region-separated
  (visible / occluded / OOF) PSNR / SSIM / LPIPS(VGG).
- **Source preservation:** merged render at source vs source RGB (must not degrade visible).
- **OOF-vs-OCC alpha coverage:** fraction of occluded/OOF pixels now covered by hidden Gaussians.
- **Deletion counterfactual:** merged-with-hidden vs merged-without-hidden (delete G_hidden) at
  targets, region-separated. This is the causal test.
- **≥10 fixed visualizations** (deterministic scene ids, chosen BEFORE looking at metrics).

## 4. Pre-registered hyperparameters (fixed; not tuned on results)

- Depth predictor for G_vis: UniDepth v1 (Flash3D default). Depth scale via COLMAP sparse (RANSAC),
  same as Flash3D eval.
- Hidden Gaussian scale: footprint = depth * (1 / focal_px) (≈ 1 source-pixel span at that depth).
- Hidden opacity (pre-sigmoid): +4.0 (≈ 0.98). Fixed.
- Confidence threshold for keeping target points: keep top 90% by depth confidence (drop bottom 10%).
- Coverage threshold to classify "occluded from source": a target point is hidden if its source
  reprojection either (a) lands outside source frame [OOF], or (b) lands in-frame but its depth is
  > source depth at that pixel + margin (0.05 * depth) [occluded].
- Target frames: wide-baseline set = {+10, +20, +30, random-far} where available (in addition to the
  standard MINE +5/+10 for comparability).
- Resolution 256×384, 5% border crop, LPIPS VGG (charter §8, matches Flash3D).

### 4.1 CRITICAL design issue — cross-frame depth scale consistency (discovered during impl)

UniDepth is *monocular*: its per-frame depth is only up-to-scale consistent across frames. Source
depth and target depth MUST live in the SAME world scale, or merging G_vis (from source) with
G_hidden (from target) is geometrically invalid (points land at wrong world positions → the whole
oracle is meaningless). This is a camera/scale error-attribution item (charter §7), and is THE most
likely oracle bug.

**Pre-registered mitigation (both must be implemented + verified before trusting P1–P7):**
1. **Metric-scale mode (primary):** use UniDepth's metric output with the *same* per-frame intrinsics;
   verify that source↔target relative pose (from RE10K, metric) is consistent with UniDepth metric
   depth by checking reprojection error (metric P6 ≤ 2px). RE10K poses are already scaled by COLMAP;
   Flash3D applies `scale_pose_by_depth` (RANSAC scale from COLMAP sparse). The oracle must apply the
   SAME per-scene scale to BOTH source depth and the pose translations so geometry is consistent.
2. **Sanity gate (must pass before reporting any P1–P7):** render G_vis alone back to SOURCE view →
   PSNR ≥ 30 dB (self-reconstruction). If this fails, the source geometry/scale/K/render is wrong and
   NO downstream oracle number is trustworthy. Then render G_vis to a target using RE10K pose → if
   this is near-random, the pose↔depth scale is inconsistent → fix scale before proceeding.

If metric mode cannot achieve P6 ≤ 2px, fall back to per-scene global scale align: solve a single
scalar s minimizing reprojection error of a few target points into source; apply s to target depth.
Document whichever mode is used. Do NOT tune s per-target (that would be cheating toward the GT).

## 5. PASS / FAIL thresholds (FIXED — cannot be lowered)

Let `merged` = G_vis ∪ G_hidden oracle scene; `vis_only` = G_vis alone.

**PASS (representation is expressive enough to justify Paper A) requires ALL of:**

| # | Criterion | Threshold |
|---|-----------|-----------|
| P1 | Merged overall PSNR at wide-baseline targets | ≥ 24 dB (well above Flash3D's ~20.8 whole-image on MINE, since oracle has true geometry) |
| P2 | Region-separated occluded PSNR (merged) | ≥ 22 dB |
| P3 | Region-separated OOF PSNR (merged) | ≥ 20 dB |
| P4 | **Causal deletion Δ** (merged − vis_only) on occluded+OOF union PSNR | **≥ 3.0 dB** (charter's load-bearing bar) |
| P5 | Source preservation: merged vs source PSNR | ≥ 28 dB (no visible-region regression) |
| P6 | Reprojection error of oracle hidden points | ≤ 2 px median |
| P7 | Hidden coverage of occluded+OOF pixels | ≥ 70% |

**FAIL interpretation (decisive):**
- If **P1–P3 fail** (oracle ceiling itself low): explicit per-point Gaussian representation cannot
  express wide-baseline hidden regions even with perfect geometry → representation problem, not
  learning → Paper A explicit-Gaussian premise is in doubt → reframe toward Paper B generative or a
  different representation. This would be a major, honest negative result.
- If **P4 fails** (deletion Δ < 3 dB even with oracle hidden): the hidden Gaussians are not
  load-bearing under this eval → either the wide-baseline split isn't wide enough, or the visible
  Gaussians already extrapolate well → re-examine the split/mask (read-only diagnosis, NOT threshold
  change).
- If **P5 fails:** merge/scale/opacity is corrupting the visible region → instrument bug → fix before
  any model work.
- If **P6/P7 fail:** oracle construction bug (coordinate transform / confidence / coverage) → fix
  instruments; do not proceed to model.

## 6. Budget

- No training. Pure forward + rasterization over a fixed dev subset (≤ 200 scene-disjoint scenes).
- Estimated ≤ 1 GPU-hour. If > 3 GPU-hours, stop and profile.

## 7. What this unlocks

- **If PASS:** freeze Paper A charter with the evidence-based wedge = "camera/ray-conditioned
  canonical-volume hidden Gaussian predictor supervised by wide-baseline targets, provably
  load-bearing via deletion, cross-view consistent" — targeting the oracle ceiling. Then implement
  the model (Phase 4).
- **If FAIL:** do NOT implement the model. Write the negative result into decision_log, reframe
  charters (possibly promoting Paper B / different representation), and re-research.

---

*No number in §5 may be edited after the first Oracle run. Any change requires a new pre-registration
entry with justification, per charter §6.*
