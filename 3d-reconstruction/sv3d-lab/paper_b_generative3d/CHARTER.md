# Paper B — Generative 3D Reconstruction Charter (v2)

## Scope
Single-view scene-level 3D reconstruction with a generative/learned component that
produces a unified renderable Gaussian scene from one source image, resolving the
"regression=stable-but-smear vs generation=sharp-but-inconsistent" contradiction.

## Core Method: Generative-Prior Distilled Gaussian Refinement
Flash3D (frozen backbone) produces per-pixel gaussians that are geometrically stable
but smear/stretch at novel views. A single-step generative image model (Difix-style,
SD-Turbo LoRA) creates sharp pseudo-GT target views from Flash3D renders. These are
used as training supervision to learn a **feed-forward refinement head** that corrects
the gaussians (positions, scales, colors) in-place.

Key: the generative prior is a **training-time teacher only** — at inference the method
is fully feed-forward from one source image, producing one shared consistent 3D scene.

## Binding Constraints
1. **Source-only inference.** `build_scene(source)` takes NO target RGB/depth/feature.
2. **One shared scene.** Every target view renders from the same `GaussianScene` instance.
3. **Frozen Flash3D backbone weights.** The Flash3D network that predicts initial
   gaussians is frozen. The trainable component is a lightweight refinement head
   (predicts per-gaussian corrections from source features). Gaussian parameter values
   change; backbone network weights do not.
4. **Deletion is causally load-bearing.** Removing refinement-head corrections must
   produce a measurable degradation (restores the backbone-only smeared render).
5. **Generative prior is training-only.** The Difix/SD-Turbo fixer produces pseudo-GT
   supervision during training. It is NOT used at test/inference time.
6. **No target leakage.** Perturbing target supervision or reordering target cameras
   cannot change the output scene.
7. **44 GB VRAM ceiling.** OOM twice → reduce config; no third attempt.
8. **Feed-forward inference.** No per-scene optimization at test time (single forward
   pass: source → backbone → refinement head → clean gaussians → render any target).

## How This Solves the Contradiction
- **Geometric stability/consistency** ← one shared 3DGS, single forward pass, Flash3D
  backbone provides the structural prior; consistency is baked into the 3D representation.
- **Visual sharpness/no smear** ← refinement head trained with generative-prior
  pseudo-GT that is sharp (Difix has seen the GT-like distribution); the head learns
  to fix the systematic smear patterns of Flash3D.
- The generative prior provides the *distribution knowledge* ("what should sharp novel
  views look like") but it's distilled into the 3D representation so consistency is
  maintained — unlike 2D generation which would produce inconsistent per-view outputs.

## Training Pipeline
1. **Stage 0 (pseudo-GT generation):** For training scenes, render Flash3D at target
   views (produces smeared renders). Run Difix fixer to produce sharp pseudo-GT. Store
   as (scene_id, target_fid, sharp_render) pairs.
2. **Stage 1 (refinement head training):** Train feed-forward refinement head on:
   - Render loss: render(refined_gaussians, target_pose) vs sharp_pseudo_GT (L1+LPIPS)
   - Visible preservation: render(refined_gaussians, source_pose) ≈ source_image
   - Regularization: scale penalty, opacity entropy
3. **Optional Stage 2:** Fine-tune Difix on Flash3D smear specifically (if needed).

## Acceptance Criteria
- Visible PSNR not inferior to backbone by >0.20 dB (no "fix" that breaks source)
- Visible VGG-LPIPS not inferior by >0.005
- Novel-view PSNR/LPIPS significantly better than Flash3D (the smear region improves)
- FID/KID: ≥2 metrics better than Flash3D on novel views
- Deletion counterfactual: removing head corrections worsens novel views by >1 dB
- Visual: smear/stretch artifacts visibly reduced in preregistered samples
- Shared scene + no target leakage verified
- Feed-forward single-pass, <44GB, no per-scene optimization

## Method-Selection History
Previous attempts (dual-layer add-Free-points, flow residual, per-primitive refine)
all failed because Flash3D's failure mode is smear (wrong high-alpha gaussians), not
holes. The refinement head directly corrects the wrong gaussians rather than adding
new ones beside them. See state.md for full diagnostic chain.
