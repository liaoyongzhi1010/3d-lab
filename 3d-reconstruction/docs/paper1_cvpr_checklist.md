# Paper 1 → CVPR/ICCV Readiness Checklist

Target: CVPR / ICCV main track for "Selective Geometry-Guided Generation for
Single-View Scene Reconstruction". This file tracks every requirement a top venue
expects and its current status. Honest — no item is checked unless truly done.

Legend: [x] done · [~] in progress · [ ] todo · (GPU) needs GPU (currently down)

---

## 1. Core claim & novelty
- [x] Clear, defensible novelty: test-time *selective* + *gated* geometry injection into a frozen generative backbone (vs train-time global alignment like Geometry Forcing).
- [x] Mechanism explanation (not just a number): invisible-gap → benefit, r=0.983.
- [x] Honest negative/limit: scene-level observable gate is imperfect (proxy r=0.46), motivating frame-level predictor.

## 2. Quantitative evidence
- [x] Large N: 166 scenes (from 16), mechanism reproduced (r=0.982, 10× scale).
- [x] Region-separated metrics (visible vs invisible) — the key framing.
- [x] Difficulty split (hard/mid/easy) showing where it helps/hurts.
- [x] Gate ablation (always/rule/oracle_inv/oracle_gap).
- [~] N → 150+ for statistical strength (GPU) — batch7 running (→N≈148).
- [x] **LPIPS + SSIM** in visible/invisible regions (e204): invisible SSIM 0.58→0.73, LPIPS 0.095→0.078, visible unchanged.
- [x] **Bootstrap 95% CI** on main deltas (e203): mechanism r CI [0.972,0.992], hard/easy buckets significant.
- [~] **Geometric/temporal consistency**: temporal-consistency proxy measured (e205) — neutral/negative, reported honestly in limitations. True reprojection error needs depth (future).

## 3. Baselines & comparisons
- [x] Ablations that isolate each component (asymmetric injection, gate, learned weights).
- [x] Naive-combination baseline = "always inject" row (explicit).
- [x] At least one external comparison framed correctly (Gen3R-alone = single-view baseline; Flash3D = injection-source reference; always/rule/ours/oracle rows with bootstrap 95% CI). Paper1 §4.4.1, e207.
- [x] Standard full-image metrics (PSNR/SSIM/LPIPS(VGG)/FID/KID) vs published components (§4.5.1, e210). **Ours wins LPIPS and FID** over both Gen3R and Flash3D (standard generative-NVS comparison axis). No invented metrics.

## 4. Qualitative results
- [x] GT | baseline | ours panels with disocclusion mask + error maps (e201).
- [x] 3+ diverse scenes (closet, staircase, outdoor patio, playground).
- [x] **Teaser figure (fig0)**: input + novel views with disocc PSNR callouts (e206).
- [x] **Failure case** figure (train_0bc64, -8.5 dB → motivates gate).
- [ ] Video / multi-frame consistency strip (optional).

## 5. Method presentation
- [x] Method text (asymmetric injection, clean-space confidence, InjectionWeightNet, gate).
- [x] **Pipeline/architecture figure** (draw.io, validated).
- [x] Precise math: injection eq, confidence, gate rule, loss.
- [x] Algorithm box (pseudo-code) for the test-time procedure (§3.6).

## 6. Implementation & reproducibility
- [x] Implementation details section: backbones, resolution, steps, hardware, runtime.
- [x] Dataset/protocol section: RE10K split, N scenes, frame sampling, mask source, metric definitions.
- [x] Reproducibility: deterministic eval, released dataset + scripts. (code release on acceptance)

## 7. Writing & structure
- [x] Abstract, intro, related work, method, experiments, limitations, conclusion.
- [x] Related work coverage: pixelSplat/MVSplat/DepthSplat/Flash3D, RePaint/ViewCrafter/CAT3D/GenWarp, Geometry Forcing, REPA, reliability/routing.
- [x] Contributions: 4 crisp claims.
- [x] LaTeX (CVPR/ICCV template) + BibTeX. `docs/paper1_latex/` compiles to 8-page main.pdf (0 undefined refs); refs.bib complete; TikZ pipeline figure + qual success/failure panels embedded; template-swap notes in README.

## 8. Ethics/soundness
- [x] Honest reporting (positive + negative), no over-claiming.
- [x] Compute/carbon note (minor), license of data/models. (§4.1 compute/carbon/licensing paragraph)

---

## Priority order (given GPU currently down)

**Now (no GPU needed):**
1. Pipeline/architecture figure (draw.io) + algorithm box + precise math.
2. Implementation-details & protocol sections.
3. Teaser figure from existing panels.
4. Tighten related work + contributions.
5. Bootstrap CI on N=166 deltas (CPU, e203).

**When GPU returns:**
6. batch7/8 → N≥150.
7. LPIPS/SSIM region metrics on all scenes.
8. Reprojection/revisit geometric-consistency metric.
9. 1-2 more diverse qualitative + a failure case.
10. (optional) external baseline renders.
