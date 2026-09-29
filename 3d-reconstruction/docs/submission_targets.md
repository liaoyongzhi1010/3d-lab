# Submission Targets & Figure/Table Checklist (three papers)

Honest venue assessment based on **current validated data** (updated 2026-08-18).
No inflation: each paper lists "current-ready" and "after-completion" targets.

---

## Paper 1 — Selective Geometry-Guided Generation

**Current-ready:** 3DV / WACV / BMVC (main track), or CVPR/ICCV/ECCV workshop.
**After-completion target:** CVPR / ICCV / ECCV main track.

**Why:** validated at N=166 (mechanism r=0.985→0.982 reproduced across 10× scale), visible-region
lossless (+0.016 dB, exact no-harm), selective gate removes worst-case, InjectionWeightNet as a
learnable module. Difficulty split (+2.49 hard / −2.52 easy) is a strong argument.

**Completion checklist:**
- [x] N=16 → N=166 large-scale validation
- [x] Mechanism scatter figure (fig1, r=0.982, N=166)
- [x] Difficulty-bucket figure (fig2)
- [x] Gate comparison figure (fig3)
- [x] Qualitative panel with disocclusion mask + error maps (e201)
- [x] Related-work paragraph vs Geometry Forcing (train-time vs test-time/selective)
- [x] Push N to ~170 (batch7+8 done → N=166)
- [x] Component ablation vs single-view baseline (e207, §4.4.1, bootstrap CIs)
- [x] Final LaTeX (docs/paper1_latex, 8-page pdf, TikZ pipeline + qual panels embedded)
- [ ] Add geometric-consistency metric (borrow Geometry Forcing revisit/reprojection error)
- [ ] Optional external methods (pixelSplat/MVSplat/ViewCrafter renders; GPU/setup)

**Figures/Tables (conference layout):**
1. Teaser: input → baseline vs ours on a hard disocclusion scene (from e201).
2. Method pipeline: Gen3R denoising + Flash3D asymmetric injection + gate (schematic, TODO draw.io).
3. Tab1 main: always / rule / oracle gates × {mean, worst, visible-Δ} at N.
4. Tab2 difficulty split (hard/mid/easy).
5. fig1 mechanism, fig2 difficulty, fig3 gate.
6. Qualitative panels (2-3 scenes) with mask + error maps.

---

## Paper 3 — Learning Disocclusion Reliability

**Current-ready:** WACV / BMVC / 3DV, or a "reliability/uncertainty in generative
models" workshop at a top venue.
**After-completion target:** top-venue main track (novelty of the problem is high).

**Why:** frame-level ReliabilityNet now beats the rule (+1.60 vs +0.89), reaches
94% of oracle, ROC-AUC 0.949 / PR-AUC 0.961 (2,898 frames / 74 scenes),
and offers a tunable mean-vs-worst frontier + 41% teacher-call savings. Mechanism
corr +0.978 across 2,898 frames. Patch-level explored (§4.4): AUC 0.72, frame is the sweet spot.

**Completion checklist:**
- [x] scene→frame granularity study (learning beats rule only at frame level)
- [x] grouped leave-one-scene-out CV (no frame leakage)
- [x] ROC/PR/AUC + operating frontier + compute-saved figures (fig4/5/6)
- [x] extended features (camera motion, disocc ratio, interactions) — PR-AUC↑ to 0.96
- [x] scale frames to 2,898 / 74 scenes (batches 5+6+7+8)
- [x] patch-level ReliabilityNet explored (e208, §4.4 — honest granularity boundary)
- [ ] pixel-level ReliabilityNet (optional, spatially selective injection)
- [x] final LaTeX (`docs/paper3_latex/`, 4-page pdf, fig4/5/6 embedded, 0 undefined refs)

**Figures/Tables:**
1. Teaser: "predict where the generative prior helps" — a frame with predicted reliability heatmap.
2. fig4 operating frontier, fig5 ROC+PR, fig6 compute-saved.
3. Tab: always / rule / FrameNet / risk-averse / oracle × {mean, worst, AUC, saved%}.
4. Feature-ablation table (base vs extended).

---

## Paper 2 — Gate-aware Disocclusion Prior Distillation

**Current-ready:** WACV / 3DV / BMVC (main track).
**After-completion target:** WACV / 3DV / BMVC main track (add residual-Gaussian variant for a stretch to a top venue).

**Why (honest):** validated image-level gate-aware student now reaches **+5.77 dB
held-out invisible-region PSNR (4 scenes, 161 frames), trailing the teacher by
only 0.10 dB, with zero visible-region change and a 2833× speedup**. A documented
negative result (Gaussian color-adapter is an architectural dead-end) serves as a
principled ablation. This is main-track material.

**Completion checklist:**
- [x] image-level gate-aware student MVP (E-030)
- [x] full-frame student trainer (e031b, gen3r/f3d basis, gate-aware)
- [x] Gaussian color-adapter negative result (e150) — becomes an ablation
- [x] high-gain teacher data (21 scenes re-run with --save_teacher_npy) → student holdout +5.77 dB
- [x] runtime table: student 0.087 s vs teacher 247 s = 2833× (e202)
- [x] qualitative panels: GT | baseline | ours | disocc mask | error maps (e201)
- [ ] optional: disocclusion residual-Gaussian upgrade (predict new Gaussians, not recolor)
- [x] final LaTeX (`docs/paper2_latex/`, 3-page pdf, distillation+runtime+ablation tables, 0 undefined refs)

**Figures/Tables:**
1. Teaser: slow teacher → fast student pipeline.
2. Tab main: unconditional vs gate-aware student, holdout, {invisible-Δ, visible-Δ}.
3. Tab runtime: teacher vs student latency/throughput.
4. Ablation: image-level vs Gaussian-color adapter (why not the Gaussian layer).
5. Qualitative panels.

---

## Overall

| Paper | Now | After completion | Confidence |
|---|---|---|---|
| Paper 1 | CVPR/ICCV/ECCV (2 datasets + std metrics + gate sweep done) | CVPR/ICCV/ECCV | high |
| Paper 3 | WACV/BMVC/3DV | top-venue main | medium-high |
| Paper 2 | WACV/3DV/BMVC | WACV/3DV/BMVC (+top-venue stretch) | medium-high |

**Paper 1 top-venue readiness (updated):** now has (i) cross-dataset generalization
(RealEstate10K indoor + ACID outdoor, mechanism reproduces without retuning),
(ii) standard full-image metrics PSNR/SSIM/LPIPS(VGG)/FID/KID with LPIPS+FID wins,
(iii) published-methods positioning table (honest protocol note), (iv) gate-threshold
sensitivity sweep (broad plateau, not cherry-picked), on top of N=166 scale + bootstrap
CIs. Remaining stretch items are optional (real external reproduction, denoise-step /
injection-timing sweep needing GPU reruns).

For a thesis: the three together form a complete, honest story (positive + negative
results) — strong for a master's/PhD chapter set. For journals (no conference
deadline): deepen and target IJCV / TPAMI.
