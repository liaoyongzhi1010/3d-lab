# Three-Paper Decision Memo

## Overall Position

All three papers are based on single-view 3D scene reconstruction, but they must remain independent:

1. Paper 1: quality-first generative refinement.
2. Paper 2: deployment-first gate-aware distillation into a feed-forward student.
3. Paper 3: reliability learning / scheduler for disocclusion failure prediction.

**Updated 2026-08-17.** Paper 1 is submission-ready (now validated at N=92). Paper 3 has been upgraded from a problem statement to a **working frame-level method that beats the rule baseline and approaches oracle** — its maturity jumped the most this cycle. Paper 2's Gaussian-color adapter was tested and found to be an architectural dead-end (documented negative result); Paper 2 will ship on the validated image-level gate-aware student, scaled up, with the Gaussian-color result as a principled ablation.

---

## Paper 1: Selective Geometry-Guided Generation

**Status:** submission-ready; validated at N=92.

**Method:** Gen3R diffusion backbone + Flash3D evidence + asymmetric injection + clean-space confidence + learned InjectionWeightNet + observable reliability gate.

**Training module:** InjectionWeightNet.

**Validated results:**

- N=16: always-inject invisible PSNR +1.58 dB; visible delta +0.016 dB; observable gate +2.07 dB, worst -0.41 dB; mechanism r=0.985.
- **N=92 large-scale validation:** always +0.97 dB; oracle_gap>5 +1.79 dB, worst -0.17; **mechanism r=0.980 (reproduced across 6x larger, harder set)**. Difficulty split: hard(<12dB) +3.07 dB, easy(>=20dB) -2.31 dB — sharpens the selective-gating argument.
- Learned injection: matches hand-crafted invisible PSNR and improves perceptual/visible metrics.

**Remaining submission work:**

1. (done) Expand to N=92 via train pool; N still growing (batches 6+ in flight).
2. Add comparison to Flash3D-alone where protocol allows (note: Flash3D/Gen3R are components, not competitors — framed as ablations).
3. Freeze final figures and write LaTeX paper.

**Recommendation:** submit this first.

---

## Paper 2: Gate-aware Disocclusion Prior Distillation

**Status:** MVP validated; method direction corrected after a decisive negative result.

**Method:** distill Paper 1's slow generative disocclusion prior into a fast feed-forward student, invoked only when a reliability gate predicts the teacher helps.

**Training modules:** RouterNet + image-level gate-aware student (validated). Gaussian-color adapter tested and rejected.

**Validated results:**

- Unconditional image student fails on holdout: -1.60 dB.
- Gate-aware student recovers holdout: +1.66 dB.
- Fast rule router `vis_gap > 0` matches oracle on current holdout: +4.08 dB, worst 0.00.

**Negative result (documented):** a color-only Gaussian adapter that edits Flash3D `features_dc` on the source plane cannot generate disocclusion content in target views — even pure overfitting reaches only ~7% of the teacher-baseline gap and does not generalize (holdout -1.72 dB). Root cause: disoccluded target pixels correspond to Gaussians never observed in the source, so source-plane recoloring has no leverage. This is exactly the feed-forward limitation the generative teacher fixes.

**Corrected plan:**

1. Ship on the validated image-level gate-aware student, **scaled from 5 keyframes to full per-frame teacher renders** (batch6 `--save_teacher_npy`, ~20 scenes × ~48 frames; script `e031b_student_fullnpy.py`).
2. Use the Gaussian-color negative as an ablation ("why not intervene at the Gaussian color layer").
3. Optional bonus: disocclusion *residual* Gaussians (predict new Gaussians rather than recolor) — attempt only if time permits.
4. Add runtime table proving deployment benefit; qualitative panels GT | baseline | teacher | student | gated student.

**Recommendation:** publishable on the image-level student + ablation. Residual-Gaussian upgrade is optional upside.

---

## Paper 3: Learning Disocclusion Reliability

**Status:** upgraded to a working frame-level method (biggest maturity gain this cycle).

**Method:** learn to predict whether a scene/frame will benefit from generative disocclusion refinement, from test-time observable features, under grouped (leave-one-scene-out) CV.

**Training module:** Disocclusion ReliabilityNet (scene-level and frame-level).

**Validated results:**

- Scene-level (N=92): learned ReliabilityNet +1.18 dB ≈ rule +1.18 dB, both far from oracle +1.82 dB — data-limited, features saturate.
- **Frame-level (746 frames, 18 scenes): ReliabilityNet acc 0.799, +1.75 dB mean = 91% of oracle (+1.92), clearly beating the rule (+1.39).** Risk-averse operating point: worst-case from -11.68 (always) to -0.82 while still +0.98.
- Mechanism correlation stable at both granularities: +0.980 (scene) / +0.981 (frame).

**Key finding:** descending from scene to frame granularity provides the sample size that lets learning beat the rule and approach oracle, plus a tunable mean-vs-worst frontier a fixed rule cannot offer. This is a genuine, publishable contribution.

**Remaining submission work:**

1. Scale frame dataset across more batches (in flight; target several thousand frames).
2. Add depth-uncertainty / opacity / camera-motion features.
3. Report PR/AUC curves and the full mean-vs-worst operating frontier.
4. Optional pixel-level ReliabilityNet.

**Recommendation:** now a real paper, not just a direction. Continue scaling.

---

## Final Recommendation

Updated staging:

1. **Paper 1 — submission-ready.** Strongest method; validated at N=92 with reproduced mechanism. Write it up first.
2. **Paper 3 — working method.** Frame-level ReliabilityNet beats the rule and approaches oracle; scale data and add curves.
3. **Paper 2 — publishable MVP + corrected direction.** Ship image-level gate-aware student scaled up, with the Gaussian-color negative as an ablation; residual-Gaussian upgrade optional.

Current truth: Paper 1 is near-ready; Paper 3 has a working, honestly-validated frame-level result; Paper 2 has a validated MVP plus a documented negative that sharpens its story. All three now have an independent method, a training module, and results — but "100% submission-ready" still requires the data scaling and write-up in progress.
