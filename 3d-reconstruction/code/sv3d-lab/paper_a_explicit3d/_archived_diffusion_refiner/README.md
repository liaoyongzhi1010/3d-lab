# ARCHIVED — 2D single-step diffusion refiner route (killed D026)

These files are kept for provenance only. **Do not develop further.**

Killed 2026-07-19 (decision_log D026) because:
1. **Novelty collision (verified on arXiv):** One-Shot Refiner (arXiv:2601.14161, Jan 2026,
   "Boosting Feed-forward Novel View Synthesis via One-Step Diffusion") and Leveling3D
   (arXiv:2603.16211, Mar 2026) already publish "one-step / geometry-aware diffusion refinement
   of feed-forward 3DGS renders."
2. SD1.5 + LCM-LoRA is not equivalent to Difix3D+ (which fine-tunes SD-Turbo on ~80k degraded↔GT
   pairs with reference conditioning).
3. Test-time per-view 2D post-processing is 3D-agnostic (cross-view inconsistency) and breaks the
   "single feed-forward pass → explicit 3DGS renders any view" task definition; the refined pixels
   are not produced by the predicted 3DGS.

Direction after D026: per-scene hidden-Gaussian **optimization** go/no-go oracle → then either an
amortized visible-anchored hidden-completion method (diffusion only as a *training-time* teacher,
inference stays pure 3DGS), or abandon completion if the oracle fails. See decision_log.md D026.
