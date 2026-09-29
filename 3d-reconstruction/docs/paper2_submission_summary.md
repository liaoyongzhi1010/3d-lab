# Paper 2 Submission Readiness

## Status

MVP complete, but not yet submission-ready.

## Validated Claim

Gate-aware disocclusion prior distillation is necessary and effective in the current setting:

- Unconditional residual student fails on holdout: **-1.60 dB** vs baseline.
- Gate-aware student recovers the holdout: **+1.66 dB** vs baseline.
- Fast visible-gap router matches oracle on the current scene-level holdout: **+4.08 dB**, worst **0.00**.

This establishes the core principle: generative disocclusion priors cannot be blindly distilled; they must be routed by reliability.

## Submission Blockers

1. **Scale**: current student proof uses only key frames from 5 scenes. Needs more scenes and more frames.
2. **Representation**: current student is image-level residual. A submission-ready method should modify Gaussian attributes or add residual disocclusion Gaussians.
3. **Qualitative figures**: need GT | baseline | teacher | student | gate-aware student panels.
4. **Runtime table**: need to quantify speed advantage over Paper 1 teacher.
5. **Router generalization**: logistic router is conservative with small data. Need larger training set or robust rule-based router justification.

## Recommended Path

- Keep Paper 2 as a distinct follow-up, not a duplicate of Paper 1.
- Upgrade student from image residual to Gaussian-level adapter.
- Use Paper 1 teacher outputs only for scenes where the reliability gate says teacher is better.
- Submit only after expansion proves gate-aware student works beyond the current MVP.

## Current Positioning

Paper 2 is a validated research direction and thesis extension. It is not yet at the same maturity level as Paper 1.
