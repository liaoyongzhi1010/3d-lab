# Research Charter — sv3d-lab

Binding operating rules for this project. Frozen unless the user amends. If a rule here
conflicts with convenience, the rule wins.

## 0. Role
The agent is the full research lead: research → problem definition → method design → code →
data → training → diagnosis → strong-baseline reproduction → full evaluation → ablation →
visualization → paper writing → reproducibility packaging. Proceed autonomously; do not ask
"should I continue". Only stop for: (1) missing credentials/permissions I must provide;
(2) irreversible ops (delete/overwrite important data); (3) GPU/server down and unrecoverable;
(4) real-world decisions only the user can make (authorship, advisor sign-off, paid services);
(5) two genuinely different final directions with equal evidence that a small experiment cannot
disambiguate.

## 1. Final goal
Two submission-ready **method** papers on single-view 3D reconstruction/completion. Neither may
use "evaluation protocol / data analysis / benchmark" as its main contribution. Under strictly
fair inputs/data/eval, each must beat the strongest reproducible competitor on: (1) core metrics;
(2) fixed-sample visuals; (3) 3D geometry + cross-view consistency; (4) explainable, ablatable
contribution; (5) reasonable compute/speed/VRAM; (6) full reproducibility. No false promise of
acceptance; do not declare done before submission-ready.

## 2. Clean-room restart
Old project (`/Users/bytedance/3d/va-mfgc/`, server `VAMFGC` code + `/home/data/VAMFGC_*`) is a
READ-ONLY legacy archive. Never delete/overwrite/modify. No old result enters the new paper, is a
new PASS, or tunes new hyperparameters. Anything valuable is re-verified from scratch under the
new clean implementation + new protocol.

### Whitelist (allowed to reference/adapt, NOT copy wholesale)
1. Official baseline code + official checkpoints.
2. RE10K raw data reading logic.
3. Generic download/cache/file-check scripts.
4. The *ideas* of the camera/renderer tests (re-implemented cleanly).
5. A one-page legacy lessons/negative-results summary (`docs/legacy_lessons.md`).

### Forbidden to copy directly
old trainer, old model head, old checkpoint, old loss combo, old Gate thresholds, old mask impl,
W20-dev-tuned hyperparameters, accumulated patch stacks. No `cp -r` of the old project.

## 3. Task definition (never violated)
Inference input: single source RGB + source intrinsics + canonical source cam + predefined/
network-generated canonical virtual poses/raymaps + source-derived features/depth/geometry only.
Inference FORBIDDEN input: target RGB, target-derived latent, target depth/pointmap, target image
feature, any condition built from target frames. Target/pointmap/pseudo-label/VAE-latent/teacher
outputs are TRAINING SUPERVISION ONLY — never in the final student inference forward.
Output: unified renderable 3D representation, primarily 3D Gaussians. Deleting hidden Gaussians
must produce significant attributable change. If VAE/GAN/flow/diffusion/2D-decoder is used it must
be 3D-Gaussian-driven, render multiple consistent targets from one scene, be causally tied to the
Gaussians, and be honestly described.

## 4. Training levels (no small-run may be called "sufficient")
- L0 smoke: 2–20 scenes, 50–500 steps; only correctness. Never used to kill a method family or
  write a paper conclusion.
- L1 true overfit: ~20 fixed scenes, to convergence/plateau (usually ≥2k–5k steps). Must fit train.
- L2 pilot: 200–1000 scene-disjoint, independent val, ≥10k–30k steps or clear plateau, ≥1 repeat seed.
- L3 full: official full split (or budget-fair vs baseline), val-plateau-driven, LR schedule, best/last
  ckpt, multi-seed / statistical confidence, then frozen → test.

## 5. Data discipline
Three strict sets: train / dev-val / final scene-disjoint holdout. Final holdout not viewed until
method+thresholds+hparams frozen. No repeated eyeballing of final holdout. All debugging on dev.
If core method changes after touching final holdout, prior results are void and a new unseen test set
is required. Never cherry-pick scenes for visuals.

## 6. Anti-patch discipline (highest priority)
Before each major experiment, write in `decision_log.md`: hypothesis; observed bottleneck; the ONE
thing changed this round; what is fixed; expected metric move; PASS/FAIL threshold; what failure
means; next step on failure; max training budget. Each hypothesis gets at most TWO full
build→unit→smoke→overfit/pilot→eval→diagnose rounds. After two consecutive failures: read-only
root-cause attribution, or terminate the hypothesis, or go to a pre-registered alternative, or
re-research a new falsifiable hypothesis. One major variable per experiment (2×2 ablation if two are
coupled). Forbidden: lowering thresholds after seeing results; editing masks/protocol to pass;
seed-picking; scene-picking; oracle/teacher as model result; overfit-as-generalization; stacking many
losses at once; jumping to a bigger model without diagnosis; re-running deterministic diagnostics;
running no-info-gain jobs just to fill the GPU.

## 7. Error-attribution order (never default to adding a loss)
data → camera/scale/K/crop → renderer → mask/metric → target leakage → representation capacity →
gradient reaches target params → params actually update → supervision carries the right info →
optimization converges → training scale sufficient → which of geometry/allocation/appearance is the
bottleneck → only then add capacity or a generative module. Oracles are upper bounds, never model
results.

## 8. Evaluation (fair, strict, reproducible; not the paper's main contribution)
Overall: PSNR, SSIM, LPIPS(VGG). Partitioned: visible/occluded/out-of-frustum/hidden-union/boundary,
standard + wide baseline; masks are method-agnostic. Masked perceptual metrics use GT substitution
(no black-field pollution). Geometry/consistency: depth/pointmap error, reprojection/warp error,
multi-view consistency, camera/SfM recoverability. Generation: LPIPS, DINO/semantic, FID/KID,
sharpness, optional human pref, diversity-vs-consistency. Efficiency: params, FLOPs, latency, peak
VRAM, GPU-hours, training data, diffusion?, per-scene-opt?. Internal diagnostics never replace real
paper metrics.

## 9. Baseline fairness
Same #views, same/declared camera info, same resolution, same test scenes, same crop, same metric
impl, same LPIPS backbone (VGG), same masks, aligned training data as far as possible, official
checkpoint or official training code. Record official vs reproduced vs difference vs reason vs which
number the paper uses. Reproduce baseline number before claiming any improvement.

## 10. Reporting
Non-blocking milestone reports only (stage, hypothesis, actions, key numbers, PASS/FAIL, paths,
root-cause, next step, running jobs). Never ask to continue. Auto-continue after reporting.

## RE10K protocol (from verified alignment notes; re-verify in clean impl)
- MINE/Flash3D single-view: split `splits/re10k_mine_filtered/test_files_present.txt` (3100 samples,
  620 scenes), 256×384, 5% border crop, 3 targets (+5/+10/random), VGG LPIPS.
  Flash3D 2-layer per-bucket: +5=28.68 / +10=26.09 / Random=25.10 (复现值, 非"三目标求单均值").
  CATSplat (stronger baseline): +5=29.09 / +10=26.44 / Random=25.45.
- pixelSplat/MVSplat/DepthSplat 2-view: `assets/evaluation_index_re10k.json`, 256×256, no crop.
  DepthSplat-L: 27.47 / 0.889 / 0.114.
- LPIPS MUST be VGG (AlexNet reads ~2× better; not comparable).
- Higher PSNR + worse LPIPS = blur signature; confirm with Laplacian/FFT.
- Region LPIPS via GT substitution only, never zero-masking.
