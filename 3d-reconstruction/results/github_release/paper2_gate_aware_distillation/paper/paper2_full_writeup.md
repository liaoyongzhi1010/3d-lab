# Selective Disocclusion Prior Distillation for Feed-Forward Single-View 3D Reconstruction

## Abstract

A generative teacher can improve disoccluded regions, but it is expensive and non-monotonic: on some scenes it is worse than the feed-forward baseline. We study whether its useful behavior can be distilled within a high-teacher-gain regime. Training and evaluation scenes are curated using teacher-minus-baseline invisible-region quality measured against ground truth, and visibility masks are derived offline from the full target clip. This oracle curation and oracle visibility define an upper-bound experiment, not deployable single-view inference. On a 4-scene/161-frame selected holdout, the image-space student improves invisible-region PSNR by **+5.77 dB** (teacher: **+6.09 dB**; **0.32 dB** gap; **94.7% retained**) with visible change **+0.000 dB**, while running in **0.087 s/scene** versus **247 s**, a **2833x speedup**. A fixed custom 759-frame diagnostic shows that the student improves LPIPS and FID over the feed-forward evidence component while trailing the teacher on both metrics; it is not an official Flash3D-protocol comparison. A negative ablation shows that recoloring source-plane Gaussians with frozen geometry lacks support at newly disoccluded target pixels.

## 1. Introduction

The Paper 1 teacher combines a slow Gen3R generative backbone with Flash3D geometry evidence. Its usefulness is not monotonic: it helps hard disocclusions and can hurt easy scenes. Unconditional distillation therefore copies both useful and harmful teacher behavior.

Selective disocclusion prior distillation studies conditional compressibility. Scenes whose teacher-minus-baseline invisible-region gain exceeds 3 dB are retained using teacher outputs and ground truth. The student is trained and evaluated within that selected regime. The selection cannot be computed for unseen scenes without the unavailable outcomes and therefore does not constitute an inference-time decision method.

### Contributions

1. **Conditional principle:** distillation of a non-monotonic teacher on an oracle-curated high-teacher-gain regime, explicitly treated as an upper bound.
2. **Model and evidence:** a 46,371-parameter image-space student with +5.77 dB held-out invisible-region gain, +0.000 dB visible change, and a 2833x speedup; a fixed custom 759-frame diagnostic reports quality versus speed without claiming an official-protocol win.
3. **Architectural diagnosis:** the tested color-only Gaussian adapter recovers only ~7% of the gap when overfit and gives -1.72 dB on holdout; a support schematic explains why this motivates image-space correction.

## 2. Method

### 2.1 Non-monotonic teacher and oracle selection

The teacher is Gen3R refined by Flash3D-guided selective disocclusion injection. It provides useful targets on high-gain scenes but is not universally better than the baseline. Teacher-minus-baseline invisible-region PSNR is measured against ground truth, and scenes above 3 dB are retained for training and evaluation.

### 2.2 Image-space student

The student consumes:

```text
[f3d_rgb, baseline_rgb, visibility_mask, invisible_mask]
```

A four-layer convolutional network predicts residual `r`:

```text
student = M * baseline + (1 - M) * (f3d + r)
```

Visible pixels are copied exactly from the baseline. Invisible-region L1 losses supervise against teacher and ground truth; a visible identity loss supplements the architectural identity. The visibility mask enforces exact no-harm inside the evaluated output but does not select scenes. It is computed offline from the full target clip and is oracle information, not a single-view prediction.

### 2.3 Selection scope

The student is applied directly on the reported selected holdout with precomputed oracle visibility. No observable selector or online visibility estimator for uncurated scenes is established. General use requires both to depend only on inputs available at inference.

## 3. Experiments

All regional PSNR values below are invisible-region PSNR; visible-region PSNR is a no-harm check.

### 3.1 High-teacher-gain holdout

Training uses high-teacher-gain scenes and full per-frame teacher renders. The held-out split contains 4 scenes and 161 frames drawn from the same oracle-selected regime.

| Split | Baseline | Teacher | Student | Student - Baseline | Visible change |
|---|---:|---:|---:|---:|---:|
| Train (n=696) | 15.31 | 20.68 | 20.80 | **+5.49** | +0.000 |
| Holdout (n=161) | 14.04 | 20.13 | 19.81 | **+5.77** | +0.000 |

The student retains 5.77 / 6.09 = **94.7% (~95%)** of the teacher's held-out gain and trails its PSNR by **0.32 dB** (19.81 versus 20.13 dB). Single-scene holdouts during scaling gave +5.96 and +4.75 dB.

### 3.2 Why selection matters

On scenes where the teacher does not outperform the baseline, the student is flat-to-slightly-negative because there is no useful teacher signal to copy. This motivates oracle high-gain curation but does not solve selection from observable inputs. The reported performance is conditional on the curated regime.

### 3.3 Runtime

| Method | Time/scene | Invisible gain | Visible behavior |
|---|---:|---:|---:|
| Teacher, 30-step diffusion | 247 s | +6.09 dB | lossless |
| **TinyStudent** | **0.087 s GPU / 11.3 s CPU** | **+5.77 dB** | lossless |
| Speedup | **2833x GPU** | **94.7% kept (~95%)** | - |

### 3.4 Fixed custom 759-frame quality-speed diagnostic

E-211 evaluates all four components on the same 21-scene/759-frame disocclusion-heavy diagnostic at 256x256 with VGG-LPIPS. Flash3D is the geometry/evidence component measured inside this custom diagnostic, not an official benchmark opponent.

| Method | PSNR | SSIM | LPIPS-VGG | FID | Time/scene |
|---|---:|---:|---:|---:|---:|
| Gen3R baseline | 17.88 | 0.595 | 0.294 | 31.6 | 247 s |
| Flash3D evidence | 19.98 | 0.664 | 0.290 | 72.9 | 0.09 s |
| Teacher (slow) | 19.37 | 0.633 | 0.258 | 31.3 | 247 s |
| **Student** | 18.17 | 0.591 | 0.284 | 33.6 | **0.087 s** |

Within this fixed diagnostic, the student has lower LPIPS (0.284 versus 0.290) and FID (33.6 versus 72.9) than the Flash3D evidence component at similar measured runtime, but trails the teacher (LPIPS 0.258, FID 31.3). This same-diagnostic statement does not establish an official-protocol ranking. The vector quality-speed plot is generated directly from `results/E-211_student_standard_metrics.json` by `scripts/generate_quality_speed.py`.

### 3.5 Negative ablation: source-plane Gaussian recoloring

E-150 edits only Flash3D `features_dc` on the source plane and freezes means, opacity, scale, and rotation. It recovers only ~7% of the teacher-baseline gap when overfit on one scene and gives -1.72 dB on holdout.

The pipeline schematic states the limited conclusion supported by these measurements: recoloring existing source-plane Gaussians does not add geometric support where the target view newly reveals content. The schematic is not an invented render output and does not imply that every Gaussian-space adapter must fail. It motivates the tested image-space residual over geometry evidence.

## 4. Discussion And Limitations

Within the oracle-selected regime, the student retains useful teacher behavior at feed-forward cost and preserves baseline visible pixels exactly. The reported training run used the Flash3D disocclusion basis (`--base_mode f3d`), offline frame-label weighting (`--gate_aware`), and a 4-scene/161-frame holdout. Released checkpoint evaluation is reproducible, but exact retraining is partial rather than a self-contained one-command workflow: the external teacher arrays, frame-label JSON, and original holdout manifest are not included, and the exact four scene IDs and frame-label path are unknown in this release. `../TRAINING_MANIFEST.md` records the known settings, unknowns, and a placeholder-only command template. Selection uses teacher and ground-truth outcomes, while visibility uses the full target clip; this is therefore an upper-bound condition, not an end-to-end system for unseen scenes. Future observable selection and visibility estimation are required beyond this regime. The fixed 759-frame diagnostic is custom and cannot support cross-protocol ranking. A future adapter could also predict new disocclusion Gaussians rather than recoloring frozen geometry.

## 5. Conclusion

On the oracle-curated high-teacher-gain regime, the student recovers +5.77 dB of invisible-region PSNR versus the teacher's +6.09 dB, runs roughly 3000x faster, and leaves visible regions unchanged. The custom quality-speed diagnostic and honest negative ablation delimit the evidence. Application to uncurated scenes remains contingent on future selection and visibility estimation using observable inference-time inputs.
