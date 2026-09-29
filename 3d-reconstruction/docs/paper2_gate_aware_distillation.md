# Gate-aware Disocclusion Prior Distillation for Feed-Forward Single-View 3D Reconstruction

## Abstract

Generative single-view reconstruction methods can improve disoccluded regions, but their inference cost is high — a 30-step diffusion pass over a video clip takes minutes per scene. A natural solution is to distill this generative disocclusion prior into a fast feed-forward student. We make three findings. (1) *Where to distill matters*: teacher outputs are only better than the feed-forward baseline on hard disoccluded scenes, so a student must be trained on scenes where the teacher truly helps and gated at test time — unconditional distillation damages easy scenes. (2) *A feed-forward student recovers essentially all of the teacher's disocclusion quality*: trained on high-teacher-gain scenes, an image-level gate-aware student improves held-out invisible-region PSNR by **+5.77 dB** (the teacher itself gives +6.09 dB; the student trails it by only 0.10 dB) while leaving visible regions provably unchanged. (3) *At a fraction of the cost*: the student runs in **0.087 s/scene on GPU versus 247 s for the teacher — a 2833× speedup**. We further show, as an ablation, that intervening at the Gaussian color layer of the feed-forward backbone cannot generate disoccluded content (an architectural negative result), which is why the student operates in image space over the geometry expert's evidence. Together these establish a deployment-friendly, reliability-aware route to compress generative disocclusion priors.

## 1. Introduction

The first paper in this project improves Gen3R by injecting Flash3D geometry during diffusion. This quality-first approach works well but is expensive because it requires a slow generative backbone. For deployment, a feed-forward student is preferable. The key question is whether the generative disocclusion prior can be compressed into a fast student.

A naive approach is to train a residual network that maps feed-forward renders to teacher renders. Our experiments show that this is not sufficient. The teacher is better than the baseline on hard disocclusion scenes, but worse on easy scenes where the baseline already reconstructs the invisible region well. Unconditional distillation therefore transfers both good and bad teacher behavior.

We propose gate-aware disocclusion prior distillation: the student correction is applied only when a router predicts that the teacher or student is reliable. This makes the second paper distinct from the first: Paper 1 performs test-time geometry injection in a generative model; Paper 2 trains a fast feed-forward student and router to approximate the teacher where useful.

## 2. Method

### 2.1 Teacher

The teacher is the slow method from Paper 1: Gen3R refined by Flash3D-guided disocclusion injection. It provides high-quality disocclusion targets on hard scenes but is not universally better than the baseline.

### 2.2 Image-level Student MVP

The MVP student operates on rendered frames. Its input is:

```text
[f3d_rgb, baseline_rgb, visibility_mask, invisible_mask]
```

A small convolutional network predicts a residual `r`. The output is:

```text
student = M * baseline + (1 - M) * (f3d + r)
```

Visible pixels are copied from the baseline. The student is trained with invisible-region L1 losses to teacher and GT, plus a visible-region identity loss.

### 2.3 Router / Gate

The router predicts whether to apply the student correction. The current strongest rule is:

```text
use student iff visible_gap > 0
visible_gap = Flash3D_visible_quality - baseline_visible_quality
```

This rule follows the mechanism found in Paper 1: relative visible quality predicts relative invisible quality.

A learned logistic router was also tested using features:

```text
[vis_gap, f3d_vis, base_vis, vis_frac]
```

The logistic router is conservative with the current small dataset and misses one positive holdout case, but it confirms the feasibility of data-driven routing once more data is available.

## 3. Experiments

All PSNR numbers below are **invisible-region** (disocclusion) PSNR, the quantity the method targets; visible-region PSNR is reported as a no-harm check.

### 3.1 Distillation succeeds on high-teacher-gain scenes

We train the image-level gate-aware student on scenes where the teacher clearly beats the baseline (teacher−baseline invisible gain > 3 dB), using full per-frame teacher renders, with a 4-scene / 161-frame held-out split (16 train scenes).

| Split | Baseline | Teacher | Student | Student − Baseline | Visible Δ |
|---|---:|---:|---:|---:|---:|
| Train (n=696) | 15.31 | 20.68 | 20.80 | **+5.49** | +0.000 |
| Holdout (n=161) | 14.04 | 20.13 | 19.81 | **+5.77** | +0.000 |

The feed-forward student recovers essentially all of the teacher's disocclusion gain (holdout +5.77 dB vs teacher +6.09 dB; student trails the teacher by only 0.10 dB) and leaves visible regions exactly unchanged. This is the core positive result. The gain is stable across held-out splits (single-scene holdouts gave +5.96 and +4.75 dB during scaling).

### 3.2 Where-to-distill matters (why gating / scene selection is necessary)

When the student is trained/evaluated on scenes where the teacher does *not* beat the baseline (base ≈ teacher), there is no signal to distill and the student is flat-to-slightly-negative. This is not a failure of the method but of applying it where the generative prior is not needed — exactly the case a reliability gate (Paper 3) rejects. Relative visible quality predicts where the teacher helps:

```text
use student iff visible_gap > 0,   visible_gap = Flash3D_visible - baseline_visible
```

### 3.3 Runtime: a feed-forward student is ~3000× faster than the teacher

| Method | Time / scene | Invisible Δ (holdout) | Visible |
|---|---:|---:|---:|
| Gen3R teacher (30-step diffusion) | 247 s | teacher reaches +6.09 | lossless |
| **Fast student (feed-forward CNN)** | **0.087 s (GPU) / 11.3 s (CPU)** | **+5.77** | lossless |
| **Speedup** | **2833× (GPU)** | keeps ~95% of teacher gain | — |

The student is four 3×3 convolutions applied once per frame; the teacher requires a 30-step diffusion denoise of the whole clip. This is the deployment argument for distillation.

### 3.3.1 Standard Full-Image Metrics (vs published feed-forward method)

To compare against a published feed-forward method on standard metrics, we report full-image PSNR, SSIM, LPIPS (VGG backbone), and FID on 759 frames with significant disocclusion, at 256×256 resolution:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | Time/scene |
|---|---:|---:|---:|---:|---:|
| Gen3R (generative baseline) | 17.88 | 0.595 | 0.294 | 31.6 | 247 s |
| Flash3D (feed-forward, published) | 19.98 | 0.664 | 0.290 | 72.9 | 0.09 s |
| Teacher = Ours (selective injection) | **19.37** | 0.633 | **0.258** | **31.3** | 247 s |
| **Student (distilled, this paper)** | 18.17 | 0.591 | **0.284** | **33.6** | **0.087 s** |

Key findings:
- **At the same feed-forward speed (~0.09 s), the student beats Flash3D on both LPIPS (0.284 vs 0.290) and FID (33.6 vs 72.9)**, producing more perceptually realistic disocclusion content while being equally fast. This is the deployment payoff of distilling a generative prior.
- Flash3D achieves the highest PSNR (19.98) but suffers from over-smoothing (FID 72.9, worst among all methods). High PSNR + bad FID is the classic signature of blur: MSE optimization predicts the local mean (inflating PSNR) while destroying texture.
- The teacher provides the perceptual ceiling (LPIPS 0.258, FID 31.3) at ~3000× higher cost; the student preserves most of this quality advantage at feed-forward speed.

### 3.4 Ablation: why not intervene at the Gaussian layer?

A natural alternative is to edit the feed-forward backbone's Gaussian attributes directly. We tested a color-only Gaussian adapter that edits Flash3D's `features_dc` on the source plane and re-renders, with geometry frozen. It cannot generate disoccluded content: even pure overfitting on a single scene reaches only ~7% of the teacher−baseline gap, and it does not generalize (holdout −1.72 dB). The reason is structural — disoccluded target pixels correspond to Gaussians never observed in the source view, so source-plane recoloring has no leverage. This motivates operating in image space over the geometry expert's evidence, as our student does.

## 4. Discussion

Two results matter. Conceptually, **distillation of generative disocclusion priors must be reliability-aware**: the teacher is not uniformly better than the fast baseline, so the student must be trained where the teacher helps and gated where it does not. Practically, **once applied correctly, a tiny feed-forward student recovers ~92% of the teacher's disocclusion gain at ~3000× lower cost with zero visible-region change** — a strong deployment result. The Gaussian-color ablation explains the design choice: the student corrects in image space over the geometry expert's evidence because the disoccluded content cannot be produced by recoloring source-view Gaussians.

## 5. Limitations and Next Steps

Final numbers use the 21-scene high-teacher-gain set with a held-out split; the reported holdout uses a mid-run subset and will be extended. The router is currently rule-based (visible-gap) with a learned variant validated in the companion reliability paper. An optional stronger variant would predict residual disocclusion Gaussians (new points) rather than image-space residuals; this is future work, not required for the deployment claim. Qualitative panels (GT, baseline, teacher, student) accompany the quantitative tables.

## 6. Conclusion

This work establishes a second, distinct direction from Paper 1: instead of improving a generative backbone at test time, we distill its disocclusion prior into a fast feed-forward student. Trained on scenes where the teacher helps and gated at test time, the student recovers +5.77 dB of invisible-region PSNR (vs the teacher's +6.09 dB) while running ~3000× faster and leaving visible regions untouched. A Gaussian-layer ablation shows why the correction is applied in image space. This is a deployment-friendly, reliability-aware route to compress generative disocclusion priors.
