# Learning Disocclusion Reliability for Single-View 3D Reconstruction

## Abstract

Single-view 3D reconstruction systems fail unevenly: some disoccluded regions can be reconstructed by a feed-forward model, while others require generative completion. This paper studies the reliability prediction problem: before invoking a slow generative teacher, can we predict whether it will help — and where? We formulate disocclusion reliability learning as predicting teacher usefulness from test-time observable features. We show that at the *scene* level the problem is data-limited: a simple visible-gap rule is already a strong policy and a learned scene-level ReliabilityNet only matches it (N=166 scenes: rule +0.87 dB vs learned +1.13 dB, both below oracle +1.70 dB). The key result is that moving to the *frame* level unlocks learning: with 2,898 frame-level samples a logistic frame ReliabilityNet reaches **0.87 accuracy, ROC-AUC 0.95, PR-AUC 0.96**, and **+1.60 dB mean invisible-PSNR gain — 94% of the oracle's +1.70 dB — clearly beating the visible-gap rule (+0.89 dB) while saving 41% of teacher calls**. A tunable risk-averse operating point drives the worst case from always-inject's −23.1 dB toward 0 while remaining net-positive. Across both granularities the governing mechanism is stable: the correlation between the Flash3D-vs-baseline invisible-region gap and the actual teacher improvement is +0.98. This establishes disocclusion reliability as a learnable scheduling problem and delivers a working frame-level predictor.

## 1. Motivation

Paper 1 shows how to inject geometry into a generative reconstruction model. Paper 2 shows why distillation into a fast student must be gated. Both depend on the same question: **where and when should the system trust a generative prior?** This paper isolates that question.

The central hypothesis is that disocclusion reliability can be learned from observable signals such as visible-region quality, disocclusion ratio, and per-frame reconstruction consistency. A good reliability predictor should approach the oracle's mean gain while letting the operator dial down worst-case failures and avoid unnecessary slow teacher calls.

## 2. Problem Definition

Given a single input image, a target camera trajectory, and fast reconstruction outputs, predict whether invoking the generative teacher will improve disoccluded regions — at scene or frame granularity.

Label (both granularities):

```text
y = 1 if teacher_invisible_PSNR - baseline_invisible_PSNR > 0.1 dB
```

Observable features:

```text
f3d_vis     Flash3D visible-region PSNR (proxy vs reprojected input at test time)
f3d_inv     Flash3D invisible-region PSNR
vis_frac    fraction of visible pixels
vis_gap     f3d_vis - baseline_vis  (relative visible reliability)
```

## 3. ReliabilityNet

We train a logistic ReliabilityNet on the observable features. Scene-level uses leave-one-scene-out CV. Frame-level uses **grouped leave-one-scene-out CV** (all frames of a held-out scene are excluded from training) so no frame leaks across the split — the honest generalization protocol.

## 4. Results

### 4.1 Scene-level: the problem is data-limited (N=166 scenes)

| Policy | Mean Δ | Worst | Inject |
|---|---:|---:|---:|
| Always teacher | +0.75 | −11.72 | 166/166 |
| Rule `vis_gap > 0` | +0.87 | −8.51 | 162/166 |
| Scene ReliabilityNet (LOOCV) | +1.13 | −6.80 | 114/166 |
| Oracle | **+1.70** | 0.00 | 98/166 |

At the scene level the learned model only slightly edges the rule and both remain below oracle: scene-level features saturate. (The learned net's small margin over the rule does grow monotonically with data: parity at N=92, +0.11 dB at N=129, +0.19 dB at N=149, +0.26 dB at N=166.)

### 4.2 Frame-level: learning unlocks the gain (2,898 frames, 74 scenes)

| Policy | Mean Δ | Worst | Acc | ROC-AUC | PR-AUC | Saved |
|---|---:|---:|---:|---:|---:|---:|
| Always teacher | +0.48 | −23.09 | — | — | — | 0% |
| Rule `vis_gap > 0` | +0.89 | −10.63 | — | — | — | 14% |
| Frame ReliabilityNet (base feats) | +1.60 | −2.61 | **0.867** | **0.949** | 0.958 | 41% |
| **Frame ReliabilityNet (+cam/motion feats)** | +1.60 | −2.61 | 0.861 | 0.947 | **0.961** | 41% |
| Oracle | +1.70 | 0.00 | — | — | — | 43% |

The frame-level ReliabilityNet **beats the rule by a large margin (+1.60 vs +0.89) and reaches 94% of the oracle gain while saving 41% of teacher calls**, at ROC-AUC 0.95 / PR-AUC 0.96. Two data-scaling effects confirm the thesis: (i) the learned model's advantage over the rule grows with data; (ii) extended camera-motion / disocclusion-ratio features help most in the 1,443–2,231 frame regime (they lift PR-AUC to 0.96). Notably, as the pool grows to 2,898 frames the newly added scenes have a lower average teacher gain (always-inject drops to +0.48 dB), yet the learned FrameNet still recovers +1.60 dB — i.e. the harder the pool, the more the learned gate matters relative to always-injecting. A tunable threshold traces a full mean-vs-worst frontier (Fig. operating-frontier), letting an operator pick maximum quality or near-zero-risk operation.

### 4.2.1 Comparison with standard routing baselines (matched compute budget)

Reliability routing is an instance of expert deferral / model cascades. We therefore compare against the standard routing baselines used in that literature, all matched to the same ~59% teacher-call budget (i.e. 41% compute saved):

| Routing policy | Mean invisible Δ | Note |
|---|---:|---|
| Random routing @59% | +0.264 | call teacher on a random 59% |
| Always call teacher | +0.481 | full compute (0% saved) |
| Visible-gap top-K @59% | +1.082 | rank by observable `vis_gap` |
| **FrameNet (ours) @59%** | **+1.60** | learned reliability |
| Oracle @ budget | +1.703 | upper bound |

At the *same* compute budget, our learned scheduler (+1.60 dB) beats random routing (+0.26), the full always-call policy (+0.48), and the strongest observable-feature ranking rule (+1.08), reaching 94% of the oracle. This is the standard cascade-comparison result: better quality at equal cost, or equal quality at lower cost.

### 4.3 Stable mechanism at both granularities

The correlation between the observable-ish invisible gap `f3d_inv − base_inv` and the actual teacher improvement is **+0.982 (scene, N=166)** and **+0.978 (frame, N=2,898)**. The mechanism that governs when generative disocclusion helps is the same at both scales and is highly predictable.

### 4.4 Patch-level: the granularity has a lower bound (13,831 patches, 21 scenes)

We tested whether reliability can be predicted *below* the frame, at 8×8 patch granularity, using only test-time-observable patch features (Flash3D-vs-baseline L1 disagreement and its variance, baseline gradient/texture, disocclusion fraction, patch position, Flash3D activity). Labels mark whether the teacher reduces MSE-to-GT in the patch's disoccluded pixels. Under grouped leave-one-scene-out CV on the 21 high-gain scenes:

| Granularity | Acc | ROC-AUC | PR-AUC | Net gain | Oracle |
|---|---:|---:|---:|---:|---:|
| Frame (2,898 samples) | 0.87 | **0.949** | 0.961 | **+1.60** (94% oracle) | +1.70 |
| Patch (13,831 samples) | 0.73 | **0.724** | 0.843 | +3.20 vs always +3.05 | +4.15 |

Patch-level reliability is **substantially harder to predict** (ROC-AUC 0.72 vs 0.95 at frame level) and the learned patch gate barely improves over always-injecting (+3.20 vs +3.05 dB, far from the +4.15 dB oracle). The reason is structural: on high-gain scenes 71% of disoccluded patches are already teacher-favorable, so the within-frame reliability signal is weak and the spatial selection has little headroom. This is an **honest negative boundary result**: it shows the frame is the sweet spot for this scheduling problem — coarse enough to have a strong, learnable, test-time-observable signal, fine enough to capture the per-view variation that the scene level misses. Pixel/patch-level selection would need richer per-pixel evidence (e.g. depth uncertainty) to pay off.

## 5. Discussion

The scientific arc is: (i) disocclusion reliability is a real supervised problem; (ii) at the scene level it is data-limited and a visible-gap rule is a strong interpretable baseline; (iii) descending to frame level provides the sample size that lets a learned predictor beat the rule and approach the oracle, while exposing a tunable mean-vs-worst-case frontier that a fixed rule cannot offer; (iv) descending further to patch level, prediction becomes hard (ROC-AUC 0.72) and the gain over always-inject nearly vanishes, so the frame is the sweet spot. This frontier is the practical contribution: an operator can choose maximum quality (+1.60 dB) or near-zero-risk (worst-case −0.28 dB at a conservative threshold).

## 6. Next Steps Toward Submission

1. Expand frame-level data across more batches (2,898 frames / 74 scenes so far; target several thousand frames).
2. Add depth-uncertainty, opacity/coverage features (camera-motion features already added and shown to help at scale).
3. Report full PR/AUC curves and the complete mean-vs-worst operating frontier.
4. Pixel/patch-level ReliabilityNet explored (§4.4): predictable signal is weak below the frame; deferred pending richer per-pixel evidence.
5. Integrate the predictor into Paper 1 and Paper 2 as a learned scheduler.

## 7. Current Conclusion

Paper 3 delivers a working frame-level disocclusion ReliabilityNet that beats the rule baseline, approaches the oracle (94%), and provides a tunable risk frontier — established under an honest grouped-CV protocol with a stable +0.98 mechanism correlation. Remaining work is scaling the frame dataset and adding features/curves for a full submission.

## Appendix: Reproduction

- Scene-level: `e142_build_combined.py` → `e040_reliability_net.py` / `e041_reliability_policy.py` on `E-142_combined_N169.json` (N=166).
- Frame-level: per-frame logging in Gen3R runner (`e012_vesg.py` `score()`), `e143_probe_perframe.py`, `e144_build_frame_dataset.py`, `e148_merge_frames.py`, `e149_augment_features.py`, `e145b_frame_net_full.py` on `E-149_frames_b5678_aug.json` (2,898 frames, 74 scenes).
- Patch-level: `e208_pixel_reliability.py` on high-gain teacher/evidence npy (`E-161_gen3r_highgain/teacher_npy`, `E-160_highgain_evidence`), 13,831 patches / 21 scenes → `E-208_pixel_reliability.json`.
