# Auditing Disocclusion Reliability Evidence for Single-View Reconstruction

## Abstract

We audit whether a slow generative teacher's usefulness can be predicted from inference-time evidence. The released GT-quality-probe diagnostic reaches ROC-AUC 0.947, trapezoidal PR-AUC 0.961, and +1.604 dB at 41% calls saved, but its PSNR inputs require GT. On the same 2,898 frames and 74 scene-grouped folds, the camera-only observable model reaches accuracy 0.612, ROC-AUC 0.650, trapezoidal PR-AUC 0.663, and +0.715 dB while retaining a -23.094 dB worst case. E-223 combines RGB disagreement with `visibility.npy`; because VGGT derives that mask from the complete target clip, this is an RGB disagreement + oracle-visibility diagnostic, not single-image observable evidence. On 910 frames from 21 selected high-gain scenes it obtains ROC-AUC 0.426, average precision 0.929, and calls every frame at threshold 0.5. The audit supports diagnostic separability but no deployment claim.

## 1. Audit Scope

Teacher gain is defined by GT-evaluated invisible-region PSNR. We separate four tiers: oracle true gain, GT-quality-probe diagnostic, camera-only observable, and RGB disagreement + oracle-visibility diagnostic.

## 2. Column Correction

Historical frame `vis_gap` is exactly:

```text
vis_gap = f3d_vis - base_inv
```

It is not `f3d_vis - base_vis`. Here `f3d_vis`, `f3d_inv`, and `base_inv` are PSNR quantities computed against GT; derived ratios and products retain this dependence.

## 3. Four Evidence Tiers

### 3.1 Oracle True Gain

The oracle calls the teacher using true gain and reaches about +1.70 dB on the 2,898-frame population. It defines evaluation headroom only.

### 3.2 GT-Quality-Probe Diagnostic

E-145b combines GT-derived quality probes with camera features under grouped leave-one-scene-out CV. It obtains ROC-AUC 0.947, trapezoidal PR-AUC 0.961, and +1.604 dB while skipping 41% of calls. This establishes diagnostic frame separability, not inference-time routing.

### 3.3 Camera-Only Observable

E-224 admits only camera translation and rotation. On 2,898 frames from 74 scenes it reaches accuracy 0.612, ROC-AUC 0.650, trapezoidal PR-AUC 0.663, and +0.715 dB. Its threshold-0.5 policy calls 2,028 frames and retains the -23.094 dB worst case.

### 3.4 RGB Disagreement + Oracle-Visibility Diagnostic

E-223 uses baseline/Flash3D RGB disagreement, gradients, camera motion, and partitions defined by `visibility.npy`. VGGT computes visibility from the complete target clip, crossing the single-image inference boundary. Changing GT and teacher arrays while holding these features fixed proves only GT/teacher-array invariance; changing visibility changes predictor features.

E-223 uses 910 frames from exactly 21 scenes selected by complete-asset availability in the high-gain `teacher_npy` directory. It is selected, not population-aligned with E-145b/E-224. The exact IDs, retained frame counts, source identifier, and hashes are released in `results/E-223_selected_scene_manifest.json`.

## 4. Results and Metric Definitions

| Experiment | Population | Tier | Accuracy | ROC-AUC | Precision-recall metric | Threshold-0.5 policy |
|---|---|---|---:|---:|---|---|
| E-145b | 2,898 frames / 74 scenes | GT-quality-probe diagnostic | 0.861 | 0.947 | Trapezoidal PR-AUC 0.961 | +1.604 dB; 41% saved |
| E-224 | 2,898 frames / 74 scenes | Camera-only observable | 0.612 | 0.650 | Trapezoidal PR-AUC 0.663 | +0.715 dB; worst -23.094; 2,028 calls |
| E-223 | 910 frames / 21 selected scenes | RGB disagreement + oracle-visibility diagnostic | 0.965 | 0.426 | Average precision 0.929 | Calls all 910 frames |

E-223 computes average precision as mean precision at positive ranks. E-145b and E-224 integrate the precision-recall curve with the trapezoidal rule. E-223's 96.5% positive rate explains its high accuracy and average precision; ROC-AUC and the all-call decision expose weak ranking.

## 5. Granularity Boundary

E-208 reports ROC-AUC 0.724 on 13,831 patches from the same 21 selected scenes, while E-145b reports 0.947 on 2,898 frames from 74 scenes. The populations, units, features, prevalence, and gain definitions differ. This is a diagnostic boundary, not a controlled granularity leaderboard.

## 6. Interpretation

Outcome-adjacent GT quality probes show that teacher usefulness is structured. Camera motion alone gives modest full-population discrimination, and E-223 fails to rank its selected population despite access to oracle visibility. Stronger genuinely observable uncertainty signals, tested on the same unselected population, are required before operational routing can be claimed.

## 7. Reproduction

- GT-quality-probe diagnostics: `e040_reliability_net.py` and `e145b_frame_net_full.py`.
- Camera-only observable baseline: `e224_camera_only_observable.py`, self-contained from released JSON.
- RGB disagreement + oracle-visibility diagnostic: `e223_observable_frame_reliability.py`, requiring external rendered and VGGT visibility assets.
- Integrity tests: `test_e223_observable_frame_reliability.py`, `test_e224_camera_only_observable.py`, and `test_scientific_claims.py`.

See [`PROTOCOL.md`](../PROTOCOL.md) for exact `/tmp` commands and population caveats.
