# Paper 3 Scientific Audit Protocol

## Audit Question

Do released results demonstrate inference-time prediction of teacher usefulness, or diagnostic separability using information beyond single-image inference? Claims are assigned to four evidence tiers.

## Four Evidence Tiers

1. **Oracle true gain.** Labels and true gains use teacher and baseline invisible-region PSNR against GT. This is an evaluation-only upper bound.
2. **GT-quality-probe diagnostic.** E-040 and E-145b use PSNR columns computed against GT. E-145b reports ROC-AUC 0.947, trapezoidal PR-AUC 0.961, +1.604 dB, and 41% calls saved. These are diagnostic, not deployment results.
3. **Camera-only observable.** E-224 uses only target-camera translation and rotation on all 2,898 frames. It reports ROC-AUC 0.650 and trapezoidal PR-AUC 0.663.
4. **RGB disagreement + oracle-visibility diagnostic.** E-223 combines baseline/Flash3D RGB disagreement and cameras with `visibility.npy`. VGGT derives this mask from the complete target clip, so it is not available at single-image inference. E-223 reports ROC-AUC 0.426 and average precision 0.929 on a selected population. It is never an observable or deployment result.

## Released Column Semantics

- `base_inv`, `teacher_inv`, and `delta` are GT-evaluated outcomes.
- `f3d_vis` and `f3d_inv` are GT-derived PSNR quality probes.
- Historical `vis_gap` is exactly `f3d_vis - base_inv`. It is not `f3d_vis - base_vis`, and it is not observable.
- `f3d_vi_ratio` and `f3d_vi_prod` inherit dependence on quality probes.
- `cam_trans` and `cam_rot_deg` are the only E-149 columns admitted by E-224's observable whitelist.

## Populations and Metrics

| Experiment | Population | Positive rate | Evidence | Result |
|---|---|---:|---|---|
| E-145b | 2,898 frames, 74 scenes | 55.5% | GT quality probes + camera | ROC 0.947; trapezoidal PR-AUC 0.961; mean +1.604 dB |
| E-224 | Same 2,898 frames, 74 scenes | 55.5% | Camera only | Accuracy 0.612; ROC 0.650; trapezoidal PR-AUC 0.663; mean +0.715 dB; worst -23.094 dB; 2,028 calls |
| E-223 | 910 frames, 21 selected high-gain scenes | 96.5% | RGB disagreement + oracle visibility + camera | ROC 0.426; average precision 0.929; threshold 0.5 calls all frames |
| E-208 | 13,831 patches, same 21 selected scenes | 71.1% | RGB disagreement + oracle visibility | ROC 0.724 |

E-223 computes average precision as mean precision at positive ranks. E-145b and E-224 compute trapezoidal PR-AUC. These metrics are not placed in one unlabeled PR-AUC column. Populations, unit granularities, prevalence, available artifacts, and some gain definitions also differ, so the rows are not a controlled leaderboard.

## E-223 Selected Population

The exact 21 scene IDs are in `results/E-223_selected_scene_manifest.json`, generated from the released E-223 `scene_frame_counts`. The source population is the set of scene IDs with complete required assets discovered from `gt_*.npy` in the available high-gain directory `/home/data/E-161_gen3r_highgain/teacher_npy`. This is selected, not population-aligned with E-145b/E-224. The manifest records the source identifier, each scene's retained frame count, a canonical scene-ID hash, a canonical scene-count hash, and the hash of the released result before this metadata correction.

## Grouped Cross-Validation

E-145b, E-224, E-223, and E-208 hold out complete scenes. Scores and policy statistics are concatenated only after every unit from the held-out scene has been excluded from fitting. The E-223 feature-invariance test changes GT and teacher arrays while holding RGB, visibility, and cameras fixed; it proves only that predictors do not directly consume GT/teacher arrays. A separate test changes visibility and verifies that predictor features change, detecting the inference-boundary crossing.

## Reproduction

Commands write generated outputs to `/tmp`.

```bash
python3 scripts/e040_reliability_net.py --data results/E-142_combined_N169.json --out /tmp/e040.json
python3 scripts/e145b_frame_net_full.py --data results/E-149_frames_b5678_aug.json --out /tmp/e145b.json --ext
python3 scripts/e224_camera_only_observable.py --out /tmp/e224.json
python3 -m unittest discover -s scripts -p 'test_*.py'
python3 scripts/e221_granularity_figure.py --out /tmp/fig_granularity.pdf
python3 scripts/e222_routing_strip.py --out /tmp/fig_routing_strip.pdf
```

E-224 is self-contained from `results/E-149_frames_b5678_aug.json`. E-223 additionally requires external baseline, teacher, GT, Flash3D, VGGT `visibility.npy`, and camera arrays under the paths accepted by `scripts/e223_observable_frame_reliability.py`:

```bash
python3 scripts/e223_observable_frame_reliability.py --teacher-dir /external/E-161_gen3r_highgain/teacher_npy --evidence-dir /external/E-160_highgain_evidence --data-root /external/gen3r_re10k/re10k --out /tmp/e223.json
```

Expected E-224 checks: N=2,898; 74 scenes; accuracy 0.6118; ROC-AUC 0.6496; trapezoidal PR-AUC 0.6631; mean +0.7152 dB; worst -23.0944 dB; 2,028 calls; 870 calls saved (30.0%). Expected E-223 released checks: N=910; 21 selected scenes; ROC-AUC 0.4255; average precision 0.9290; all 910 frames called at threshold 0.5.

## Figure Policy

The pipeline uses four boxes and places E-223 on the diagnostic side of the inference boundary. The granularity figure reports ROC-AUC only and explicitly marks differing populations. The routing strip is an E-145b GT-quality-probe diagnostic, not inference-time behavior.
