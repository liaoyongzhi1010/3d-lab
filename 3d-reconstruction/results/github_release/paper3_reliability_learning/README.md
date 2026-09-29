# Auditing Disocclusion Reliability Evidence

Paper 3 audits whether teacher usefulness in single-view reconstruction can be predicted from information available at inference. GT-derived quality probes separate favorable frames well, but only the camera-only E-224 predictors are genuinely observable in this release. E-223 uses `visibility.npy`, which VGGT derives from the complete target clip, so it crosses the single-image inference boundary. No deployment or operational-routing claim is made.

## Four Evidence Tiers

| Tier | Population | Inputs | ROC-AUC | Precision-recall metric | Policy at 0.5 |
|---|---|---|---:|---|---|
| Oracle true gain | 2,898 frames / 74 scenes | GT-evaluated true teacher gain | -- | -- | +1.70 dB upper bound |
| GT-quality-probe diagnostic | 2,898 frames / 74 scenes | GT-derived PSNR probes + camera | 0.947 | Trapezoidal PR-AUC 0.961 | +1.604 dB, 41% calls saved |
| Camera-only observable | 2,898 frames / 74 scenes | Translation + rotation | 0.650 | Trapezoidal PR-AUC 0.663 | +0.715 dB, worst -23.094 dB, 2,028 calls |
| RGB disagreement + oracle-visibility diagnostic | 910 frames / 21 selected high-gain scenes | RGB, VGGT complete-clip visibility, camera | 0.426 | Average precision 0.929* | Calls all 910 frames |

`*` E-223 is 96.5% positive, so average precision is dominated by prevalence. It is selected, not population-aligned, and is not comparable to the full-population rows.

The 0.947 result is a diagnostic upper bound, not deployable performance. In the released frame JSON, `f3d_vis`, `f3d_inv`, and related PSNR quantities are GT-derived quality probes. Historical `vis_gap` is exactly `f3d_vis - base_inv`; its name does not denote a visible-vs-visible gap.

## Layout

```text
paper/      Audit manuscript, Markdown version, and derived figures.
scripts/    Diagnostic and camera-only observable experiments plus integrity tests.
results/    Released evidence, including the E-223 selected-scene manifest.
PROTOCOL.md Evidence tiers, populations, leakage controls, and commands.
```

## Reproduce

Run from this directory. Commands write generated outputs to `/tmp`, not tracked release paths.

```bash
python3 scripts/e040_reliability_net.py --data results/E-142_combined_N169.json --out /tmp/e040.json
python3 scripts/e145b_frame_net_full.py --data results/E-149_frames_b5678_aug.json --out /tmp/e145b.json --ext
python3 scripts/e224_camera_only_observable.py --out /tmp/e224.json
python3 -m unittest discover -s scripts -p 'test_*.py'
python3 scripts/e221_granularity_figure.py --out /tmp/fig_granularity.pdf
python3 scripts/e222_routing_strip.py --out /tmp/fig_routing_strip.pdf
```

E-224 is self-contained from released JSON. A full E-223 rerun requires external baseline, teacher, GT, Flash3D, VGGT `visibility.npy`, and camera assets; see `PROTOCOL.md`. Its released aggregate and exact selected population are `results/E-223_observable_frame_reliability.json` and `results/E-223_selected_scene_manifest.json`.

## Interpretation

The GT-quality-probe experiment establishes strong diagnostic frame structure. Camera-only evidence is moderately discriminative but retains the full worst-case loss. E-223 ranks below chance on its selected population and defaults to calling every frame. The release does not support an operational scheduler.
