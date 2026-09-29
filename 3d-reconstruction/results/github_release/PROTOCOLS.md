# Protocol Declaration

The repository contains results from four protocol families. They answer different
questions and must not be compared as one leaderboard.

| Protocol family | Context views | Split/index | Resolution and crop | Target rule | LPIPS | Comparison status |
|---|---:|---|---|---|---|---|
| Official MINE / Flash3D single-view | 1 | 3,204 samples from 641 scenes | 256x384, 5% border crop | +5, +10, and U[-30,30] buckets reported separately | VGG | Published values are quoted for positioning only; no official reproduction is claimed. |
| pixelSplat / DepthSplat two-view | 2 | `evaluation_index_re10k.json` | 256x256, no border crop | 3 interpolated targets | VGG | Published values are quoted for positioning only; they are not compared as head-to-head wins against single-view results. |
| Released long-sequence disocclusion diagnostic | 1 source image, but oracle access to the full target clip for visibility in Paper 1/2 and E-223 | 166-scene released manifest; full-image metrics cover 759 rendered frames from 21 scenes | Source frames are aspect-preservingly resized so the short side is 560 pixels, then center-cropped to 560x560. E-210 evaluates the native saved 560x560 arrays with no additional crop or resize; E-211 bilinearly resizes RGB arrays to 256x256 and nearest-neighbor resizes visibility masks to 256x256. | 49-frame sequences selected for significant disocclusion, with custom visibility and region analyses | VGG | Oracle-visibility component/distillation studies and GT-derived upper bounds; E-223 also uses oracle visibility. Only camera-only E-224 is genuinely observable. None establishes deployable routing. |
| ACID mechanism diagnostic | 1 | Converted ACID test trajectories; released metrics contain 8 evaluated mechanism scenes and 215 frames | E-212 copies native JPEGs and records their native width/height and denormalized intrinsics; it does not resize or crop. The shared E-012 renderer then aspect-preservingly resizes the short side to 560 pixels and center-crops to 560x560. E-215 evaluates those saved arrays without another resize or crop. Native source dimensions are read per scene but their values are not retained in released result JSON; ACID radial-distortion coefficients are ignored during conversion. | Up to the first 49 timestamp-matched frames per converted trajectory; the exact released scene/frame selection beyond the 8-scene/215-frame aggregate is unknown because IDs and conversion logs are absent from E-215. | VGG | Cross-dataset mechanism/difficulty diagnostic for Paper 1 only, not an official ACID leaderboard claim. |

## Interpretation Rules

- A number is comparable only when views, split/index, resolution, crop, target
  rule, metric backbone, checkpoint/version, and test-time optimization status
  match.
- Flash3D and Gen3R are components in the released long-sequence diagnostic.
  Lower LPIPS/FID there does not establish a win over published Flash3D under the
  official MINE protocol.
- pixelSplat, MVSplat, and DepthSplat use a two-view interpolation protocol. Their
  published values provide context, not a direct ranking against these
  single-view diagnostics.
- ACID evidence is limited to Paper 1's mechanism/difficulty transfer analysis.
  Papers 2 and 3 do not claim ACID validation.
- LPIPS means LPIPS-VGG throughout released standard metrics. Scripts must not
  silently substitute another backbone.
- Paper 1 uses visibility computed from the full 49-frame target clip. That oracle
  visibility is unavailable in deployable single-image inference; its `gap>5`
  fallback additionally uses GT quality gaps and is not a test-time inference gate.
- Paper 2 uses the same full-target-clip oracle visibility and a regime curated from
  teacher/GT outcomes. It is not a deployable single-image system or unseen-scene selector.
- Paper 3 E-145b uses GT-derived quality probes and is an oracle-quality upper bound.
  E-223 uses RGB/camera inputs plus full-target-clip oracle visibility on a selected
  910-frame population, so it is not genuinely observable. Only E-224 uses genuinely
  observable camera transforms on all 2,898 frames; its unsafe result is not routing evidence.

## Released Diagnostic Metadata

| Result | Input views | Source | Resolution/crop | Targets | Status | Test-time optimization |
|---|---:|---|---|---|---|---|
| Paper 1 scene mechanism and gate analyses | 1 source image plus full-target-clip oracle visibility | [`E-142_combined_N169.json`](paper1_selective_generation/results/E-142_combined_N169.json), [`re10k_scene_manifest.json`](manifests/re10k_scene_manifest.json) | 560x560 working frames from short-side resize plus center crop; oracle region visibility is computed/saved at 70x70 and resized as needed | Per-scene aggregation over long sequences | Oracle diagnostic; not deployable single-image inference | None in released analysis |
| Paper 1 full-image metrics | 1 | [`E-210_standard_metrics.json`](paper1_selective_generation/results/E-210_standard_metrics.json) | Native saved 560x560 RGB arrays; no additional E-210 crop/resize and no official border-crop equivalence claimed | 759 rendered frames / 21 scenes | Reproduced component study | None in released analysis |
| Paper 2 student metrics | 1 source image plus full-target-clip oracle visibility | [`E-211_student_standard_metrics.json`](paper2_gate_aware_distillation/results/E-211_student_standard_metrics.json) | Every GT/component RGB frame is bilinearly resized from 560x560 to 256x256; oracle visibility is nearest-neighbor resized to 256x256; no additional crop and no official MINE equivalence claimed | Same 759-frame custom diagnostic after teacher/GT-based oracle curation | Conditional component study; included checkpoint; not deployable single-image inference | Feed-forward student; none |
| Paper 3 quality-probe routing | 1 | [`E-145b_N2898_ext.json`](paper3_reliability_learning/results/E-145b_N2898_ext.json), [`E-149_frames_b5678_aug.json`](paper3_reliability_learning/results/E-149_frames_b5678_aug.json) | GT-derived quality probes from the custom diagnostic | 2,898 frames / 74 scenes | Grouped leave-one-scene-out upper bound; not observable | Diagnostic call/skip simulation only |
| Paper 3 camera-only audit | 1 | [`E-224_camera_only_observable.json`](paper3_reliability_learning/results/E-224_camera_only_observable.json), [`frame_reliability_manifest.json`](manifests/frame_reliability_manifest.json) | Observable translation and rotation; GT teacher gain only defines labels/evaluation | 2,898 frames / 74 scenes | Grouped leave-one-scene-out negative result; ROC-AUC 0.650 and worst routed gain -23.094 dB | Diagnostic call/skip simulation only |
| Paper 3 RGB/oracle-visibility analysis | 1 source image plus full-target-clip oracle visibility | [`E-223_observable_frame_reliability.json`](paper3_reliability_learning/results/E-223_observable_frame_reliability.json) | Baseline/evidence RGB and camera plus oracle visibility; GT teacher gain defines labels/evaluation | 910 frames / 21 selected high-gain scenes | Not genuinely observable; selected-population negative result; ROC-AUC 0.426 and all 910 calls at threshold 0.5 | Diagnostic call/skip simulation only |
| Paper 1 ACID transfer | 1 | [`E-215_acid_metrics.json`](paper1_selective_generation/results/E-215_acid_metrics.json) | Native JPEGs are copied by E-212 without image resampling; E-012 then applies short-side-to-560 resize plus a 560x560 center crop, and E-215 evaluates the saved arrays without further spatial preprocessing. Native dimensions and exact crop offsets are unknown from released JSON; radial distortion is ignored. | 215 frames / 8 evaluated mechanism scenes; exact IDs unknown from released aggregate | Reproduced mechanism diagnostic | None in released analysis |

The upstream checkpoint versions used to render the released JSONs are not
embedded in those JSONs. This limits exact fresh-render reproduction and is
reported rather than guessed. See [`MODEL_ZOO.md`](MODEL_ZOO.md) and
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
