# Tables and Figures Reproduction Index

Run commands from the repository root. Outputs under `/tmp` avoid changing the
checkout. Every entry below is a complete executable command. Commands marked
"Required" need licensed datasets, upstream repositories/checkpoints, or render
arrays that are not distributed in this release; set these paths first:

```bash
export GEN3R_ROOT=/absolute/path/to/Gen3R
export VGGT_ROOT=/absolute/path/to/vggt
export FLASH3D_ROOT=/absolute/path/to/flash3d
export RE10K_ROOT=/absolute/path/to/re10k
export ACID_SCENE_ROOT=/absolute/path/to/converted_acid
export TEACHER_DIR=/absolute/path/to/teacher_npy
export EVIDENCE_DIR=/absolute/path/to/flash3d_evidence
export ACID_RESULTS=/absolute/path/to/acid_e012_vesg.json
export FRAME_PROBE=/absolute/path/to/frame_probe.json
export GEN3R_FRAME_RESULTS=/absolute/path/to/gen3r_per_frame.json
export FRAME_LABELS=/absolute/path/to/training_frame_labels.json
export HOLDOUT_SCENES=comma_separated_holdout_scene_ids
export RELEASE_SCENE_ID=scene_id_with_required_prefix
```

Paper table/figure numbering may evolve, so entries are indexed by scientific
content rather than an unstable number alone.

## Paper 1: Selective Generation

| Evidence | Exact command | Output | External renders |
|---|---|---|---|
| Mechanism correlation, confidence intervals, difficulty buckets | `python3 paper1_selective_generation/scripts/e203_bootstrap_ci.py --data paper1_selective_generation/results/E-142_combined_N169.json --out /tmp/e203.json` | `/tmp/e203.json` and stdout | No |
| Component and oracle-fallback ablation table | `python3 paper1_selective_generation/scripts/e207_baseline_ablation.py --data paper1_selective_generation/results/E-142_combined_N169.json --out /tmp/e207.json` | `/tmp/e207.json` | No; `gap>5` uses a GT quality-gap oracle |
| Oracle quality-gap threshold sweep | `python3 paper1_selective_generation/scripts/e216_gate_sensitivity.py --data paper1_selective_generation/results/E-142_combined_N169.json --out /tmp/e216.json` | `/tmp/e216.json` | No; proposed future fallback analysis, not an inference gate |
| Full-image PSNR table | `python3 paper1_selective_generation/scripts/e209_fullimage_psnr.py --data paper1_selective_generation/results/E-142_combined_N169.json --out /tmp/e209.json` | `/tmp/e209.json` | No |
| VGGT visibility precompute | `GEN3R_ROOT="$GEN3R_ROOT" VGGT_ROOT="$VGGT_ROOT" PYTHONPATH="$GEN3R_ROOT:$VGGT_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 paper1_selective_generation/scripts/precompute_visibility.py --data "$RE10K_ROOT"` | `visibility.npy` under each scene | Required; VGGT consumes the full target clip |
| E-012 selective-generation diagnostic | `GEN3R_ROOT="$GEN3R_ROOT" VGGT_ROOT="$VGGT_ROOT" PYTHONPATH="$GEN3R_ROOT:$VGGT_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 paper1_selective_generation/scripts/e012_vesg.py --ckpt "$GEN3R_ROOT/checkpoints" --data "$RE10K_ROOT" --f3d_dir "$EVIDENCE_DIR" --scene_names "$RELEASE_SCENE_ID" --out /tmp/e012_vesg` | Results under `/tmp/e012_vesg` | Required |
| LPIPS-VGG / FID / KID table | `python3 paper1_selective_generation/scripts/e210_standard_metrics.py --data_root "$RE10K_ROOT" --tdir "$TEACHER_DIR" --edir "$EVIDENCE_DIR" --out /tmp/e210.json` | `/tmp/e210.json` | Required |
| ACID mechanism/metric diagnostic | `python3 paper1_selective_generation/scripts/e215_acid_metrics.py --results "$ACID_RESULTS" --tdir "$TEACHER_DIR" --edir "$EVIDENCE_DIR" --scene_root "$ACID_SCENE_ROOT" --out /tmp/e215.json` | `/tmp/e215.json` | Required |
| Teaser from released qualitative panels (E-219) | `python3 paper1_selective_generation/scripts/e219_topvenue_teaser.py --fig-dir paper1_selective_generation/paper/figs --out /tmp/e219_teaser.png` | `/tmp/e219_teaser.png` | No; source panels are released, fresh panels require renders |
| Combined mechanism/protocol figure (E-220) | `python3 paper1_selective_generation/scripts/e220_protocol_table.py --data paper1_selective_generation/results/E-142_combined_N169.json --gate paper1_selective_generation/results/E-216_gate_sensitivity.json --out /tmp/e220_protocol.pdf` | `/tmp/e220_protocol.pdf` | No; gate panel is a GT-derived oracle diagnostic |
| Inspectable qualitative ranking | `python3 paper1_selective_generation/scripts/e218_select_obvious_quals.py --teacher_dir "$TEACHER_DIR" --data "$RE10K_ROOT" --out /tmp/e218_quals` | Ranking JSON and panels under `/tmp/e218_quals` | Required |
| Per-scene qualitative panel | `python3 paper1_selective_generation/scripts/e201_qual_panel.py --teacher_dir "$TEACHER_DIR" --sid "$RELEASE_SCENE_ID" --data "$RE10K_ROOT" --out /tmp/e201_panel.png` | `/tmp/e201_panel.png` | Required |

## Paper 2: Distillation

| Evidence | Exact command | Output | External renders |
|---|---|---|---|
| TinyStudent checkpoint validation (exact released path) | `python3 paper2_gate_aware_distillation/scripts/check_checkpoint.py` | Shape, parameter count, SHA256 | No |
| Student full-image evaluation (exact released path with external renders) | `python3 paper2_gate_aware_distillation/scripts/e211_student_standard_metrics.py --data_root "$RE10K_ROOT" --tdir "$TEACHER_DIR" --edir "$EVIDENCE_DIR" --student paper2_gate_aware_distillation/checkpoints/student.pt --out /tmp/e211.json` | `/tmp/e211.json` | Required; evaluated in an oracle-curated high-teacher-gain regime, with no inference gate claim |
| Quality-speed figure from released E-211 metrics | `python3 paper2_gate_aware_distillation/scripts/generate_quality_speed.py --metrics paper2_gate_aware_distillation/results/E-211_student_standard_metrics.json --out /tmp/fig_quality_speed.pdf` | `/tmp/fig_quality_speed.pdf` | No |
| Runtime benchmark | `python3 paper2_gate_aware_distillation/scripts/e202_runtime_bench.py --frames 48 --size 256 --teacher_sec 247 --reps 50` | Runtime stdout | No data; hardware dependent; benchmarks a freshly initialized architecture |
| TinyStudent retraining template (not exact reproduction) | `python3 paper2_gate_aware_distillation/scripts/e031b_student_fullnpy.py --teacher_dir "$TEACHER_DIR" --f3d_dir "$EVIDENCE_DIR" --data "$RE10K_ROOT" --out /tmp/e031b_student --base_mode f3d --gate_aware --frame_dataset "$FRAME_LABELS" --holdout "$HOLDOUT_SCENES"` | Checkpoint and results under `/tmp/e031b_student` | Exact retraining unavailable: the experiment's holdout and frame-label manifests are absent from the release; this template requires equivalent user-supplied manifests |
| Gaussian color-adapter negative ablation | `RELEASE_ROOT="$PWD"; cd "$FLASH3D_ROOT" && FLASH3D_ROOT="$FLASH3D_ROOT" PYTHONPATH="$FLASH3D_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 "$RELEASE_ROOT/paper2_gate_aware_distillation/scripts/e150_gaussian_adapter.py" --cfg "$FLASH3D_ROOT/eval_official/.hydra/config.yaml" --data_root "$RE10K_ROOT" --scenes "$RELEASE_SCENE_ID" --teacher_dir "$TEACHER_DIR" --out /tmp/e150_adapter` | Checkpoint and `train_log.json` under `/tmp/e150_adapter` | Required, including the upstream Flash3D environment, model, config, and CUDA setup |

## Paper 3: Reliability Learning

| Evidence | Exact command | Output | External renders |
|---|---|---|---|
| Scene-level grouped reliability table | `python3 paper3_reliability_learning/scripts/e040_reliability_net.py --data paper3_reliability_learning/results/E-142_combined_N169.json --out /tmp/scene_reliability.json` | `/tmp/scene_reliability.json` | No; oracle-quality diagnostic |
| Frame-level ROC/PR and routing frontier | `python3 paper3_reliability_learning/scripts/e145b_frame_net_full.py --data paper3_reliability_learning/results/E-149_frames_b5678_aug.json --out /tmp/frame_reliability.json --ext` | `/tmp/frame_reliability.json` and stdout | No; GT-derived oracle-quality diagnostic, not deployable |
| Granularity figure (E-221) | `python3 paper3_reliability_learning/scripts/e221_granularity_figure.py --out /tmp/e221_granularity.pdf` | `/tmp/e221_granularity.pdf` | No; the script reads released result paths internally; GT-derived oracle-quality diagnostic |
| Routing strip (E-222) | `python3 paper3_reliability_learning/scripts/e222_routing_strip.py --data paper3_reliability_learning/results/E-149_frames_b5678_aug.json --out /tmp/e222_routing_strip.pdf` | `/tmp/e222_routing_strip.pdf` | No; GT-derived oracle-quality diagnostic, not image-level evidence |
| RGB/camera plus oracle-visibility analysis (E-223) | `python3 paper3_reliability_learning/scripts/e223_observable_frame_reliability.py --teacher-dir "$TEACHER_DIR" --evidence-dir "$EVIDENCE_DIR" --data-root "$RE10K_ROOT" --out /tmp/e223_rgb_oracle_visibility.json` | `/tmp/e223_rgb_oracle_visibility.json` | Required; full-target-clip visibility makes this non-observable; selected-population negative result |
| Genuinely observable camera-only audit (E-224) | `python3 paper3_reliability_learning/scripts/e224_camera_only_observable.py --data paper3_reliability_learning/results/E-149_frames_b5678_aug.json --out /tmp/e224_camera_only.json` | `/tmp/e224_camera_only.json` | No; full-population negative result |
| Pixel/patch reliability diagnostic | `python3 paper3_reliability_learning/scripts/e208_pixel_reliability.py --data_root "$RE10K_ROOT" --teacher_dir "$TEACHER_DIR" --evid_dir "$EVIDENCE_DIR" --out /tmp/e208.json` | `/tmp/e208.json` | Required |
| Frame dataset assembly | `python3 paper3_reliability_learning/scripts/e144_build_frame_dataset.py --gen3r "$GEN3R_FRAME_RESULTS" --probe "$FRAME_PROBE" --out /tmp/e144_frames.json` | `/tmp/e144_frames.json` | Required |
| Frame feature augmentation | `python3 paper3_reliability_learning/scripts/e149_augment_features.py --frames /tmp/e144_frames.json --data "$RE10K_ROOT" --out /tmp/e149_frames.json` | `/tmp/e149_frames.json` | Required |

## Standalone Pipeline Figures

Each command compiles the corresponding released TikZ source without writing
build artifacts into the checkout. Because `-output-directory` changes only where
TeX writes artifacts, each PDF is named after its source file inside the stated
directory:

```bash
mkdir -p /tmp/paper1_pipeline && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/paper1_pipeline paper1_selective_generation/paper/figs/fig_pipeline.tex
mkdir -p /tmp/paper2_pipeline && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/paper2_pipeline paper2_gate_aware_distillation/paper/figs/fig_pipeline.tex
mkdir -p /tmp/paper3_pipeline && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/paper3_pipeline paper3_reliability_learning/paper/figs/fig_pipeline.tex
```

The resulting files are exactly `/tmp/paper1_pipeline/fig_pipeline.pdf`,
`/tmp/paper2_pipeline/fig_pipeline.pdf`, and
`/tmp/paper3_pipeline/fig_pipeline.pdf`; no PDF is written beside the source.
Other new final figures are generated by
the E-219, E-220, quality-speed, E-221, and E-222 rows above.

## Release Checks

```bash
python3 scripts/generate_manifests.py --check
python3 scripts/smoke_test.py
python3 -m json.tool manifests/re10k_scene_manifest.json >/dev/null
python3 -m json.tool manifests/frame_reliability_manifest.json >/dev/null
```

Released aggregate analyses can be reproduced without private assets where the
last column says "No." Fresh rendering and rows marked "Required" depend on the
specified upstream checkpoints, render arrays, and licensed RealEstate10K/ACID
data; no command silently substitutes those inputs.
