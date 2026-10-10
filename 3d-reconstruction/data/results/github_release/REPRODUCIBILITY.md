# Reproducibility

Start with the [protocol declaration](PROTOCOLS.md): official MINE/Flash3D
single-view, pixelSplat/DepthSplat two-view, the released 49-frame diagnostic,
and the ACID mechanism diagnostic are separate evaluation families.

## Smoke Reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 scripts/smoke_test.py
```

`requirements-analysis.txt` pins the exact direct-package versions reported by
the tested analysis environment. It covers the released CPU analyses, figures,
metadata checks, and smoke test; it is not a full transitive lock file. External
Gen3R/Flash3D/VGGT CUDA dependencies remain documented and installed separately
from their upstream repositories because those stacks are not claimed to be
universally pin-compatible.

The smoke test loads the 13 released headline JSONs, explicitly including E-209,
E-216, E-223, and E-224; asserts selected headline keys with numeric tolerances;
compares both checked-in manifests byte-for-byte with deterministic regeneration,
including scene split assignments and per-scene frame counts; parses CFF metadata;
verifies local links and images in every repository `README.md`; and checks, loads,
and runs TinyStudent as `[1,8,64,64] -> [1,3,64,64]`. It is a release-integrity
check, not fresh-render or complete paper-test coverage. It finishes with
`SMOKE PASS`.

## Level 1: Released Metrics and Figures

This level is self-contained. Released JSONs reproduce aggregate numbers,
confidence intervals, component ablations, oracle fallback sweeps, routing audits,
and analysis plots without private data or external model weights. Paper 1/2 use
visibility computed from the full target clip and are not deployable single-image
systems. E-145b uses GT-derived quality probes. E-223 combines RGB/camera features
with the same oracle visibility on 910 frames from 21 selected high-gain scenes.
Only E-224 is genuinely observable; its camera-only result on 2,898 frames is unsafe.

```bash
# Paper 1: scene mechanism, component ablation, and gate sweep
python3 paper1_selective_generation/scripts/e203_bootstrap_ci.py \
  --data paper1_selective_generation/results/E-142_combined_N169.json \
  --out /tmp/e203_bootstrap.json
python3 paper1_selective_generation/scripts/e207_baseline_ablation.py \
  --data paper1_selective_generation/results/E-142_combined_N169.json \
  --out /tmp/e207_ablation.json
python3 paper1_selective_generation/scripts/e216_gate_sensitivity.py \
  --data paper1_selective_generation/results/E-142_combined_N169.json \
  --out /tmp/e216_gate_sweep.json

# Paper 3: quality-probe upper bounds and observable camera-only audit
python3 paper3_reliability_learning/scripts/e040_reliability_net.py \
  --data paper3_reliability_learning/results/E-142_combined_N169.json \
  --out /tmp/scene_reliability.json
python3 paper3_reliability_learning/scripts/e145b_frame_net_full.py \
  --data paper3_reliability_learning/results/E-149_frames_b5678_aug.json \
  --out /tmp/frame_reliability.json --ext
python3 paper3_reliability_learning/scripts/e224_camera_only_observable.py \
  --data paper3_reliability_learning/results/E-149_frames_b5678_aug.json \
  --out /tmp/e224_camera_only.json
```

See [`TABLES_AND_FIGURES.md`](TABLES_AND_FIGURES.md) for the complete mapping
from scientific content to scripts, released inputs, outputs, and external-render
requirements.

## Level 2: End-to-End Rendering

Fresh frame generation requires upstream projects, licensed data, and their
checkpoints. Their CUDA-specific dependency stacks are governed by the upstream
projects and are intentionally not merged into the pinned analysis environment.
These large third-party assets are not redistributed:

- Gen3R repository and checkpoint for the generative backbone/teacher.
- Flash3D repository and checkpoint for geometry evidence.
- VGGT repository/checkpoint for visibility preprocessing.
- RealEstate10K and/or ACID frames and camera metadata under their licenses.

Core scripts:

- `paper1_selective_generation/scripts/precompute_visibility.py`
- `paper1_selective_generation/scripts/render_gen3r_scene_evidence.py`
- `paper1_selective_generation/scripts/e012_vesg.py`
- `paper1_selective_generation/scripts/e212_acid_to_gen3r.py`

Typical sequence:

```text
scene images + camera metadata
  -> precompute_visibility.py
  -> render_gen3r_scene_evidence.py
  -> e012_vesg.py (baseline + selective injection, optionally save arrays)
  -> e210_standard_metrics.py / e215_acid_metrics.py
```

Paper 1's external repository roots are configured with environment variables,
not uniformly with CLI root arguments:

```bash
export GEN3R_ROOT=/absolute/path/to/Gen3R
export VGGT_ROOT=/absolute/path/to/vggt
export FLASH3D_ROOT=/absolute/path/to/flash3d
export PYTHONPATH="$GEN3R_ROOT:$VGGT_ROOT:$FLASH3D_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

Dataset, checkpoint, scene, and output locations that have CLI flags should still
be passed explicitly. Some scripts preserve original experiment path defaults for
traceability, so do not infer that every path can be supplied by a root argument.
Missing external assets must not be replaced with another checkpoint, metric
backbone, crop, or protocol.

## TinyStudent Checkpoint

The included checkpoint is
[`paper2_gate_aware_distillation/checkpoints/student.pt`](paper2_gate_aware_distillation/checkpoints/student.pt).
Its SHA256 is
`1aa66b23bf6d30fe24a55aa7966ff398238a8102ea89efb213b916a375af76f2`.
Architecture, parameter count, input format, and external dependencies are in
[`MODEL_ZOO.md`](MODEL_ZOO.md).

## Manifests and Evaluation Conventions

- Run `python3 scripts/generate_manifests.py` to regenerate both manifests, or add
  `--check` to compare their canonical content without writing.
- [`re10k_scene_manifest.json`](manifests/re10k_scene_manifest.json) is generated
  from the released Paper 1 scene rows and contains 166 sorted IDs with source
  splits.
- [`frame_reliability_manifest.json`](manifests/frame_reliability_manifest.json)
  is generated from the released Paper 3 frame rows and contains the source-split
  assignment and frame count for each of 74 scenes, totaling 2,898 frames.
- Scene-level splits avoid train/test leakage.
- Frame reliability uses grouped leave-one-scene-out evaluation; frames from one
  scene do not cross train/evaluation groups.
- LPIPS uses VGG with images scaled to `[-1,1]` in the metric scripts.
- Region-attributed LPIPS uses GT substitution rather than zero masking.
- Paper 1/2 visibility is computed from the full target clip, so neither result is
  deployable single-image inference. Paper 1's quality-gap fallback and Paper 3's
  E-145b quality probes additionally consume GT-derived quantities.
- E-223 uses rendered RGB/camera inputs and full-target-clip oracle visibility on
  only 21 selected high-gain scenes; it ranks below chance and calls every frame
  at threshold 0.5. It is not a genuinely observable result.
- E-224 is the only genuinely observable audit. Its camera-only model on all 2,898
  frames is moderately discriminative and retains the full -23.094 dB worst loss.
- The source JSONs do not contain all upstream checkpoint versions or seeds;
  manifests mark missing metadata as unavailable rather than guessing it.

## Protocol Caveats

- The released long-sequence component diagnostic starts from one context image
  and disocclusion-heavy 49-frame trajectories, but Paper 1/2 visibility uses the
  full target clip. It is not the official MINE protocol or deployable single-image inference.
- pixelSplat/MVSplat/DepthSplat published results use two context views and three
  interpolated targets; they are positioning references only.
- ACID results are Paper 1 cross-dataset mechanism/difficulty evidence from
  converted trajectories, not an official ACID leaderboard result.
- The released ACID metrics contain eight evaluated mechanism scenes and 215
  frames. No broader count is inferred from unreleased conversion attempts.
