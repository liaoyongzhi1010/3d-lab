# TinyStudent Training Manifest

## Reproducibility Status

Released checkpoint evaluation is reproducible with `checkpoints/student.pt` and
`scripts/check_checkpoint.py`. Exact retraining is not self-contained and cannot
be reproduced from this repository alone. It requires external teacher arrays,
the frame-label JSON, and the original holdout manifest, which are not included.
The compatible Flash3D arrays, Gen3R baseline and ground-truth arrays, RealEstate10K
data with offline full-target-clip visibility masks, and exact upstream revisions
are also not included.

## Reported Settings

The released paper narrative and E-031b implementation establish these settings.
Only `--base_mode f3d`, frame-label weighting, and the holdout size are identified
as properties of the reported model; remaining numeric values are current E-031b
script settings and are not recoverable from the state dict:

- Entry point: `scripts/e031b_student_fullnpy.py`.
- Disocclusion basis: `--base_mode f3d`.
- Frame weighting: `--gate_aware`, using labels loaded from `--frame_dataset`.
- Holdout: 4 scenes and 161 frames from the oracle-selected high-teacher-gain regime.
- Input size: 256x256.
- Optimizer: Adam with learning rate `1e-3`.
- Training duration: 300 epochs.
- Loss weights: `--w_gt 0.2` and `--w_id 2.0`.
- Architecture: the 46,371-parameter TinyStudent described in `MODEL_CARD.md`.

The optimizer, epoch, size, and loss-weight values above are the E-031b script
settings. The checkpoint contains only model tensors, so it cannot independently
prove the complete invocation.

## Unreleased Information

The following values needed to reconstruct the reported run are unknown in this
release:

- The exact four holdout scene IDs and their original holdout manifest.
- The exact `--frame_dataset` path and frame-label JSON contents.
- The random seed, optimizer state, and sample-order state.
- Exact snapshots or revisions for RealEstate10K, teacher generation, Flash3D,
  Gen3R, and visibility preprocessing.
- The original paths and byte-level contents of all training arrays.

The two scene IDs defaulted by the older `e030_student_distill.py` MVP and scene
names shown by the E-150 adapter example do not identify the reported E-031b
four-scene holdout and must not be substituted for it.

## Training Template

The command below is a template, not a self-contained exact reproduction command.
Replace every angle-bracketed placeholder with the corresponding external artifact;
`<FOUR_COMMA_SEPARATED_SCENE_IDS>` must contain the unreleased original four IDs.

```bash
python3 scripts/e031b_student_fullnpy.py \
  --teacher_dir <TEACHER_ARRAY_DIR> \
  --f3d_dir <FLASH3D_ARRAY_DIR> \
  --data <RE10K_ROOT_WITH_VISIBILITY_ARRAYS> \
  --frame_dataset <FRAME_LABEL_JSON> \
  --holdout <FOUR_COMMA_SEPARATED_SCENE_IDS> \
  --out <OUTPUT_DIR> \
  --base_mode f3d \
  --gate_aware \
  --epochs 300 \
  --lr 1e-3 \
  --size 256 \
  --w_gt 0.2 \
  --w_id 2.0
```

This template documents known arguments only. It does not recover the missing
artifacts or establish bit-exact retraining.

## Top-Level Command Correction

`TABLES_AND_FIGURES.md` is outside this scoped directory. Its Paper 2
TinyStudent-training row should point readers to this manifest rather than present
a self-contained command. Correct replacement text for its command cell:

`See paper2_gate_aware_distillation/TRAINING_MANIFEST.md for the non-self-contained training template; exact retraining requires unreleased external artifacts and holdout metadata.`
