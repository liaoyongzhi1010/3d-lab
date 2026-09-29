# TinyStudent Model Card

## Model And Format

TinyStudent is a four-layer image-space residual CNN (`8 -> 48 -> 48 -> 48 -> 3`,
`3x3` kernels) with 46,371 trainable parameters. `checkpoints/student.pt` is a
PyTorch `state_dict` only, not a serialized module; instantiate the architecture
in `scripts/check_checkpoint.py` or `scripts/e031b_student_fullnpy.py`, then call
`load_state_dict`. The release was validated with Python 3.9.6 and PyTorch 2.8.0.
Compatibility with other PyTorch versions is not guaranteed; `weights_only=True`
is used when supported.

## Inputs And Output

Input is a float tensor shaped `[N, 8, H, W]`, ordered exactly as:

1. Channels 0-2: Flash3D RGB evidence in `[0, 1]`.
2. Channels 3-5: Gen3R baseline RGB in `[0, 1]`.
3. Channel 6: binary visibility mask `M`, where `1` means target content visible from the source.
4. Channel 7: binary invisibility mask `1-M`.

Training resizes RGB to a square `--size` using bilinear interpolation and masks
using nearest-neighbor interpolation. Inputs originating as `[0, 255]` arrays are
scaled to `[0, 1]`. The network predicts a three-channel residual `r`; the reported
image is `M * baseline + (1-M) * clamp(f3d + r, 0, 1)`. Thus visible output pixels
are copied from the baseline exactly. The baseline is mandatory and ground truth
must never be substituted for it.

The visibility masks were derived offline using the full target clip. They are
oracle inputs, not outputs of this checkpoint or an online single-view estimator.

## Training Provenance

The teacher supplies target images. An oracle dataset-curation procedure computes
teacher-minus-baseline invisible-region PSNR against ground truth and retains scenes
above 3 dB; the student is trained and evaluated only in that selected regime.
Ground truth is also used as an invisible-region auxiliary loss. The reported run
uses `--base_mode f3d`, `--gate_aware`, and a 4-scene/161-frame holdout. The exact
scene IDs, frame-label JSON path and contents, random seed, optimizer state, source
dataset snapshot, and external repository revisions are not recorded in this
release. Released checkpoint evaluation is reproducible, but exact retraining is
partial rather than a self-contained one-command workflow. It requires external
teacher arrays, the frame-label JSON, and the original holdout manifest, which are
not included. See [`TRAINING_MANIFEST.md`](TRAINING_MANIFEST.md) for the known
settings, unknowns, and placeholder-only command template.

This is an oracle-curated, oracle-visibility upper-bound distillation experiment.
It has no deployable selector and is not end-to-end single-view inference.

## Evaluation And Runtime

The released E-211 JSON is a custom 21-scene/759-frame, 256x256 diagnostic using
VGG-LPIPS. It is not the official Flash3D evaluation protocol. Held-out
invisible-region gain is +5.77 dB versus +6.09 dB for the teacher, a 0.32 dB gap
that retains 94.7% (~95%) of teacher gain, with visible-region change +0.000 dB.
Student runtime is 0.087 s/scene on the reported GPU setup versus 247 s/scene for
the teacher (2833x); CPU runtime was reported as 11.3 s/scene. The GPU model,
CUDA/cuDNN versions, power mode, warm-up policy used for the headline measurement,
and teacher hardware were not recorded, so these timings are not portable.

## Checkpoint Integrity

- Path: `checkpoints/student.pt`
- Format: PyTorch tensor-keyed `state_dict`
- Parameters: `46,371`
- Synthetic input/output: `(1, 8, 64, 64)` to `(1, 3, 64, 64)`
- SHA256: `1aa66b23bf6d30fe24a55aa7966ff398238a8102ea89efb213b916a375af76f2`

Run `python3 scripts/check_checkpoint.py` from this directory.

## Dependencies And Acquisition

PyTorch, NumPy, and the external Flash3D/Gen3R pipelines are required for real
inputs. Flash3D and Gen3R code, weights, datasets, teacher arrays, baseline arrays,
and oracle visibility masks are not bundled. Their acquisition locations,
licenses, and exact versions were not recorded in this package; obtain them from
the respective upstream projects and verify their terms. See `README.md` for the
validated local package versions and commands.

## Intended Use And Limitations

Use the checkpoint to inspect the released architecture or reproduce inference
when all eight input channels have been generated compatibly. Do not present it as
a standalone reconstruction model, a deployable selector, or evidence of quality
on uncurated scenes. Its performance is conditioned on target-outcome-based scene
curation and full-clip oracle visibility. The Gaussian color adapter is only a
negative ablation and is not an alternative released model.

## License

Repository-authored code, documentation, and this checkpoint are released under
the repository's MIT License (`../LICENSE`). Third-party models, data, and code
remain subject to their own licenses.
