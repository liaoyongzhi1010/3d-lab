# Selective Disocclusion Prior Distillation for Feed-Forward Single-View 3D Reconstruction

**Central thesis:** a tiny feed-forward student can retain useful behavior from a
non-monotonic teacher within an oracle-selected high-teacher-gain regime. Scene
selection uses teacher and ground-truth outcomes for dataset curation and is an
upper-bound condition, not a deployable inference mechanism. Visibility masks are
also oracle inputs computed offline from the full target clip. Within the evaluated
output, they copy baseline pixels and enforce exact visible-region no-harm; this is
not deployable single-view inference.

> Paper 2 of a three-paper single-view 3D reconstruction project.

## Key Results

- Held-out invisible-region gain: **+5.77 dB** (teacher: **+6.09 dB**; **0.32 dB** gap), retaining **94.7% (~95%)** of the teacher gain with visible change **+0.000 dB**.
- Runtime: **0.087 s/scene** on GPU versus **247 s/scene**, a **2833x speedup**.
- Fixed custom diagnostic: 21 scenes, **759** disocclusion-heavy frames, 256x256, VGG-LPIPS. The student improves over the Flash3D geometry/evidence component on LPIPS (**0.284** versus **0.290**) and FID (**33.6** versus **72.9**) at similar cost, while trailing the teacher (**0.258**, **31.3**).
- Negative ablation: a source-plane Gaussian color adapter with frozen geometry recovers only **~7%** of the gap when overfit and gives **-1.72 dB** on holdout.

The 759-frame result is a same-diagnostic component analysis, not an official
Flash3D-protocol result or a cross-protocol benchmark win.

| Method | PSNR | SSIM | LPIPS-VGG | FID | Time/scene |
|---|---:|---:|---:|---:|---:|
| Gen3R baseline | 17.88 | 0.595 | 0.294 | 31.6 | 247 s |
| Flash3D evidence | 19.98 | 0.664 | 0.290 | 72.9 | 0.09 s |
| Teacher (slow) | 19.37 | 0.633 | 0.258 | 31.3 | 247 s |
| **Student** | 18.17 | 0.591 | 0.284 | 33.6 | **0.087 s** |

## Installation

Tested for release validation with Python 3.9.6, PyTorch 2.8.0, NumPy 2.0.2,
Pillow 11.3.0, and Matplotlib 3.9.4. Create an isolated environment and install
the lightweight validation dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install "torch==2.8.0" "numpy==2.0.2" "Pillow==11.3.0" "matplotlib==3.9.4"
```

The full E-211 metric script additionally requires `lpips`, `scikit-image`,
`torchmetrics[image]`, and `torch-fidelity`:

```bash
python3 -m pip install lpips scikit-image "torchmetrics[image]" torch-fidelity
```

Paper generation requires `pdflatex`, `bibtex`, TikZ/PGF, and the LaTeX
`geometry`, `graphicx`, `amsmath`, `amssymb`, `booktabs`, `multirow`, `xcolor`,
`hyperref`, `caption`, and `float` packages. The validated build used TeX Live
2026. External Flash3D and Gen3R repositories/checkpoints and the dataset are not
bundled; their exact source revisions were not recorded, so metric regeneration
requires supplying compatible artifacts explicitly.

## Checkpoint

The included TinyStudent has **46,371 parameters**. See
[`MODEL_CARD.md`](MODEL_CARD.md), then verify its architecture, synthetic output,
and SHA256:

```bash
python3 scripts/check_checkpoint.py
```

Expected output includes:

```text
Parameters: 46,371
Output shape: (1, 3, 64, 64)
SHA256: 1aa66b23bf6d30fe24a55aa7966ff398238a8102ea89efb213b916a375af76f2
CHECKPOINT PASS
```

## Validate And Build

Run from `paper2_gate_aware_distillation/`:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 scripts/check_checkpoint.py
python3 scripts/check_claim_integrity.py
python3 -m py_compile scripts/*.py tests/*.py
python3 scripts/generate_quality_speed.py
(cd paper/figs && pdflatex -interaction=nonstopmode -halt-on-error fig_pipeline.tex)
(cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex && bibtex main && pdflatex -interaction=nonstopmode -halt-on-error main.tex && pdflatex -interaction=nonstopmode -halt-on-error main.tex)
```

Released checkpoint evaluation is reproducible with the bundled state dict and
validation scripts. Exact retraining is only partially documented and is not a
self-contained one-command workflow: external arrays, the frame-label JSON, and
the original four-scene holdout manifest are not included. See
[`TRAINING_MANIFEST.md`](TRAINING_MANIFEST.md) for known settings, explicit
unknowns, and a placeholder-only training template using the reported
`--base_mode f3d` and `--gate_aware` options.

## Repository Layout

- `checkpoints/student.pt`: released TinyStudent state dict.
- `results/E-211_student_standard_metrics.json`: source for the fixed custom diagnostic and quality-speed vector figure.
- `paper/`: LaTeX paper, synchronized Markdown write-up, and vector figures.
- `scripts/e030_student_distill.py`, `e031b_student_fullnpy.py`: student training.
- `scripts/e150_gaussian_adapter.py`: color-only Gaussian-adapter negative ablation.
- `scripts/e202_runtime_bench.py`: runtime measurement.
- `scripts/e211_student_standard_metrics.py`: full-image diagnostic metrics.
- `scripts/generate_quality_speed.py`: regenerates the quality-speed scatter from the E-211 JSON.
- `scripts/check_checkpoint.py`: released-checkpoint integrity and shape check.
- `scripts/check_claim_integrity.py`: multiline claim scan plus E-211/checkpoint consistency checks.
- `tests/test_release_integrity.py`: missing-baseline and claim-integrity regressions.

## Evidence Boundaries

The reported student result is conditional on oracle dataset curation using
teacher-minus-baseline invisible-region quality measured against ground truth and
on visibility masks derived offline from the full target clip. No observable
selector or online visibility estimator for unseen scenes is established. The
result is an upper-bound compression experiment, not deployable single-view
inference.

The Gaussian schematic explains existing E-150 evidence only: recoloring existing
source-plane Gaussians with frozen geometry does not create support for newly
disoccluded target pixels. It is not a render comparison and does not claim that
all Gaussian-space adapters fail. External Flash3D, Gen3R, and visibility models
are not included.
