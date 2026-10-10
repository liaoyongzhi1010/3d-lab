# Paper 1 LaTeX Package

The paper is synchronized with `paper1_full_writeup.md`, `../README.md`, and `../PROTOCOL.md` as an offline oracle-visibility mechanism study, not a deployable single-view method.

## Build

Run from the repository root:

```bash
cd paper1_selective_generation/paper/figs
pdflatex -interaction=nonstopmode -halt-on-error fig_pipeline.tex
cd ..
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
cd ../../..
```

## Figures

- `fig_teaser_aligned.png`: three released GT/Gen3R/injected successes and one injected failure motivating a future fallback; no routing decision is claimed. The teaser is constrained to released GT/Gen3R/injected assets; missing input/Flash3D assets are a release limitation, so the approved strongest honest teaser does not fabricate replacements.
- `fig_pipeline.pdf`: target-clip oracle visibility, an exact latent bypass, and oracle fallback analysis; it does not claim exact decoded RGB.
- `fig_mechanism_combined.pdf`: mechanism scatter, difficulty buckets, and oracle threshold operating points from released JSONs.
- `fig_failures_limitations.pdf`: released injected failure plus generated temporal and ACID aggregate panels; no raw ACID images.
- `panel_success_boxed.png`: sequence-level GT/component/mask/error diagnostic.

Rebuild generated figures from the repository root:

```bash
python3 paper1_selective_generation/scripts/e219_topvenue_teaser.py
python3 paper1_selective_generation/scripts/e220_protocol_table.py
python3 paper1_selective_generation/scripts/e221_failure_limitations.py
```

## Protocol Scope

The 759-frame result is a fixed offline oracle-visibility component diagnostic, not official Flash3D/MINE evaluation or deployable source-only inference. VGGT consumed the complete target clip to build `visibility.npy`. Published numbers appear only in a separate positioning table with protocol metadata and no cross-protocol winner. See `../PROTOCOL.md`.

## Template Note

The checked-in source uses a standalone two-column draft preamble. To use the official CVPR/ICCV template, add its style files, enable `\usepackage[review]{cvpr}`, restore the template fonts, and use `ieee_fullname`.
