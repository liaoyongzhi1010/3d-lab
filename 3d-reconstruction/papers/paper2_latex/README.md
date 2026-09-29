# Paper 2 LaTeX package — Gate-aware Disocclusion Prior Distillation

Self-contained draft that compiles with a base TeX install.

## Build
```
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
Produces `main.pdf` (3 pages, all tables + bibliography, zero undefined refs).

## Files
- `main.tex` — full paper (distillation table, runtime table, Gaussian-adapter negative-result ablation).
- `refs.bib` — shared BibTeX.
- `figs/` — (currently table-only; add teacher/student qualitative panels here when rendered).

## Switching to the official CVPR/ICCV template
1. Drop the official `cvpr.sty` next to `main.tex`.
2. Replace the standalone preamble with the template's `\documentclass` + `\usepackage[review]{cvpr}`.
3. Re-enable `\usepackage{times}`; remove `\renewcommand{\ttdefault}{cmtt}`.
4. Set `\bibliographystyle{ieee_fullname}`.

## Still to add before camera-ready
- Qualitative panels (GT / baseline / teacher / student) as a figure.
- Extend the held-out split beyond the mid-run 4-scene/161-frame subset.
