# Paper 1 LaTeX package — Selective Geometry-Guided Generation

Self-contained draft that compiles with a base TeX install.

## Build
```
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
Produces `main.pdf` (5 pages, all figures + bibliography, zero undefined refs).

## Files
- `main.tex` — full paper (abstract, method, algorithm, all tables/figures, limitations).
- `refs.bib` — BibTeX for all cited work.
- `figs/` — figure PDFs (`fig0_teaser`, `fig1_mechanism`, `fig2_difficulty`, `fig3_gate`),
  the TikZ pipeline (`fig_pipeline.tex` → `fig_pipeline.pdf`), and qualitative panels
  (`panel_success.png`, `panel_failure.png`).

## Rebuilding the pipeline figure
```
cd figs && pdflatex fig_pipeline.tex   # produces fig_pipeline.pdf (single page)
```
Source of truth is `docs/figs/paper1_pipeline.drawio`; the TikZ version reproduces it self-contained so it compiles anywhere.

## Switching to the official CVPR/ICCV template
1. Drop the official `cvpr.sty` (+ `eso-pic`, `cvpr_eso.sty`) next to `main.tex`.
2. Replace the standalone preamble with `\documentclass[10pt,twocolumn,letterpaper]{article}` + `\usepackage[review]{cvpr}` per the template.
3. Re-enable `\usepackage{times}` and remove the `\renewcommand{\ttdefault}{cmtt}` line (draft-only font workaround for minimal TeX installs).
4. Set `\bibliographystyle{ieee_fullname}` (ships with the template) instead of `plain`.

## Still to add before camera-ready
- ROC/PR and compute-saving figures (`fig5_roc_pr.pdf`, `fig6_saved.pdf`) if the frame-level reliability discussion is expanded.
- Update N when batch7+ scene expansion completes (currently N=129).
