# Paper 3 LaTeX package — Learning Disocclusion Reliability

Self-contained draft that compiles with a base TeX install.

## Build
```
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
Produces `main.pdf` (4 pages, all tables/figures + bibliography, zero undefined refs).

## Files
- `main.tex` — full paper (scene/frame/patch granularity study, operating frontier, mechanism).
- `refs.bib` — shared BibTeX.
- `figs/` — `fig4_frontier.pdf`, `fig5_roc_pr.pdf`, `fig6_saved.pdf`.

## Switching to the official CVPR/ICCV template
1. Drop the official `cvpr.sty` next to `main.tex`.
2. Replace the standalone preamble with the template's `\documentclass` + `\usepackage[review]{cvpr}`.
3. Re-enable `\usepackage{times}`; remove `\renewcommand{\ttdefault}{cmtt}`.
4. Set `\bibliographystyle{ieee_fullname}`.

## Still to add before camera-ready
- A teaser: a frame with a predicted-reliability heatmap.
- Update N when scene/frame expansion continues (currently N=166 scenes / 2,898 frames).
