# Paper 3 LaTeX Package: Scientific Audit

The manuscript separates four tiers: oracle true gain, GT-quality-probe diagnostic, camera-only observable, and RGB disagreement + oracle-visibility diagnostic. E-223 uses VGGT visibility derived from the complete target clip and is not observable at single-image inference. No operational-routing or deployment claim is made.

## Build

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Evidence Provenance

- `../PROTOCOL.md` defines evidence tiers, populations, grouped-CV controls, and metric integration.
- `../scripts/e221_granularity_figure.py` creates the non-comparable-population boundary figure.
- `../scripts/e222_routing_strip.py` creates the GT-quality-probe diagnostic strip.
- `../results/E-224_camera_only_observable.json` is the full-population camera-only observable result.
- `../results/E-223_observable_frame_reliability.json` is the selected RGB disagreement + oracle-visibility diagnostic.
- `../results/E-223_selected_scene_manifest.json` records E-223's exact 21 selected scenes, source, frame counts, and hashes.

E-223 reports average precision; E-145b and E-224 report trapezoidal PR-AUC. Historical `vis_gap` is `f3d_vis - base_inv`, not a visible-vs-visible quantity; both terms are GT-dependent quality/evaluation probes.
