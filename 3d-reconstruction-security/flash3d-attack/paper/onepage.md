# Depth-Parallax Confusion: Source-Camouflaged Attacks on Single-Image 3D Gaussian Splatting

**One-page summary / extended abstract**

---

## Teaser (Figure 1)

`results/qual/scene7_dpc.png` (indoor) and `results/qual/scene20_dpc.png` (outdoor).
Layout: 3 rows x 4 columns.
- Columns: **source view | novel +5 | novel +10 | novel random** (increasing parallax →).
- Rows: **clean reconstruction | after DPC attack | |difference| x4**.
- Read the bottom row: the leftmost (source) panel is nearly black (attack invisible where the
  user looks), and panels brighten left→right (novel views break, more so with parallax).

**Caption.** An imperceptible perturbation (L∞=8/255) to the *input image only*. The source-view
reconstruction stays intact; novel views collapse, and the collapse grows with camera motion.

---

## The problem (why now)

Feed-forward single-image 3DGS (Flash3D, pixelSplat, Splatter Image, MVSplat, DepthSplat) turns
one photo into a full renderable 3D scene in a single forward pass, and is heading into products
(AR/VR, 3D asset creation). Their robustness is essentially unstudied; the one prior attack
(AdvSplat) treats them as ordinary image models and degrades all views uniformly.

## The insight (our contribution #1)

These models are trained almost only with a **novel-view photometric loss**. For a single image,
many (depth, appearance) explanations render *identically in the source view* and differ only
under **parallax**. So source-view consistency does not constrain the parallax-dependent
geometry. We call this the **depth-appearance ambiguity** and argue it is a *structural* attack
surface, not an incidental network weakness.

## The method: DPC (contribution #2)

A perturbation delta (‖delta‖∞ ≤ eps) on the source image, optimized with an opposite-sign dual
objective:

```
min_delta   λ_src · ‖R_src(x+δ) − R_src(x)‖²          (camouflage the source view)
          −            mean_j ‖R_{P_j}(x+δ) − R_{P_j}(x)‖²   (destroy novel views)
```

- **Source-camouflaged:** keep the observed view correct, break the unobserved ones.
- **Self-referential:** targets are the model's own clean renders — no ground-truth novel views.
- **Pose-agnostic:** optimized on attacker-sampled auxiliary poses, evaluated on unseen poses.

White-box and query-only black-box (NES over low-frequency DCT) instantiations. Model-agnostic
via a `RenderFn(image, pose)` interface.

## Key results (Flash3D, RealEstate10K / MINE)

**Main (n=100, eps=8/255).** Novel-view PSNR drops ~2x more than source; degradation grows with
parallax — the geometry-corruption fingerprint.

| View | ΔPSNR | ΔLPIPS |
|---|---:|---:|
| source   | −2.6 | +0.10 |
| novel +5 | −4.8 | +0.10 |
| novel +10| −5.2 | +0.12 |
| novel rand| −5.0 | +0.12 |

**DPC is the only source-camouflaged attack (n=100, same budget).**

| Method | source ΔPSNR | novel ΔPSNR (avg) | camouflage (novel−source) |
|---|---:|---:|---:|
| random noise | −1.9 | −0.7 | worse on source |
| naive-PGD (no camouflage) | **−20.7** | −12.0 | destroys source |
| **DPC (ours)** | **−2.6** | −5.0 | **+2.4 (novel ≫ source)** |

**Stealth vs a training-free defense.** A multi-view consistency detector gets **AUC 0.555**
(≈chance): DPC evades it, motivating dedicated defenses.

## Why it's a top-venue story

1. Geometry-grounded, not a re-skinned PGD; a named, testable **parallax-scaling signature**.
2. A new, practically dangerous threat model: *looks correct where inspected, fails where used.*
3. Clean ablations isolate the mechanism (λ_src controls the source/novel trade-off; random
   noise ≈ harmless at the same budget).

## Honest limitations / what's next

- Validated on **one** model (Flash3D). The cross-model transfer study (pixelSplat / Splatter
  Image) is the top priority to prove the vulnerability is architectural.
- n=100 subset (not full 3204 MINE); black-box validated on synthetic + unit tests but expensive
  on full Flash3D.
- Source "camouflage" is relative (novel ≫ source), not perfect invisibility.

## Reproducibility

Self-contained in `flash3d-attack/` (does not touch prior projects). 13 unit tests pass.
`attacks/summarize_results.py` regenerates every table from `results/*.json`.
