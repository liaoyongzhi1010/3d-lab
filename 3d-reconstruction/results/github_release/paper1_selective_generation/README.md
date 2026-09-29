# Oracle-Visibility Selective Injection: A Single-View Reconstruction Diagnostic

**Question:** Under oracle visibility, when does feed-forward geometry evidence improve a generative reconstruction, and what does masked latent injection reveal?

This is an offline mechanism study, not deployable single-view inference. `visibility.npy` is produced by running VGGT on the complete target clip, which crosses the single-image inference boundary. Geometry is injected only into oracle-invisible latents; target-GT selection is a separate oracle analysis.

## Three Claims

1. Concept: oracle-visibility selective injection as a mechanism diagnostic, not deployable single-view inference.
2. Mechanism: an exact visible latent bypass, with decoded RGB behavior measured rather than guaranteed.
3. Evidence: a scale-stable quality-gap mechanism and perceptual component gains on a fixed long-sequence diagnostic, bounded by failed zero-shot ACID transfer.

## Protocol First

Read [`PROTOCOL.md`](PROTOCOL.md) before comparing numbers. The reproduced studies are custom diagnostics:

- **Fixed offline oracle-visibility diagnostic:** 166 valid RealEstate10K scenes, one designated source frame and 48 targets. VGGT processes all frames to produce the masks.
- **Fixed 759-frame component diagnostic:** 21 disocclusion-heavy long sequences evaluated with full-image PSNR/SSIM/LPIPS-VGG/FID/KID. This is **not** official Flash3D/MINE evaluation or source-only inference.
- **ACID zero-shot diagnostic:** the released E-215 aggregate has 8 scenes, -0.443 dB always-inject mean, and 3/8 wins; this is failed transfer, not an official leaderboard result.
- Published Flash3D, pixelSplat, MVSplat, and DepthSplat numbers are quoted positioning only and remain separated by protocol family.

## Headline Evidence

| Claim-level check | Released result |
|---|---:|
| Evidence-gap / injection-benefit correlation | **r = 0.982**, N=166 |
| Hard / mid / easy always-inject benefit | **+2.49 / +0.11 / -2.52 dB** |
| Default quality-gap oracle (`tau=5`) | **+1.658 dB** mean, -0.393 dB worst |
| E-210 always injected candidate | PSNR 17.77 to 19.22; LPIPS-VGG 0.360 to 0.341; injected output evaluated directly, no oracle fallback |
| Visible path / RGB check | latent-path exact; measured visible change **+0.016 dB** on the direct 16-scene run |
| Large-scale visible value | 14.48 baseline/selected by construction assumption; not an independent RGB measurement |
| ACID E-215 zero-shot aggregate | **8 scenes**; always mean **-0.443 dB**; **3/8 wins** |

## Visual Evidence

`paper/figs/fig_teaser_aligned.png` combines three released obvious successes and one injected failure. The failure demonstrates why a future fallback is needed; it is not a routing decision. Yellow boxes and zooms use the released fixed ROIs. The selected assets contain GT, Gen3R, and injected candidates, but no matching raw input or Flash3D images; the figure labels this limitation and does not invent replacements.

`paper/figs/fig_mechanism_combined.pdf` combines the released mechanism scatter, difficulty buckets, and oracle threshold operating points. `paper/figs/fig_failures_limitations.pdf` includes the target-clip visibility leak alongside the injected failure, temporal result, and ACID aggregate. `paper/figs/fig_pipeline.pdf` labels visibility and fallback as offline oracle inputs.

## Reproduce

```bash
python3 scripts/e203_bootstrap_ci.py \
  --data results/E-142_combined_N169.json --out /tmp/e203.json
python3 scripts/e216_gate_sensitivity.py \
  --data results/E-142_combined_N169.json --out /tmp/e216.json
python3 scripts/e219_topvenue_teaser.py
python3 scripts/e220_protocol_table.py
python3 scripts/e221_failure_limitations.py
```

Expected unchanged checks: `r=0.982` and default quality-gap oracle mean `+1.658 dB`.

### Upstream model setup

Install [Gen3R](https://github.com/JaceyHuang/Gen3R) and [VGGT](https://github.com/facebookresearch/vggt) using their official instructions. GPU entry points accept either explicit checkout paths:

```bash
python3 scripts/precompute_visibility.py \
  --gen3r_root /path/to/Gen3R --vggt_root /path/to/vggt \
  --data /path/to/re10k
python3 scripts/e012_vesg.py \
  --gen3r_root /path/to/Gen3R --vggt_root /path/to/vggt \
  --ckpt /path/to/Gen3R/checkpoints --data /path/to/re10k --out /path/to/output
```

or environment variables:

```bash
export GEN3R_ROOT=/path/to/Gen3R
export VGGT_ROOT=/path/to/vggt
python3 scripts/precompute_visibility.py --data /path/to/re10k
python3 scripts/e012_vesg.py --data /path/to/re10k --out /path/to/output
```

CLI flags override the environment. Without either, the scripts retain the legacy defaults `/root/projects/Gen3R` and `/root/projects/vggt`; `e012_vesg.py` defaults `--ckpt` to `<gen3r_root>/checkpoints`. Both scripts support `--help` before upstream dependencies are installed and report missing roots or imports with official setup links.

## Build

```bash
cd paper/figs
pdflatex -interaction=nonstopmode -halt-on-error fig_pipeline.tex
cd ..
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Layout

```text
PROTOCOL.md  Per-paper protocol and comparison declaration
paper/       LaTeX, synchronized Markdown write-up, and figures
scripts/     Released analyses and deterministic figure generators
results/     Released JSON values used by analyses and figures
```

## Limitations

- Oracle visibility from the complete target clip crosses the single-image inference boundary. Deployment requires a source-only visibility estimator for each requested target camera and a source-observable selector.
- The latent bypass is exact, but decoder spatial mixing means final visible RGB is not guaranteed. The direct 16-scene visible change is +0.016 dB; large-scale equality is an analysis construction assumption.
- Flash3D evidence renders are per-target component references, not an official single-view reproduction.
- The teaser is constrained to released GT/Gen3R/injected assets; missing input/Flash3D assets are a release limitation, so the approved strongest honest teaser does not invent replacements.
- The scene-level proxy weakens at scale.
- Per-frame injection does not optimize temporal consistency.
