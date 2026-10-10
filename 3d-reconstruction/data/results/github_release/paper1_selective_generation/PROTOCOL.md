# Paper 1 Protocol Declaration

Paper 1 is an offline oracle-visibility mechanism study. It asks when feed-forward geometry evidence helps a generative reconstruction and how masked latent injection behaves. Its results are not deployable single-view inference or official leaderboard reproductions.

## Protocol Families

| Family | Views | Split/index | Resolution/crop | Targets | Metrics and status |
|---|---:|---|---|---|---|
| Fixed long-sequence oracle-visibility/component diagnostic | 1 designated source image; complete target clip used by mask estimator | Released `E-142_combined_N169.json`; 166 valid scenes | 560-pixel working renders; metric-specific preprocessing; no MINE 5% crop | First frame as context, next 48 along a fixed 49-frame trajectory; VGGT consumes all 49 frames | Visible/invisible PSNR and oracle-substitution LPIPS-VGG/SSIM; offline diagnostic, not source-only inference |
| Fixed 759-frame full-image component diagnostic | 1 source image | 21 selected scenes, 759 rendered frames with significant disocclusion | Same stored renders and released metric preprocessing | Disocclusion-heavy frames from the long sequences | PSNR, SSIM, LPIPS-VGG, FID, KID; reproduced comparison among Gen3R, Flash3D evidence renders, and the always injected candidate |
| ACID zero-shot diagnostic | 1 source image | Released E-215 aggregate over 8 valid scenes | Converted ACID trajectories with VGGT visibility | Long trajectory aggregate | Failed zero-shot transfer: always mean -0.443 dB and 3/8 wins; no per-bucket claim; not an official ACID leaderboard result |
| Official MINE / Flash3D | 1 | Published 3,204 samples / 641 scenes | 256x384, 5% border crop | +5, +10, and U[-30,30] buckets | LPIPS-VGG; quoted positioning only, not reproduced |
| Official pixelSplat-family | 2 | `evaluation_index_re10k.json` | 256x256, no border crop | 3 interpolated targets | LPIPS-VGG; quoted positioning only, not reproduced |

## Component Semantics

- Gen3R is the frozen generative baseline.
- Flash3D is the feed-forward geometry evidence component. In the reproduced diagnostics, its evidence renders are built per target from scene-folder geometry and are not a directly comparable official single-view Flash3D evaluation.
- E-210 `ours` is the `adaptive2` injected output evaluated directly on every included frame: it is the always injected candidate and has no oracle fallback.
- E-207/E-209 selection numbers separately use a target-GT quality-gap oracle.
- The visible branch follows the Gen3R latent exactly at the masked injection operation. VAE/decoder spatial mixing means final visible RGB invariance is not proven. The direct 16-scene run measured a +0.016 dB visible-PSNR change.
- The N=166 visible value of 14.48 dB for the selected output was assigned from the baseline as a construction assumption; it is not an independent decoded-RGB measurement.
- Returning the unchanged baseline is an oracle/proposed fallback analysis, not a deployed policy.

## Inference Boundary

`scripts/precompute_visibility.py` runs VGGT jointly on the complete target clip, then backprojects frame 0 using depths and poses inferred from that clip. The resulting `visibility.npy` therefore contains target-clip information and crosses the single-image inference boundary. Every current mask, region metric, and injected output is an **oracle visibility** or offline diagnostic result.

A deployable continuation must replace this file with a source-only visibility estimator: given only the source image, source camera, and requested target camera, it must predict target-visible support without reading target RGB frames. A source-observable selector is also needed to replace target-GT quality-gap policies. Neither component is evaluated in this release.

## Positioning Policy

Published pixelSplat, MVSplat, DepthSplat, and Flash3D values are retained only to identify the neighboring protocol families. They are never bolded against this work and do not support a cross-protocol win claim.

| Method | Views | Split/index | Resolution/crop | Targets | PSNR | SSIM | LPIPS-VGG | Exact source/version/table status |
|---|---:|---|---|---|---:|---:|---:|---|
| pixelSplat | 2 | official eval index | 256x256, none | 3 interpolated | 26.09 | 0.863 | 0.136 | camera-ready/retrained README; arXiv:2312.12337v4 Table 1 agrees; exact checkpoint file N/A |
| MVSplat | 2 | official eval index | 256x256, none | 3 interpolated | 26.39 | 0.869 | 0.128 | arXiv:2403.14627v2 Table 1; checkpoint N/A |
| DepthSplat-L | 2 | official eval index | 256x256, none | 3 interpolated | 27.47 | 0.889 | 0.114 | arXiv:2410.13862v3 Table 6; checkpoint N/A |
| Flash3D | 1 | MINE, 3,204 samples / 641 scenes | 256x384, 5% crop | +5 | 28.46 | 0.899 | 0.100 | arXiv:2406.04343v2 Table 2 |
| Flash3D | 1 | MINE, 3,204 samples / 641 scenes | 256x384, 5% crop | +10 | 25.94 | 0.857 | 0.133 | arXiv:2406.04343v2 Table 2 |
| Flash3D | 1 | MINE, 3,204 samples / 641 scenes | 256x384, 5% crop | U[-30,30] | 24.93 | 0.833 | 0.160 | arXiv:2406.04343v2 Table 2 |

## Teaser Asset Limitation

The teaser is constrained to released GT/Gen3R/injected assets; missing input/Flash3D assets are a release limitation and the approved design therefore uses the strongest honest teaser available. The released selected qualitative assets contain GT, Gen3R, and injected-result panels with fixed yellow regions of interest. They do not contain the raw input frame, a matching Flash3D render, or a routing decision for those selected cases. `scripts/e219_topvenue_teaser.py` therefore presents the injected failure only as motivation for a future fallback and does not invent missing images or claim a selected output.

## Reproduction

```bash
python3 scripts/e203_bootstrap_ci.py --data results/E-142_combined_N169.json --out /tmp/e203.json
python3 scripts/e216_gate_sensitivity.py --data results/E-142_combined_N169.json --out /tmp/e216.json
python3 scripts/e219_topvenue_teaser.py
python3 scripts/e220_protocol_table.py
```

Expected unchanged headline checks are mechanism correlation `r=0.982` and default quality-gap oracle mean `+1.658 dB`.

### Upstream checkout configuration

Install Gen3R from <https://github.com/JaceyHuang/Gen3R> and VGGT from <https://github.com/facebookresearch/vggt>. Configure both GPU entry points with `--gen3r_root /path/to/Gen3R --vggt_root /path/to/vggt`, or set `GEN3R_ROOT` and `VGGT_ROOT`. Flags take precedence; absent flags and environment variables, the legacy `/root/projects/Gen3R` and `/root/projects/vggt` locations remain the defaults. For `scripts/e012_vesg.py`, omit `--ckpt` to use `<gen3r_root>/checkpoints` or pass an explicit checkpoint path. `--help` does not require either upstream repository.

## Qualitative Selection Mapping

`results/E-218_obvious_qual_ranking.json` is the deterministic scanner ranking. `results/E-218_selected_qual_manifest.json` records the three de-duplicated published assets, their scene/frame/ROI records, and their ranking entries. Selection is ranking-derived followed by documented manual semantic de-duplication; it is not an unrecorded choice.
