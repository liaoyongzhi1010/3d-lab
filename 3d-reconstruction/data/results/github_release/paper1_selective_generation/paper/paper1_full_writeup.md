# Oracle-Visibility Selective Injection: A Single-View Reconstruction Diagnostic

## Abstract

Single-view reconstruction is asymmetric: visible regions have image evidence, whereas disoccluded regions require a prior. We study asymmetric latent injection as an **oracle-visibility diagnostic**. The released `visibility.npy` is produced by running VGGT on the complete target clip and therefore crosses the single-image inference boundary. The visible latent path is exact, but VAE/decoder spatial mixing prevents a final-RGB guarantee; the direct 16-scene run measured a +0.016 dB visible-PSNR change. A target-GT quality-gap oracle predicts injection benefit at r=0.982 over 166 scenes. On a fixed 759-frame component diagnostic, the always injected output improves Gen3R PSNR from 17.77 to 19.22 dB and LPIPS-VGG from 0.360 to 0.341. A deployable method requires source-only visibility and selection estimators, neither of which is evaluated.

## 1. Central Thesis

**When does feed-forward geometric evidence improve a generative single-view reconstruction, and how can it be injected without harming already-correct visible regions?**

Geometry helps when its evidence is better than the generative prior in newly revealed regions. Asymmetric injection protects visible content; oracle fallback analysis quantifies the opportunity for a future predictor but does not protect a deployed system.

The three contributions are:

1. Concept: oracle-visibility selective injection as a mechanism diagnostic, not deployable single-view inference.
2. Mechanism: an exact visible latent bypass, with decoded RGB behavior measured rather than guaranteed.
3. Evidence: a scale-stable quality-gap mechanism and perceptual component gains on a fixed long-sequence diagnostic, bounded by failed zero-shot ACID transfer.

## 2. Method

### Stage 1: Construct offline oracle evidence

Flash3D supplies target-aligned evidence. VGGT supplies the visible/invisible mask by jointly processing the complete target clip, so this stage is not source-only. Flash3D is an evidence component, not an official reproduced leaderboard baseline.

### Stage 2: Selective generation

Gen3R first produces a clean baseline latent. For visibility mask `M` (1 means visible), current latent `z`, encoded evidence `z_f3d`, and weight `w`, injection is

```text
z_new = M * z + (1 - M) * ((1 - w) * z + w * z_f3d)
```

For `M=1`, `z_new = z` exactly at the masked latent operation. A fixed clean-space agreement weight controls invisible injection. The VAE/decoder may mix spatially, so this does not prove exact final visible RGB; the direct 16-scene run measured +0.016 dB visible-PSNR change. A future predictor could select either the merged candidate or unchanged Gen3R fallback; released selection results are target-GT oracle analyses, not deployment.

## 3. Experiments: Claim-First Order

### 3.1 Protocol and implementation

The primary evaluation is a **fixed offline oracle-visibility long-sequence diagnostic**, not official MINE/Flash3D evaluation or deployable single-view inference. One frame is designated as source, but VGGT consumes the complete 49-frame target clip to produce `visibility.npy`. This crosses the single-image inference boundary. The main set has 166 valid scenes. Working renders are 560 pixels; masks are nearest-neighbor resized. Region metrics use masked PSNR and oracle-substitution LPIPS-VGG/SSIM with 10,000-scene-resample bootstrap confidence intervals. See [`../PROTOCOL.md`](../PROTOCOL.md).

### 3.2 Main component diagnostic

The fixed full-image study contains 759 disocclusion-heavy frames from 21 long sequences. All rows use the same stored outputs and metric preprocessing. It is neither official MINE nor an official Flash3D reproduction.

| Variant | Views | Frame policy | Resolution | Metric type | PSNR | SSIM | LPIPS-VGG | FID | KID | Runtime | Status |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---|---|
| Gen3R baseline | 1 | 759 disocclusion frames | 560 working | full image/VGG | 17.77 | 0.653 | 0.360 | 31.8 | 0.0020 | ~247 s/scene | reproduced component |
| Flash3D evidence reference | 1 | same | 560 working | full image/VGG | 19.76 | 0.679 | 0.444 | 54.8 | 0.0096 | unavailable | per-target evidence, not official single-view |
| Always injected candidate | 1 | same | 560 working | full image/VGG | 19.22 | 0.676 | 0.341 | 31.1 | 0.0028 | unavailable | injected output |

E-210 evaluates the stored `adaptive2` injected output directly on every included frame, with no oracle fallback. This directly evaluated injected output lifts Gen3R PSNR by +1.45 dB and lowers LPIPS-VGG from 0.360 to 0.341 while keeping FID close. E-207/E-209 oracle-selected metrics remain separate. At scene level, oracle-selective invisible PSNR is 15.44 dB versus 13.78 for Gen3R, a +1.66 dB gain with 95% CI [1.32, 2.02]. The large-scale 14.48 dB visible value was assigned from the baseline as a construction assumption, not independently measured. The direct 16-scene run measured +0.016 dB visible change. On 21 high-gain scenes, invisible SSIM changes 0.579 to 0.725 and LPIPS-VGG 0.0948 to 0.0784.

### 3.3 Oracle selectivity and proposed fallback

On the initial fixed 16-scene diagnostic, always injection yields +1.58 dB mean and -6.88 dB worst case. A target-GT visible-quality proxy yields +2.07 dB mean and -0.41 dB worst case, but is unavailable at raw test time. On 166 scenes, the released default quality-gap oracle threshold `tau=5` yields **+1.658 dB mean**, -0.393 dB worst case, and a 52% injection rate. Thresholds 3-6 remain within 0.1 dB of the mean optimum. These values motivate a proposed fallback interface; they are not deployed routing results.

### 3.4 Mechanism, difficulty, and scale

The Flash3D-evidence minus Gen3R invisible PSNR gap predicts always-inject benefit at **r=0.982** (95% CI [0.973, 0.990], N=166). Hard, mid, and easy baseline buckets change by +2.49, +0.11, and -2.52 dB. The correlation remains 0.985, 0.980, 0.983, 0.984, and 0.982 at N=16, 92, 129, 149, and 166. `fig_mechanism_combined.pdf` combines the scatter, difficulty buckets, and quality-gap oracle operating points from released JSONs.

### 3.5 ACID failed zero-shot transfer

Converted ACID trajectories form a cross-dataset diagnostic, not an official leaderboard result. The released E-215 aggregate contains **8 scenes**: always injection averages **-0.443 dB** and wins **3/8** scenes. This is failed zero-shot transfer and evidence that selectivity is dataset-dependent, not mechanism success; no per-bucket ACID claim is released.

### 3.6 Qualitative successes and failures

The aligned teaser uses only released selected assets. It shows three localized recoveries with fixed yellow ROIs and one injected failure that demonstrates why a future fallback is needed; no routing decision is claimed. The teaser is constrained to released GT/Gen3R/injected assets; missing input/Flash3D assets are a release limitation, so the approved design uses the strongest honest teaser available rather than inventing replacements. The sequence panel separately shows GT, Gen3R, injected candidate, disocclusion, and both error maps.

### 3.7 Published positioning only

Published pixelSplat, MVSplat, DepthSplat, and Flash3D values are retained only in a separate positioning table. The two-view methods use the official pixelSplat index, 256x256, no crop, three interpolated targets, and LPIPS-VGG. Flash3D uses the MINE split (3,204 samples/641 scenes), one view, 256x384, 5% crop, LPIPS-VGG, and separate +5, +10, and U[-30,30] target buckets (officially quoted as 28.46/0.899/0.100, 25.94/0.857/0.133, and 24.93/0.833/0.160). Sources are Flash3D arXiv:2406.04343v2 Table 2; pixelSplat camera-ready/retrained README (arXiv:2312.12337v4 Table 1 agrees; exact checkpoint file N/A); MVSplat arXiv:2403.14627v2 Table 1, checkpoint N/A; and DepthSplat-L arXiv:2410.13862v3 Table 6, checkpoint N/A. Values are quoted, not reproduced; no cross-protocol winner is marked.

## 4. Limitations

The complete target clip is consumed by VGGT to produce `visibility.npy`; this oracle visibility crosses the single-image inference boundary. Deployment needs a source-only visibility estimator that predicts target-visible support from only the source image/camera and requested target camera. The released scene-level policies also use target-GT quality probes; a source-observable selector is future work. Neither estimator is evaluated. The latent mask is exact, but final visible RGB is not guaranteed; the measured direct-run change is +0.016 dB, while large-scale equality is a construction assumption. Flash3D evidence is a per-target component reference, not an official single-view reproduction. The teaser is constrained to released GT/Gen3R/injected assets; missing input/Flash3D assets are a release limitation, and this reviewer-visible exception permits the strongest honest teaser rather than invented replacements. `fig_failures_limitations.pdf` combines the released injected failure with generated statistics: second-order disocclusion energy is 0.0251 for Gen3R and 0.0267 for the injected output over 21 scenes, with the injected output more stable in 9/21; ACID has 8 scenes, -0.443 dB always-injected mean, and 3/8 wins. No raw ACID images are used.

## 5. Conclusion

Under oracle visibility, geometry evidence is useful conditionally, not universally. Region-asymmetric injection provides an exact latent bypass while measured RGB behavior remains decoder-dependent. Oracle selectivity quantifies a future fallback opportunity; source-only visibility and selection are required before deployment.
