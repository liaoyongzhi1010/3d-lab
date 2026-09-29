# Paper 2 (draft): Distilling Generative Priors into a Feed-Forward 3D Scene Completer

> Working title: **"Gate-Aware Distillation: Fast Feed-Forward 3D Completion of
> Single-View Scenes from a Generative Teacher"**
> Draft v1 (2026-08-22). Combines open-sourced top-venue components (Flash3D
> feed-forward 3DGS + Gen3R/diffusion generative teacher) — the "diffusion +
> feed-forward" combination. All numbers from real runs (E-064/E-402/E-403/E-410/
> E-411/E-420). Honest: single-view disocclusion is hard; gains are modest but
> consistent, and the contribution is the *distillation recipe* + speed.

## Abstract (draft)

Generative diffusion models can complete the disoccluded regions of a single-view
3D scene reconstruction, but they are slow (seconds per scene) and unreliable. We
distill a diffusion generative teacher into a **fast feed-forward 3D hole-Gaussian
completer** (7.8M params, 13.7 ms/frame) that predicts, for every disoccluded
pixel, an explicit 3D Gaussian (depth, covariance, opacity, color) rendered
through a real 3DGS rasteriser — a **true 3D representation with correct
parallax**, not 2D image inpainting. Naively distilling every teacher sample is
suboptimal: **26% of teacher completions are actually harmful** (worse than the
feed-forward baseline vs ground truth). We introduce **gate-aware distillation**:
supervise the student only on samples the teacher completes reliably. Using just
**23% of the data**, the gate-aware student **exceeds** the all-data baseline
(disoccluded-region gain +0.129±0.016 dB vs +0.039±0.012 dB, ~3.3× over 4 seeds),
while running 3–4 orders of magnitude faster than the diffusion teacher. Our
recipe combines open-source top-venue building blocks (feed-forward 3DGS +
diffusion prior) into a deployable single-view scene completer.

## 1. Introduction (draft outline)
- Single-view scene reconstruction: feed-forward (Flash3D/CATSplat) fast & faithful on visible surface, but incomplete at disocclusions.
- Diffusion / generative (Gen3R, Scene-Splatter) complete but slow + unreliable + not guaranteed 3D-consistent.
- **Combine the two**: distill the generative teacher into a feed-forward 3D completer → fast, 3D, deployable.
- Problem: teacher is unreliable (26% harmful). Contribution: **gate-aware distillation** = distill only reliable teacher samples.
- Contributions:
  1. A feed-forward **3D hole-Gaussian** completer distilled from a diffusion teacher (true 3D, 13.7 ms).
  2. **Gate-aware distillation**: reliability-filtered supervision; 23% data beats 100% baseline.
  3. Full evaluation: multi-seed, speed, held-out; honest single-view limits.

## 2. Related Work
- Feed-forward single-view 3DGS: Splatter Image (CVPR24), Flash3D, **CATSplat (ICCV25)**.
- Diffusion scene generation/completion: Gen3R, Scene-Splatter, GenFusion, **Difix3D+ (CVPR25)**.
- Distillation of generative priors; knowledge distillation for NVS.
- **Combining diffusion + feed-forward** — our niche: distill diffusion completeness into feed-forward speed, in true 3D.

## 3. Method

### 3.1 Feed-forward 3D hole-Gaussian completer (真 3D, not 2D)
- Input: single image → Flash3D base 3D Gaussians (visible surface, frozen).
- Disoccluded pixels identified by feed-forward visibility.
- Student head $g$: for each hole pixel predict G-buffer (depth offset, RGB→SH, scale, opacity, quaternion).
- Back-project hole pixel to 3D via predicted depth: $xyz=((u-c_x)/f_x\,z,(v-c_y)/f_y\,z,z)$; render through the real 3DGS rasteriser.
- **This is explicit 3D**: hole content has 3D position + covariance; renders with correct parallax under viewpoint change (verified, Sec 4.5); visible region kept exactly (Flash3D parity).

### 3.2 Distillation from a generative teacher
- Teacher $T$ (diffusion, Gen3R lineage) completes the disoccluded region in a target view.
- Student learns teacher's hole appearance: direct per-hole-pixel color loss + through-render L1/LPIPS + coverage + depth-anchor/smoothness. (Ill-posed hole→GT regression is converted to well-posed teacher imitation.)

### 3.3 Gate-aware distillation (core contribution)
- Not all teacher completions are trustworthy: per-sample teacher reliability
  $q = \mathrm{PSNR}^{\text{hole}}_T - \mathrm{PSNR}^{\text{hole}}_B$ vs GT.
- **26% of samples have $q<0$** (teacher worse than baseline) → naive distillation learns hallucinations.
- Gate-aware: train only on $q \ge \tau$ (reliable) samples. Optionally $q$-weighted loss.
- (Ties to Paper 1's gate: reliability-driven selective use of the generative prior, here at *distillation* time.)

## 4. Experiments

### 4.1 Protocol
- Teacher cache: 1000 (scene,target) samples, deterministic 85/15 train/test split by hash (165 held-out).
- Metrics vs **ground truth** in the disoccluded (hole) region: PSNR, LPIPS; Δ over feed-forward base.
- Seeds {0,1,2,3}. A6000.

### 4.2 Teacher reliability analysis
- 1000 samples: mean teacher gain −0.104 dB; **22.9% help (>0.5 dB), 26.1% hurt (<−0.5 dB)**. Motivates gate-aware.

### 4.3 Main result: gate-aware vs baseline distillation
| Student | Train samples | Δ vs base (hole PSNR, mean±std over seeds) |
|---|---|---|
| baseline (all) | 835 | **+0.039 ± 0.012** |
| **gate-aware (q≥0.5)** | **191 (23%)** | **+0.129 ± 0.016** |
→ Gate-aware uses **23% of data** yet **~3.3× the net gain**, consistent across 4 seeds. LPIPS matched (0.482–0.483).

### 4.4 Speed (deployment)
- Student feed-forward completion: **13.7 ms/frame** (median 13.6, A6000).
- Diffusion teacher (Gen3R): ~247 s/scene. Student completion is **3–4 orders of magnitude faster**. (Honest: 247s is full-scene generation; the point is feed-forward vs iterative-diffusion latency class.)

### 4.5 3D consistency (真 3D verification)
- Orbit rendering (yaw ±4° + translation) of the completed hole Gaussians shows **correct parallax** (adjacent-view mean diff 0.155±0.043, smooth/bounded — not the degenerate zero of a 2D paste nor flicker of broken 3D).
- Honest limitation: current eval composites over the cached 2D base image; a full scene multi-view render (visible Flash3D Gaussians + hole Gaussians together) is the natural extension.

### 4.6 Ablations
- Gate threshold $\tau$ sweep (data fraction vs gain).
- With/without through-render LPIPS, coverage loss (from teacher-cache design).

## 5. Conclusion
Combining an open-source feed-forward 3DGS reconstructor with a diffusion
generative teacher, we distill a fast (13.7 ms) **true-3D** single-view scene
completer. **Gate-aware distillation** — supervising only on reliable teacher
samples — beats all-data distillation with a fraction of the data, consistently.
Limitations: absolute disoccluded-region PSNR remains low (single-view
disocclusion is intrinsically ill-posed, cf. Paper 1's upper-bound analysis);
gains are modest but robust; full-scene multi-view rendering is future work.

## Assets / provenance
- Teacher cache: `/home/data/E-064_teacher_cache` (1000 npz: teacher_rgb, hole, base_rgb, gt_rgb, depth_tgt, K_tgt).
- Student train/eval: `_e402_student_unc.py` (gate_aware/use_unc flags), `_e403_eval_student.py`.
- Teacher quality: `_e410_teacher_quality.py` → `E-410_teacher_quality.json`.
- Multi-seed: `E-420_{base,gate}_s{1,2,3}` + seed-0 `E-402_baseline`/`E-411_gateaware`.
- 3D consistency: `_e412_multiview_consistency.py` → `E-412_mv/`.
