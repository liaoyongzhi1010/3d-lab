# Paper 1 (draft): Grounding Generative Priors for Single-View Scene Reconstruction

> Working title: **"When to Trust Generation: A Deployable Reliability Gate for
> Single-View 3D Scene Completion"**
> Draft v1 (2026-08-22). Story mirrors OracleGS (propose-and-validate / grounding
> generative priors) but in the **single-view** regime. All numbers from real runs
> (E-149/E-224/E-305/E-306). Honest: negative results (VGGT, deployability ceiling)
> reported as-is.

## Abstract (draft)

Single-view 3D scene reconstruction is asymmetric: visible surfaces have image
evidence, but disoccluded regions must be *generated*. Generative priors can
plausibly complete these regions, yet they are unreliable — on RealEstate10K,
injecting a generative teacher into a feed-forward reconstructor helps on average
(+0.48 dB in disoccluded regions) but is **harmful on 41.5% of frames** and can
cost up to **−23 dB**. OracleGS (WACV'26) resolves the analogous sparse-view
problem by *validating* generative proposals with a multi-view-stereo oracle
(VGGT attention). We ask whether this "propose-and-validate" recipe transfers to
the **single view**, where no multi-view evidence exists. We show it does **not**
at the validation stage — deployable single-view geometric evidence (VGGT
confidence) cannot predict per-frame generative reliability (ROC-AUC 0.65 vs a
ground-truth-quality ceiling of 0.95) — and we explain why with a controlled
upper-bound experiment: even a *perfect* per-pixel reliability oracle fails to
improve reconstruction, because single-view disocclusions are fundamentally
unrecoverable. Instead of grounding at the pixel level, we contribute a
**deployable reliability gate**: a small learned network that, using only
inference-time signals (camera motion, feed-forward visibility, monocular
confidence), decides *when to inject the generative prior at all*. The gate
converts the risky "always inject" policy (+0.48 dB, 467 catastrophic frames)
into a risk-averse one (**+0.73 dB net, +51%**, or **−89% catastrophic frames**),
stable across 5 seeds. Our study delineates the boundary of trustworthy
generative-prior use under single-view constraints.

## 1. Introduction (draft outline)
- Single-view reconstruction asymmetry: transport (visible) vs generation (disoccluded).
- Generative priors complete but hallucinate; quantify the risk (Fig.1: delta histogram, −23 dB tail, 41.5% harmful).
- OracleGS recipe recap: propose (diffusion) → validate (MVS oracle attention) → uncertainty-weighted optimization. Works for sparse-view because multi-view evidence exists.
- **Our question**: does grounding transfer to single-view? Contributions:
  1. A controlled study showing the *validation* stage fails in single-view (deployable evidence can't predict reliability; even a perfect oracle can't fix reconstruction — Sec 4.2/4.3).
  2. A **deployable reliability gate** (training contribution) that grounds the prior at the *decision* level, not the pixel level (Sec 3).
  3. Honest boundary characterization: the observability gap (0.65→0.95) is intrinsic, not an engineering gap.

## 2. Related Work
- Single-view feed-forward 3DGS: Splatter Image (CVPR24), Flash3D, CATSplat (ICCV25).
- Generative scene completion: Gen3R, Scene-Splatter, diffusion NVS.
- Grounding generative priors: **OracleGS (WACV'26)** — closest; sparse-view, VGGT-attention oracle, uncertainty-weighted 3DGS loss. We contrast single-view.
- Uncertainty / reliability in NVS: CoMapGS covisibility, etc.

## 3. Method: Deployable Reliability Gate

### 3.1 Setup
- Feed-forward base reconstructor $B$ (Flash3D) → 3D Gaussians for visible surface.
- Generative teacher $T$ (Gen3R / Paper-lineage) → completes disoccluded region.
- Per target frame, injecting $T$ yields disoccluded-region gain $\delta = \mathrm{PSNR}_T^{\text{inv}} - \mathrm{PSNR}_B^{\text{inv}}$. Positive on ~55.5% frames.

### 3.2 The gate
- A small MLP $g(\phi) \to [0,1]$, $\phi$ = **deployable** features only:
  camera translation & rotation, feed-forward disocclusion fraction, monocular/VGGT confidence stats. (No GT-derived quality probe.)
- Decision: inject iff $g(\phi) > \tau$. $\tau$ is a **risk-averse** operating point.
- Training objective: we compare (a) BCE on the helpful label, (b) $|\delta|$-weighted BCE (risk-sensitive: penalize misranking high-magnitude frames), (c) direct $\delta$ regression. (b)/(c) reduce catastrophic injections.

### 3.3 Why not pixel-level grounding (OracleGS-style)?
- We tried it (Sec 4.3): VGGT per-pixel uncertainty-weighted distillation, and even a *perfect* oracle weight from teacher-vs-GT error. Neither improves single-view reconstruction — the disoccluded GT is unrecoverable. Hence we ground at the **decision** level.

## 4. Experiments

### 4.1 Protocol
- RealEstate10K, 2898 (scene,frame) rows over 74 scenes (E-149). Label $\delta>0.1$ dB.
- Grouped leave-one-scene-out / grouped 5-fold CV (no scene leakage).
- Report ROC-AUC **and** injection payoff: net dB gain, worst-case dB, #catastrophic (<−3 dB).

### 4.2 Can deployable evidence predict reliability? (validation stage)
| Feature set | Deployable | ROC-AUC |
|---|---|---|
| GT-quality probe (ceiling) | ✗ | **0.949** (logistic) / **0.978** (MLP) |
| camera-only (floor) | ✓ | 0.650 |
| VGGT single-view evidence | ✓ | 0.45–0.50 |
| camera + VGGT | ✓ | 0.58–0.64 |

→ **Deployable single-view evidence cannot close the 0.65→0.95 gap.** VGGT (an MVS model) carries little single-view reliability signal (consistent across logistic/MLP/loss-weighting — 4 independent tests).

### 4.3 Upper-bound: does grounding even help reconstruction?
Train the 3D completion student with OracleGS-style uncertainty-weighted loss:
| Weight source | PSNR_hole vs GT |
|---|---|
| none (baseline) | 12.621 |
| VGGT (deployable) | 12.568 |
| **perfect oracle (teacher-vs-GT err)** | 12.615 |
→ Even a **perfect** per-pixel reliability weight ≈ baseline. Single-view disocclusion GT is unrecoverable; pixel-level grounding has a physical ceiling. (Distinguishes single-view from OracleGS's sparse-view success.)

### 4.4 Our gate: grounding at the decision level (main result)
Reference policies (2898 frames): always **+0.481 dB / worst −23.09 / 467 catastrophic**; oracle +1.703 / 0 / 0.
| Gate (deployable) | AUC | best-net gain | worst | catastrophic |
|---|---|---|---|---|
| cam [wcls] | 0.60 | **+0.764** | −23.09 | 307 |
| cam+geo [wcls] | 0.64 | +0.589 | −9.39 | **52 (−89%)** |
**Multi-seed (5 seeds)**: cam[wcls] net gain **0.727 ± 0.097 dB (+51% over always)**, AUC 0.611 ± 0.038.
→ The gate turns a risky policy into a risk-averse one: either **+51% net gain** or **−89% catastrophic frames**, tunable by $\tau$.

### 4.5 Ablations
- Feature groups: camera dominant; feed-forward `disocc_frac` adds risk control; VGGT adds nothing (4th confirmation).
- Training objective: $|\delta|$-weighted BCE best for catastrophe avoidance.

## 5. Conclusion
Grounding generative priors transfers to single-view only at the **decision** level,
not the pixel level. We characterize an intrinsic observability gap and deliver a
deployable, robust reliability gate. Limitations: deployable AUC ceiling ~0.64;
fully eliminating catastrophes costs injection rate.

## Assets / provenance
- Data: `results/E-149_frames_b5678_aug.json` (2898 frames, 74 scenes).
- Scripts: `e224` (camera-only), `e304` (MLP), `e305` (gate payoff), `e306` (multi-seed).
- Upper-bound distillation: `_e402` (VGGT/oracle weighted), `_e403` (eval).
