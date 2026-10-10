# Literature Matrix — sv3d-lab (as of 2026-07)

Sources verified via arXiv API / abstract pages / project pages. Metrics marked *unverified* were
NOT confirmed from an official table this session — pull the PDF before citing. Only Flash3D (MINE
3204, VGG) and DepthSplat-L (pixelSplat 2-view, VGG) numbers are reproduction-grounded.

## Competitor table

| Method | Venue/Year/arXiv | #views | 1-img infer | Output repr | Explicit3D | Gen. unseen | Beyond-frustum | Diffusion | Train data (~scenes) | Code+ckpt | RE10K headline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Flash3D** | 3DV'25 / 2406.04343 | 1 | yes | multi-layer per-pixel 3DGS (depth-offset) | yes | partial (offset layers) | **no** (in-frustum) | no | RE10K ~67K | yes | PSNR 20.81/SSIM .743/LPIPS .253 MINE-3204 VGG (verified) |
| pixelSplat | CVPR'24 / 2312.12337 | 2 | no | per-pixel 3DGS | yes | no | no | no | RE10K+ACID | yes | ~25.9 2-view (unverified) |
| MVSplat | ECCV'24 / 2403.14627 | 2(N) | no | per-pixel 3DGS (cost vol) | yes | no | no | no | RE10K+ACID | yes | ~26.4 2-view (unverified) |
| DepthSplat | CVPR'25 / 2410.13862 | 2–12 | no | per-pixel 3DGS (mono+MVS) | yes | no | no | no | RE10K+DL3DV+ScanNet | yes | 27.47/.889/.114 2-view VGG (verified) |
| latentSplat | ECCV'24 / 2403.16292 | 2 | no | variational 3D feature GS + gen decoder | yes(latent) | partial | partial | no (VAE/GAN decoder) | RE10K | yes | unverified |
| Bolt3D | ICCV'25 / 2503.14445 | 1+ | yes | 3DGS from latent diffusion | yes | yes | yes | **yes** | large MV + dense pseudo-GT | project | unverified |
| Wonderland | CVPR'25 / 2412.12091 | 1 | yes | per-pixel 3DGS from video-diff latents (LRM) | yes | yes | yes | **yes** | multi | project | unverified |
| FlashWorld | 2025 / 2510.13678 | 1/text | yes | 3DGS (3D-oriented MV diffusion) | yes | yes | yes | **yes** | RE10K/DL3DV+ | project | unverified |
| Splatter Image | CVPR'24 / 2312.13150 | 1 | yes | per-pixel 3DGS | yes | partial | no | no | objects | yes | N/A (objects) |
| MINE | ICCV'21 / 2103.14910 | 1 | yes | continuous-depth MPI | no (2.5D) | partial (in-frustum) | no | no | RE10K+KITTI | yes | protocol origin |
| GS-LRM | ECCV'24 / 2404.19702 | 2–4 | no | per-pixel 3DGS (transformer) | yes | no | no | no | Objaverse+RE10K | project | unverified |
| CAT3D | NeurIPS'24 / 2405.10314 | 1–few | yes | MV-diffusion → NeRF | yes(after) | yes | yes | **yes** | large mixed | project | unverified |
| ZeroNVS | CVPR'24 / 2310.17994 | 1 | yes | 3D-aware diffusion → SDS NeRF | yes(SDS) | yes | yes | **yes** | CO3D+RE10K+ACID | yes | N/A |
| GenWarp | NeurIPS'24 / 2405.17251 | 1 | yes | 2D gen warp, NO persistent 3D | **no** | yes | yes | **yes** | MV video | yes | FID-based |
| ViewCrafter | 2024 / 2409.02048 | 1–sparse | yes | pointcloud + video-diff → opt 3DGS | partial | yes | yes | **yes** | large video | yes | unverified |
| **studentSplat** ⚠ | 2026 / 2601.11772 | 1 | yes | per-pixel 3DGS + extrapolation net (teacher-student) | yes | yes (extrapolation) | yes | **no** | RE10K-style MV teacher | ? | claims single-view SOTA (unverified) |
| **CATSplat** ⚠ | 2024 / 2412.12906 | 1 | yes | per-pixel 3DGS + VLM text + 3D point guidance | yes | partial | no | no | RE10K+ACID | ? | claims single-view SOTA (unverified) |
| PRISM | 2026 / 2606.25430 | 1 | yes | MV latents (warp+residual) → 3D | yes | yes | yes | no@infer (distilled) | synthetic | ? | 36s/scene (unverified) |

⚠ = direct single-view no/low-diffusion competitor discovered in recon; **must read in full before freezing Paper A**.

Adjacent 2026 flagged: FastPano3D (2606.30352), CylinderSplat (2603.05882, occlusion-completion volume), HY-World2.0 (2604.14268), Prometheus (2412.21117). No paper named "CompleteSplat" exists.

## Evaluation-gap finding (from second recon)
- **"No published method reports SEPARATE visible/invisible metrics with a mask"** — visibility-
  partitioned (visible/disocclusion/out-of-frustum) reporting for feed-forward single-view 3DGS is
  an *open gap* (confirmed via multiple zero-hit targeted searches). BUT it's an evaluation
  contribution, so per charter it can only be a *secondary* axis of a method paper, not the main claim.
- Closest prior art to cite/differentiate: EUVS benchmark (2412.05256, extrapolation, camera-level not
  region-level), Somraj/Kanchana disocclusion eval (WACV'22 2110.08805, video not 3DGS),
  "Appreciate the View" (3DV'26 2511.12675, better global metric not partition), NerfBaselines
  (2406.17345, protocol reproducibility).
- delete-region / causal-contribution as a *correctness* metric: essentially unclaimed (only exists in
  pruning/efficiency lit). Usable as a distinctive diagnostic.
- Consensus: completing unseen single-view regions is "generative-prior territory" (ConfCtrl 2603.09819,
  UMAMI 2512.20107, ArtiFixer 2603.00492). Flash3D is the non-generative exception and reports only
  global metrics — the activity we target.

## Consolidated risk read
- Paper A moat = "no-diffusion, explicit, single feed-forward, real occluded+beyond-frustum 3D" holds
  vs all diffusion methods; contested by **studentSplat / CATSplat** (must differentiate on region-level
  wide-baseline occluded+OOF geometry, not visible-region fidelity alone).
- Paper B collides with latentSplat (variational multi-hypothesis shared-3D) + Bolt3D/CAT3D (generative
  shared-3D from single image). Differentiator must be: explicit reliable hidden-geometry grounding +
  calibrated multi-hypothesis + source-only, evaluated with FID + cross-view consistency on hidden region.

## Immediate follow-ups (before freezing charters)
1. Full-text read: studentSplat 2601.11772, CATSplat 2412.12906, latentSplat 2403.16292, Flash3D suppl.
2. Verify whether any of the above already reports region-split (visible/occluded) numbers in tables.
3. Confirm Bolt3D/FlashWorld exact single-image RE10K protocol + numbers (are they comparable to MINE-3204?).

## 2026-07-18 recon refresh (post-L2-negative-result, Family 4 = amodal, deciding new directions)

New/late works most relevant to single-view HIDDEN completion (amodal), from a fresh 2024-2026 survey:

| Method | Venue/Year/arXiv | 1-img | Output | Amodal hidden? | Appearance? | RE10K NVS? | Code | Note |
|---|---|---|---|---|---|---|---|---|
| **VolFill** | 2026 / 2605.31466* | yes | volumetric latent (TUDF grid) + latent DiffTransformer | **yes (geometry)** | **NO** | no (geometry benches: SCRREAM/NRGB-D) | new/unverified | amodal GEOMETRY only — appearance+NVS is open |
| **NOVA3R** | ICLR'26 / 2603.04179* | set (not strictly 1) | non-pixel-aligned global rep → diffusion 3D decoder → complete pointcloud | yes (incl. invisible) | partial (points) | no | project | attacks pixel-aligned duplication; multi-image, pointcloud not GS |
| **GENA3D** | ECCV'26 / 2511.21945* | 1 (object) | 2D-prior + 3D-coherence cross-attn | yes | yes | object-level only | project | good mechanism template, not scene/RE10K |
| **RelaxFlow** | ICML'26 / 2603.05425* | text+img (object) | training-free dual-branch amodal | yes | yes | object | project | controllable disocclusion idea; not scene NVS |

*arXiv ids from recon, MARKED UNVERIFIED — pull PDFs before citing any number/claim.

**Key structural finding:** amodal single-view work is (a) geometry-centric (VolFill: no appearance),
(b) object-centric (GENA3D/RelaxFlow), or (c) multi-image/pointcloud (NOVA3R). **Scene-level amodal
APPEARANCE completion from ONE image, 3D-grounded, with RE10K region-separated photometric NVS, is
essentially unclaimed.**

**Our own decisive negative result (E006/D012) as motivation:** we empirically proved that a
deterministic feed-forward hidden-Gaussian predictor recovers +4.4 dB on OVERFIT scenes but EXACTLY
0.0 dB on scene-disjoint held-out (opacity collapses to ~0). Combined with our oracle (E002: correct
hidden geometry gives +11.5 dB — representation is capable), this is a clean, publishable motivation:
*the hidden region is representable but NOT deterministically learnable from one image → it requires a
generative/uncertainty-aware formulation.* No competitor states this crisp overfit-vs-generalize
collapse for single-view hidden Gaussians.

## CHOSEN DIRECTIONS (2026-07-18, autonomous decision — supersede prior Paper A/B framing)

- **Direction 1 (first paper target): Uncertainty/latent hidden representation on a UNIFIED
  single-view GS.** Replace Flash3D's DETERMINISTIC offset layers with a SAMPLED/latent hidden head
  (variational or few-step generative), trained with a generative objective so hidden regions are
  drawn from a distribution, not regressed to the mean (which collapses/blurs). Crux experiment: does
  a sampled head score >0 held-out deletion Δ where the deterministic head scored 0 (E006)? Beats
  Flash3D on invisible-region LPIPS; single-image, 3D-grounded (renders one shared 3D set to all
  targets). Closest competitors: Flash3D (deterministic), latentSplat (2-view → single-view).
- **Direction 2 (second paper target): 3D-grounded amodal APPEARANCE scene completion from one
  image.** VolFill (2026) completes amodal GEOMETRY only; we add generative hidden APPEARANCE on
  reliable geometry, evaluated with RE10K region-separated NVS + cross-view consistency. Closest:
  VolFill (geometry-only), CAT3D/ViewCrafter (per-view, slow, not single-pass 3D-grounded).

Independence: D1 = feed-forward unified GS with calibrated uncertainty in disoccluded regions;
D2 = generative amodal appearance model. Different mechanism, different contribution.

**Verify-before-cite list:** Flash3D offset-layer determinism (our motivation — CONFIRMED from
upstream code read: offset layers are a deterministic regression head, no sampling); latentSplat
single-view feasibility; VolFill/NOVA3R exact scope + code release; CATSplat/studentSplat RE10K
protocol match.

## 2026-07-18 — REFRAME after 3 collapses (D014): the two FINAL paper directions

After E006/E008/E009 (deterministic + global-CVAE + spatial-CVAE hidden APPEARANCE all collapse to
Δ≈0 held-out), hidden APPEARANCE from one image + photometric loss is proven not learnable. Both
papers reframed; collision-checked via web/arXiv agents (2026-07-18).

### PAPER A — Calibrated aleatoric uncertainty for feed-forward single-image 3DGS (VERDICT: OPEN)
Collision check: no paper does {feed-forward + single-image + 3DGS + CALIBRATED uncertainty +
AUSE/ECE/NLL on RE10K}. Closest: latentSplat (2-view, variational→generative, uncertainty
QUALITATIVE only §4.4, no calibration/NLL/AUSE); SGS 2403.18476 (AUSE in GS but PER-SCENE optim,
LLFF); NeRF UQ FisherRF/Bayes'Rays/ActiveNeRF (per-scene); heteroscedastic Gaussian-NLL
(Kendall&Gal'17) never applied to NVS/3DGS. Must-cite/differentiate: latentSplat, SGS, FisherRF,
Bayes'Rays, Splatter Image, Flash3D, Kendall&Gal'17, Ilg'18(AUSE origin). Reviewer risk = latentSplat
(pre-empt: single-img vs 2-view; calibrated regression vs generative sampling — run THEIR variance
through our AUSE/ECE to show uncalibrated; NLL vs VAE). Motivation = our own 3-collapse negative
result. Implemented D015 (uncertainty_model + train_uncertainty + eval_uncertainty).

### PAPER B — Single-image → shared-3DGS SCENE generation w/ distilled generative prior (VERDICT: PARTIALLY-OCCUPIED, open lane)
| Method | Venue/Yr | arXiv | 1-img | scene | ONE shared 3D | feed-fwd? | RE10K | prior mechanism |
|---|---|---|---|---|---|---|---|---|
| Scene Splatter | CVPR'25 | 2504.02764 | y | y | y (global 3DGS) | per-scene iter | no | video-diff refine→recon |
| Wonderland | CVPR'25 | 2412.12091 | y | y | y (feed-fwd 3DGS) | feed-fwd | yes | video-diff latents→LRM |
| Bolt3D | ICCV'25 | 2503.14445 | y(1+) | y | y (samples GS) | feed-fwd | partial | latent diffusion over 3D rep |
| CAT3D | NeurIPS'24 | 2405.10314 | y | y | no (MV-gen→recon) | MV-gen+recon | partial | multiview diffusion→recon |
| ZeroNVS | CVPR'24 | 2310.17994 | y | y | y (NeRF via SDS) | per-scene SDS | no | SDS from 3D-aware diff |
| RealmDreamer | 3DV'25 | 2404.07199 | y(+text) | y | y (3DGS distill) | per-scene | no | inpaint+depth diff distill→3DGS |
| GenWarp | NeurIPS'24 | 2405.17251 | y | y | NO (per-view 2D) | feed-fwd 2D | no | T2I gen warp (the foil) |
| VolFill | 2026 | 2605.31466 | y | y | y (TUDF grid) | feed-fwd diff | no | GEOMETRY ONLY, no appearance |
| NOVA3R | ICLR'26 | 2603.04179 | multi | y | y (point cloud) | feed-fwd | no | geometry points, no appearance |

**Unoccupied (verified, `RealEstate10K + disoccluded` = 0 arXiv hits):** appearance-level amodal
completion into ONE shared explicit 3DGS scene, single-image, RE10K region-separated
(visible/disoccluded/OOF) + cross-view consistency. Novelty stmt: "distill a generative image prior
into a single shared explicit 3DGS scene and be the FIRST to quantify WHERE hallucination happens —
region-separated RE10K + cross-view consistency (TSED/reprojection), directly measuring the shared-3D
advantage over per-view 2D inpainting that prior work only claims qualitatively." Reviewer risk =
Wonderland+Scene Splatter+RealmDreamer (differentiate: feed-forward distilled inference vs their
per-scene optim; region-separated eval as first-class contribution). Consistency metrics in lit:
TSED (thresholded symmetric epipolar dist), reprojection/warp-consistency, cross-view FID,
feature-match inliers (LoFTR/SuperGlue).
**Feasibility ranking (1×A6000):** (1 lowest-risk) reconstruct pseudo-views from off-the-shelf
single-image NVS diffusion (ViewCrafter/ZeroNVS/CAT3D-samples) → distill into Flash3D GS; (2) SDS
from pretrained 2D diffusion distilled into feed-forward net; (3) finetune MV diffusion (too heavy);
(4 highest-risk/novelty) DMD into 3DGS scenes (no precedent). Biggest technical risk: generative
supervision must NOT degrade visible-region PSNR (28.7) — MASK supervision so prior drives ONLY
disoccluded/OOF Gaussians while visible keep reconstruction loss. Blur signature (higher PSNR worse
LPIPS) would be exposed by region-separated metric.
