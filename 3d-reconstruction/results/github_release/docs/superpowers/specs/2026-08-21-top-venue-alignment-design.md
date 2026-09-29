# Top-Venue Paper and Repository Alignment Design

Date: 2026-08-21

## 1. Objective

Align the three-paper release with the scientific presentation and release
engineering standards demonstrated by Flash3D, Gen3R, pixelSplat, DepthSplat,
and ViewCrafter, without changing experimental results or implying invalid
cross-protocol comparisons.

The release must communicate one coherent program:

> Use a generative prior selectively: first improve hard disocclusions, then
> distill the prior for speed, and finally learn when the prior is worth using.

## 2. Non-Goals

- Do not imitate another paper's visual identity or prose.
- Do not report quoted baseline numbers as head-to-head wins unless the protocol
  is identical.
- Do not hide negative results, protocol differences, or external dependencies.
- Do not add unverified metrics or estimate missing experiments.
- Do not change model outputs, checkpoints, or reported experiment values in
  this presentation-alignment pass.

## 3. Reference Patterns

### Flash3D

Adopt:

- one claim-level teaser;
- a left-to-right method figure with stable colors;
- experiment sections mapped directly to claims;
- separate tables for in-domain, cross-domain, stronger-input comparisons, and
  ablations;
- captions that state rows, columns, protocol, and what to inspect;
- README order: paper/demo, install, data, pretrained evaluation, training,
  citation.

### Gen3R

Adopt:

- one central representation/mechanism thesis;
- training-stage and inference-stage separation in the method figure;
- appearance, geometry, camera control, and reconstruction evidence separated
  into distinct sections;
- conceptual ablations against simpler pipelines;
- supplement ordered as implementation details, full comparisons, ablations,
  additional results, and failures;
- repository examples organized by task mode.

### pixelSplat / DepthSplat / ViewCrafter

Adopt:

- question-driven ablations paired with visual evidence;
- explicit geometry/depth/error visualizations;
- fixed crops and highlighted challenging regions;
- runtime/memory/input-view columns in main result tables;
- commands labeled by paper table/figure;
- model/checkpoint matrices and known limitations.

## 4. Scientific Claim and Protocol Policy

### 4.1 Protocol families must remain separate

Every result table must identify its protocol family:

1. **Official MINE / Flash3D single-view protocol**
   - 1 context image;
   - 256x384;
   - 5% border crop;
   - +5, +10, and U[-30,30] target buckets reported separately;
   - LPIPS-VGG.

2. **pixelSplat / DepthSplat two-view protocol**
   - 2 context images and 3 interpolated targets;
   - 256x256;
   - no border crop;
   - official evaluation index;
   - LPIPS-VGG.

3. **This repository's long-sequence disocclusion diagnostic**
   - 1 context image;
   - 49-frame sequence;
   - frames with significant disocclusion;
   - custom visibility partition and region analyses;
   - standard full-image PSNR/SSIM/LPIPS-VGG/FID/KID plus diagnostic region
     metrics.

4. **ACID cross-dataset mechanism diagnostic**
   - converted ACID camera trajectories;
   - VGGT-derived visibility;
   - same selective-injection mechanism;
   - mechanism/difficulty transfer, not an official ACID leaderboard claim.

### 4.2 Claim wording

Allowed:

- "On our fixed long-sequence disocclusion diagnostic, selective injection
  improves LPIPS/FID relative to its Gen3R and Flash3D components."
- "The difficulty-dependent mechanism transfers from RE10K to ACID without
  retuning."
- "At matched teacher-call budget, FrameNet outperforms random and observable
  ranking policies."

Not allowed:

- "We beat Flash3D on RE10K" without an official MINE-aligned reproduction.
- "We beat pixelSplat/MVSplat/DepthSplat" across different input-view and target
  protocols.
- "All three papers have ACID validation" when only Paper 1 contains that
  evidence.

### 4.3 Table metadata

Every external-comparison table must expose:

- input views;
- split/index;
- resolution and crop;
- target rule or frame gap;
- metric backbone (LPIPS-VGG);
- quoted vs reproduced status;
- checkpoint/paper version;
- test-time optimization status when applicable.

## 5. Shared Visual System

Use a consistent semantic palette across all papers and the repository:

- blue: geometry / observable evidence;
- violet: generative teacher;
- orange: trained student or reliability predictor;
- yellow: gate, mask, or highlighted ROI;
- green: final output.

Rules:

- vector PDF for papers, PNG for GitHub;
- fixed method order across qualitative figures;
- same crop coordinates across GT and all methods;
- yellow boxes only around diagnostic regions;
- error maps include a legend or explicit caption semantics;
- no decorative boxes on statistical plots;
- captions must specify task, conditioning, rows, columns, annotations, and the
  supported conclusion.

## 6. Paper 1 Design

### 6.1 Narrative

Central question:

> When does feed-forward geometric evidence improve a generative
> single-view reconstruction, and how can it be injected without harming
> already-correct regions?

The introduction and abstract should follow:

1. disocclusion is the core asymmetric failure;
2. feed-forward and generative components have complementary strengths;
3. asymmetric latent injection preserves visible regions;
4. reliability determines when injection should be used;
5. experiments establish the mechanism, selectivity, perceptual gain, and
   cross-dataset transfer.

Contributions are reduced to three non-overlapping claims:

1. conceptual: selective disocclusion refinement with an exact visible no-harm
   construction;
2. technical: asymmetric latent injection plus a reliability gate;
3. empirical: stable mechanism at scale, perceptual/component gains on a fixed
   diagnostic, and cross-dataset difficulty transfer.

### 6.2 Main figures

- **Figure 1 — teaser:** one input plus three obvious successful targets and one
  rejected failure; columns GT / Flash3D / Gen3R / Ours; fixed yellow ROIs and
  2-3x crops; one-line metric annotations.
- **Figure 2 — method:** two stages: evidence construction and test-time
  selective generation; visible branch visibly bypasses injection; gate/fallback
  explicitly shown.
- **Figure 3 — mechanism/difficulty:** mechanism scatter plus difficulty buckets,
  not two unrelated figures.
- **Figure 4 — qualitative diagnostic:** GT / components / Ours / visibility /
  error, with fixed crops and self-contained caption.
- **Figure 5 — failures/limitations:** accepted vs rejected scenes, temporal
  consistency limitation, and ACID transfer cases.

### 6.3 Main tables

- Table 1: protocol declaration and component comparison on the custom diagnostic.
- Table 2: main standard full-image metrics on the same rendered frames; all
  methods marked as components/variants, not official leaderboard methods.
- Table 3: reliability policies and threshold sweep.
- Table 4: cross-dataset mechanism transfer.
- Published official numbers move to a clearly labeled positioning table or
  supplementary protocol appendix.

## 7. Paper 2 Design

Central question:

> Can the useful part of a slow generative disocclusion prior be distilled into
> a feed-forward model without copying teacher failures?

Required presentation:

- teaser with GT / Flash3D / Teacher / Student and fixed crops;
- pipeline separates training-only teacher from inference-time student;
- main table combines quality, runtime, speedup, VRAM, and visible-region
  preservation;
- quality-speed Pareto plot;
- Gaussian color-adapter negative result paired with a visual failure example;
- explicit train/holdout scene split and checkpoint metadata.

The README must provide the included student checkpoint, a checksum, architecture,
input format, expected smoke output, and the evaluation command.

## 8. Paper 3 Design

Central question:

> What is the right granularity for deciding whether an expensive generative
> teacher will help?

Required presentation:

- teaser/routing strip showing consecutive frames, predicted probabilities,
  call/skip decisions, and oracle labels;
- pipeline from observable features to routing decision;
- one matched-budget routing table;
- ROC/PR and mean-vs-worst frontier;
- granularity table (scene/frame/patch) with sample count and leakage control;
- false-positive/false-negative examples with marked ROIs;
- grouped scene split emphasized in both paper and README.

## 9. Repository Design

Top-level order:

1. title and one-sentence thesis;
2. paper/project/PDF/checkpoint links;
3. one compact visual program overview;
4. three paper cards;
5. strongest teaser and qualitative evidence;
6. quickstart and smoke reproduction;
7. model/checkpoint matrix;
8. table/figure reproduction commands;
9. protocol declaration;
10. data/license/citation/known limitations.

Required new release artifacts:

- `LICENSE`;
- `CITATION.cff`;
- `PROTOCOLS.md`;
- `MODEL_ZOO.md` with checkpoint hash and source;
- `MANIFESTS/` containing scene IDs, seeds, protocol labels, and expected outputs;
- table/figure command index;
- smoke-test script covering JSON analysis and student checkpoint loading.

The homepage should not show all long qualitative panels inline. It should show
one hero comparison and link to per-paper galleries.

## 10. Error Handling and Integrity

- Missing external checkpoints produce actionable messages with URLs and expected
  paths.
- Scripts must not silently fall back to a different LPIPS backbone or protocol.
- Quoted baseline values include paper/table/version metadata.
- Reproduced values include checkpoint hash, manifest, and command.
- Custom diagnostics are labeled in filenames, captions, and tables.
- Figure-selection manifests are committed; visually selected examples are
  separated from pre-registered main-paper examples.

## 11. Verification Matrix

| Assertion | Verification |
|---|---|
| No invalid cross-protocol win claims | automated grep plus manual table audit |
| Every external table exposes protocol metadata | table-by-table checklist |
| Three papers compile cleanly | full LaTeX + BibTeX build, zero undefined refs |
| All main figures are readable and self-contained | PDF/PNG visual review |
| README links and commands work | link/file checks plus copied-command smoke run |
| Student checkpoint is valid | load state dict and run a synthetic forward pass |
| Analysis numbers remain unchanged | rerun released JSON analyses and compare |
| Release contains no private data or secrets | git diff, secret/path scan, file-size audit |
| Repository is synchronized | local and remote commit hashes match |

## 12. Implementation Order

1. Protocol and claim audit across all prose/tables.
2. Paper 1 teaser, method figure, main-table reorganization, and captions.
3. Paper 2 teaser/Pareto/negative visual and checkpoint documentation.
4. Paper 3 routing strip, matched-budget table, granularity and failure visuals.
5. Shared repository release engineering and project-page README.
6. Final compile, smoke tests, visual review, commit, and push.
