# Top-Venue Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the three papers and repository to match top-venue standards for protocol hygiene, evidence hierarchy, figure readability, narrative coherence, and public reproducibility without changing experimental results.

**Architecture:** Four isolated work streams are integrated at the end. The release stream owns only top-level protocol/reproducibility metadata. Paper streams own only their paper directory and implement claim-specific narrative, figures, captions, and per-paper commands. A final integration pass audits all claims and rebuilds every PDF.

**Tech Stack:** LaTeX, TikZ, Python 3, NumPy, Pillow, Matplotlib, Git/GitHub, Markdown.

---

## File Ownership Map

### Release/protocol stream

- Modify: `README.md`
- Modify: `REPRODUCIBILITY.md`
- Modify: `requirements.txt`
- Create: `PROTOCOLS.md`
- Create: `MODEL_ZOO.md`
- Create: `TABLES_AND_FIGURES.md`
- Create: `LICENSE`
- Create: `CITATION.cff`
- Create: `manifests/README.md`
- Create: `manifests/re10k_scene_manifest.json`
- Create: `manifests/frame_reliability_manifest.json`
- Create: `scripts/smoke_test.py`

### Paper 1 stream

- Modify: `paper1_selective_generation/README.md`
- Modify: `paper1_selective_generation/paper/README.md`
- Modify: `paper1_selective_generation/paper/main.tex`
- Modify: `paper1_selective_generation/paper/paper1_full_writeup.md`
- Modify: `paper1_selective_generation/paper/figs/fig_pipeline.tex`
- Create: `paper1_selective_generation/paper/figs/fig_teaser_aligned.png`
- Create: `paper1_selective_generation/paper/figs/fig_mechanism_combined.pdf`
- Create: `paper1_selective_generation/scripts/e219_topvenue_teaser.py`
- Create: `paper1_selective_generation/scripts/e220_protocol_table.py`
- Create: `paper1_selective_generation/PROTOCOL.md`

### Paper 2 stream

- Modify: `paper2_gate_aware_distillation/README.md`
- Modify: `paper2_gate_aware_distillation/paper/README.md`
- Modify: `paper2_gate_aware_distillation/paper/main.tex`
- Modify: `paper2_gate_aware_distillation/paper/paper2_full_writeup.md`
- Modify: `paper2_gate_aware_distillation/paper/figs/fig_pipeline.tex`
- Create: `paper2_gate_aware_distillation/MODEL_CARD.md`
- Create: `paper2_gate_aware_distillation/scripts/check_checkpoint.py`
- Create: `paper2_gate_aware_distillation/paper/figs/fig_quality_speed.pdf`

### Paper 3 stream

- Modify: `paper3_reliability_learning/README.md`
- Modify: `paper3_reliability_learning/paper/README.md`
- Modify: `paper3_reliability_learning/paper/main.tex`
- Modify: `paper3_reliability_learning/paper/paper3_full_writeup.md`
- Modify: `paper3_reliability_learning/paper/figs/fig_pipeline.tex`
- Create: `paper3_reliability_learning/paper/figs/fig_granularity.pdf`
- Create: `paper3_reliability_learning/paper/figs/fig_routing_strip.pdf`
- Create: `paper3_reliability_learning/scripts/e221_granularity_figure.py`
- Create: `paper3_reliability_learning/scripts/e222_routing_strip.py`
- Create: `paper3_reliability_learning/PROTOCOL.md`

---

### Task 1: Freeze protocol and claim boundaries

**Files:**
- Create: `PROTOCOLS.md`
- Modify: `README.md`
- Modify: `REPRODUCIBILITY.md`

- [ ] **Step 1: Create a protocol matrix**

Document four separate protocol families with exact views, split, resolution,
crop, target rule, LPIPS backbone, and comparison status:

```markdown
| Protocol | Views | Split | Resolution/crop | Targets | Use |
|---|---:|---|---|---|---|
| MINE/Flash3D | 1 | 3204 samples / 641 scenes | 256x384, 5% crop | +5/+10/U[-30,30] | quoted positioning only |
| pixelSplat/DepthSplat | 2 | evaluation_index_re10k.json | 256x256, no crop | 3 interpolated | quoted positioning only |
| Long-sequence diagnostic | 1 | released N=166 manifest | 560 working render / reported metric preprocessing | 49-frame, disocclusion-heavy | reproduced component study |
| ACID mechanism diagnostic | 1 | released 10-scene manifest | converted ACID trajectory | long trajectory | cross-dataset mechanism only |
```

- [ ] **Step 2: Correct top-level claims**

Replace unqualified wording such as:

```text
beats the published Flash3D
```

with:

```text
on the fixed long-sequence diagnostic, has lower LPIPS/FID than the Flash3D evidence component
```

Also replace “All experiments ... with ACID validation” with a statement that
Paper 1 alone contains ACID transfer evidence.

- [ ] **Step 3: Add prominent protocol links**

Add a “Protocol declaration” link near the first quantitative claims in
`README.md` and `REPRODUCIBILITY.md`.

- [ ] **Step 4: Verify no invalid claim remains**

Run:

```bash
rg -n "beats? (the )?(published )?(Flash3D|pixelSplat|MVSplat|DepthSplat)|All experiments.*ACID" README.md REPRODUCIBILITY.md paper*/README.md paper*/paper/main.tex
```

Expected: no unqualified cross-protocol win claim.

- [ ] **Step 5: Commit**

```bash
git add PROTOCOLS.md README.md REPRODUCIBILITY.md
git commit -m "docs: clarify evaluation protocols and claims"
```

### Task 2: Add release metadata and manifests

**Files:**
- Modify: `requirements.txt`
- Create: `LICENSE`
- Create: `CITATION.cff`
- Create: `MODEL_ZOO.md`
- Create: `TABLES_AND_FIGURES.md`
- Create: `manifests/README.md`
- Create: `manifests/re10k_scene_manifest.json`
- Create: `manifests/frame_reliability_manifest.json`

- [ ] **Step 0: Declare metadata validation dependency**

Add the following explicit dependency to `requirements.txt`:

```text
PyYAML>=6.0
```

- [ ] **Step 1: Add MIT license and citation metadata**

Use an MIT license for original release code and a CFF entry titled:

```yaml
title: "Selective Generation, Distillation, and Reliability for Single-View Scene Reconstruction"
type: software
version: 0.1.0
repository-code: "https://github.com/liaoyongzhi1010/3d"
license: MIT
```

- [ ] **Step 2: Add model/checkpoint matrix**

Document the included `TinyStudent` checkpoint and external dependencies:

```markdown
| Component | Included | Source | Hash | Role |
| TinyStudent | yes | this repository | SHA256 | Paper 2 inference |
| Gen3R | no | official repository/checkpoint | external | generative backbone |
| Flash3D | no | official repository/checkpoint | external | geometry evidence |
| VGGT | no | official repository/checkpoint | external | visibility preprocessing |
```

Calculate the TinyStudent SHA256 with:

```bash
shasum -a 256 paper2_gate_aware_distillation/checkpoints/student.pt
```

- [ ] **Step 3: Export manifests from released JSONs**

`re10k_scene_manifest.json` contains sorted scene identifiers and source split
from `paper1_selective_generation/results/E-142_combined_N169.json`.

`frame_reliability_manifest.json` contains sorted unique scene identifiers,
frame count, feature version, and grouping rule from
`paper3_reliability_learning/results/E-149_frames_b5678_aug.json`.

- [ ] **Step 4: Index exact table and figure commands**

Map every paper table/figure to script, input JSON, output artifact, and whether
it requires external renders.

- [ ] **Step 5: Validate metadata**

Run:

```bash
python3 -m json.tool manifests/re10k_scene_manifest.json >/dev/null
python3 -m json.tool manifests/frame_reliability_manifest.json >/dev/null
python3 - <<'PY'
import yaml
for path in ['CITATION.cff']:
    yaml.safe_load(open(path))
print('metadata ok')
PY
```

Expected: `metadata ok`.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt LICENSE CITATION.cff MODEL_ZOO.md TABLES_AND_FIGURES.md manifests
git commit -m "docs: add release metadata and manifests"
```

### Task 3: Build a repository smoke test

**Files:**
- Create: `scripts/smoke_test.py`
- Modify: `README.md`
- Modify: `REPRODUCIBILITY.md`

- [ ] **Step 1: Implement smoke assertions**

The script must:

```python
# 1. load all released JSON files used by headline results;
# 2. assert N=166 scenes and N=2898 frames;
# 3. load TinyStudent checkpoint into TinyStudent;
# 4. run a synthetic [1,8,64,64] forward pass and assert [1,3,64,64];
# 5. verify every README-linked image/PDF exists;
# 6. print expected headline values and "SMOKE PASS".
```

- [ ] **Step 2: Run the smoke test**

Run:

```bash
python3 scripts/smoke_test.py
```

Expected final line: `SMOKE PASS`.

- [ ] **Step 3: Document the command**

Place the smoke command before full environment instructions in the top-level
README.

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke_test.py README.md REPRODUCIBILITY.md
git commit -m "test: add release smoke test"
```

### Task 4: Reframe Paper 1 around one causal thesis

**Files:**
- Modify: `paper1_selective_generation/paper/main.tex`
- Modify: `paper1_selective_generation/paper/paper1_full_writeup.md`
- Modify: `paper1_selective_generation/README.md`
- Create: `paper1_selective_generation/PROTOCOL.md`

- [ ] **Step 1: Rewrite abstract structure**

Use five sentences/blocks:

1. asymmetric visible/disoccluded problem;
2. complementary feed-forward/generative components;
3. asymmetric injection and fallback gate;
4. mechanism and scale evidence;
5. perceptual diagnostic plus cross-dataset mechanism transfer.

Explicitly label the 759-frame comparison as a fixed long-sequence component
diagnostic.

- [ ] **Step 2: Reduce contributions to three**

Use:

```text
Concept: selective disocclusion refinement with exact visible no-harm.
Mechanism: asymmetric latent injection plus reliability fallback.
Evidence: scale-stable mechanism, fixed-diagnostic perceptual improvement, and ACID transfer.
```

- [ ] **Step 3: Reorder experiments by claims**

Order:

1. Protocol and implementation;
2. Main component diagnostic;
3. Reliability/gate ablation;
4. Mechanism/difficulty/scale;
5. ACID transfer;
6. qualitative successes and failures;
7. limitations.

Move quoted official baseline numbers into a positioning subsection with full
protocol metadata.

- [ ] **Step 4: Synchronize Markdown and README**

The Markdown write-up and per-paper README must use identical claim wording and
protocol labels.

- [ ] **Step 5: Compile and audit**

Run:

```bash
cd paper1_selective_generation/paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
grep -ci undefined main.log
```

Expected: exit 0 and `0` undefined references.

- [ ] **Step 6: Commit**

```bash
git add paper1_selective_generation
git commit -m "docs(paper1): align narrative and protocol framing"
```

### Task 5: Upgrade Paper 1 teaser and method figure

**Files:**
- Create: `paper1_selective_generation/scripts/e219_topvenue_teaser.py`
- Create: `paper1_selective_generation/paper/figs/fig_teaser_aligned.png`
- Modify: `paper1_selective_generation/paper/figs/fig_pipeline.tex`
- Modify: `paper1_selective_generation/paper/main.tex`

- [ ] **Step 1: Generate fixed-layout teaser**

Use the preselected visual manifest to create:

```text
rows: input/task, GT, Flash3D/geometry reference, Gen3R, Ours
columns: window recovery, doorway recovery, sofa/window recovery, gate-rejected failure
```

Each success column uses one fixed yellow ROI and zoom inset; the failure column
is labeled `Gate rejects` and shows baseline fallback.

- [ ] **Step 2: Redraw method figure as two stages**

Left stage: source image/cameras → Flash3D evidence + VGGT visibility.

Right stage: Gen3R baseline → asymmetric injection in invisible region →
reliability gate → injected output or baseline fallback.

Show visible pixels bypassing injection.

- [ ] **Step 3: Replace teaser and update caption**

The caption must define rows, columns, yellow boxes, gate behavior, and supported
conclusion without relying on body text.

- [ ] **Step 4: Visual and build verification**

Convert the relevant PDF page to PNG and inspect text/crop readability at 100%.
Compile Paper 1 and confirm nine or fewer main-paper pages before references when
using the target template; the standalone draft may differ.

- [ ] **Step 5: Commit**

```bash
git add paper1_selective_generation/paper paper1_selective_generation/scripts/e219_topvenue_teaser.py
git commit -m "docs(paper1): upgrade teaser and method figure"
```

### Task 6: Consolidate Paper 1 mechanism and protocol tables

**Files:**
- Create: `paper1_selective_generation/scripts/e220_protocol_table.py`
- Create: `paper1_selective_generation/paper/figs/fig_mechanism_combined.pdf`
- Modify: `paper1_selective_generation/paper/main.tex`

- [ ] **Step 1: Generate combined mechanism figure**

Combine mechanism scatter, difficulty buckets, and gate operating points in a
single multi-panel vector figure with shared typography and panel labels `(a-c)`.

- [ ] **Step 2: Add protocol-aware main table**

The component diagnostic table includes:

```text
method/variant, views, frame policy, resolution, metric type, PSNR, SSIM,
LPIPS-VGG, FID, runtime, quoted/reproduced
```

Do not mix official MINE or two-view numbers into this table.

- [ ] **Step 3: Move official positioning table**

Keep official pixelSplat/MVSplat/DepthSplat/Flash3D values in a separate table
with protocol metadata and no bolding against Ours.

- [ ] **Step 4: Rebuild and verify unchanged values**

Rerun:

```bash
python3 paper1_selective_generation/scripts/e203_bootstrap_ci.py --data paper1_selective_generation/results/E-142_combined_N169.json --out /tmp/e203.json
python3 paper1_selective_generation/scripts/e216_gate_sensitivity.py --data paper1_selective_generation/results/E-142_combined_N169.json --out /tmp/e216.json
```

Expected mechanism `r=0.982` and default gate mean `+1.658 dB`.

- [ ] **Step 5: Commit**

```bash
git add paper1_selective_generation
git commit -m "docs(paper1): consolidate evidence and tables"
```

### Task 7: Align Paper 2 narrative, visuals, and model card

**Files:**
- Modify: `paper2_gate_aware_distillation/paper/main.tex`
- Modify: `paper2_gate_aware_distillation/paper/paper2_full_writeup.md`
- Modify: `paper2_gate_aware_distillation/README.md`
- Modify: `paper2_gate_aware_distillation/paper/figs/fig_pipeline.tex`
- Create: `paper2_gate_aware_distillation/MODEL_CARD.md`
- Create: `paper2_gate_aware_distillation/scripts/check_checkpoint.py`
- Create: `paper2_gate_aware_distillation/paper/figs/fig_quality_speed.pdf`

- [ ] **Step 1: Reframe central question**

Use the thesis:

```text
Distill only the useful portion of a non-monotonic teacher, rather than copying
all teacher behavior.
```

- [ ] **Step 2: Correct Flash3D comparison wording**

Describe Flash3D as the geometry/evidence component evaluated on the same custom
diagnostic, not an official benchmark opponent.

- [ ] **Step 3: Add quality-speed Pareto figure**

Plot runtime versus LPIPS/FID for Gen3R, Flash3D evidence, teacher, and student.
Use distinct annotations for slow teacher and feed-forward student.

- [ ] **Step 4: Strengthen negative ablation**

Pair the Gaussian color-adapter result with a schematic or existing visual
showing why source-plane recoloring has no support in newly disoccluded pixels.

- [ ] **Step 5: Add model card and checkpoint checker**

`check_checkpoint.py` loads the checkpoint, verifies 46,371 parameters, performs
a synthetic forward pass, and prints SHA256.

- [ ] **Step 6: Compile and verify**

Expected: zero undefined references; checkpoint checker prints output shape
`(1, 3, 64, 64)`.

- [ ] **Step 7: Commit**

```bash
git add paper2_gate_aware_distillation
git commit -m "docs(paper2): align distillation evidence and release"
```

### Task 8: Align Paper 3 granularity and routing evidence

**Files:**
- Modify: `paper3_reliability_learning/paper/main.tex`
- Modify: `paper3_reliability_learning/paper/paper3_full_writeup.md`
- Modify: `paper3_reliability_learning/README.md`
- Modify: `paper3_reliability_learning/paper/figs/fig_pipeline.tex`
- Create: `paper3_reliability_learning/PROTOCOL.md`
- Create: `paper3_reliability_learning/scripts/e221_granularity_figure.py`
- Create: `paper3_reliability_learning/scripts/e222_routing_strip.py`
- Create: `paper3_reliability_learning/paper/figs/fig_granularity.pdf`
- Create: `paper3_reliability_learning/paper/figs/fig_routing_strip.pdf`

- [ ] **Step 1: Reframe around granularity**

Use the thesis:

```text
Reliability becomes learnable at frame granularity; scene is too coarse and
patch is too weakly observed.
```

- [ ] **Step 2: Create granularity figure**

Plot scene/frame/patch sample count, ROC-AUC, and retained oracle gain with clear
labels and grouped-CV note.

- [ ] **Step 3: Create routing decision strip**

Use released frame rows to show an illustrative ordered sequence of predicted
score, call/skip decision, true gain, and oracle label. Mark examples as
illustrative analysis, not image-level qualitative evidence.

- [ ] **Step 4: Emphasize matched-budget comparison**

Keep random, always, visible-gap top-K, FrameNet, and oracle in one table at the
same 59% call budget where applicable.

- [ ] **Step 5: Document leakage control**

State grouped leave-one-scene-out evaluation in abstract/experiments/README and
list the released scene manifest.

- [ ] **Step 6: Rebuild and verify**

Run the released scene/frame scripts and assert:

```text
scene N=166
frame N=2898
ROC-AUC=0.947
PR-AUC=0.961
FrameNet mean=+1.604 dB
saved=41%
```

- [ ] **Step 7: Commit**

```bash
git add paper3_reliability_learning
git commit -m "docs(paper3): align granularity and routing evidence"
```

### Task 9: Rebuild the top-level project page

**Files:**
- Modify: `README.md`
- Modify: `REPRODUCIBILITY.md`
- Modify: `TABLES_AND_FIGURES.md`

- [ ] **Step 1: Reorder homepage**

Order:

1. thesis and links;
2. compact program overview;
3. one hero teaser;
4. three paper cards;
5. quick smoke test;
6. checkpoint/model matrix;
7. protocol declaration;
8. reproduction index;
9. limitations/license/citation.

Move long qualitative images out of the first viewport into per-paper galleries.

- [ ] **Step 2: Add exact commands**

Add one command for smoke test and links to table/figure commands.

- [ ] **Step 3: Add expected output**

The README includes:

```text
SMOKE PASS
Paper1 scenes: 166
Paper3 frames: 2898
TinyStudent output: (1, 3, 64, 64)
```

- [ ] **Step 4: Verify links**

Run `scripts/smoke_test.py` and a local Markdown link/file checker.

- [ ] **Step 5: Commit**

```bash
git add README.md REPRODUCIBILITY.md TABLES_AND_FIGURES.md
git commit -m "docs: align project page with paper evidence"
```

### Task 10: Final integration and push

**Files:**
- Verify all modified and generated files.

- [ ] **Step 1: Build all pipeline figures**

Run `pdflatex` in each `paper/figs` directory for `fig_pipeline.tex`.

- [ ] **Step 2: Build all papers**

Run full LaTeX + BibTeX builds and assert zero undefined references.

- [ ] **Step 3: Run Python checks**

```bash
python3 -m py_compile paper1_selective_generation/scripts/*.py paper2_gate_aware_distillation/scripts/*.py paper3_reliability_learning/scripts/*.py scripts/*.py
python3 scripts/smoke_test.py
```

- [ ] **Step 4: Audit claims and protocol metadata**

Run the invalid-claim grep from Task 1 and manually inspect all external
comparison tables.

- [ ] **Step 5: Audit release safety**

```bash
git diff --check
find . -type f -size +90M -print
find . -name '*.env' -o -iname '*secret*' -o -iname '*credential*'
```

Expected: no output for oversized or sensitive files.

- [ ] **Step 6: Push and verify**

```bash
git push origin main
git status --short --branch
git rev-list --left-right --count origin/main...main
```

Expected: clean tree and `0 0` divergence.
