# sv3d-lab

Clean-room research project: **single-view 3D reconstruction / 3D scene completion**.
Target: two submission-ready method papers (CCF-B or above).

- Paper A (`paper_a_explicit3d/`): explicit-3D single-view wide-baseline completion.
- Paper B (`paper_b_generative3d/`): 3D-grounded generative hidden-appearance completion.

## Status
Bootstrapped 2026-07-18. See `decision_log.md` for the running log and `research_charter.md`
for the binding rules. The operating guide requires an `experiment_registry.csv`, but no registry
file is present in this checkout; historical runs therefore require manual recovery from logs.

## Hard task definition (never violated)
- Inference input = **single source RGB** (+ source intrinsics + canonical/virtual poses derived
  only from source). NO target RGB / target depth / target latent / target feature at inference.
- Output = unified renderable **3D Gaussian** scene. Target data is training supervision only.
- Deleting hidden Gaussians must cause a significant, attributable change (no fake 3D).

## Layout
- `common/` shared data, geometry, metrics and training utilities
- `eval/` unified evaluation harness
- `paper_a_explicit3d/`, `paper_b_generative3d/` per-paper method code
- `scripts/` data preparation and setup utilities
- `tests_cvae_leakage.py` under Paper A is the only dedicated leakage test file currently present

## Environments
- Local (this repo): macOS arm64, code authoring + orchestration.
- Server: `ssh -p 10244 root@10.44.6.60`, 1× RTX A6000 (48GB), CUDA 11.8.
  Server code mirror: `~/sv3d-lab/`. Data/runs: `/home/data/sv3d-lab/`.
- Local and server must sit on the same git commit for any recorded run.

## Legacy
Old project `va-mfgc` / `/home/data/VAMFGC_*` is a server-side read-only archive. Its referenced
`docs/legacy_lessons.md` is not present in this checkout; retained lessons are summarized in
`decision_log.md` and the top-level `data/provenance/` archive.
