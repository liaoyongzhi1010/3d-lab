# sv3d-lab

Clean-room research project: **single-view 3D reconstruction / 3D scene completion**.
Target: two submission-ready method papers (CCF-B or above).

- Paper A (`paper_a_explicit3d/`): explicit-3D single-view wide-baseline completion.
- Paper B (`paper_b_generative3d/`): 3D-grounded generative hidden-appearance completion.

## Status
Bootstrapped 2026-07-18. See `decision_log.md` for the running log, `research_charter.md`
for the binding rules, `experiment_registry.csv` for every run.

## Hard task definition (never violated)
- Inference input = **single source RGB** (+ source intrinsics + canonical/virtual poses derived
  only from source). NO target RGB / target depth / target latent / target feature at inference.
- Output = unified renderable **3D Gaussian** scene. Target data is training supervision only.
- Deleting hidden Gaussians must cause a significant, attributable change (no fake 3D).

## Layout
- `common/` shared data/geometry/rendering/metrics/training code
- `tests/` correctness unit tests (camera/renderer/leakage/gradients/dataset)
- `baselines/` official baseline reproduction (Flash3D first)
- `eval/` unified evaluation harness
- `paper_a_explicit3d/`, `paper_b_generative3d/` per-paper method code
- `configs/`, `scripts/`, `docs/`, `papers/`

## Environments
- Local (this repo): macOS arm64, code authoring + orchestration.
- Server: `ssh -p 10244 root@10.44.6.60`, 1× RTX A6000 (48GB), CUDA 11.8.
  Server code mirror: `~/sv3d-lab/`. Data/runs: `/home/data/sv3d-lab/`.
- Local and server must sit on the same git commit for any recorded run.

## Legacy
Old project `va-mfgc` / `/home/data/VAMFGC_*` is **read-only archive**. See `docs/legacy_lessons.md`.
Nothing from it is reused except the whitelist in `research_charter.md`.
