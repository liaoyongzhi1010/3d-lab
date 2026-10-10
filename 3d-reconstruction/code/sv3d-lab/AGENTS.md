# AGENTS.md — sv3d-lab operating guide

## What this project is
Clean-room single-view 3D reconstruction/completion research → two method papers.
Read `research_charter.md` (binding rules) before doing anything. This file = practical how-to.

## Golden rules
1. Single-view inference only. No target leakage, ever. Paper A currently has `tests_cvae_leakage.py`; the broader documented `tests/leakage/` suite is not present in this checkout.
2. Legacy `va-mfgc` / `VAMFGC_*` = read-only. Never copy trainers/heads/losses/thresholds.
3. One major variable per experiment. Log it in `decision_log.md` BEFORE running.
4. Oracles/teachers are upper bounds, never reported as model results.
5. Every new recorded run must add a row to `experiment_registry.csv` with git commit + config + paths. The historical registry is missing from this checkout and must not be inferred.
6. Reproduce a baseline number before claiming any improvement.
7. Never lower a threshold / edit a mask / change protocol to pass a gate.

## Server
- SSH: `ssh -p 10244 root@10.44.6.60` (drops often; use ConnectTimeout, run long jobs via nohup).
- GPU: 1× RTX A6000 48GB, dedicated. Log utilization; don't run no-info-gain jobs.
- Code mirror: `~/sv3d-lab/` (keep at same git commit as local). Data/runs: `/home/data/sv3d-lab/`.
- Flash3D upstream (baseline + RE10K reader + official ckpt): `/root/projects/flash3d/`
  venv `/root/projects/flash3d/.venv`, ckpt `checkpoints/model_re10k_v2.pth`,
  MINE split `splits/re10k_mine_filtered/test_files.txt`.
- RE10K: full train `/home/data/RealEstate10K_full/frames/train` (69,272 scenes),
  meta `/home/data/RealEstate10K/RealEstate10K/`.

## Code authoring rules
- Write files locally with the editor, `scp` to server; never heredoc Python over SSH (quote corruption).
- Local Python is 3.9 (authoring only); real runs use the server venv (torch/CUDA).
- LSP errors about torch/PIL/diffusers dynamic attrs are false positives locally (no stubs); the
  server venv runs fine. Legacy `va-mfgc` LSP errors are irrelevant.
- Environment for server runs:
  `cd /root/projects/flash3d && source .venv/bin/activate && export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`

## Experiment protocol
- Before: decision_log entry (hypothesis/change/fixed/threshold/budget).
- Smoke (MICRO) first, then the real run. Long runs: nohup + checkpoint + a run-local resume note.
- After: registry row + PASS/FAIL + next step. Personally inspect any PNG you cite.
- The historical `experiment_registry.csv` is absent from this checkout. Create it before recording new runs; do not reconstruct old rows from memory.

## Files that matter
- `research_charter.md` binding rules · `decision_log.md` running log
- `literature_matrix.md` competitors · top-level `data/provenance/` historical lessons
- `paper_a_explicit3d/CHARTER.md`, `paper_b_generative3d/CHARTER.md` (frozen after collision check)
