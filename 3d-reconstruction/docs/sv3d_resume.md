# RESUME — Paper A: source-anchored TEST-TIME Gaussian completion (LOCKED D032)

## LOCKED STRATEGY (2026-07-19, D032) — read this first
**Method = source-anchored TEST-TIME Gaussian completion** (per-scene optimization on top of frozen
Flash3D). GUARANTEED deliverable, proven to beat Flash3D:
- Standard MINE protocol (n=100, E017): overall +0.35 dB PSNR, +0.030 LPIPS, source +0.006 dB.
- Wide gaps (E014): +1.68 dB overall / +17.8 dB hidden / +0.056 LPIPS.
- **Leakage-free hold-out-view test (running)**: optimize hidden G on targets {1,2}, EVALUATE on unseen
  target {3}. Smoke (n=3): +5.5 dB hidden / +0.42 dB overall / +0.024 LPIPS on the UNSEEN view = genuine
  3D generalization, NOT target-fitting. This is the paper's core validity experiment.
- Feed-forward amortized version = DEMOTED (collapsed 5x: E006/8/9/E015/E016). Optional future work only.

## HONESTY FRAMING (non-negotiable for the paper)
The per-scene optimizer uses target-view RGB as its objective → NOT Flash3D's single-view protocol.
Frame HONESTLY as test-time optimization. The leakage-free claim = the hold-out-view protocol (optimize
on some views, evaluate on a DISJOINT view). That is the fair "improves generalization over Flash3D"
story. Do NOT claim single-forward-pass for this method. Differentiators vs UAR/One-Shot/Leveling3D/
Difix3D+: NO diffusion, hard source-null guarantee, region-causal deletion test.

## RUNNING JOBS
- holdout_wide_n100 (setsid nohup): leakage-free hold-out-view eval, n=100 wide700, optimize on {1,2}
  eval on {3}. Out: /home/data/sv3d-lab/evaluations/holdout_wide_n100/summary.json. DECISIVE.

## NEXT ACTIONS (in order)
1. Read holdout_wide_n100 result. If overall/LPIPS gain on the held-out view is positive across n=100 →
   leakage-free win confirmed → paper's core table.
2. Also run holdout on STANDARD protocol (test_files_present, holdout_frame 3) for the headline number.
3. Scale winning protocol toward full 620 test set (per-scene opt ~5-15s/scene; 620 feasible overnight).
4. Ablations: source-null on/off, w_hidden, steps, L2 vs LPIPS (E014c: LPIPS 2.2x LPIPS gain), holdout
   distance. Qualitative side-by-side (Flash3D vs ours on held-out view). Efficiency vs UAR.
5. Write paper (papers/paper_a/) + finalize code/README. Notify user.

## KEY FACTS / GOTCHAS
- CORE method code: `paper_a_explicit3d/oracle/hidden_opt_oracle.py` (per-scene hidden-Gaussian optimizer;
  --holdout_frame N = leakage-free protocol; --loss l2|lpips; --split; per-param LRs xyz2e-4/scale5e-3/
  rot1e-3/opa5e-2/rgb1e-2; source-null = MSE(merged_src,vis_src)+alpha penalty+projected clamp).
  Reuses oracle_core.render_gaussians_relpose (differentiable) + common/geometry/visibility.py masks +
  Flash3D UniDepth v1 for G_vis (frozen).
- Teachers cached (optional feed-forward work): /home/data/sv3d-lab/teachers/wide700_l2 (464 scenes,
  mean hidden deletion +16dB). gen_oracle_teacher.py + train_distill.py exist but distillation collapsed.
- render_predicted consumes ALREADY-ACTIVATED opacity/scale/color; differentiable.
- Splits: wide700 (572 sc, gaps 20/40/60), present/standard (620 sc / 3100 pairs, gaps ~5/10). Format
  `scene src t1 t2 t3`.
- Server env: cd /root/projects/flash3d && source .venv/bin/activate && export CUDA_HOME=/usr/local/cuda-11.8
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. 1x A6000 48GB.
- SSH -p 10244 root@10.44.6.60 drops often. LAUNCH long jobs with `setsid nohup ... & disown < /dev/null`
  (plain nohup died when launching SSH timed out). Author code locally + scp.
- Flash3D baseline: official ckpt tgt5 28.68 (E001b). Our method IMPROVES on top of it.

## Do-not-repeat
- No amortized feed-forward hidden head (collapsed 5x). No 2D inference diffusion refiner (D026 collision).
  No spatial branch. One variable per experiment; never lower a pre-registered threshold; frame test-time
  optimization honestly (no single-forward-pass claim).
