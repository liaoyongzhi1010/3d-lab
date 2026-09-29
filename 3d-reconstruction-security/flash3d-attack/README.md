# flash3d-attack

**Depth-Parallax Confusion (DPC)** — a source-camouflaged adversarial attack on feed-forward
single-image 3D Gaussian Splatting (3DGS). DPC keeps the *source* view looking correct while
progressively corrupting *novel* views as the camera moves.

Project page: **`docs/index.html`** (GitHub Pages) &middot; Paper draft: `paper/paper_dpc.md`

This folder is **self-contained**. It does not modify the `sv3d-lab` project; it only references
the upstream Flash3D repo and its official checkpoint at inference time (weights are never changed).

## The idea

Feed-forward single-image 3DGS is trained only on novel-view photometric loss. As a result, many
`(depth, appearance)` explanations of one source image are photometrically indistinguishable in the
source view and diverge only under parallax — the **depth-appearance ambiguity**. DPC weaponizes it
with an opposite-sign dual objective:

```
min_delta   lambda_src * || R_0(x+delta) - R_0(x) ||^2          # keep source view (camouflage)
          -  (1/M) sum_j || R_Pj(x+delta) - R_Pj(x) ||^2         # destroy novel views
s.t.        || delta ||_inf <= eps
```

- **Source-camouflaged** — invisible where a defender inspects, destructive where views are consumed.
- **Self-referential** — targets are the model's own clean renders; no ground-truth novel views needed.
- **Pose-agnostic** — optimized on auxiliary poses disjoint from the eval poses.
- **Not PGD / not a patch** — a geometry-specific objective, not a single-task imperceptible perturbation.

## Headline result

Flash3D / RealEstate10K, eps=8/255, lambda_src=3, 40 steps, n=100:

| View | Δ PSNR |
|---|---:|
| src (observed) | **−2.6** |
| tgt5  | −4.8 |
| tgt10 | −5.2 |
| tgt_rand | −5.0 |

Novel damage (~5.0 dB) is ~1.9× the source damage, rising monotonically with parallax. DPC is the
**only** method (vs. random noise and naive-PGD) whose novel damage exceeds its source damage.

## Layout

```
flash3d-attack/
├── attacks/
│   ├── dpc_attack.py             # core DPC optimizer + random / naive-PGD baselines (model-agnostic)
│   ├── dpc_blackbox.py           # query-only NES-over-DCT variant
│   ├── eval_dpc_flash3d.py       # unified eval on Flash3D (--method dpc|naive_pgd|random|blackbox)
│   ├── eval_depth_transfer.py    # shared UniDepth-prior corruption (cross-architecture evidence)
│   ├── eval_defense.py           # multi-view consistency detector
│   ├── make_dpc_qualitative.py   # clean / attacked / diff figure grids
│   ├── probe_gradient.py         # gradient-path feasibility probe
│   ├── summarize_results.py      # stdlib results table (no torch)
│   └── attack_transforms.py      # early CP-Drift / Depth-Cue baselines (non-core)
├── tests/                        # 13 unit tests
├── results/                      # metric json + qualitative pngs
├── paper/                        # paper_dpc.md, onepage.md
├── docs/                         # GitHub Pages project site (index.html + figs/)
└── METHOD_DPC.md
```

## Run (server, Flash3D venv)

```bash
cd /root/projects/flash3d && source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# unit tests
cd /root/flash3d-attack && python -m pytest tests -q

# main DPC evaluation
python attacks/eval_dpc_flash3d.py --method dpc --max_scenes 100 --steps 40 \
  --epsilon 0.0314 --lambda_src 3.0 --out results/main_dpc_n100.json

# baselines (same budget/optimizer)
python attacks/eval_dpc_flash3d.py --method random    --max_scenes 100 --out results/main_random_n100.json
python attacks/eval_dpc_flash3d.py --method naive_pgd --max_scenes 100 --out results/main_naivepgd_n100.json

# cross-architecture evidence (shared depth prior)
python attacks/eval_depth_transfer.py --max_scenes 40 --epsilon 0.0314 --out results/depth_transfer_n40.json

# summarize (runs anywhere, stdlib only)
python attacks/summarize_results.py
```

## Honesty

- Source camouflage is *relative* (novel ≫ source), not perfect invisibility (~2.6 dB source drop at eps=8/255).
- Cross-architecture shown via the shared UniDepth prior; a full second-model (CATSplat) table is blocked by
  missing public inference assets (per-scene LLaVA features), stated rather than worked around.
- Black-box is validated on synthetic renderers + unit tests; full-Flash3D black-box is compute-bound.
- All results report n; small-n ablations are labeled. Baseline weights are never modified.
