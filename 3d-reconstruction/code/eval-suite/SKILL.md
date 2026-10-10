---
name: sv3d-eval-suite
description: Use when evaluating any single-view 3D reconstruction / novel-view-synthesis method (Flash3D, CATSplat, pixelSplat, or our own) on RealEstate10K and needing numbers that align with published papers and are cross-method comparable. Covers the unified official-metric evaluator, the export format contract, and the reliability guardrails.
tags: [evaluation, 3d-reconstruction, nvs, re10k]
version: "1.0.0"
---

# SV3D-Eval-Suite: 统一评测工程使用指南

一句话：**任何单视图重建/新视角合成方法，只要能导出目标视角渲染图，就用这套工程测出可对齐顶会论文、且与竞品可比的指标。**

工程根目录（服务器）：`/root/projects/sv3d-eval-suite/`（或本机 `/Users/bytedance/3d/sv3d-eval-suite/`）。
运行环境：`/root/projects/flash3d/.venv`（含 torch/hydra/pointnet/cleanfid/DISTS_pytorch）。

## 核心原则（不可违背）

1. **只调官方 evaluator 与顶会公认指标，绝不自写 metric、不自造评测协议。**
   - 重建精度 = Flash3D/CATSplat 官方 `evaluation/evaluator.py`（PSNR/SSIM/LPIPS-VGG + 5% crop）。
   - 生成质量 = cleanfid FID/KID + DISTS_pytorch（latentSplat 同款）。
2. **主表用官方 MINE split**（`test_files_present.txt`，对齐论文）；wide700 只作附录压力测试。
3. **跨方法必须同 split / 同源帧 / 同目标帧 / 同 GT**，否则数字不可比。

## 统一导出格式（所有方法都遵守）

```
<export_dir>/<scene>/pred/00X.png   # 该方法渲染的目标视角
<export_dir>/<scene>/gt/00X.png     # 对应真值
# 000=source(自还原), 001=tgt5, 002=tgt10, 003=tgt_rand
# target_frame_ids=[1,2,3]
```

## 评测一个方法：三步

### 步骤 1 — 出重建精度主表（对齐论文，推荐直接用官方 evaluate.py 的 metrics JSON）

竞品各自跑官方 `evaluate.py`，它按每行累积平均输出 `metrics_re10k_test_*.json`（PSNR/SSIM/LPIPS，分 src/tgt5/tgt10/tgt_rand）。**不依赖导图**（present split 每场景 5 行，导图按 scene 命名会覆盖，但官方指标按行累积不受影响）。

关键：避开官方 evaluate.py 写死的 `os.chdir(output_dir)` 导致的 checkpoint 静默不加载坑：
- Flash3D：`hydra.job.chdir=false` + run_dir 建 `ln -sfn .../checkpoints`。
- CATSplat：`run.checkpoint=绝对路径` + `dataset.data_path=/home/data/RealEstate10K` + pointnet 用绝对路径（见 `patches/catsplat_unidepth_encoder.py`）。

### 步骤 2 — 出生成质量（FID/KID/DISTS，需要图）

用 `patches/` 里打过补丁的 evaluate.py（`SKIP_PLY=1` 跳过 ply，`FLASH3D_VIS_DIR`/`CATSPLAT_VIS_DIR` 指定导图目录），`++eval.save_vis=true` 导图，再：

```bash
python metrics/eval_genquality.py --root <export_dir> --method <名字> \
  --out results/<名字>_genq.json --workdir /tmp/_genq/<名字>
```

### 步骤 3 — 编表

```bash
python metrics/eval_recon.py --root <export_dir> --method <名字> --out results/<名字>_recon.json   # 若走导图路线
python metrics/compile_table.py --recon a=... b=... --genq a=... b=... --out results/COMPARISON.md
```

## 评测我们自己的新方法

写推理脚本：对 MINE split 每行，用新方法渲染 src/tgt5/tgt10/tgt_rand 并存成统一格式（GT 用 dataloader 的 `("color",f_id,0)`，与竞品同一张真值）。然后走上面步骤 1-3。`eval_recon.py` 直接调 Flash3D 官方 evaluator，所以我们和竞品口径一致。

## 出表前必过的可比性自检

- 源视图 src 自还原 PSNR **~36-39dB**（每个方法都要过；低于 35dB = 权重没加载/管线错，数字作废）。
- 所有方法**同一 split 文件**（md5 一致）。
- 抽样核对不同方法**同场景同帧 GT 逐像素一致**（GT-MAE≈0）；用官方 evaluate.py 的 metrics JSON 出主表则天然一致。
- LPIPS 用 VGG；FID 用 cleanfid（不用 torchmetrics FID，会联网卡死）。

## 已知坑（都已固化进 patches/README）

1. **os.chdir 坑**：官方 evaluate.py `main()` 写死 `os.chdir(output_dir)`，破坏所有相对路径（checkpoints/pointnet/data）→ 权重静默不加载 → src 仅 ~11dB。修复见步骤 1。
2. **present split 每场景 5 行**：导图按 scene 命名互相覆盖 → 主表用官方 metrics JSON，不靠导图。
3. **FID**：torchmetrics FID 联网下权重卡死 → 用 cleanfid + 本地 `/tmp/inception-2015-12-05.pt`。
4. **后台任务勿用 `pkill -f <脚本名>`**（会误杀自己，脚本名含关键词）。
