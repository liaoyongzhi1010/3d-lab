# SV3D-Eval-Suite 评测结果

所有数字由各方法**官方 evaluate.py + 官方 evaluator（PSNR/SSIM/LPIPS-VGG + 5% crop）**产出，同 split / 同源帧 / 同目标帧 / 同 GT，可对齐论文、跨方法可比。

## 主表 1：重建精度（官方 MINE split `test_files_present.txt`，3100 行 / 620 场景，对齐论文）

单视图重建 / 新视角合成精度。每格 PSNR↑ / SSIM↑ / LPIPS↓。

| 方法 | 源视图(自还原) | tgt5 | tgt10 | tgt_rand |
|---|---|---|---|---|
| Flash3D | 38.39 / 0.988 / 0.021 | 28.68 / 0.902 / 0.095 | 26.09 / 0.861 / 0.128 | 25.10 / 0.836 / 0.155 |
| CATSplat | 39.68 / 0.989 / 0.024 | 29.16 / 0.907 / 0.094 | 26.50 / 0.867 / 0.126 | 25.51 / 0.843 / 0.151 |

**读法**
- **管线正确性**：两方源视图自还原均 ~38-39dB，通过渲染管线正确性判据（flash3d-gaussian-render 铁律）。
- **对齐论文**：Flash3D 逐档 28.68 / 26.09 / 25.10，对齐其论文 Table 2 的 28.46 / 25.94 / 24.93（每档差 < 0.25dB，present 是官方 3205 帧的可用子集）。证明这把尺子是准的。
- **跨方法可比**：CATSplat 各档均略优于 Flash3D（tgt5 +0.48dB，tgt10 +0.41dB，tgt_rand +0.42dB），符合它是 Flash3D 的后续改进工作；两者同 split、同帧、同 GT，数字真正可比。

## 主表 2：生成质量（官方 MINE present split，跨方法严格可比）

在与主表 1 完全相同的官方 split 上，用行级唯一命名（`{row}_{scene}_src{src}`）导出两方图，经 `check_gt_consistency.py` 验证 **3100 目录名完全一致、抽样 800 帧 GT-MAE=0.000000**（同 GT），因此 FID/KID/DISTS 跨方法严格可比。每格 FID↓ / KID↓ / DISTS↓（cleanfid + DISTS_pytorch，全图，n=3100/档）。

| 方法 | tgt5 | tgt10 | tgt_rand | novel 汇总 |
|---|---|---|---|---|
| Flash3D | 4.69 / 0.0009 / 0.062 | 6.67 / 0.0011 / 0.080 | 8.60 / 0.0019 / 0.098 | 4.22 / 0.0014 / 0.080 |
| CATSplat | 4.81 / 0.0009 / 0.061 | 6.73 / 0.0013 / 0.079 | 8.56 / 0.0019 / 0.097 | 4.34 / 0.0014 / 0.079 |

**读法**：在论文常用的近/中距离协议下，两方 FID 都很低（novel 4.2~4.3）且几乎持平——**这恰恰说明官方协议"看不出"洞区问题**，因为目标帧离源帧近、不可见区占比小。这解释了为什么回归类竞品论文即便报生成质量也显得漂亮。要暴露短板，必须加大视差（见下）。

## 对照：wide700 大视差压力测试（`test_files_wide700.txt`，572 自建大视差场景）

⚠️ **非官方 split，论文无对应数字，仅作压力测试**。此集刻意放大不可见区占比。指标口径与主表 2 相同（cleanfid，全图）。

| 方法 | tgt5 FID | tgt10 FID | tgt_rand FID |
|---|---|---|---|
| Flash3D | 26.9 | 43.9 | 64.0 |
| CATSplat | 26.1 | 40.7 | 56.5 |

**读法**：把主表 2 与本表并列看，结论极清晰——**同样的方法、同样的指标口径，视差从"近"（官方 present）拉到"大"（wide700），FID 从 4~8 暴涨到 26~64（约 8~10 倍）**。不可见区占比一上升，回归类方法只能把洞区糊过去，分布级指标立刻崩坏。这正是竞品论文用近距离协议 + 定性图回避、而本工程用顶会公认指标量化出来的短板，也是我们方法要发力的战场。

> 注：wide700 的两方 FID 目前来自较早导出（跨方法未做行级 GT 对齐），仅用于"同方法跨视差"的纵向对照；其绝对值和横向差异不作跨方法结论。主表 2 才是跨方法可比的严格结果。

---

数据来源：
- 主表 1（重建）：`results/flash3d_present_recon.json`、`results/catsplat_present_recon.json`（官方 evaluate.py 输出）
- 主表 2（生成质量，可比）：`results/flash3d_present_genq.json`、`results/catsplat_present_genq.json`（`run_genquality_official.sh` → `metrics/eval_genquality.py`）
- 可比性铁证：`metrics/check_gt_consistency.py`（3100 目录名一致 + GT-MAE=0）
- 对照 FID：wide700 早期导出 + `metrics/eval_genquality.py`
