# SV3D-Eval-Suite

统一评测工程：单视图三维重建 / 新视角合成（Single-View NVS + disocclusion 生成）方法的跨方法可比评测。

一句话目标：**任何一个新方法，只要能导出目标视角渲染图，就能立刻用这套工程测出可对齐顶会论文、且与竞品可比的指标。**

---

## 1. 为什么要这套工程

课题要证明我们的方法在"不可见区（disocclusion）生成"上超越近三年竞品（Flash3D / CATSplat / pixelSplat / MVSplat 等）。要让这个结论可信，评测必须先过三关：

1. **能对齐论文** —— 用竞品官方评测代码复现它们论文里的数字（证明尺子是准的）。
2. **跨方法可比** —— 所有方法走同一套指标、同一测试集、同一目标帧、同一 GT（否则数字不可比）。
3. **能暴露短板** —— 借鉴生成类顶会的 FID/KID/DISTS，量化竞品没报告、但恰是我们优势所在的维度（洞区生成质量）。

**核心原则：只复用各方法官方 evaluator 与顶会公认指标实现，绝不自写 metric，也不自造评测协议。**

---

## 2. 设计（怎么构建的）

### 2.1 指标唯一真源：官方 evaluator

- **重建精度（PSNR / SSIM / LPIPS）**：直接调用 Flash3D / CATSplat 官方 `evaluation/evaluator.py`。经核实，两个仓库的 evaluator 指标逻辑完全一致：
  - PSNR = `-10·log10(mean((pred-gt)²))`
  - SSIM = torchmetrics `data_range=1.0`
  - LPIPS = torchmetrics VGG 骨干，输入归一化到 [-1,1]
  - 统一 5% 边缘裁剪（`margin=0.05`）
- **生成质量（FID / KID / DISTS）**：借鉴生成类顶会（latentSplat / GenWarp）：
  - FID / KID = `cleanfid`（本地权重，与历史实验口径一致）
  - DISTS = `DISTS_pytorch`（latentSplat 同款包）
  - LPIPS = VGG（所有论文一致）

### 2.2 指标层与推理层解耦

评测工程不与任何单一模型耦合。约定一个统一导出格式，任何方法把目标视角渲染图与真值图按此格式落盘，即可喂进同一指标层：

```
<method_export_dir>/<scene>/pred/00X.png   # 该方法渲染的目标视角
<method_export_dir>/<scene>/gt/00X.png     # 对应真值
# 000=source(源视图自还原), 001=tgt5, 002=tgt10, 003=tgt_rand
```

### 2.3 三个组件

| 组件 | 文件 | 作用 |
|---|---|---|
| 重建精度 | `metrics/eval_recon.py` | 吃 `(pred,gt)` 图对 → 官方 evaluator → 逐档 PSNR/SSIM/LPIPS |
| 生成质量 | `metrics/eval_genquality.py` | 吃 `(pred,gt)` 图对 → cleanfid FID/KID + DISTS + LPIPS，按 gap 分档 |
| 编表 | `metrics/compile_table.py` | 汇编多方法 JSON → Markdown 对比表 |
| 编排 | `run_all.sh` | 串起导图→两层指标→编表 |

### 2.4 测试集（对齐论文的关键）

- **主表用官方 `test_files.txt` / `test_files_present.txt`（MINE 标准 split）**。Flash3D、CATSplat 论文数字都在这个 split 上报告；两个仓库的 split 文件 md5 完全一致。
  - `test_files.txt`：3204 行（官方全量，含少量缺帧会 crash）
  - `test_files_present.txt`：3100 行 = 620 唯一场景 × 每场景 5 个源帧配置（可用子集，推荐）
- 协议：`target_frame_ids=[1,2,3]`，`eval_frames=[src, tgt5, tgt10, tgt_rand]`。
- `test_files_wide700.txt`（572 大视差场景）是**自建的压力测试集**，论文无对应数字，仅作附录用途，不进主表。

---

## 3. 可靠性铁律（这套工程踩过的坑，已固化为检查）

1. **源视图自还原 PSNR 必须 ~36–39dB**。这是渲染管线（投影/位姿/光栅化/权重加载）正确的判据。任何一次评测先看 src PSNR，低于 35dB 说明权重没加载或管线错，数字全部作废。
2. **checkpoint 静默不加载坑**：官方 `evaluate.py` 的 `main()` 里有写死的 `os.chdir(output_dir)`，切目录后相对路径 `checkpoints/`、`pointnet/ckpt/save.pth`、`data/RealEstate10K` 全部失效 → 模型跑未训练权重（src 仅 ~11dB）。修复：在 run_dir 建 `checkpoints` 软链、pointnet 用绝对路径、`dataset.data_path` 传绝对路径。
3. **跨方法可比铁律**：所有方法必须同 split、同源帧、同目标帧、同 GT。上线前抽样核对两方 GT 逐像素一致（GT-MAE≈0）才可比。⚠️ 注意 present split 每场景有 5 行，按 `<scene>/gt` 命名导图会互相覆盖 → **主表用官方 evaluate.py 输出的 metrics JSON（按行累积平均，不受覆盖影响），不靠导图算重建精度**。
4. **FID 用 cleanfid 不用 torchmetrics**：torchmetrics 的 FID 首次实例化要联网下 Inception 权重，离线环境会卡死；cleanfid 用本地权重且与历史口径一致。

---

## 4. 目录结构

```
sv3d-eval-suite/
├── README.md                        # 本文件
├── run_all.sh                       # 编排：导图→指标→编表
├── metrics/
│   ├── eval_recon.py                # 重建精度（官方 evaluator）
│   ├── eval_genquality.py           # 生成质量（cleanfid FID/KID + DISTS）
│   └── compile_table.py             # 编表
├── patches/                         # 对官方 evaluate.py 的最小补丁（加导图目录/跳过ply/绝对路径修复）
│   ├── flash3d_evaluate.py
│   ├── catsplat_evaluate.py
│   └── catsplat_unidepth_encoder.py
├── splits/                          # MINE 标准 split + 自建压力测试集
│   ├── test_files.txt
│   ├── test_files_present.txt
│   └── test_files_wide700.txt
├── adapters/                        # 各方法推理→导图的说明（见 docs）
├── results/                         # 评测产物（JSON + 对比表）
└── docs/
    └── USAGE.md                     # 怎么接一个新方法
```

---

## 5. 快速开始

见 [`docs/USAGE.md`](docs/USAGE.md)。核心三步：

1. 让方法用官方 evaluate.py（或你自己的方法）在 MINE split 上导出 `<scene>/{pred,gt}/00X.png`；
2. `python metrics/eval_recon.py --root <导出目录> --method <名字> --out results/<名字>_recon.json`；
3. `python metrics/eval_genquality.py --root <导出目录> --method <名字> --out results/<名字>_genq.json`，最后 `compile_table.py` 汇总。
