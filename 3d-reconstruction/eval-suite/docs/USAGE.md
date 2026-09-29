# USAGE：怎么用这套工程评测一个方法

面向"我们发明了一个新方法，想立刻测出可对齐论文、且与竞品可比的数字"。

## 前置

- 环境：`/root/projects/flash3d/.venv`（Flash3D / CATSplat 共用，含 torch/hydra/pointnet/cleanfid/DISTS_pytorch）。
- 数据：RealEstate10K，软链 `data/RealEstate10K -> /home/data/RealEstate10K`。
- 权重：Inception（cleanfid）在 `/tmp/inception-2015-12-05.pt`，已 cp 到 cleanfid 包目录。

## 统一约定：任何方法都导出成这个格式

```
<method_export_dir>/<row_dir>/pred/00X.png
<method_export_dir>/<row_dir>/gt/00X.png
# 000=source, 001=tgt5, 002=tgt10, 003=tgt_rand
# 所有方法必须用同一 split（MINE test_files[_present].txt），target_frame_ids=[1,2,3]
```

### ⚠️ 关键：`<row_dir>` 必须是"行级唯一名"，否则跨方法 FID 不可比

present split **每个场景有多达 5 行**（同一 `scene_hash`，不同源帧 `src_idx`）。如果导图目录只用 `scene_hash` 命名，同场景 5 行会写同名 `pred/00X.png` **互相覆盖**——不同方法覆盖后剩下的行还可能不同，导致两方图集根本不是同一批，FID 无从比较。这正是早期 wide700 对比失败（同场景 GT-MAE=60~97）的根因。

**规范：每一行一个唯一目录，命名 `{row:05d}_{scene_hash}_src{src_idx}`。** 两个仓库的 dataset 都按 split 文件顺序遍历 `_seq_key_src_idx_pairs[k] = (scene_hash, [src,tgt5,tgt10,tgt_rand])`，所以 `row 序号 + scene + src` 在任何方法里都指向同一 (源帧, 目标帧) 组合，天然一一对应。

- Flash3D / CATSplat：用本仓库 `patches/` 版 evaluate.py，已按此命名（见 `patches/*.py` 里 `seq_name = f"{k:05d}_{_scene_hash}_src{int(_idxs[0])}"`）。
- 自己的新方法：推理脚本遍历 split 时，用**行号 k**（而非场景名）构造目录名。

**出可比 FID 前，务必抽查两方同名目录同帧 GT-MAE≈0**（脚本见"可比性自检清单"）。

---

## 场景 A：评测一个已有官方仓库的竞品（Flash3D / CATSplat）

各方法用自己官方 `evaluate.py` 出数，这是对齐它论文的正道。

### A.1 拿"重建精度主表"（对齐论文，推荐）

直接跑官方 `evaluate.py`，它自己按每行累积平均输出 `metrics_re10k_test_*.json`（PSNR/SSIM/LPIPS 按 src/tgt5/tgt10/tgt_rand）。这是主表数字来源，**不依赖导图**（present split 每场景 5 行，导图会覆盖，但指标按行累积正确）。

```bash
# Flash3D（注意 checkpoints 软链避开 os.chdir 坑）
cd /root/projects/flash3d && source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-11.8
mkdir -p <run_dir> && ln -sfn /root/projects/flash3d/checkpoints <run_dir>/checkpoints
python evaluate.py hydra.run.dir=<run_dir> hydra.job.chdir=false \
  +experiment=layered_re10k +dataset.crop_border=true \
  dataset.test_split_path=splits/re10k_mine_filtered/test_files_present.txt \
  model.depth.version=v1 ++eval.save_vis=false

# CATSplat（权重用绝对路径；pointnet/data 已在 patches 修成绝对路径）
cd /root/projects/CATSplat
python evaluate.py hydra.run.dir=<run_dir> hydra.job.chdir=true \
  +experiment=layered_re10k +dataset.crop_border=true \
  dataset.data_path=/home/data/RealEstate10K \
  dataset.test_split_path=./splits/re10k_mine_filtered/test_files_present.txt \
  model.depth.version=v1 ++eval.save_vis=false \
  run.checkpoint=/root/projects/CATSplat/ckpts/CATSplat.pth
```

**先看 log 里源视图 src 的 PSNR ~36-39dB**，否则权重没加载，数字作废。

### A.2 拿"生成质量（FID/KID/DISTS）"——跨方法可比

FID 需要图，且要**两方图集严格对齐**。用带 `patches/` 的 evaluate.py（`SKIP_PLY=1` 跳过 ply、`FLASH3D_VIS_DIR`/`CATSPLAT_VIS_DIR` 指定导图目录、行级唯一命名），`++eval.save_vis=true` 导出 `<row_dir>/{pred,gt}/00X.png`。

一键跑法（导两方图 + 算 FID/KID/DISTS，已封装）：

```bash
bash run_genquality_official.sh   # 官方 present split，两方行级对齐导图 + cleanfid
# 产物：flash3d_genq_official.json / catsplat_genq_official.json
```

单独导某一方（示例 Flash3D，CATSplat 见脚本内）：

```bash
export SKIP_PLY=1
FLASH3D_VIS_DIR=<export_dir> python evaluate.py \
  hydra.run.dir=<run_dir> hydra.job.chdir=false \
  +experiment=layered_re10k +dataset.crop_border=true \
  dataset.test_split_path=splits/re10k_mine_filtered/test_files_present.txt \
  model.depth.version=v1 ++eval.save_vis=true
```

再算指标：

```bash
python metrics/eval_genquality.py \
  --root <export_dir> --method <名字> \
  --out results/<名字>_genq.json \
  --workdir /tmp/_genq/<名字>
```

---

## 场景 B：评测我们自己的新方法

只要新方法能对 MINE split 的每个样本，渲染出目标视角图并落成统一格式，就跟竞品完全同一把尺子。

1. 写一个推理脚本：对 split 每行，用你的方法渲染 src/tgt5/tgt10/tgt_rand，存 `<export>/<scene>/{pred,gt}/00X.png`。
   - **务必输出 000=src 自还原**，用它做管线自检（~36-39dB）。
   - GT 直接存 dataloader 给的 `("color", f_id, 0)`，保证和竞品同一张真值。
2. 重建精度：`python metrics/eval_recon.py --root <export> --method ours --out results/ours_recon.json`
3. 生成质量：`python metrics/eval_genquality.py --root <export> --method ours --out results/ours_genq.json --workdir /tmp/_genq/ours`

`eval_recon.py` 直接调 Flash3D 官方 `evaluation/evaluator.py`（PSNR/SSIM/LPIPS-VGG + 5% crop），所以我们方法和竞品指标口径完全一致。

---

## 场景 C：一键编排 + 出对比表

```bash
bash run_all.sh   # 按脚本内配置：等导图→跑 eval_recon + eval_genquality→compile_table
python metrics/compile_table.py \
  --recon flash3d=results/flash3d_recon.json --recon catsplat=results/catsplat_recon.json --recon ours=results/ours_recon.json \
  --genq  flash3d=results/flash3d_genq.json  --genq  catsplat=results/catsplat_genq.json  --genq  ours=results/ours_genq.json \
  --out results/COMPARISON.md
```

---

## 可比性自检清单（出表前必过）

- [ ] 所有方法用**同一 split 文件**（md5 一致）。
- [ ] 源视图 src 自还原 PSNR ~36-39dB（每个方法都要过）。
- [ ] 导图目录用**行级唯一名** `{row:05d}_{scene}_src{src}`，不是纯场景名（否则同场景多行覆盖）。
- [ ] 抽样核对不同方法**同场景同帧的 GT 逐像素一致**（GT-MAE≈0）；若用官方 evaluate.py 的 metrics JSON 出主表，则天然一致（同 loader）。
- [ ] LPIPS 用 VGG；FID 用 cleanfid。
- [ ] 主表用官方 MINE split（对齐论文）；wide700 只作附录压力测试。

跨方法 GT 一致性抽查（跨方法 FID 前必跑）：

```bash
python - <<'PY'
import numpy as np; from PIL import Image; from pathlib import Path
A="<methodA_export>"; B="<methodB_export>"
names=sorted(p.name for p in Path(A).iterdir())
assert names==sorted(p.name for p in Path(B).iterdir()), "目录名不一致→不可比"
bad=0
for nm in names:
    for fr in ["000","001","002","003"]:
        fa,fb=Path(A)/nm/"gt"/f"{fr}.png",Path(B)/nm/"gt"/f"{fr}.png"
        if not (fa.exists() and fb.exists()): continue
        a=np.asarray(Image.open(fa).convert("RGB"),np.float32)
        b=np.asarray(Image.open(fb).convert("RGB"),np.float32)
        m=np.abs(a-b).mean() if a.shape==b.shape else 999
        if m>1e-3: bad+=1; print("MISMATCH",nm,fr,m)
print("ALL GT MATCH" if bad==0 else f"{bad} mismatches")
PY
```
