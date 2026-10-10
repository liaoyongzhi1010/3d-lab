# DPC Code

Depth-Parallax Confusion（DPC）是针对单图前馈 3DGS 的源视图约束攻击。代码保持模型权重冻结，只优化输入图像扰动。

## 结构

```text
code/flash3d-attack/
├── attacks/                 # 攻击、评测、分析与定性图生成
├── tests/                   # CPU 单元测试与布局/证据完整性测试
├── METHOD_DPC.md            # 方法细节和早期实验记录
├── eval_*.py                # 迁移/黑盒入口
├── query_*.py               # 查询预算与基线分析
└── run_*.sh                 # 历史服务器运行脚本
```

机器可读结果位于 [`../../data/results/`](../../data/results/)，实验记录位于 [`../../onepage/experiments/`](../../onepage/experiments/)。

## 本地测试

```bash
cd 3d-reconstruction-security
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest code/flash3d-attack/tests -q -p no:cacheprovider
```

不依赖 Flash3D 的测试可在 CPU 环境执行；Flash3D adapter 集成测试在缺少上游环境时会跳过。

## 汇总已有结果

```bash
cd 3d-reconstruction-security/code/flash3d-attack
python3 attacks/summarize_results.py
```

脚本默认读取 `../../data/results/`。若将代码复制到历史服务器布局，可通过 `DPC_RESULTS_DIR=/path/to/results` 指定结果目录。

## 主实验

需要 Linux、CUDA 11.8、官方 Flash3D 仓库与 checkpoint、RealEstate10K 数据：

```bash
cd /root/projects/flash3d
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-11.8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /path/to/3d-reconstruction-security/code/flash3d-attack
python attacks/eval_dpc_flash3d.py --method dpc --max_scenes 100 --steps 40 \
  --split /root/projects/flash3d/splits/re10k_mine_filtered/test_files_present.txt \
  --epsilon 0.0314 --lambda_src 3.0 \
  --out ../../data/results/main_dpc_n100.json
```

现有 GPU 脚本仍保留历史绝对路径 `/root/projects/flash3d` 和 `/root/flash3d-attack`。迁移没有改写这些实验实现；在服务器运行前需将代码放到历史路径，或先单独完成路径参数化并重新验证。

## 已知复现限制

- 未记录原运行的 Flash3D commit 和 checkpoint SHA256。
- 未记录随机种子和精确的 100-scene ID 清单。
- 没有项目级 dependency lock；GPU 依赖由上游 Flash3D 环境提供。
- `run_*.sh` 中部分 geometry-hybrid sweep 的输出未进入仓库，不能作为当前结论。
