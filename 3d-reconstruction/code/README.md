# Code

本目录包含三个完整单元，迁移时未拆散其内部相对路径：

- `sv3d-lab/`：单图场景级 3DGS 研究实现，含 Paper A/B、共享几何与评测代码。
- `eval-suite/`：Flash3D/CATSplat 对齐评测、split、patch 和已有评测结果。
- `experiments/`：历史服务器实验脚本。多数依赖 `/root/projects/*` 和 `/home/data/*`，用于追溯，不保证本地可直接运行。

## 可执行发布包

最完整的分析复现入口位于 [`../data/results/github_release/`](../data/results/github_release/)：

```bash
cd 3d-reconstruction/data/results/github_release
python3 -m pip install -r requirements.txt
python3 scripts/smoke_test.py
```

## 服务器兼容

`sv3d-lab` 内仍有 `/root/sv3d-lab`、`/root/projects/flash3d` 等历史绝对路径。为保持已有实验可运行，本次整理未批量改写模型代码。服务器可继续使用原镜像路径或建立兼容软链接；新实验应逐步改为环境变量。

历史 `eval-suite/run_all.sh` 因调用未保留的 `_e052_*` 文件而删除。请直接使用 `code/eval-suite/metrics/` 下的三个脚本。
