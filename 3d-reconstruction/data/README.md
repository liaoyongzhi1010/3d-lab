# Data

本目录保存可公开、可追溯且适合 Git 的实验证据；原始数据集和大型第三方权重不进入仓库。

## 内容

- `results/github_release/`：当前最完整的自包含发布包，含 Paper 1–3 的 JSON、manifest、分析脚本、论文、图片和 TinyStudent checkpoint。
- `provenance/`：历史实验日志、材料汇总和失败经验。它们用于追溯研究过程，不自动等同于可复现实验结果。

## 外部数据

- RealEstate10K：原始视频帧和相机元数据不在 Git 中。
- ACID：原始帧和转换数据不在 Git 中。
- Gen3R、Flash3D、VGGT：上游代码和权重不在 Git 中。

获取方式、环境变量和复现边界见 [`results/github_release/REPRODUCIBILITY.md`](results/github_release/REPRODUCIBILITY.md)。上游 checkpoint 版本、部分随机种子、Paper 2 原始 holdout/frame-label manifest 和部分服务器数组未被完整记录，因此不能声明精确端到端重跑。

## 大文件策略

新实验不得直接提交原始数据、缓存数组、视频、PLY 或大型权重。每条实验记录应保存数据来源、版本、命令、代码 commit、配置、外部存储位置和校验值。当前仅保留 184 KB 的 TinyStudent checkpoint，因为它是发布包可执行 smoke test 的必要组成。
