# Data

本目录只保存可公开、体量适合 Git、且能追溯当前结论的实验产物；不保存 RealEstate10K 原始帧、Flash3D 权重或第三方模型缓存。

## 数据来源

- 数据集：RealEstate10K，实验记录声明使用 Flash3D 的 `splits/re10k_mine_filtered/test_files_present.txt`。
- 上游模型：官方 Flash3D checkpoint，运行环境中的历史路径为 `/root/projects/flash3d/checkpoints/model_re10k_v2.pth`。
- 原始数据和权重未复制到本仓库。其许可证、下载方式和版本应以各上游项目为准。

## 当前内容

- `results/`：11 份可用于当前审阅的机器可读 JSON，包含主实验、基线、消融、共享深度变化、防御和区域分析。
- `results/invalid/`：1 份审计未通过的 VGGT 迁移 JSON。原脚本复用了首场景 context，文件仅保留原始产物和失败记录，不支撑结论。
- `qualitative/`：5 张正式定性面板。每张图按 source、tgt5、tgt10、tgt_rand 展示 clean、attacked 和差异。
- `qualitative/archive/`：9 张唯一但未被当前 One-page 选用的感知、v2 和 v3 可视化。因缺少正式选择 manifest，它们只归档、不支撑主页结论。

## 已知溯源缺口

现有 JSON 未完整记录上游 Flash3D commit、checkpoint SHA256、随机种子、CUDA/cuDNN 版本和精确的 100-scene ID 清单。相关字段不能从目录名推断，因此保持为“未知”。在补齐这些信息前，本目录支持结果核查和分析复现，不支持声称精确端到端重跑。

## 大文件策略

原始帧、第三方权重、`.npy/.npz`、视频和三维资产不进入 Git。后续数据应在对应实验记录中保存来源、版本、获取方式、校验值和外部存储位置。
