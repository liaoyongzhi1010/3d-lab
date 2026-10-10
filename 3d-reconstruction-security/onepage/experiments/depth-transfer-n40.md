# 共享深度先验迁移（n=40）

## 目的
判断由 Flash3D 渲染目标优化出的扰动，是否同时破坏其使用的 UniDepth 深度先验。

## 数据与代码版本
- 数据：RealEstate10K present split。
- 代码：[`../../code/flash3d-attack/attacks/eval_depth_transfer.py`](../../code/flash3d-attack/attacks/eval_depth_transfer.py)。
- 结果：[`../../data/results/depth_transfer_n40.json`](../../data/results/depth_transfer_n40.json)。
- 仓库版本：`7477a9d`；上游 Flash3D/UniDepth 版本和 checkpoint hash 未记录。

## 配置
40 scenes，epsilon=8/255，40 steps，lambda_src=3.0。

## 运行方式
```bash
python attacks/eval_depth_transfer.py --max_scenes 40 --epsilon 0.0314 \
  --split /root/projects/flash3d/splits/re10k_mine_filtered/test_files_present.txt \
  --out ../../data/results/depth_transfer_n40.json
```
需要外部 Flash3D 环境及其 UniDepth 依赖。

## 指标
攻击前后深度 AbsRel=0.958；65% 像素的深度变化超过 10%；攻击图与原图的 source RGB PSNR=32.53 dB。

## 效果图
没有归档与该实验一一对应的深度图或点云可视化，标记为“待补”。

## 观察
在源 RGB 仍较接近的情况下，深度先验输出发生大幅变化，说明扰动影响了几何输入链路。

## 失败点
这不是第二个重建模型的端到端迁移；它只验证同一共享深度先验的变化。没有保存每个 scene 的明细和运行版本。

## 结论
结果支持“DPC 可显著扰动共享深度先验”，不支持“已经证明在所有 UniDepth 重建器上端到端迁移”。
