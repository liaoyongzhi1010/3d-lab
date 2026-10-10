# Flash3D 到 VGGT 迁移（n=30，审计未通过）

**审计状态：无效。** 只保留原始产物和失败原因，不用于支撑当前结论。

## 目的
原实验试图评估在 Flash3D surrogate 上生成的 DPC 扰动，是否会改变独立 VGGT-1B 目标的深度和点图输出。

## 数据与代码版本
- 数据：30 个 RealEstate10K scenes；结果仅保存 `scene_idx`，未保存稳定 scene ID。
- 代码：[`../../code/flash3d-attack/eval_vggt_transfer.py`](../../code/flash3d-attack/eval_vggt_transfer.py)。
- 结果：[`../../data/results/invalid/BB-002-vggt-transfer.json`](../../data/results/invalid/BB-002-vggt-transfer.json)。
- 仓库版本：`7477a9d`；Flash3D 和 VGGT 上游 commit/checkpoint hash 未记录。

## 配置
Surrogate=Flash3D，target=VGGT-1B，30 scenes，epsilon=4/255，40 steps，lambda_src=3.0。

## 运行方式
```bash
cd 3d-reconstruction-security/code/flash3d-attack
python eval_vggt_transfer.py --max_scenes 30 --epsilon 0.015686 \
  --steps 40 --lambda_src 3.0 --seed 0 \
  --out /tmp/BB-002-vggt-transfer-rerun.json
```
需要本地缓存的 VGGT-1B 和完整 Flash3D CUDA 环境。先输出到 `/tmp`，完成逐场景审计后才能作为新的有效结果归档；不得覆盖 `data/results/invalid/` 中的历史失败产物。

## 指标
平均 point-map displacement=0.0864；平均 depth rank reversal=0.0668；source PSNR=37.91 dB。相对随机扰动，位移比为 1.189，rank-reversal 比为 1.086。

## 效果图
没有与该结果绑定的 VGGT 点图、深度图或三维模型，标记为“待补”。

## 观察
JSON 中两项迁移指标高于其随机基线，但实现审计表明这些汇总值不能用于跨场景结论。

## 失败点
生成该历史 JSON 时，`eval_vggt_transfer.py` 的场景循环虽更新了当前 `inputs`，但内部 `render_fn` 使用 `adapter.example_inputs`，即 adapter 初始化时保存的首个场景 context。后续场景的 source image 因而与首场景相机/上下文混用。当前代码已修复该缺陷并增加 seed，但尚未重跑；历史实验还缺少稳定 scene ID、模型版本和可视化。

## 结论
该结果审计未通过，必须修复场景 context、固定随机性并重新运行后才能评估跨模型迁移。当前 JSON 仅作为失败实验原始记录保留。
