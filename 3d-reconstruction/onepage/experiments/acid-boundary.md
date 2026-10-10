# ACID 跨数据集边界（E-215）

## 目的
检查 Paper 1 的选择性注入规律是否在转换后的 ACID 轨迹上保持。

## 数据与代码版本
- 结果：[`../../data/results/github_release/paper1_selective_generation/results/E-215_acid_metrics.json`](../../data/results/github_release/paper1_selective_generation/results/E-215_acid_metrics.json)。
- 代码：[`../../data/results/github_release/paper1_selective_generation/scripts/e215_acid_metrics.py`](../../data/results/github_release/paper1_selective_generation/scripts/e215_acid_metrics.py)。
- 版本：`7477a9d`；精确 scene ID 和转换日志未保存在结果 JSON 中。

## 配置
结果 JSON 记录 8 个有效 scenes、215 frames 和 LPIPS-VGG。转换轨迹的短边缩放、560×560 中心裁剪及非官方 ACID leaderboard 边界来自 [`PROTOCOLS.md`](../../data/results/github_release/PROTOCOLS.md)。

## 运行方式
```bash
cd 3d-reconstruction/data/results/github_release
python3 paper1_selective_generation/scripts/e215_acid_metrics.py \
  --results "$ACID_RESULTS" --tdir "$TEACHER_DIR" --edir "$EVIDENCE_DIR" \
  --scene_root "$ACID_SCENE_ROOT" --out /tmp/e215.json
```

## 指标
机制区域平均变化 `-0.443 dB`，仅 3/8 scenes 提升；Gen3R / injected LPIPS 为 0.383 / 0.408。

## 效果图
没有与这 8 个场景精确绑定且带 manifest 的定性面板，标记为“待补”。

## 观察
平均结果为负，说明 RE10K 上的选择规律没有直接跨数据集保持。

## 失败点
样本只有 8 scenes，精确 scene ID、原尺寸和 crop offset 缺失；不能替代仓库日志中声称的 ACID 40-scene 后续实验。

## 结论
这是重要负结果：当前机制具有数据集依赖性，不支持广泛跨域泛化结论。
