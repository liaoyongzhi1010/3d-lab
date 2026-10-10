# 可观测可靠性审计（E-224）

## 目的
检验推理时真正可观测的相机运动特征，能否安全判断何时调用生成教师。

## 数据与代码版本
- 结果：[`../../data/results/github_release/paper3_reliability_learning/results/E-224_camera_only_observable.json`](../../data/results/github_release/paper3_reliability_learning/results/E-224_camera_only_observable.json)。
- 输入：[`../../data/results/github_release/paper3_reliability_learning/results/E-149_frames_b5678_aug.json`](../../data/results/github_release/paper3_reliability_learning/results/E-149_frames_b5678_aug.json)。
- 代码：[`../../data/results/github_release/paper3_reliability_learning/scripts/e224_camera_only_observable.py`](../../data/results/github_release/paper3_reliability_learning/scripts/e224_camera_only_observable.py)。
- 版本：`7477a9d`。

## 配置
2,898 frames、74 scenes；仅使用 translation 与 rotation；按 scene grouped leave-one-scene-out CV；GT teacher gain 只用于标签和评估。

## 运行方式
```bash
cd 3d-reconstruction/data/results/github_release
python3 paper3_reliability_learning/scripts/e224_camera_only_observable.py \
  --data paper3_reliability_learning/results/E-149_frames_b5678_aug.json \
  --out /tmp/e224.json
```

## 指标
ROC-AUC 0.6496，trapezoidal PR-AUC 0.6631；threshold 0.5 时 mean gain +0.715 dB，worst gain -23.094 dB，节省 870/2,898 次调用。

## 效果图
[`paper3_granularity.png`](../../data/results/github_release/assets/paper3_granularity.png)。GT-derived routing strip 只作诊断边界说明，不作为可部署效果图。

## 观察
Camera-only 特征有中等区分能力，但保留严重最坏损失。

## 失败点
高 AUC 的 E-145b 使用 GT-derived quality probe；E-223 使用完整目标 clip 的 oracle visibility。两者都不能替代 E-224 的可观测结论。

## 结论
当前可观测证据不足以建立安全 selector；这是可靠性审计的负结论。
