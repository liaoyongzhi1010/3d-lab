# 区域损伤分析（n=80）

## 目的
将目标视角划分为源可见、遮挡和源视锥外区域，检查攻击损伤是否集中在缺乏源视图约束的位置。

## 数据与代码版本
- 数据：RealEstate10K present split。
- 代码：[`../../code/flash3d-attack/attacks/eval_region_metrics.py`](../../code/flash3d-attack/attacks/eval_region_metrics.py) 与 [`region_masks.py`](../../code/flash3d-attack/attacks/region_masks.py)。
- 结果：[`../../data/results/region_metrics_n80.json`](../../data/results/region_metrics_n80.json)。
- 仓库版本：`7477a9d`；运行 commit、seed 和 checkpoint hash 未记录。

## 配置
80 scenes，epsilon=8/255，40 steps，lambda_src=3.0，4 个辅助位姿，最小区域比例 1%。

## 运行方式
```bash
python attacks/eval_region_metrics.py --max_scenes 80 --epsilon 0.031372549 \
  --split /root/projects/flash3d/splits/re10k_mine_filtered/test_files_present.txt \
  --out ../../data/results/region_metrics_n80.json
```
需要 Flash3D、相机和深度数据。

## 指标
可见区 ΔPSNR=-5.48 dB（n=240 frames）；遮挡区 -4.46 dB（n=3）；源视锥外 -10.56 dB（n=122）。

## 效果图
现有正式主实验面板未附带区域 mask，因此不作为本实验直接效果图。区域 mask 与并排结果均标记为“待补”。

## 观察
源视锥外区域的平均 PSNR 损伤约为可见区域的 1.9 倍，与缺少源图约束的假设一致。

## 失败点
遮挡区仅 3 个合格目标帧，不能据此形成稳定结论；各区域 n 不同，区域比例也不同。当前 JSON 没有逐帧值或场景清单。

## 结论
证据支持源视锥外区域更易受损；遮挡区域证据不足，应保持为待扩展分析。
