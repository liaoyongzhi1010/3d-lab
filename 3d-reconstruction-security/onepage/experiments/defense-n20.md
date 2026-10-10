# 多视角一致性防御（n=20）

## 目的
测试无需训练的多视角曲率分数能否区分干净输入和 DPC 攻击输入。

## 数据与代码版本
- 数据：RealEstate10K present split。
- 代码：[`../../code/flash3d-attack/attacks/eval_defense.py`](../../code/flash3d-attack/attacks/eval_defense.py)。
- 结果：[`../../data/results/defense_n20.json`](../../data/results/defense_n20.json)。
- 仓库版本：`7477a9d`；实际运行 commit、seed 和 checkpoint hash 未记录。

## 配置
20 scenes，epsilon=8/255，lambda_src=3.0。JSON 未保存 steps 和完整 detector 配置。

## 运行方式
历史入口为：
```bash
python attacks/eval_defense.py --max_scenes 20 \
  --split /root/projects/flash3d/splits/re10k_mine_filtered/test_files_present.txt \
  --out ../../data/results/defense_n20.json
```
完整参数需结合脚本默认值核对；外部 Flash3D 环境不可省略。

## 指标
Detector ROC-AUC=0.555；clean score mean=0.04839，attacked score mean=0.05005。

## 效果图
该实验没有保存 ROC 曲线、分数分布或逐场景图，标记为“待补”。

## 观察
检测器只略高于随机水平，干净和攻击分数均值接近。

## 失败点
样本只有 20；JSON 没有逐样本分数、置信区间和完整 detector 配置，因此不能复核 AUC 计算，也不能泛化为所有防御无效。

## 结论
当前训练自由的一致性分数未能可靠检测 DPC；这只是一个负基线，不是对防御空间的完整评估。
