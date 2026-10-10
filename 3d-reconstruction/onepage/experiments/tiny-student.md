# TinyStudent 质量与速度（E-211）

## 目的
评估在 oracle-curated、oracle-visibility 条件下，小型前馈学生能否近似慢速生成教师。

## 数据与代码版本
- 结果：[`../../data/results/github_release/paper2_gate_aware_distillation/results/E-211_student_standard_metrics.json`](../../data/results/github_release/paper2_gate_aware_distillation/results/E-211_student_standard_metrics.json)。
- Checkpoint：[`../../data/results/github_release/paper2_gate_aware_distillation/checkpoints/student.pt`](../../data/results/github_release/paper2_gate_aware_distillation/checkpoints/student.pt)。
- 模型卡：[`../../data/results/github_release/paper2_gate_aware_distillation/MODEL_CARD.md`](../../data/results/github_release/paper2_gate_aware_distillation/MODEL_CARD.md)。
- 版本：`7477a9d`；checkpoint SHA256 为模型卡记录的 `1aa66b23...af76f2`。

## 配置
21 scenes、759 frames、256×256、LPIPS-VGG；46,371 参数；训练数据由 teacher/GT gain 进行 oracle 筛选。

## 运行方式
```bash
cd 3d-reconstruction/data/results/github_release
python3 paper2_gate_aware_distillation/scripts/check_checkpoint.py
python3 paper2_gate_aware_distillation/scripts/generate_quality_speed.py \
  --metrics paper2_gate_aware_distillation/results/E-211_student_standard_metrics.json \
  --out /tmp/quality-speed.pdf
```

## 指标
Teacher：PSNR 19.373、LPIPS 0.258、FID 31.328、247 s/scene；Student：PSNR 18.166、LPIPS 0.284、FID 33.614、0.087 s/scene。

## 效果图
[`paper2_quality_speed.png`](../../data/results/github_release/assets/paper2_quality_speed.png)

## 观察
Student 明显快于 teacher，但质量未完全达到 teacher；相对 Gen3R baseline 的提升也依指标不同。

## 失败点
原始 holdout scene 清单、frame labels、seed、sample order 和 optimizer state 缺失，无法精确重训。当前实验依赖 oracle visibility 和 oracle curation，不是未知场景的可部署 selector。

## 结论
结果支持条件化的教师压缩/速度研究，不支持已解决真实单图选择与端到端训练复现。
