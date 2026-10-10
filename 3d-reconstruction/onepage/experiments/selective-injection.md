# 选择性几何注入：难度与机制（E-203）

## 目的
检验几何证据注入的收益是否随场景难度变化，并量化组件质量差与收益的相关性。

## 数据与代码版本
- 结果：[`../../data/results/github_release/paper1_selective_generation/results/E-203_bootstrap_N166.json`](../../data/results/github_release/paper1_selective_generation/results/E-203_bootstrap_N166.json)。
- 输入：E-142 的 166-scene 汇总；scene manifest 位于 [`../../data/results/github_release/manifests/re10k_scene_manifest.json`](../../data/results/github_release/manifests/re10k_scene_manifest.json)。
- 代码：[`../../data/results/github_release/paper1_selective_generation/scripts/e203_bootstrap_ci.py`](../../data/results/github_release/paper1_selective_generation/scripts/e203_bootstrap_ci.py)。
- 版本：成果首次进入本仓库的提交为 `7477a9d`；上游渲染 checkpoint 版本未记录。

## 配置
166 scenes；难度桶为 hard `<12`、mid `12–20`、easy `>=20`。可见性来自完整目标 clip，选择信号包含 GT-derived quality probe。

## 运行方式
```bash
cd 3d-reconstruction/data/results/github_release
python3 paper1_selective_generation/scripts/e203_bootstrap_ci.py \
  --data paper1_selective_generation/results/E-142_combined_N169.json \
  --out /tmp/e203.json
```

## 指标
机制相关性 `r=0.9817`；hard 场景平均 `+2.486 dB`，mid `+0.108 dB`，easy `-2.517 dB`；oracle gap-5 平均 `+1.658 dB`。

## 效果图
- [`fig2_difficulty.pdf`](../../data/results/github_release/paper1_selective_generation/paper/figs/fig2_difficulty.pdf)
- [`fig_failures_limitations.pdf`](../../data/results/github_release/paper1_selective_generation/paper/figs/fig_failures_limitations.pdf)

## 观察
注入收益明显依赖场景难度；简单场景平均受损，说明 always-inject 不可靠。

## 失败点
可见性使用完整目标序列，quality-gap 依赖 GT。该结果是 offline oracle diagnostic，不是可部署 selector。

## 结论
当前证据支持“生成/几何注入具有强选择性”，不支持“单图推理时已经知道何时注入”。
