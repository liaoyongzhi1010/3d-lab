# 固定 759-frame 全图质量诊断（E-210）

## 目的
在同一批 disocclusion-heavy frames 上比较 Gen3R、Flash3D evidence 和注入输出的全图质量。

## 数据与代码版本
- 结果：[`../../data/results/github_release/paper1_selective_generation/results/E-210_standard_metrics.json`](../../data/results/github_release/paper1_selective_generation/results/E-210_standard_metrics.json)。
- 代码：[`../../data/results/github_release/paper1_selective_generation/scripts/e210_standard_metrics.py`](../../data/results/github_release/paper1_selective_generation/scripts/e210_standard_metrics.py)。
- 版本：`7477a9d`；新鲜渲染所需 Gen3R/Flash3D/VGGT 版本未完整记录。

## 配置
结果 JSON 记录 21 scenes、759 frames 和 LPIPS-VGG。560×560 保存尺寸、无额外 crop 及非官方 MINE 的边界来自 [`PROTOCOLS.md`](../../data/results/github_release/PROTOCOLS.md)，不由 E-210 JSON 单独证明。

## 运行方式
已有 JSON 可直接检查。重新计算指标需要外部数组：
```bash
cd 3d-reconstruction/data/results/github_release
python3 paper1_selective_generation/scripts/e210_standard_metrics.py \
  --data_root "$RE10K_ROOT" --tdir "$TEACHER_DIR" --edir "$EVIDENCE_DIR" \
  --out /tmp/e210.json
```

## 指标
Gen3R：PSNR 17.768、LPIPS 0.360、FID 31.774；注入输出：PSNR 19.224、LPIPS 0.341、FID 31.143；Flash3D evidence：PSNR 19.761、LPIPS 0.444、FID 54.841。

## 效果图
- [`panel_success_boxed.png`](../../data/results/github_release/paper1_selective_generation/paper/figs/panel_success_boxed.png)
- [`panel_failure_boxed.png`](../../data/results/github_release/paper1_selective_generation/paper/figs/panel_failure_boxed.png)

## 观察
注入输出在这组固定诊断中相对 Gen3R 改善 PSNR、LPIPS 和 FID；但 Flash3D evidence 呈现 PSNR 较高、LPIPS/FID 较差的质量权衡。

## 失败点
该诊断使用 oracle visibility，且不是官方 MINE 排行协议。定性图是选择案例，不能替代总体统计。

## 结论
结果支持同一诊断集中的组件质量改善，不支持与不同协议论文数字做直接排名。
