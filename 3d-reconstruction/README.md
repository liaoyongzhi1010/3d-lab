# 生成式引导的单视图场景级三维重建

本项目研究生成先验在单视图场景重建中的收益、压缩与可靠性边界。当前最完整的证据来自选择性注入诊断、TinyStudent 蒸馏和可观测可靠性审计；CycleFusion 与 Paper A 的部分成果仍缺本地原始产物。

## 入口

- [One-page 总览](onepage/index.html)
- [逐实验记录](onepage/experiments/)
- [数据与证据说明](data/README.md)
- [代码说明](code/README.md)
- [论文与毕设](papers/README.md)
- [完整 release 包](data/results/github_release/README.md)

## 目录

```text
3d-reconstruction/
├── README.md
├── data/       # release 结果、manifest、checkpoint 与历史 provenance
├── code/       # sv3d-lab、统一评测和历史实验入口
├── onepage/    # 静态总览和逐实验记录
└── papers/     # Difix/Paper A 草稿、毕设与方法说明
```

## 当前可核查结论

- 选择性注入具有明显难度依赖：166-scene 诊断中，hard 场景平均 `+2.486 dB`，easy 场景平均 `-2.517 dB`。
- 21-scene / 759-frame 固定诊断中，注入输出相对 Gen3R 的 PSNR、LPIPS-VGG 与 FID 均有改善，但该协议不是官方 MINE。
- TinyStudent 在 oracle-curated 条件下实现 0.087 s/scene 的前向速度，但没有解决真实 selector，且精确重训资料不完整。
- 真正可观测的 camera-only selector 只有 ROC-AUC 0.650，并保留 `-23.094 dB` 最坏损失，当前不适合安全部署。
- CycleFusion 的设计和汇总日志已保留，但原始 E838/E841/E856/E861/E863 证据尚未恢复，因此不列为主结果。

## 验证

```bash
cd 3d-reconstruction/data/results/github_release
python3 scripts/smoke_test.py
python3 scripts/test_workspace_layout.py
```

## 本地查看

```bash
python3 -m http.server 8766 --directory 3d-reconstruction
```

打开 `http://127.0.0.1:8766/onepage/`。该命令只提供本地静态预览，不会发布或上传。
