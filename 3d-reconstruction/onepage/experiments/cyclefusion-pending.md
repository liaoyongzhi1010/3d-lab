# CycleFusion 3D-cycle 门控（原始产物待恢复）

**证据状态：原始产物待恢复。** 本记录只索引本地设计与汇总日志，不把汇总数字列入 One-page 主结果。

## 目的
研究在测试时优化中，能否用跨视角 3D cycle consistency 降低生成伪观测的幻觉污染。

## 数据与代码版本
- 设计：[`../../papers/notes/cyclefusion_design_spec.md`](../../papers/notes/cyclefusion_design_spec.md)。
- 汇总日志：[`../../data/provenance/OracleGS_singleview_project.md`](../../data/provenance/OracleGS_singleview_project.md)。
- 毕设汇总：[`../../papers/thesis/thesis_chapter_outline.md`](../../papers/thesis/thesis_chapter_outline.md)。
- 缺失：E838/E841/E856/E861/E863 原始 JSON、完整配置、脚本快照和五列面板。

## 配置
汇总文本描述 Flash3D 初始化、Difix 锚定、生成候选、逐像素 cycle 权重和 per-scene 800-step 优化。因原始配置未归档，不能将此描述当作可执行配置。

## 运行方式
待补。当前仓库没有可独立执行并生成汇总数字的 CycleFusion 入口。

## 指标
内部汇总曾记录 big30/big50 FID 和 15-view AUROC，但原始结果不在 checkout；本页不复述为已验证结果。

## 效果图
待补。E841 的 `Flash3D | Difix | Naive | CycleFusion | GT` 面板、差异热力图和置信度图未在当前仓库找到。

## 观察
设计和长日志说明该方向主要关注 anti-ghosting 与幻觉定位；日志同时记录 CycleFusion 与 Difix-only 接近、端到端 LPIPS 增益很小。

## 失败点
缺少原始结果、配置、脚本和图像；不同文档还存在 Difix 与 CycleFusion 数值口径差异。无法独立核验或重跑。

## 结论
CycleFusion 当前只能作为“正在恢复证据的研究方向”，不能进入可复现主结果或对外 headline。
