# 中文总—分项目主页设计

日期：2026-08-21

## 目标

将仓库首页重构为中文学术项目主页。读者应在首屏理解三篇工作的共同主线，并可依次查看每篇论文的 Pipeline、明显效果、核心数据和论文/代码入口。

## 页面结构

1. 项目标题与一句话主线。
2. 三篇关系总览：Paper 1 诊断生成机制，Paper 2 压缩有效行为，Paper 3 审计可靠性证据。
3. 总表：论文、核心问题、主要结论、入口。
4. Paper 1：简介、Pipeline、三个显著局部恢复案例、标准指标/机制数据表。
5. Paper 2：简介、Pipeline、质量—速度效果图、蒸馏与运行时间数据表。
6. Paper 3：简介、Pipeline、granularity/evidence-tier 与 routing 图、审计数据表。
7. 复现入口、协议边界、模型依赖、License/Citation。

## 视觉规范

- Pipeline 保持语义配色：蓝色几何/证据、紫色生成教师、橙色学生/预测器、黄色选择或 ROI、绿色输出。
- Paper 1 使用真实 RGB 对比图，固定黄色框与相同 crop。
- Paper 2 没有公开的 student RGB 定性渲染，使用质量—速度图，不伪造视觉对比。
- Paper 3 是证据审计，使用 granularity 和 routing/evidence 图，不把诊断上界包装为部署效果。
- GitHub 首页统一使用 PNG；论文继续保留 vector PDF。

## 科学边界

- Paper 1/2 使用由完整目标序列预计算的 oracle visibility，只能称为离线诊断/上界。
- Paper 1 的 quality-gap 选择使用 GT 质量探针，属于 oracle selectivity。
- Paper 2 使用 oracle 高收益数据筛选，不存在已验证的推理时 selector。
- Paper 3 的 0.947 AUC 是 GT-quality-probe 诊断；真正可观测 camera-only AUC 为 0.650；RGB + oracle visibility 的选定子集 AUC 为 0.426。
- 所有外部方法数字保持协议隔离，不作跨协议胜负结论。

## 验收

- README 采用明确的总—分层级。
- 每篇均有 Pipeline、效果图、数据表和入口。
- 所有图片链接存在并可在 GitHub Markdown 中展示。
- `python3 scripts/smoke_test.py` 通过。
- `git diff --check` 通过，无大文件或敏感文件。
