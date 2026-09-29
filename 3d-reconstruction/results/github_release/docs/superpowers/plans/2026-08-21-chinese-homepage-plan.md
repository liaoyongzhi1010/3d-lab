# 中文总—分项目主页实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 GitHub 主页改成中文总—分结构，每篇论文都集中展示简介、Pipeline、明显效果图、核心数据表和入口。

**Architecture:** 顶层 README 先给共同主线和三篇关系总览，再使用统一模板分别描述 Paper 1/2/3。论文 PDF 保留 vector 图，主页在 `assets/` 使用 PNG。协议和 oracle-visibility 边界集中放在末尾，并在各数据表附近保留必要提示。

**Tech Stack:** GitHub Markdown、PNG、pdftoppm、Python smoke test、Git。

---

### Task 1: 导出主页图片

**Files:**
- Update: `assets/paper1_pipeline.png`
- Update: `assets/paper2_pipeline.png`
- Create: `assets/paper2_quality_speed.png`
- Update: `assets/paper3_pipeline.png`
- Create: `assets/paper3_granularity.png`
- Create: `assets/paper3_routing_strip.png`

- [ ] **Step 1: 从论文 PDF 导出 PNG**

```bash
pdftoppm -png -singlefile -r 150 paper1_selective_generation/paper/figs/fig_pipeline.pdf assets/paper1_pipeline
pdftoppm -png -singlefile -r 150 paper2_gate_aware_distillation/paper/figs/fig_pipeline.pdf assets/paper2_pipeline
pdftoppm -png -singlefile -r 150 paper2_gate_aware_distillation/paper/figs/fig_quality_speed.pdf assets/paper2_quality_speed
pdftoppm -png -singlefile -r 150 paper3_reliability_learning/paper/figs/fig_pipeline.pdf assets/paper3_pipeline
pdftoppm -png -singlefile -r 150 paper3_reliability_learning/paper/figs/fig_granularity.pdf assets/paper3_granularity
pdftoppm -png -singlefile -r 150 paper3_reliability_learning/paper/figs/fig_routing_strip.pdf assets/paper3_routing_strip
```

- [ ] **Step 2: 视觉检查**

使用图片读取工具检查六张图无裁切、无重叠、文字清楚。

### Task 2: 重写中文总览

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 写项目总览**

包含：中文标题、一句话主线、Paper 1 → Paper 2 → Paper 3 的关系、三篇总表和 PDF/代码入口。

- [ ] **Step 2: 保留科学边界**

首屏不堆叠限制，但总表中明确：Paper1/2 为 oracle-visibility 离线诊断，Paper3 是可靠性证据审计。

### Task 3: 写 Paper 1 分节

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 论文简介与 Pipeline**

说明单视图重建中的 disocclusion 问题与 offline oracle-visibility 注入机制。

- [ ] **Step 2: 放三个明显效果案例**

依次展示：窗户恢复 +12.58 dB、门廊/地板 +12.53 dB、沙发/窗户 +5.54 dB。

- [ ] **Step 3: 放核心数据表**

表格包含：N=166 mechanism r、759-frame LPIPS/FID、质量差 oracle、ACID aggregate 8 scenes、visible-region measured +0.016 dB。

### Task 4: 写 Paper 2 分节

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 论文简介与 Pipeline**

说明 oracle-curated、oracle-visibility upper-bound distillation，不声称部署 selector。

- [ ] **Step 2: 放质量—速度效果图**

使用 `assets/paper2_quality_speed.png`，说明它是固定 759-frame 自定义诊断。

- [ ] **Step 3: 放蒸馏数据表**

包含：holdout +5.77 dB、teacher +6.09 dB、94.7%、0.087 s、2833x、LPIPS/FID 与 teacher/component 的关系。

### Task 5: 写 Paper 3 分节

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 论文简介与 evidence-tier Pipeline**

说明目标是审计“什么证据真的可用于推理时可靠性判断”。

- [ ] **Step 2: 放 granularity 和 routing 诊断图**

使用 `assets/paper3_granularity.png` 和 `assets/paper3_routing_strip.png`，后者明确是 GT-quality probe 诊断，不是部署路由。

- [ ] **Step 3: 放证据分层数据表**

包含：oracle；GT-quality probe AUC .947；camera-only AUC .650；RGB+oracle-visibility AUC .426；population/call/worst gain 信息。

### Task 6: 收尾与验证

**Files:**
- Modify: `README.md`
- Create: `docs/superpowers/specs/2026-08-21-chinese-homepage-design.md`
- Create: `docs/superpowers/plans/2026-08-21-chinese-homepage-plan.md`

- [ ] **Step 1: 写复现与协议入口**

链接 `REPRODUCIBILITY.md`、`PROTOCOLS.md`、`TABLES_AND_FIGURES.md`、`MODEL_ZOO.md`、License/Citation。

- [ ] **Step 2: 验证链接和 smoke**

```bash
python3 scripts/smoke_test.py
git diff --check
```

预期：`SMOKE PASS`，无 diff 错误。

- [ ] **Step 3: 检查仓库安全**

```bash
find . -type f -size +90M -print
find . -iname '*.env' -o -iname '*secret*' -o -iname '*credential*'
```

预期：无输出。

- [ ] **Step 4: 提交并推送**

```bash
git add README.md assets docs/superpowers/specs/2026-08-21-chinese-homepage-design.md docs/superpowers/plans/2026-08-21-chinese-homepage-plan.md
git commit -m "docs: rebuild Chinese project homepage"
git push origin main
```
