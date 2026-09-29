# 实验方案评审记录：单视图3DGS不可见区前馈补全

## 0. 使用规则

从本版本开始，后续每一个新实验方案都必须先进入本文档，完成**人工review**后再执行。

流程：

1. Agent 提交方案草案
2. 人工review：确认是否符合课题目标、论文叙事、顶会可发表性
3. 只有状态变为 **Approved** 后，才允许启动训练/大规模评测
4. 被否决方案保留，不删除，避免重复踩坑

状态定义：

| 状态 | 含义 |
|------|------|
| Draft | agent 初稿，尚未讨论 |
| Pending Review | 等人工review |
| Approved | 可以执行 |
| Rejected | 不执行，保留原因 |
| Parked | 暂停，后续可恢复 |
| Running | 已启动实验 |
| Done | 实验完成，有结果 |

---

# 方案01：开源顶会Teacher蒸馏 + Residual-over-mean Flow Student

**状态**：Pending Review  
**提出时间**：2026-07-22  
**人工结论**：待定  
**是否已执行**：部分误启动了内部flow训练；后续应暂停大规模继续，待人工确认teacher选择后再正式执行。

## 1. 方案一句话

用开源顶会级生成/NVS方法作为 teacher，离线产生不可见区监督，再训练我们的前馈 student，一次性输出洞区显式 3D Gaussians。

Student 采用 **Residual-over-mean Flow**：mean head 先预测稳定但偏糊的高斯参数，flow head 只生成残差细节，最终输出：

```text
hole_gaussians = mean + gate * residual_flow
```

## 2. 为什么提出这个方案

当前 student v1 图像已确认：纯回归 student 会输出大面积模糊 blob，没有纹理，FID/KID 输 Flash3D。

已知失败链：

| 路线 | 问题 |
|------|------|
| 纯回归 student | 均值糊，纹理消失 |
| 纯 pixel/parameter flow | speckle/花斑，平滑区域乱生成 |
| 自研 sdfit teacher | 不够公认，论文里可能被质疑 teacher 自造 |

因此新方案希望同时解决三个问题：

1. teacher 需要更公认：优先选开源顶会方法
2. student 需要前馈合规：推理不使用teacher、不优化、不用SD
3. 生成需要稳定：mean 保结构，flow 只补残差细节

## 3. Teacher 选择原则

Teacher 不应优先使用我们自研 sdfit 作为论文主线 teacher。更稳的选择是开源、顶会、强公认方法。

Teacher 必须满足：

| 条件 | 原因 |
|------|------|
| 开源可跑 | 审稿可复现 |
| 顶会或强公开方法 | 避免“自造teacher”质疑 |
| 能产生不可见区内容 | 不是只做可见区重建 |
| 最好能输出多视图一致结果 | 方便蒸馏成3DGS |
| teacher 可慢 | 只离线使用，student推理合规即可 |

候选 teacher：

| 候选 | 类型 | 优点 | 风险 | 当前动作 |
|------|------|------|------|----------|
| ViewCrafter | 单图/视频生成NVS | 生成先验强，多视图内容较丰富 | 输出2D，需要转3DGS标签；环境需核实 | 待smoke |
| CAT3D | 多视图生成 | 顶会/强生成路线，适合不可见内容 | 可能非3DGS，推理慢 | 待核实开源和资源 |
| GenWarp | 生成式NVS/warp | 论文相关性强，适合作teacher | 输出/代码适配待查 | 待核实 |
| Complete-GS / ProSplat | 3D补全/3DGS相关 | 如果开源且可跑，最接近理想teacher | 开源状态和协议需确认 | 待查 |
| sdfit(自研) | per-scene优化teacher | 已有label，可调试student | 不作为论文主teacher，最多fallback | Parked |

## 4. Student 输入输出

### 输入

Student 推理时只使用单图经过 Flash3D/CATSplat backbone 得到的信息：

```text
cond = [base_render, hole_mask, depth]
```

可扩展：Flash3D feature、uncertainty/confidence。

### 输出

Student 输出洞区每个像素对应的显式高斯参数：

```text
gbuf = [depth_z, color/SH, scale, opacity, rotation]
```

然后通过反投影生成 3D Gaussian：

```text
hole pixel (u,v) + predicted depth z + camera K
→ target-camera 3D point
→ transform to source coordinate
→ merge with Flash3D Gaussians
→ render by same 3DGS rasterizer
```

## 5. Residual-over-mean Flow 细节

### Mean Head

预测稳定粗结果：

```text
mean = 模糊但结构/颜色大致正确的高斯参数
```

作用：提供几何和颜色的低频结构，避免纯flow从噪声生成整个洞区导致不稳定。

### Flow Head

只预测残差：

```text
residual = teacher_target - mean
```

作用：补充高频纹理、边缘、细节。

### Gate

逐像素/逐通道控制残差强度：

```text
final = mean + sigmoid(gate) * residual
```

期望行为：

| 区域 | gate |
|------|------|
| 平墙、天花板 | 小，避免speckle |
| 砖墙、植被、木纹 | 大，补纹理 |

## 6. 训练监督

Teacher 离线生成伪标签，student 学：

```text
teacher target gaussian parameters
teacher target render
```

损失：

```text
loss_mean = L1(mean, teacher_target)
loss_flow = flow_matching(residual)
loss_render = L1/LPIPS(render(student), render(teacher))
```

注意：若 teacher 输出是2D多视图，需要先做 teacher-to-3DGS label conversion：

```text
teacher target image + depth + camera pose + hole mask
→ back-project
→ construct hole Gaussians
→ student label
```

## 7. 推理流程

```text
输入单张图
→ Flash3D/CATSplat 得到可见区3DGS
→ 检测新视角洞区
→ Student 一次前馈输出洞区3DGS
→ 合并可见区高斯和洞区高斯
→ 得到完整显式3DGS
```

推理阶段不使用：

- SD
- GAN
- 多视图输入
- per-scene optimization
- test-time optimization

## 8. 论文卖点

初步表述：

> We distill the invisible-region prior of an open-source state-of-the-art generative NVS teacher into a feed-forward explicit 3DGS completion student.

中文解释：

把开源顶会生成模型的不可见区先验，蒸馏成单图前馈显式3DGS补全网络。

贡献点：

1. Teacher-to-3DGS distillation：把开源生成teacher转成显式高斯监督
2. Residual-over-mean Flow：mean保证稳定结构，flow只补残差细节
3. 单图前馈显式3DGS补全：推理合规、快速、可渲染任意视角
4. 使用开源顶会评测器报告 FID/KID/DISTS/PSNR/LPIPS

## 9. 需要人工review的问题

请重点review以下问题：

1. Teacher 是否必须限定为开源顶会方法？
2. Teacher 候选优先级是否认可：ViewCrafter / CAT3D / GenWarp / ProSplat / Complete-GS？
3. sdfit 是否只作为内部调试，不进入论文主线？
4. Student 的 residual-over-mean flow 是否符合“3DGS补全”主线？
5. 是否允许先跑 teacher smoke，再决定最终teacher？
6. 评测仍然只用开源顶会评测器，不自造mask协议，是否保持？

## 10. 暂定执行门槛

只有人工review通过后，才执行以下动作：

1. 核实 teacher 开源代码和权重可跑
2. 每个 teacher 跑10个场景 smoke
3. 人工看图选teacher
4. 批量生成 3DGS pseudo-label
5. 训练 residual-flow student
6. 再进入大规模评测

## 11. 当前状态备注

- student v1 已评测：纯回归 student FID/KID 输 Flash3D，定性图是模糊blob。
- 这说明 A1 纯回归路线不够。
- residual-flow 的旧版 v4 在 ep50 曾有隐藏区 FID 小赢 Flash3D，但充分训练 ep300 过拟合变差；说明该方向有信号但需要更稳的 teacher/训练策略。
- 后续不应直接把自研 sdfit teacher 写成主线，需优先换成开源顶会 teacher。
