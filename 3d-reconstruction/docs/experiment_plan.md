# 实验规划：单视图3DGS不可见区前馈生成

## 一、课题定位与硬约束

**课题**：单视图三维重建中，对不可见区(disocclusion)的前馈生成式显式3DGS补全。

**硬约束（不可松动）**：

1. 推理输入：单视图（无多视图/视频）
2. 推理方式：训练式前馈（不接受 per-scene optimization / test-time optimization）
3. 输出表示：显式3D Gaussian Splatting
4. 硬件：单卡 A6000 48GB（红线 44GB）
5. 不用 GAN（过时范式，v1已证FID微赢但视觉糊）

**效果目标**：

- 视觉真实感（锐利、无speckle/糊/涂抹，论文figure可放）
- 不可见区 FID/KID 显著优于竞品（Flash3D/CATSplat 大视差下FID暴涨8-10倍）
- 跨视图一致性超越 2D-lift 方法
- 可见区 PSNR 保持 parity（masked-blend 保 Flash3D 26.6dB）
- 全图 PSNR 诚实报（不硬凑，owning 结构性劣势）

## 二、调研核心结论（60+ 篇，三轮 deep-research 核实原文）

### 2.1 成功范式

能出"清晰不可见区"的强方法全部在 **latent 空间**（扩散/视频/结构化 3D latent）生成，借视频/多视图扩散大先验：

- 参数空间回归 → 糊（M1 AGC 印证，10.5dB 落差于 oracle）
- 像素空间 flow → speckle 噪声（M2 flow ep300 印证，FID 301 > Flash3D 283）

### 2.2 前馈与生成的矛盾解法（2025-2026 主流）

| 路线 | 代表工作 | 核心思路 |
|------|----------|----------|
| 蒸馏 | Difix3D+(CVPR25), FlashWorld, Lyra, CrowdGaussian(CVPR26) | 违规teacher离线产标签→训合规student |
| 单步扩散增强 | ProSplat(arXiv25.06), GIFSplat | 前馈骨架 + 一步diffusion细化 |
| Latent前馈 | Wonderland, Bolt3D(ICCV25), Diff4Splat(CVPR26) | 在预训练VAE latent空间生成 |
| Residual-over-mean | latentSplat(ECCV24), GenWarp(NeurIPS24) | 确定性回归mean + 生成器只学残差 |

### 2.3 立论空间

截至 2026.07，「单视图→3DGS + 不可见区专门前馈生成」在顶会**仍无正面命题工作**。最接近的：

- Complete-GS (arXiv 2508): 正面竞品但用 diffusion 采样非前馈
- VidSplat (SIGGRAPH26): 迭代优化式
- ProSplat: 两阶段但不在单视图 MINE 协议

### 2.4 RE10K 最新数字

单视图 MINE 协议：Flash3D 28.68/26.09/25.10 | CATSplat 29.16/26.50/25.51（我方复现对齐论文）。无更新单视图方法刷新 → 我们若报"可见区 parity + 不可见区 FID 显著降"即 SOTA 贡献。

## 三、历史方法复盘与定性（5个方法，真实数据）

| 方法 | 核心 | 结果 | 状态 | 根因 |
|------|------|------|------|------|
| M1 AGC回归头 | 参数空间 3D-GT MSE | invis+8.2dB但糊 | 已否 | 单GT回归赌均值，小头3.3M到顶 |
| M2 Flow-matching | 像素空间 rectified flow | diversity 19x但speckle | 待复测 | 旧评测不公平(3K vs 15K)；新尺子下ep50反赢FID |
| M3 GAN completion | 2D PatchGAN对抗 | 隐藏区FID微赢(283→273)但竖条纹涂抹 | 已否 | 2D image-space + 无强预训练先验 |
| M4 sdfit(per-scene) | SD-anchor + 多视图拟合 | 大洞视觉锐利，跨视图一致性+6.9dB | 当teacher | 违前馈约束，推理含SD+250iter优化 |
| M5 Student蒸馏 | teacher label → 前馈student | 进行中 | 主线 | 正在执行 |

## 四、先决步骤：重测历史 ckpt（澄清"谁被冤枉"）

### 4.1 为什么必须先做

- M2 flow-matching 曾被"只训3K + 旧评测(10样本/GT-MAE不等于0/FID口径乱)"冤枉判死
- 新评测工程已闭环（官方split 3100样本 / cleanfid / check_gt_consistency GT-MAE=0 / 统一尺子）
- 历史 ckpt 全在服务器，GPU 全空闲，重测成本极低（纯推理+评测，无需重训）
- 若 M2 在新尺子下表现好 → 可能改变主线优先级

### 4.2 重测计划

| ckpt | 路径 | 评测内容 | 用时估计 |
|------|------|----------|----------|
| M2 flow ep50 | /home/data/E-052_flow_v4_smoke/ | wide700 FID/KID + present PSNR + 定性图 | 1h |
| M2 flow ep300 | /home/data/E-052_flow_v4/ | 同上 | 1h |
| M4 sdfit (teacher) | _e050_flash3d_sdfit.py | 确认teacher label质量(已知好，复验) | 0.5h |
| M5 student v1 | /home/data/E-052_student_v1/ | 新尺子全套 | 1h |

**判据**：M2 flow 在 wide700 隐藏区 FID < Flash3D(283) 且定性图无致命 speckle → 升级为候选方案组件。

## 五、候选方案排序

### 方案 A（主线，最高优先）：Teacher→Student 蒸馏

**假设**：用 M4 sdfit（违规但锐利）当 teacher 离线在 TRAIN 场景产伪标签（锐利洞区 3DGS 参数），训前馈 student 一次输出洞区高斯。推理零 SD 零优化合规。

**为何可行**：

- E-022 铁证：参数空间 3D-GT 监督破均值回归（+8.2dB），teacher 把多模态 p(hole|input) 坍缩成单个锐利确定目标 → student 回归 well-posed
- Difix3D+(CVPR25)/CrowdGaussian(CVPR26) 同构路线已被顶会接受

**关键设计选择**：

| 维度 | 选项A1 | 选项A2 | 选项A3 |
|------|--------|--------|--------|
| Student架构 | 直接参数回归(像M1但监督=teacher) | Residual-over-mean(回归mean+flow学残差) | Latent-space(先encode再decode) |
| 监督信号 | 纯参数L2 | 参数L2 + render L1/LPIPS | 参数L2 + render + FID蒸馏 |
| 条件化 | base render + mask + depth | + Flash3D 特征 | + DINO/CLIP embedding |

**推荐**：A2(Residual-over-mean) + 参数L2+render LPIPS + base render+mask+depth 条件。

理由：Residual-over-mean 天然抑制 speckle（光滑区残差约等于0，只在需要处生成细节）；参数+render双监督平衡几何精度与感知质量；条件化轻量不增显存。

**显存预算**（估算）：

- Flash3D backbone frozen: 约4GB
- Student head (UNet-like, 约50M params): 约3GB 权重 + 约8GB 激活(grad ckpt)
- Teacher labels 预缓存: 0 GPU（离线生成存盘）
- 训练 batch=4, 256x384: 总约20-25GB，远在红线内

**风险与对策**：

- Teacher label 噪声 → 按 teacher 渲染质量过滤（LPIPS < 阈值才入库）
- Student 容量不足 → 从 50M 起步，不够再加到100M
- 轻微回归均值 → 加 render LPIPS（感知loss抗糊）

**里程碑**：

1. Teacher label cache（TRAIN 场景 800个，离线1-2天）→ 已完成
2. Student smoke train（1K iter, 验证 loss 下降 + 渲染不崩）
3. Student full train（15-20K iter, loss plateau）
4. 新尺子评测（FID/KID/DISTS/一致性/定性图）
5. 3-seed + verify

### 方案 B（增强备选）：Latent-space Residual Flow

**假设**：在预训练 VAE 的 latent 空间做 residual flow-matching。先用确定性回归器预测 mean（latent 空间），再用 flow 只学 residual。

**为何考虑**：所有强方法都在 latent 空间生成（Bolt3D/latentSplat/GenWarp/Diff4Splat）；M2 像素空间 flow 的 speckle 根因 = 像素空间不平滑 → latent 空间流形天然平滑；Residual-over-mean = 光滑区残差约0 = 无 speckle 且 FID 友好。

**与方案A的关系**：B 是 A 的 student 架构升级版。若 A 效果好，不需要 B。若 A 出现残留糊/speckle，B 是结构性 fix。

**显存**：latent dim 更小(4ch vs 14ch高斯参数) → 比 A 更省。总约18-22GB。

**风险**：VAE 重建瓶颈（latent→pixel 不可逆损失）；调试周期长。

### 方案 C（探索性）：单步 Diffusion 细化（ProSplat 式）

Flash3D 前馈重建 + 一步 diffusion（SD 蒸馏的单步模型）只在洞区做 latent 细化。

低优先级：ProSplat 已做，直接用有"非自己贡献"嫌疑；单步 diffusion 蒸馏工程复杂；与 teacher→student 路线重叠。

### 方案 D（保底）：ViewCrafter Teacher + 多视图蒸馏

用 ViewCrafter_25（视频扩散，在 HF 缓存）离线生成多视图辅助帧作为 teacher label 来源替代 sdfit。

保底原因：ViewCrafter 代码仓库不在服务器 + 缺依赖 + setup 风险高；sdfit teacher 已证有效优先用已有的。

## 六、评测判据（统一，贯穿所有方案）

| 指标 | 口径 | 目标 | 锚点 |
|------|------|------|------|
| 可见区 PSNR | 官方 MINE present split 3100, masked-blend | 大于等于 Flash3D 26.6（parity） | Flash3D 28.68/26.09/25.10 |
| 隐藏区 FID | wide700 大视差, cleanfid | 小于 Flash3D 283 显著 | Flash3D 282.9 |
| 隐藏区 KID | 同上 | 小于 Flash3D 0.097 | 0.0971 |
| DISTS | latentSplat 官方实现 | 报告 | — |
| 跨视图一致性 | held-out 帧 PSNR, multigap split | 大于 2D-lift +0.85dB | 2D-lift 13.70 |
| 全图 PSNR | 官方 MINE | 诚实报，不硬凑 | Flash3D 26.6 |
| 定性图 | 按 LPIPS 分位选（防挑图），4+ 张 montage | 锐利无speckle/糊/涂抹 | — |

**Kill criteria（触发则换方案）**：

- 隐藏区 FID > Flash3D（即没有任何改善） → 换方案
- 定性图仍然糊/speckle 且训练已 plateau → 换方案
- 显存 > 44GB → 降配置或换架构

## 七、执行里程碑

| 阶段 | 内容 | 估时 | 前置 |
|------|------|------|------|
| M0 重测 | 历史 ckpt（M2/M4/M5）用新尺子重测，澄清冤案 | 0.5天 | 无 |
| M1 Teacher cache | sdfit 在 TRAIN 800 场景离线产 teacher labels | 1-2天 | M0（已完成） |
| M2 Student smoke | 方案A student 1K iter smoke | 0.5天 | M1 |
| M3 Student full | 15-20K iter 训练到 plateau | 2-3天 | M2 |
| M4 评测 | 新尺子全套 + 定性图 + 竞品对比 | 1天 | M3 |
| M5 迭代 | 根据结果决定：够好→3seed+verify；不够→试B | 1-3天 | M4 |
| M6 定稿 | 3-seed verify + 论文 figure + 飞书报告 | 1天 | M5 |

总估时：7-10天（含GPU等待时间）。

## 八、风险矩阵

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| Teacher labels 大量场景失败 | 中 | 高 | 按渲染质量过滤；失败场景不入训练集 |
| Student 容量不足（糊） | 中 | 中 | 从50M起步，渐增至100M；加render LPIPS |
| Speckle 残留 | 低 | 高 | Residual-over-mean结构(方案B)；uncertainty gating |
| 显存超标 | 低 | 中 | grad ckpt + 降batch + 混合精度 |
| 与ProSplat/CrowdGaussian撞车 | 中 | 中 | 差异化=不可见区专门+teacher是per-scene多视图(非扩散) |
| 新尺子下所有方法都赢不了Flash3D | 低 | 致命 | 大视差FID暴涨8-10倍是铁证=Flash3D大洞必糊=有空间 |

## 九、与现有工作的差异化

| 维度 | 我们 | Difix3D+(CVPR25) | CrowdGaussian(CVPR26) | ProSplat |
|------|------|---------|---------|---------|
| Teacher来源 | per-scene多视图几何拟合(3D原生) | 单步diffusion | 单步diffusion蒸馏 | 两阶段diffusion |
| 不可见区处理 | 显式mask+专门洞区高斯 | 全图一步refine | 全图一步refine | 全图(无显式洞区) |
| 3D原生保证 | 多视图held-out一致性+6.9dB | 隐式(diffusion不保证) | 隐式 | 无 |
| 评测 | FID/KID/DISTS/一致性/分区 | 仅全图PSNR | 仅全图+FID | 全图 |

**核心novelty**：在"单视图→3DGS"的严格约束下，首次将不可见区补全作为正面命题，用3D原生teacher(per-scene多视图拟合)的锐利+一致性蒸馏入前馈student，且用完整的分区评测（FID/一致性/定性）量化贡献。
