# CycleFusion: 3D Cycle-Consistency Guided Single-View Extrapolation

Date: 2026-09-05
Status: design (approved direction: 大外推 + 3D循环一致性)

## Problem (Gate 1)

- 现象/需求: 单视图前馈重建(Flash3D)在**大视角外推**(gap30/50, disocclusion 多)时严重退化;
  引入视频扩散先验(ViewCrafter)能补几何,但**会幻觉**,而幻觉内容在保真指标(PSNR)上必然掉分,
  在朴素 2D 融合里也无法被可靠抑制(E831 证伪: 高频幻觉 coarse 一致性检测不到)。
- 当前行为: test-time 优化用生成伪GT作固定视角监督,VC 幻觉被直接优化进 3D → PSNR 掉、ghosting。
- 期望行为: 一个可信度信号,能**自动区分「真几何」与「幻觉」**,只把可信的生成内容注入 3D 表示,
  在大外推下同时改善感知真实感(FID/LPIPS)且不牺牲(或少牺牲)保真度(PSNR)。
- 影响范围: test-time 优化器(新增 3D cycle-consistency 可信度)、评测(以 big30/big50 大外推为主, 报 FID+LPIPS+PSNR)。
- 不做什么: 不重训 Flash3D 网络(非蒸馏); 不换数据集(RealEstate10K big30/big50 已是足够外推);
  不做 2D Difix-VC 一致性(已证伪 E831)。

## Root cause (为什么幻觉之前抑制不了)

2D 一致性(Difix vs VC 逐像素/coarse比)只在**单一视角**内比较外观,
高频幻觉在单视角里"看起来合理"→ 检测不到。
**幻觉的本质是 3D 不一致**: VC 在视角 A 生成的内容, 若是幻觉, 从视角 B 看 / 回投到源视图会矛盾;
真几何则多视角自洽。→ 必须用 **3D/多视角** 而非 2D 单视角来做仲裁。

## Method (Gate 3): CycleFusion

核心: 用**当前 3D 高斯**作为几何媒介, 对每个生成的 novel view 像素算 **cycle-consistency 可信度**,
加权注入优化。不预融合成一张伪GT图(避免 ghosting)。

流程 (per scene, test-time):
1. Flash3D 前馈 → 初始高斯 G0 + 精确相机轨迹 (含大外推 novel poses)。
2. 生成先验伪观测: Difix (保真修复近视角) + ViewCrafter (几何生成远视角), 各视角 idx 一张。
3. **3D cycle-consistency 可信度** (核心创新):
   对生成视角 i 的像素 p:
   - 用 G0 的深度把 p 反投影成 3D 点 X (视角 i 的深度来自渲染 G0 的深度图)。
   - 把 X 投影到源视图 0 (和若干其他生成视角 j), 得到对应像素 p0。
   - cycle 误差 e(p) = 生成图_i(p) 与 [生成图_0/源图(p0)] 在可见处的外观差 (+ 可选深度/reproj几何差)。
   - 可见性: 用 G0 在视角 i 的 coverage α_i 和回投点是否落在 α_0>阈值 区域判定。
   - 可信度 conf_i(p) = 1 - smoothstep(e, e_lo, e_hi); 对**遮挡新区(源视图看不到, α_0低)**特殊处理:
     此处无法 cycle 验证 → 用 VC 的**时序一致性**(相邻生成帧 i-1,i+1 warp 到 i 的一致性)作 fallback 可信度。
4. 加权优化: L_i = conf_i ⊙ (生成图_i 与 render 的 L1/LPIPS); 源视图 anchor 全权重保真。
   幻觉像素 (cycle 矛盾) conf→低 → 不注入; 真几何 (cycle 自洽) conf→高 → 强监督。

关键区别 vs 已证伪方案:
- vs 蒸馏(E82x): test-time 保留自适应, 不压进固定权重。
- vs 2D融合(E79x/E835): 用 3D 多视角 cycle 而非 2D 单视角一致性; 不 pixel-blend → 无 ghosting。

## 最小性检查

1. 能否不改已有文件? 不能, 需新 test-time 优化器 (基于 _e720/_e835 扩展 reproject 一致性)。
2. 更少代码? cycle 可信度是一个新函数 + 优化 loop 加权; 复用现有渲染/相机/高斯。
3. 新依赖? 无; reproject 用现有相机内外参 + 渲染深度。

## 验证计划 (Gate 5)

- 主实验: 158 场景, 大外推 (big30/big50 为主) 全量。
- 对比: baseline(Flash3D) / Difix-only test-time / naive fusion(E790) / **CycleFusion(ours)**。
- 指标: **FID (主, 分布真实感)** + LPIPS + PSNR/SSIM。判据: CycleFusion 在 big30/big50 的 FID 明显优于
  naive fusion 和 Difix-only, 且 PSNR 不明显差于 Difix-only (证明幻觉被抑制)。
- 消融: cycle 可信度 on/off; 时序 fallback on/off; e_lo/e_hi 敏感度。
- 效果图: big50 上 baseline|Difix|naive-fusion|CycleFusion|GT 并排, 肉眼验证幻觉抑制 + 几何补全。

## 风险

- reproject 需要生成视角的深度: 用渲染 G0 的深度 (diff-gaussian-rasterizer 是否输出深度需确认; 否则用 override_color=depth 的 trick 或 median depth)。
- cycle 误差可能被光照/视角相关外观干扰: 用 LPIPS-like 或结构相似而非纯 RGB L1 降低敏感。
- 若 CycleFusion 仍无法在 PSNR 上不掉: 退而强调 FID/感知 + 效果图 (生成类论文可接受)。
