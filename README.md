# 3D-Lab

单视图场景级三维重建研究，涵盖重建方法、综述调研与安全攻防三个方向。

## 目录结构

```
3d-lab/
├── 3d-reconstruction/          # 三维重建（主线研究）
├── 3d-reconstruction-survey/   # 三维重建综述
├── 3d-reconstruction-security/ # 三维重建安全
```

### `3d-reconstruction/` — 三维重建

主线研究：基于生成先验引导的单视图场景级三维重建。

核心工作：
- **测试时生成引导框架**：以 Flash3D 前馈重建为几何底座，引入 Difix（保真修复）、ViewCrafter（视频扩散外推）、FlashWorld（前馈 3D 生成）等生成先验，通过 test-time 优化 3DGS 改善大视角外推质量，零额外训练成本
- **3D 循环一致性门控（CycleFusion）**：通过跨视角回投一致性估计逐像素可信度，区分「真几何」与「幻觉」，只在可信区域注入生成监督，抑制幻觉污染（AUROC 0.9227 vs 2D baseline 0.7144）
- **选择性生成注入**：不对称几何注入 + 测试时可靠性门控 + 可学习注入权重，遮挡区 PSNR +2.07 dB，可见区基本无损
- **Gate-aware 蒸馏**：按教师质量筛选可靠样本蒸馏前馈补全网络，23% 数据超越全量 baseline（Δ 提升 2.7×）

评估数据集：RealEstate10K（158 场景）+ ACID（40 场景）
主要指标：FID（分布真实感）、LPIPS、PSNR、幻觉检测 AUROC

### `3d-reconstruction-survey/` — 三维重建综述

三维重建领域的文献调研与综述材料，涵盖：
- 单视图/多视图三维重建方法梳理
- 3D Gaussian Splatting 相关工作
- 生成式三维重建（扩散模型、视频生成、多视图生成）
- 前馈 vs 生成式方法对比分析
- 竞品方法详细对比（Flash3D、CATSplat、pixelSplat、MVSplat、DepthSplat 等）

### `3d-reconstruction-security/` — 三维重建安全

3D Gaussian Splatting 的对抗攻击与安全研究，涵盖：
- 3DGS 渲染管线的对抗脆弱性分析
- 针对 3DGS 的攻击方法
- 防御与鲁棒性评估
