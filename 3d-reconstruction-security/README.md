# 三维重建安全

本项目研究单图前馈 3D Gaussian Splatting 的攻击与初步防御评估。当前核心工作是 Depth-Parallax Confusion（DPC）：在输入图像的 L-infinity 预算内优化扰动，约束源视图变化并扩大新视角渲染差异。

## 入口

- [One-page 总览](onepage/index.html)
- [实验记录](onepage/experiments/)
- [数据与结果说明](data/README.md)
- [代码与运行说明](code/flash3d-attack/README.md)
- [论文草稿](papers/paper_dpc.md)

## 目录

```text
3d-reconstruction-security/
├── README.md
├── data/       # 机器可读结果、正式定性图和外部数据说明
├── code/       # 攻击、评测、测试和复现入口
├── onepage/    # 总览页及逐实验记录
└── papers/     # 论文草稿
```

## 当前证据边界

主实验覆盖 Flash3D / RealEstate10K 的 100 个 scenes。结果支持“新视角损伤大于源视图损伤”，但源视图并非无损。共享深度先验实验只证明同一依赖链路的变化；VGGT 迁移结果因场景 context 复用缺陷审计无效。完整跨模型新视角攻击、系统防御研究和精确端到端重跑仍未完成。

## 本地查看

```bash
python3 -m http.server 8765 --directory 3d-reconstruction-security
```

打开 `http://127.0.0.1:8765/onepage/`。该命令只提供静态预览，不会发布或上传内容。
