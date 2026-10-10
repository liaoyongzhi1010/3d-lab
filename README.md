# 3D-Lab

面向三项研究工作的可维护、可复现研究空间。每个项目统一保留 `README.md`、`data/`、`code/`、`onepage/`、`papers/`。

| 研究工作 | 当前状态 | One-page |
|---|---|---|
| 基于生成模型的稀疏视角三维重建综述 | 研究骨架；文献库、检索记录和结论待补 | [打开](3d-reconstruction-survey/onepage/index.html) |
| 基于生成式引导的单视图场景级三维重建 | 已整理 release 证据、代码、论文和六条实验记录 | [打开](3d-reconstruction/onepage/index.html) |
| 三维重建安全 | 已整理 DPC 攻击、初步防御评估、结果和六条实验记录 | [打开](3d-reconstruction-security/onepage/index.html) |

## 目录

```text
3d-lab/
├── 3d-reconstruction-survey/
│   ├── README.md
│   ├── data/
│   ├── code/
│   ├── onepage/
│   └── papers/
├── 3d-reconstruction/
│   ├── README.md
│   ├── data/
│   ├── code/
│   ├── onepage/
│   └── papers/
└── 3d-reconstruction-security/
    ├── README.md
    ├── data/
    ├── code/
    ├── onepage/
    └── papers/
```

## 证据原则

- 结果结论链接到具体 JSON、manifest、checkpoint、脚本或实验记录。
- 缺少原始产物的方向明确标记“待补”或“原始产物待恢复”。
- 不跨 split、视角数量、分辨率、crop 或 LPIPS backbone 直接比较。
- 原始数据和大型第三方权重不进入 Git；各项目的 `data/README.md` 记录外部依赖与缺口。
- 保留负结果、失败案例和不可复现边界。

## 本地预览

在仓库根目录运行：

```bash
python3 -m http.server 8770 --bind 127.0.0.1 --directory .
```

打开：

- `http://127.0.0.1:8770/3d-reconstruction-survey/onepage/`
- `http://127.0.0.1:8770/3d-reconstruction/onepage/`
- `http://127.0.0.1:8770/3d-reconstruction-security/onepage/`

这些是本地静态预览。本次整理不推送、不发布、不合并。
