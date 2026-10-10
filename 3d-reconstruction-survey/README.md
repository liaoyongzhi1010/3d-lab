# 基于生成模型的稀疏视角三维重建综述

本项目用于建立可追溯的稀疏视角生成式三维重建综述。目前只有研究范围，尚无系统检索、文献库、taxonomy、图表或可引用结论。

## 入口

- [One-page 当前状态](onepage/index.html)
- [文献数据计划](data/README.md)
- [代码状态](code/README.md)
- [综述正文状态](papers/README.md)
- [检索记录](onepage/experiments/README.md)

## 目录

```text
3d-reconstruction-survey/
├── README.md
├── data/
├── code/
├── onepage/
└── papers/
```

## 研究范围

拟覆盖单视图、少视图和稀疏视角重建中的生成先验，包括扩散/视频生成、多视图生成、显式 3DGS 与隐式表示。正式撰写前必须先冻结任务定义和协议字段。

## 本地查看

```bash
python3 -m http.server 8767 --directory 3d-reconstruction-survey
```

打开 `http://127.0.0.1:8767/onepage/`。当前页面不会展示未经核查的文献结论。
