# DPC 主实验（n=100）

## 目的
验证 DPC 是否能在控制源视图损伤的同时，对未参与攻击优化的目标视角造成更大的退化。

## 数据与代码版本
- 数据：RealEstate10K，JSON 记录的 split 为 `splits/re10k_mine_filtered/test_files_present.txt`。
- 代码：[`../../code/flash3d-attack/attacks/eval_dpc_flash3d.py`](../../code/flash3d-attack/attacks/eval_dpc_flash3d.py)。
- 结果：[`../../data/results/main_dpc_n100.json`](../../data/results/main_dpc_n100.json)。
- 仓库版本：当前成果首次进入本仓库的提交为 `7477a9d`；实际服务器运行 commit、上游 Flash3D commit/checkpoint hash 和随机种子未记录。

## 配置
`method=dpc`，`epsilon=0.0314`，`steps=40`，`step_size=1/255`，`lambda_src=3.0`，4 个辅助位姿，100 scenes。

## 运行方式
在 Flash3D CUDA 环境中，从 `code/flash3d-attack/` 执行：
```bash
python attacks/eval_dpc_flash3d.py --method dpc --max_scenes 100 --steps 40 \
  --split /root/projects/flash3d/splits/re10k_mine_filtered/test_files_present.txt \
  --epsilon 0.0314 --lambda_src 3.0 \
  --out ../../data/results/main_dpc_n100.json
```
脚本仍依赖服务器上的 `/root/projects/flash3d`；当前仓库未包含完整环境和数据。

## 指标
PSNR、SSIM、LPIPS-VGG；Flash3D evaluator 使用 5% border crop。源视图 PSNR 下降 2.62 dB；tgt5、tgt10、tgt_rand 分别下降 4.78、5.16、4.97 dB。

## 效果图
- [`scene7_dpc.png`](../../data/qualitative/scene7_dpc.png)
- [`scene20_dpc.png`](../../data/qualitative/scene20_dpc.png)

## 观察
三个目标视角的平均损伤约 4.97 dB，大于源视图损伤 2.62 dB。固定 +5 到 +10 目标的损伤随视差增大。

## 失败点
源视图并非无损，仍下降约 2.6 dB；“camouflage”只能表述为相对隐藏。没有保存精确 100-scene 清单和运行时版本，无法证明精确重跑。

## 结论
当前证据支持“DPC 对目标视角的破坏大于源视图”这一相对结论，不支持“源视图完全不可察觉”或“已覆盖全部 MINE 测试集”。
