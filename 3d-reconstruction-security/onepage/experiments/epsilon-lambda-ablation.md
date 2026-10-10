# Epsilon 与 Lambda 消融

## 目的
检查攻击预算与源视图约束强度如何影响源/目标视角损伤，并验证主实验现象是否只出现在单一设置。

## 数据与代码版本
- 数据：RealEstate10K present split。
- 代码：[`../../code/flash3d-attack/attacks/eval_dpc_flash3d.py`](../../code/flash3d-attack/attacks/eval_dpc_flash3d.py)。
- 结果：[`abl_eps4_n50.json`](../../data/results/abl_eps4_n50.json)、[`main_dpc_n100.json`](../../data/results/main_dpc_n100.json)、[`dpc_eps16_lam3.json`](../../data/results/dpc_eps16_lam3.json)、[`dpc_eps16_lam1.json`](../../data/results/dpc_eps16_lam1.json)。
- 仓库版本：`7477a9d`；实际运行 commit、seed 和上游 checkpoint hash 未记录。

## 配置
共同设置为 40 steps、4 个辅助位姿。epsilon 分别为 4/255、8/255、16/255；lambda 消融在 epsilon=16/255、n=10 下比较 3.0 与 1.0。不同 epsilon 的样本量为 50、100、10。

## 运行方式
```bash
python attacks/eval_dpc_flash3d.py --method dpc --steps 40 \
  --split /root/projects/flash3d/splits/re10k_mine_filtered/test_files_present.txt \
  --epsilon <budget> --lambda_src <weight> --max_scenes <n> \
  --out ../../data/results/<result>.json
```
完整运行仍要求外部 Flash3D、checkpoint 和 RE10K。

## 指标
4/255、8/255、16/255 的源视图 PSNR 下降分别约 1.52、2.62、5.66 dB；对应 tgt10 下降约 3.14、5.16、7.56 dB。epsilon=16/255 时，lambda 从 3 降到 1，源视图损伤从 5.66 dB 增至 11.51 dB。

## 效果图
当前没有为这些消融保存独立、带 manifest 的定性图，标记为“待补”。主实验图仅用于理解布局，不能冒充该消融结果。

## 观察
预算增大时攻击和源损伤都增大；降低 lambda 会显著破坏源视图，说明 source term 确实控制相对 camouflage。

## 失败点
三个预算使用不同样本量，不能把变化全部归因于 epsilon；lambda 消融只有 10 scenes。现有 `run_eps_sweep.sh` 还记录了另一组 geometry-hybrid 参数，但其输出未归档，不能纳入结论。

## 结论
当前小规模证据支持 source term 的必要性和预算-损伤趋势，但不足以给出稳定的最优 epsilon/lambda。
