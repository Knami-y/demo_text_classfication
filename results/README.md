# 结果与图表

## `figures/` —— 报告用的对比曲线

由 TensorBoard 导出，把 7 组实验画在同一张图上，是实验报告的核心插图。

| 文件 | 内容 | 对应报告插图 |
| --- | --- | --- |
| `dev_acc曲线.png` | `Accuracy/dev`：各实验验证集准确率随 epoch 变化 | 图 2 |
| `train_acc曲线.png` | `Accuracy/train`：各实验训练集准确率随 epoch 变化 | 图 3 |
| `train_loss曲线.png` | `Loss/train`：各实验训练损失随 epoch 变化 | 图 4 |

> 报告中的"图 1（各实验 Run 与颜色对应）"只在 `.docx` 内部，没有对应的独立图片文件。

## `screenshots/` —— 单个实验的 TensorBoard 截图

| 文件 | 实验 | 说明 |
| --- | --- | --- |
| `base.png` | Baseline | 基线训练过程 |
| `base_test_acc83.png` | Baseline | 基线在测试集上的结果（通过 83% 参考指标） |
| `1e-5lr.png` | LR-1 | lr = 1e-5 |
| `0.03dropout.png` | Dropout-1 | hidden dropout = 0.03 |
| `0.3dropout.png` | Dropout-2 | hidden dropout = 0.3 |
| `4batch_size.png` | BS-1 | batch size = 4 |
| `64batch_size.png` | BS-2 | batch size = 64 |

## 读图注意事项

TensorBoard 图中同时显示 **Smoothed** 与 **Value** 两条线：

- **Value** = 原始数值，报告表格与结论都以它为准；
- **Smoothed** = 仅为视觉平滑，不作为最终数值。

另外 `Step = 0 / 1 / 2` 分别对应第 `1 / 2 / 3` 个 epoch。

## 与日志目录的对应关系

截图里的 Run 名称就是 `runs/` 下的日志目录名（`runs/` 未入库，但本地保留）。
完整对照表见项目根目录 [`README.md`](../README.md) 的"脚本 ↔ 日志 ↔ 权重 对照表"。
