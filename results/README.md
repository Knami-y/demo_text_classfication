# 结果与图表

## `figures/` —— 7 组实验的对比曲线

由 TensorBoard 导出（关闭平滑），把 7 组实验画在同一张图上，是复盘实验的主要依据。

| 文件 | 内容 |
| --- | --- |
| `dev_acc.png` | `Accuracy/dev`：验证集准确率随 epoch 变化 |
| `dev_loss.png` | `Loss/dev`：验证集损失随 epoch 变化 |
| `train_acc.png` | `Accuracy/train`：训练集准确率随 epoch 变化 |
| `train_loss.png` | `Loss/train`：训练损失随 epoch 变化 |

## `screenshots/` —— 单次运行的终端截图

| 文件 | 实验 | 内容 |
| --- | --- | --- |
| `base.png` | Baseline | 基线训练日志，最佳 Dev Acc 85.40%（Epoch 3） |
| `exp_lr_1e-5.png` | LR-1 | `lr=1e-5` 训练日志，最佳 Dev Acc 85.50%（Epoch 2） |
| `exp_lr_1e-4.png` | LR-2 | `lr=1e-4` 训练日志，最佳 Dev Acc 78.20%（Epoch 1） |
| `exp_batch_4.png` | BS-1 | `batch_size=4` 训练日志，最佳 Dev Acc 84.60%（Epoch 3） |
| `exp_dropout_0.03.png` | Dropout-1 | `hidden_dropout=0.03` 训练日志，最佳 Dev Acc 86.00%（Epoch 3） |
| `exp_dropout_0.3.png` | Dropout-2 | `hidden_dropout=0.3` 训练日志，最佳 Dev Acc 84.30%（Epoch 2） |
| `test_base_acc84.png` | Baseline | `src/test.py` 运行截图；截图为模型加载阶段，最终 Test Accuracy 84.40% 见根目录 README |

`batch_size=64` 这组没有单独的终端截图，其曲线包含在 `figures/` 的 7 条线中。

## 读图注意事项

TensorBoard 图中同时显示 **Smoothed** 与 **Value** 两条线：

- **Value** = 原始数值，README 表格与结论都以它为准；
- **Smoothed** = 仅为视觉平滑，不作为最终数值。

另外 `Step = 0 / 1 / 2` 分别对应第 `1 / 2 / 3` 个 epoch。

## 与日志目录的对应关系

截图里的 Run 名称就是 `runs/` 下的日志目录名（`runs/` 未入库，但本地保留）。
完整对照表见项目根目录 [`README.md`](../README.md) 的"实验矩阵"。

| Run 名称 | `--exp` | 权重文件 |
| --- | --- | --- |
| `bert_text_classification` | `base` | `best_model.pth` |
| `model_lr1e-5` | `exp_lr_1e-5` | `best_model_1e-5.pth` |
| `model_lr1e-4` | `exp_lr_1e-4` | `best_model_1e-4.pth` |
| `model_4size` | `exp_batch_4` | `model_4size.pth` |
| `model_64size` | `exp_batch_64` | `best_model_64size.pth` |
| `model_0.03dropout` | `exp_dropout_0.03` | `model_0.03dropout.pth` |
| `model_0.3dropout` | `exp_dropout_0.3` | `model_0.3dropout.pth` |
