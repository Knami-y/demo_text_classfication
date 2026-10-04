# 超参数对照实验

采用**控制变量法**：每组实验只改一个超参数，其余与基线保持一致
（`lr=2e-5`、`batch_size=16`、`hidden_dropout=0.1`、3 个 epoch）。

| 实验组 | 脚本 | Learning Rate | Batch Size | Hidden Dropout | TensorBoard 日志 |
| --- | --- | --- | --- | --- | --- |
| Baseline | `../train.py` | 2e-5 | 16 | 0.1 | `runs/bert_text_classification` |
| LR-1 | `exp_lr_1e-5.py` | **1e-5** | 16 | 0.1 | `runs/model_lr1e-5` |
| LR-2 | `exp_lr_1e-4.py` | **1e-4** | 16 | 0.1 | `runs/model_lr1e-4` |
| BS-1 | `exp_batch_4.py` | 2e-5 | **4** | 0.1 | `runs/model_4size` |
| BS-2 | `exp_batch_64.py` | 2e-5 | **64** | 0.1 | `runs/bodel_64size` |
| Dropout-1 | `exp_dropout_0.03.py` | 2e-5 | 16 | **0.03** | `runs/model_0.03dropout` |
| Dropout-2 | `exp_dropout_0.3.py` | 2e-5 | 16 | **0.3** | `runs/model_0.3dropout` |

## 这些脚本是怎么来的

它们都是从 `../train.py`（基线）复制后只改一两行得到的，**保持了当初实际运行的形态**，
每个文件都是可以独立执行的完整脚本，方便逐行对照差异：

```bash
# 找出与基线的差异
diff ../train.py exp_batch_4.py
```

主要差异只有三类：

1. 超参数常量（`BATCH_SIZE`、`lr`、`config.hidden_dropout_prob`）；
2. `SummaryWriter` 的日志目录；
3. `torch.save` 的 checkpoint 文件名。

## 运行

```bash
# 在项目根目录，单个实验
python src/experiments/exp_lr_1e-5.py

# 或依次跑完全部 6 组（约 1 小时）
bash scripts/run_experiments.sh
```

## 关于 Dropout 实验

只修改 `config.hidden_dropout_prob`，`attention_probs_dropout_prob` 与
`classifier_dropout` 都固定为 0.1，避免多个 dropout 机制同时变化造成混淆：

```python
config.hidden_dropout_prob = 0.3   # 或 0.03
config.classifier_dropout = 0.1
```

## 注意

`exp_lr_1e-4.py` 对应的原文件（`high_lr.py`）里 `lr` 曾被误写成 `1e-5`，
与其日志目录、权重文件名和实验报告中的 LR-2（1e-4）都不一致，现已修正为 `1e-4`。
如果要严格复现报告里 LR-2 的曲线，需要重新运行该脚本。
