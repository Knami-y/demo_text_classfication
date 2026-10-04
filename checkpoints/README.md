# 权重文件

本目录保存训练产生的 checkpoint，**共 7 个、约 8.5 GB，已被 `.gitignore` 排除，不会推到 GitHub**。

每个 checkpoint 是一个字典：

```python
{
    "epoch":    int,            # 保存时的 epoch（从 1 开始）
    "model":    state_dict,     # BERT + 分类头
    "optimizer": state_dict,    # AdamW 优化器状态
    "dev_acc":  float,          # 该 epoch 的验证集准确率
}
```

| 文件 | 大小 | 实验组 | 生成脚本 |
| --- | --- | --- | --- |
| `best_model.pth` | 1.2 GB | Baseline | `../src/train.py` |
| `best_model_1e-5.pth` | 1.2 GB | LR-1 | `../src/experiments/exp_lr_1e-5.py` |
| `best_model_1e-4.pth` | 1.2 GB | LR-2 | `../src/experiments/exp_lr_1e-4.py` |
| `model_4size.pth` | 1.2 GB | BS-1 | `../src/experiments/exp_batch_4.py` |
| `best_model_64size.pth` | 1.2 GB | BS-2 | `../src/experiments/exp_batch_64.py` |
| `model_0.03dropout.pth` | 1.2 GB | Dropout-1 | `../src/experiments/exp_dropout_0.03.py` |
| `model_0.3dropout.pth` | 1.2 GB | Dropout-2 | `../src/experiments/exp_dropout_0.3.py` |

> 由于 checkpoint 里连优化器状态一起保存，每个文件都是完整的 1.2 GB（等于一份
> `bert-base-chinese` 的全部参数），删掉优化器状态可以缩小到约 400 MB。

## 怎么用

测试集评估直接读 `best_model.pth`：

```bash
python src/test.py
```

换成别的 checkpoint 时，改 `src/test.py` 里这一行即可：

```python
checkpoint = torch.load(CKPT_DIR / "best_model.pth", map_location=device)
```

## 怎么重新生成

```bash
python src/train.py                        # 基线
bash scripts/run_experiments.sh            # 6 组对照实验
```

权重丢失也不会影响仓库的可复现性——源码、数据、图表都在版本控制里，重跑即可。
