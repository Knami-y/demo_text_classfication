# 权重文件

本目录保存训练产生的 checkpoint，**共 7 个、约 8.0 GB，已被 `.gitignore` 排除，不会推到 GitHub**。
仓库里只保留这份说明，权重需要本地重新训练或自行拷贝。

每个 checkpoint 是一个字典：

```python
{
    "epoch":     int,          # 保存时的 epoch（从 1 开始）
    "model":     state_dict,   # BERT + 分类头
    "optimizer": state_dict,   # AdamW 优化器状态
    "dev_acc":   float,        # 该 epoch 的验证集准确率
}
```

| 文件 | 大小 | 实验 | 生成命令 | 保存时的 Dev Acc |
| --- | --- | --- | --- | --- |
| `best_model.pth` | 约 1.2 GB | Baseline | `python src/train.py --exp base` | 85.40%（Epoch 3） |
| `best_model_1e-5.pth` | 约 1.2 GB | LR-1 | `python src/train.py --exp exp_lr_1e-5` | 85.50%（Epoch 2） |
| `best_model_1e-4.pth` | 约 1.2 GB | LR-2 | `python src/train.py --exp exp_lr_1e-4` | 78.20%（Epoch 1） |
| `model_4size.pth` | 约 1.2 GB | BS-1 | `python src/train.py --exp exp_batch_4` | 84.60%（Epoch 3） |
| `best_model_64size.pth` | 约 1.2 GB | BS-2 | `python src/train.py --exp exp_batch_64` | 84.60%（Epoch 3） |
| `model_0.03dropout.pth` | 约 1.2 GB | Dropout-1 | `python src/train.py --exp exp_dropout_0.03` | 86.00%（Epoch 3） |
| `model_0.3dropout.pth` | 约 1.2 GB | Dropout-2 | `python src/train.py --exp exp_dropout_0.3` | 84.30%（Epoch 2） |

> 由于 checkpoint 里连优化器状态一起保存，每个文件都是完整的约 1.2 GB
> （等于一份 `bert-base-chinese` 的全部参数）；只保留模型参数可以缩小到约 400 MB。

## 怎么用

测试集评估不再需要改代码，只要把 `--exp` 换成对应实验名，`src/test.py` 会根据
`config.json` 自动找到该实验的 `checkpoint_name`：

```bash
python src/test.py --exp base
python src/test.py --exp exp_dropout_0.03
```

若指定的文件不存在，脚本会抛出 `FileNotFoundError` 并打印完整路径。

## 怎么重新生成

```bash
python src/train.py --exp base            # 只跑基线
bash scripts/run_experiments.sh           # 依次跑 6 组对照实验
```

权重丢失不会影响仓库的可复现性——源码、配置、数据、图表都在版本控制里，重跑即可。
