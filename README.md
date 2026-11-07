# BERT 中文新闻文本分类

用 `bert-base-chinese` 做 15 类新闻文本分类，数据来源：今日头条的新闻标题和关键词。

基础参数：batch=16,lr=2e-5,hidden_dropout=0.1

对照实验参数：batch=4、64；lr=1e-5、1e-4;dropout=0.03、0.3

## 运行

```bash
conda activate learn-pytorch
pip install -r requirement/requirements.txt
```
训练和测试：

```bash
python src/train.py --exp base
python src/test.py --exp base
```

训练使用 SwanLab 记录 `train_loss`、`train_acc`、`dev_loss`、`dev_acc`、`dev_precision`、`dev_recall`、`dev_f1`  
项目名是 `bert_text_classification`，run 名就是 `--exp` 后面的实验名。

## 实验

所有实验参数均置于`json`文件夹下，文件名与实验名一致

| 实验 | 改动 | Dev Acc |
| --- | --- |--------:|
| `base` | 无 |  85.80% |
| `exp_batch_4` | batch_size=4 |  83.90% |
| `exp_batch_64` | batch_size=64 |  86.50% |
| `exp_dropout_0.03` | hidden_dropout=0.03 |  86.20% |
| `exp_dropout_0.3` | hidden_dropout=0.3 |  83.80% |
| `exp_lr_1e-5` | lr=1e-5 |  85.10% |
| `exp_lr_1e-4` | lr=1e-4 |   9.20% |

* dev_acc为每组实验的最佳dev_acc。
* 对于`exp_lr_1e-4`组(dev_acc仅`9.20%`),由于`lr`过大导致训练未收敛，该组仅作为失败对照。

更多数据及完整曲线请见 [swanlab实验记录](https://swanlab.cn/@1211698600/bert_text_classification/v1/lwk4sv/overview)

运行其他实验时候就把 `--exp` 换成表里的名字，如 `python src/train.py --exp exp_dropout_0.03`。


## 项目结构

```text
.
├── README.md
├── json/
│   ├── base.json
│   ├── exp_*.json
│   └── label_map.json
├── data/
│   ├── README.md
│   ├── train_3k.txt
│   ├── dev_1k.txt
│   └── test_1k.txt
├── requirement/
├── src/
│   ├── utils.py
│   ├── dataset.py
│   ├── model.py
│   ├── trainer.py
│   ├── train.py
│   └── test.py
├── checkpoints/   # 本地生成，已 gitignore
└── runs/          # SwanLab 日志，已 gitignore
```

## src文件介绍

| 文件 | 作用 |
| --- | --- |
| `utils.py` | `Config`、标签映射、P/R/F1 指标 |
| `dataset.py` | `NewsDataset` 和 `make_collate_fn` |
| `model.py` | `BertClassifier` 和 `build_model` |
| `trainer.py` | 训练、验证、SwanLab 记录 |
| `train.py` | 训练入口 |
| `test.py` | 测试入口 |