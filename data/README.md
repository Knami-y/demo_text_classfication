# 数据说明

今日头条中文新闻标题数据集，共 15 个类别。三份文件的**每一行**格式相同：

```text
id_!_label_id_!_label_name_!_title_!_keywords
```

| 字段 | 下标 | 说明 | 示例 |
| --- | --- | --- | --- |
| `id` | 0 | 新闻唯一编号 | `6552436691485852168` |
| `label_id` | 1 | 类别编号，**脚本实际使用的标签** | `107` |
| `label_name` | 2 | 类别英文名（脚本未使用） | `news_car` |
| `title` | 3 | 新闻标题 | `拉力风格的偏时点火系统调教之Fuel Cut篇` |
| `keywords` | 4 | 关键词，逗号分隔，可能为空 | `REV,Fuel,排气系统,ECU,Cut` |

字段之间用 `_!_` 分隔。

## 文件清单

| 文件 | 行数 | 用途 |
| --- | --- | --- |
| `train_3k.txt` | 3000 | 训练集 |
| `dev_1k.txt` | 1000 | 验证集（每个 epoch 评估、选择最优 checkpoint） |
| `test_1k.txt` | 1064 | 测试集（只在最后评估一次） |

> 注意：文件名写作 `test_1k`，但实际是 **1064 条**（实验报告里写的 1000 条略有出入）。
> 原始数据来自 [今日头条文本分类数据集](https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset)。

## 类别

标签编号共 15 个，缺少 `105` 和 `111`：

```text
100 101 102 103 104 106 107 108 109 110 112 113 114 115 116
```

`src/train.py` 会按编号字符串排序后从 0 开始编号，得到 `label2id = {"100": 0, ..., "116": 14}`，
`src/test.py` 中直接写死了同一套映射。

## 输入文本的构造方式

训练和测试脚本都把标题与关键词拼成一句话作为 BERT 的输入：

```python
text = title + " " + keywords if keywords else title
```

超长截断、不足补齐，`max_length = 128`。
