from pathlib import Path
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding


BASE_DIR = Path(__file__).resolve().parents[1]  # src/ 的上一级 = 项目根目录
DATA_DIR = BASE_DIR / "data"
CKPT_DIR = BASE_DIR / "checkpoints"
RUNS_DIR = BASE_DIR / "runs"

# 全新克隆的仓库里这两个目录可能不存在，先确保可写
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_3k.txt"
DEV_PATH = DATA_DIR / "dev_1k.txt"
TEST_PATH = DATA_DIR / "test_1k.txt"

BATCH_SIZE = 16
MODEL_PATH = "bert-base-chinese"

class NewsDataset(Dataset):

    def __init__(self,path,tokenizer,label2id,max_length=128):
        self.label2id = label2id
        self.max_length = max_length
        self.tokenizer = tokenizer

        raw_data = self.read_data(path)

        self.data  = []

        for text,label in raw_data:
            encoding = tokenizer(
                text,
                max_length = self.max_length,
                truncation = True,#自动截断
            )

            self.data.append(
                {
                    "input_ids": encoding["input_ids"],
                    "attention_mask": encoding["attention_mask"],
                    "labels": self.label2id[label]
                }
            )
    @staticmethod
    def read_data(path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip() # 删除字符间的空格

                if not line:
                    continue

                parts = line.split("_!_")

                if len(parts) < 5:
                    continue

                label = parts[1].strip()  # 类别
                title = parts[3].strip()  # 新闻标题
                keywords = parts[4].strip()  # 关键词

                if keywords:
                    text = title + " " + keywords
                else:
                    text = title

                data.append([text, label])

        return data



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        return self.data[index]

def create_collate_fn(tokenizer):
    return DataCollatorWithPadding(tokenizer = tokenizer)
