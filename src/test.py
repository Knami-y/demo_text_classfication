from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch


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



def read_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("_!_")
            label = parts[1].strip()
            title = parts[3].strip()
            keywords = parts[4].strip()

            if keywords:
                text = title + " " +keywords
            else:
                text = title

            data.append([text,label])

    return data

test_data = read_data(TEST_PATH)
print("test的数量:",len(test_data))

label2id = {
    "100": 0,
    "101": 1,
    "102": 2,
    "103": 3,
    "104": 4,
    "106": 5,
    "107": 6,
    "108": 7,
    "109": 8,
    "110": 9,
    "112": 10,
    "113": 11,
    "114": 12,
    "115": 13,
    "116": 14
}

class NewsDataset(Dataset):
    def __init__(self, data,tokenizer,label2id,max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text,label = self.data[index]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.label2id[label], dtype=torch.long),

        }#把tokenizer编码后的结果整理为一个字典
        return item
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)

test_dataset = NewsDataset(data = test_data,tokenizer=tokenizer,label2id=label2id,max_length=128)
test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH,
                                                           num_labels=len(label2id),
                                                           local_files_only=True)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

checkpoint = torch.load(CKPT_DIR / "best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model"])
model = model.to(device)
print("最佳模型 Epoch:", checkpoint["epoch"])
print("最佳模型 Dev Acc:", checkpoint["dev_acc"])
model.eval()

# Test开始
test_correct = 0
test_total = 0
print("开始测试模型：")
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids,attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits,dim=1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

test_acc = test_correct / test_total
print("The test acc:",test_acc)
