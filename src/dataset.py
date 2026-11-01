from pathlib import Path
from torch.utils.data import Dataset

BASE_DIR = Path(__file__).resolve().parents[1]

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
                truncation = True,
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
                line = line.strip()

                if not line:
                    continue

                parts = line.split("_!_")

                if len(parts) < 5:
                    continue

                label = parts[1].strip()
                title = parts[3].strip()
                keywords = parts[4].strip()

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
