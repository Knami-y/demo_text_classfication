import torch
from torch.utils.data import Dataset

class NewsDataset(Dataset):

    def __init__(self,path):
        self.data = self.read_data(path)

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

def make_collate_fn(tokenizer,label2id,max_length):
    def collate_fn(batch):
        texts = [item[0] for item in batch]
        labels = [label2id[item[1]] for item in batch]
        encoding = tokenizer(
            texts,
            padding=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        encoding["labels"] = torch.tensor(labels,dtype=torch.long)
        return encoding
    return collate_fn
