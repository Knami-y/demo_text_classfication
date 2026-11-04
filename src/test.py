import argparse
import json

import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from config import Config
from dataset import NewsDataset, make_collate_fn
from model import build_model


def main():
    parser = argparse.ArgumentParser(description = "测试模型")
    parser.add_argument("--exp",default="base",help="json/ 的实验名")
    args = parser.parse_args()

    config = Config(exp_name=args.exp)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config["model_path"],
        local_files_only = config["local_files_only"],
    )
    with open(config.label_map_path,"r",encoding = "utf-8") as f:
        label2id = json.load(f)
    test_dataset = NewsDataset(config.test_path)

    if len(test_dataset) == 0:
        raise ValueError("测试集为空")

    collate_fn = make_collate_fn(
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=config["max_length"],
    )

    test_dataloader = DataLoader(test_dataset, batch_size = config["batch_size"], collate_fn = collate_fn)

    if not config.checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到模型权重:{config.checkpoint_path}")
    model = build_model(config,label2id)
    checkpoint = torch.load(
        config.checkpoint_path,
        map_location = "cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = model.to(device)
    model.eval()

    print(f"实验名称：{args.exp}")
    print(f"测试样本数：{len(test_dataset)}")
    print(f"使用设备：{device}")
    print(f"加载模型：{config.checkpoint_path}")
    print(f"最佳 Epoch：{checkpoint['epoch']}")
    print(f"最佳 Dev Acc：{checkpoint['dev_acc'] * 100:.2f}%")

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_dataloader:
            batch = {key: value.to (device) for key, value in batch.items()}
            outputs = model(**batch)
            labels = batch["labels"]
            predictions = outputs.logits.argmax(dim = 1)

            total_loss += outputs.loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print(f"Test Loss:{total_loss / total:.4f}")
    print(f"Test Accuracy:{correct / total:.4f}")

if __name__ == "__main__":
    main()
