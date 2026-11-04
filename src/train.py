import argparse
import json
import random
import numpy as np
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from utils import Config
from dataset import NewsDataset, make_collate_fn
from model import build_model
from trainer import Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser(description="BERT 中文新闻文本分类训练")
    parser.add_argument("--exp",default="base",help="json/ 的实验名")
    args = parser.parse_args()

    config = Config(exp_name=args.exp)
    set_seed(config["seed"])

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config["model_path"],
        local_files_only=config["local_files_only"],
    )

    with open(config.label_map_path,"r",encoding = "utf-8") as f:
        label2id = json.load(f)

    train_dataset = NewsDataset(config.train_path)
    dev_dataset = NewsDataset(config.dev_path)

    collate_fn = make_collate_fn(
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=config["max_length"],
    )


    generator = torch.Generator()
    generator.manual_seed(config["seed"])
    eval_batch_size = config.to_dict().get("eval_batch_size",config["batch_size"])

    train_loader = DataLoader(
        train_dataset,
        batch_size = config["batch_size"],
        shuffle = True,
        collate_fn = collate_fn,
        generator = generator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = build_model(config,label2id)
    if args.exp == "exp_batch_64":
        model.gradient_checkpointing_enable()
    device = "cuda" if torch.cuda.is_available() else(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"实验名称：{args.exp}")
    print(f"训练样本数：{len(train_dataset)},验证样本数：{len(dev_dataset)}")
    print(f"device:{device}")

    trainer = Trainer(model = model,
                      train_loader=train_loader,
                      dev_loader=dev_loader,
                      config=config,
                      device=device,
                      )
    trainer.train()

if __name__ == "__main__":
    main()
