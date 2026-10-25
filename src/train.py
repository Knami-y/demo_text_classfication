import argparse
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from config import Config
from dataset import NewsDataset, create_collate_fn
from model import build_model
from trainer import Trainer

# 固定随机种子
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
    parser.add_argument("--exp",default="base",help="config.json 中的实验名称")
    args = parser.parse_args()

    #1.读取配置及随机种子
    config = Config(exp_name=args.exp)
    set_seed(config["seed"])

    #2.读取标签映射
    with open(config.label_map_path,"r",encoding = "utf-8") as f:
        label2id = json.load(f)

    #3.创建toeknizer、dataset
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"],local_files_only=config["local_files_only"])
    train_dataset = NewsDataset(config.train_path,tokenizer=tokenizer,
                                label2id=label2id,max_length=config["max_length"])
    dev_dataset = NewsDataset(config.dev_path,tokenizer=tokenizer,label2id=label2id,max_length=config["max_length"])

    if len(train_dataset) == 0 or len(dev_dataset) == 0:
        raise ValueError("训练集或验证集为空，请检查数据！")

    #4.动态或定长填充
    if config["dynamic_padding"]:
        collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    else:
        collate_fn = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding = "max_length",
            max_length=config["max_length"]
        )

    #训练集的随机数生成器
    generator = torch.Generator()
    generator.manual_seed(config["seed"])#创建随机种子
    eval_batch_size = config.to_dict().get("eval_batch_size",config["batch_size"])

    #Dataloader
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

    #5.构建模型，用Trainer训练
    model = build_model(config,label2id)
    if args.exp == "exp_batch_64":
        model.gradient_checkpointing_enable()#梯度重新计算，节省缓存
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

