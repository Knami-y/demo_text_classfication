import argparse
import random
import numpy as np
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from utils import Config, load_label_map, compute_metrics
from dataset import NewsDataset
from model import BertClassifier
from torch import nn
from tqdm.auto import tqdm
import swanlab

class Trainer:
    def __init__(self,model,train_loader,dev_loader,
                 config,id2label,device=None):
        self.config = config
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.id2label = id2label

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model = model.to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr = config["lr"])

        self.checkpoint_path = config.checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        config.run_dir.mkdir(parents=True, exist_ok=True)

        self.best_dev_acc = -1.0
        self.best_epoch = 0

    def train_one_epoch(self,epoch,epochs):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{epochs} Training",
            leave=False
        )

        for batch in progress_bar:
            batch = {key: value.to(self.device) for key, value in batch.items()}
            labels = batch.pop("labels")

            self.optimizer.zero_grad()
            logits = self.model(**batch)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()*labels.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(
                loss=f"{total_loss / total:.4f}",
                acc=f"{correct / total:.4f}"
            )

        train_loss = total_loss / total
        train_acc = correct / total

        return train_loss, train_acc

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch in self.dev_loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                labels = batch.pop("labels")

                logits = self.model(**batch)
                loss = self.loss_fn(logits, labels)

                predictions = logits.argmax(dim=1)

                total_loss += loss.item() * labels.size(0)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.detach().cpu().tolist())
                all_predictions.extend(predictions.detach().cpu().tolist())

        dev_loss = total_loss / total
        dev_acc = correct / total

        report,metrics = compute_metrics(
            all_labels,
            all_predictions,
            self.id2label,
        )
        print(report)

        return dev_loss, dev_acc, metrics

    @staticmethod
    def swan_log(epoch,train_loss,train_acc,dev_loss,dev_acc,metrics):
        swanlab.log(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "dev_loss": dev_loss,
                "dev_acc": dev_acc,
                "dev_precision": metrics["macro_avg"]["precision"],
                "dev_recall": metrics["macro_avg"]["recall"],
                "dev_f1": metrics["macro_avg"]["f1"],
            },
            step=epoch
        )

    def train(self):
        epochs = self.config["epochs"]
        try:
            swanlab.init(
                project="bert_text_classification",
                logdir=str(self.config.run_dir),
                config=self.config.to_dict(),
                name=self.config.exp_name,
            )

            for epoch in range(1, epochs + 1):
                print(f"\nEpoch {epoch}/{epochs}")
                train_loss, train_acc = self.train_one_epoch(epoch, epochs)
                dev_loss, dev_acc, metrics = self.evaluate()

                if dev_acc > self.best_dev_acc:
                    self.best_dev_acc = dev_acc
                    self.best_epoch = epoch
                    torch.save(
                        {
                            "epoch": epoch,
                            "model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "dev_acc": dev_acc,

                            "config": self.config.to_dict(),
                            "label2id":{
                                label: idx
                                for idx, label in self.id2label.items()
                            },
                            "id2label": self.id2label,
                        },
                        self.checkpoint_path,
                    )
                    print(f"保存最优模型: Epoch {epoch}, Dev Acc: {dev_acc:.4f}")

                self.swan_log(
                    epoch,
                    train_loss,train_acc,
                    dev_loss,dev_acc,
                    metrics
                )

                print(
                    f"Epoch {epoch}/{epochs} "
                    f"Train Loss: {train_loss:.4f} "
                    f"Train Acc: {train_acc * 100:.2f}% "
                    f"Dev Loss: {dev_loss:.4f} "
                    f"Dev Acc: {dev_acc * 100:.2f}%"
                )
        finally:
            swanlab.finish()

        print(f"训练结束，最佳Epoch {self.best_epoch} Dev Acc: {self.best_dev_acc:.4f}")
        return self.best_dev_acc

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
    parser.add_argument("--exp",default="base",help="config_exp/ 的实验名")
    args = parser.parse_args()

    config = Config(exp_name=args.exp)
    set_seed(config["seed"])

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config["model_path"],
        local_files_only=config["local_files_only"],
    )

    label2id, id2label = load_label_map(config.label_map_path)

    train_dataset = NewsDataset(config.train_path)
    dev_dataset = NewsDataset(config.dev_path)

    collate_fn = NewsDataset.make_collate_fn(
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

    model = BertClassifier(config,label2id,id2label)
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
                      id2label=id2label,
                      device=device,
                      )
    trainer.train()

if __name__ == "__main__":
    main()
