import torch
from torch import nn
from tqdm.auto import tqdm
import swanlab
from utils import compute_metrics

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
