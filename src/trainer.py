import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

class Trainer:
    def __init__(self,model,train_loader,dev_loader,
                 config,device=None):
        self.config = config
        self.train_loader = train_loader
        self.dev_loader = dev_loader

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr = config["lr"])

        self.checkpoint_path = config.checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        config.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(config.run_dir))

        self.best_dev_acc = -1.0
        self.best_epoch = 0

    def train_one_epoch(self,epoch,epochs):
        self.model.train()
        total_loss = 0
        correct = 0 #预测正确的样本数量
        total = 0  #总共的样本数量

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{epochs} Training",
            leave=True
        )

        for batch in progress_bar:
            batch = {key: value.to(self.device) for key, value in batch.items()}
            labels = batch["labels"]

            self.optimizer.zero_grad()# 清空梯度
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()#反向传播
            self.optimizer.step()#更新参数

            total_loss += loss.item()*labels.size(0)
            predictions = outputs.logits.argmax(dim=1)
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

        with torch.no_grad():
            for batch in self.dev_loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss

                predictions = outputs.logits.argmax(dim=1)
                labels = batch["labels"]

                total_loss += loss.item() * labels.size(0)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        dev_loss = total_loss / total
        dev_acc = correct / total

        return dev_loss, dev_acc

    def train(self):
        epochs = self.config["epochs"]
        print("使用设备：",self.device)

        try:
            for epoch in range(1,epochs+1):
                print(f"\nEpoch {epoch}/{epochs}")
                train_loss, train_acc = self.train_one_epoch(epoch,epochs)
                dev_loss,dev_acc = self.evaluate()

                self.writer.add_scalar("Loss/train",train_loss,epoch-1)
                self.writer.add_scalar("Accuracy/train",train_acc,epoch-1)
                self.writer.add_scalar("Accuracy/dev",dev_acc,epoch-1)
                self.writer.add_scalar("Loss/dev",dev_loss,epoch-1)

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

                print(
                    f"Epoch {epoch}/{epochs} "
                    f"Train Loss: {train_loss:.4f} "
                    f"Train Acc: {train_acc * 100:.2f}% "
                    f"Dev Loss: {dev_loss:.4f} "
                    f"Dev Acc: {dev_acc * 100:.2f}%"
                )
        finally:
            self.writer.close()

        print(f"训练结束，最佳Epoch {self.best_epoch} Dev Acc: {self.best_dev_acc:.4f}")
        return self.best_dev_acc


