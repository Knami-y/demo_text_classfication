
from idlelib.iomenu import encoding
from pathlib import Path
import torch
from scipy.sparse import data
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).parent
TRAIN_PATH = BASE_DIR / "train_3k.txt"
TEST_PATH = BASE_DIR / "test_1k.txt"
DEV_PATH = BASE_DIR / "dev_1k.txt"

BATCH_SIZE = 16
MODEL_PATH = "bert-base-chinese"




#1.读取数据
def read_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()#删除字符间的空格
            if not line:
                continue

            parts = line.split("_!_")
            label = parts[1].strip()#类别
            title = parts[3].strip()#新闻标题
            keywords = parts[4].strip()#关键词

            if keywords:
                text = title + " " + keywords
            else:
                text = title

            data.append([text, label])

    return data

train_data = read_data(TRAIN_PATH)
dev_data = read_data(DEV_PATH)
test_data = read_data(TEST_PATH)

print("train_data:", len(train_data))
print("dev_data:", len(dev_data))
print("test_data:", len(test_data))

# print("第一条训练数据:",train_data[0])

#2.建立映射
all_labels = sorted(
    set(
        label
        for _,label in train_data + dev_data + test_data
    )
)#set自动去重，整理所有类型的标签  sorted排序

label2id = {
    label: index
    for index, label in enumerate(all_labels)
}#建立文字到数字的映射并且生成字典，如label2id={“体育”:0}

id2label = {
    index: label
    for label,index in label2id.items()
}#反向映射

print("类别数量：",len(all_labels))
print("label2id:", label2id)
print("id2label:", id2label)

#3.创建dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese",local_files_only=True)#tokenizer使文字转为数字

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
train_dataset = NewsDataset(train_data,tokenizer,label2id,max_length=128)
dev_dataset = NewsDataset(dev_data,tokenizer,label2id,max_length=128)
test_dataset = NewsDataset(test_data,tokenizer,label2id,max_length=128)

#4.创建dataloader
train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
dev_loader = DataLoader(dev_dataset,batch_size=BATCH_SIZE,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)

batch = next(iter(train_loader))

print(batch["input_ids"].shape)
print(batch["attention_mask"].shape)
print(batch["labels"].shape)

#5.bert-base-chinese模型训练
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=len(label2id),
    local_files_only=True,
)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = model.to(device)

print("使用设备:", device)


batch = next(iter(train_loader))

# input_ids = batch["input_ids"].to(device)
# attention_mask = batch["attention_mask"].to(device)
#
# outputs = model(
#     input_ids=input_ids,
#     attention_mask=attention_mask
# )
#
# logits = outputs.logits
#
# print(logits.shape)


#6.创建交叉熵损失函数(交叉熵适用于多分类)
loss_fn = torch.nn.CrossEntropyLoss()

#7.创建优化器optimizer
opt = torch.optim.AdamW(model.parameters(),lr = 2e-5)

#8.开始训练过程
if __name__ == "__main__":
    Epochs = 3
    best_dev_acc = 0
    best_epoch = 0
    writer = SummaryWriter("./runs/bert_text_classification")
    for epoch in range(Epochs):
        print("epoch:", epoch + 1)
        model.train()#切换模型为训练模式
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            # 清空上一轮梯度
            opt.zero_grad()
            # 前向传播
            outputs = model(input_ids = input_ids,attention_mask = attention_mask)
            logits = outputs.logits#logits是模型对各类分别输出的原始分数
            # 计算每个损失
            loss = loss_fn(logits,labels)
            # 反向传播
            loss.backward()
            # 更新参数
            opt.step()
            # 统计
            total_loss = total_loss + loss.item()
            # 得到预测类别
            preds = torch.argmax(logits, dim=1)
            # 统计正确个数
            correct += (preds == labels).sum().item()
            #统计已训练样本数量
            total += labels.size(0)
        # 总损失和正确率
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # Dev验证
        model.eval()
        dev_correct = 0
        dev_total = 0
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids = input_ids,attention_mask = attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                dev_correct += (preds == labels).sum().item()
                dev_total += labels.size(0)

        dev_acc = dev_correct / dev_total


        writer.add_scalar("Loss/train",train_loss,epoch)
        writer.add_scalar("Accuracy/train",train_acc,epoch)
        writer.add_scalar("Accuracy/dev",dev_acc,epoch)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch = epoch + 1
            torch.save({"epoch":best_epoch,
                        "model":model.state_dict(),
                        "optimizer":opt.state_dict(),
                        "dev_acc":dev_acc},"best_model.pth")
            print(f"保存最优模型： Epoch {best_epoch}, Best dev acc: {best_dev_acc:.4f}")

        print(
            f"Epoch {epoch+1}/{Epochs} "
            f"Train loss: {train_loss:.4f} "
            f"Train acc: {train_acc * 100: .2f} "
            f"Dev acc: {dev_acc * 100: .2f} "
        )

    writer.close()





