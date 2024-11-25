# 使用预训练模型权重，使用Transfor learn训练cifar识别模型、验证（.py+训练截图）
from torchvision.models import efficientnet_b0
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

model = efficientnet_b0(weights='DEFAULT')


custom_classifier=nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280,100),
    nn.ReLU(),
    nn.Linear(100,10)
)

model.features.trainable=False
model.classifier=custom_classifier

train_data = CIFAR10(
    root="../data/cifar_10",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = CIFAR10(
    root="../data/cifar_10",
    train=False,
    download=True,
    transform=ToTensor()
)

model.to('cuda')
train_dl=DataLoader(train_data,batch_size=32,shuffle=True)
test_dl=DataLoader(test_data,batch_size=32,shuffle=False)

# 创建优化器
optimizer=Adam(model.parameters())
loss_fn=nn.CrossEntropyLoss()

writer = SummaryWriter()
train_loss,test_loss,test_acc=0,0,0

for epoch in range(10):
    pbar=tqdm(train_dl)
    for img,lbl in pbar:
        img,lbl=img.to('cuda'),lbl.to('cuda')
        logits=model(img)
        loss=loss_fn(logits,lbl)
        loss.backward()
        optimizer.step()
        model.zero_grad()
        writer.add_scalar('train_loss_qy', loss, train_loss)
        train_loss+=1
        pbar.set_description(f"epoch{epoch+1},loss{loss.item():.4f}")
    correct = 0
    total_loss = 0
    total = 0
    with torch.no_grad():
        for img, lbl in test_dl:
            img, lbl = img.to('cuda'), lbl.to('cuda')
            logits = model(img)
            loss = loss_fn(logits, lbl)
            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == lbl).sum().item()
            total += lbl.size(0)
        writer.add_scalar('test_loss_qy',total_loss/len(test_dl),test_loss)
        writer.add_scalar('test_acc_qy',correct/len(test_data),test_acc)
        test_acc+=1
        test_loss+=1

