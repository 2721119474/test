import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class customize_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1=nn.Conv2d(1,64,3,1,1)
        self.maxpool1=nn.MaxPool2d(2,stride=2)
        self.cnn2=nn.Conv2d(64,128,3,1,1)
        self.maxpool2=nn.MaxPool2d(2,stride=2)
        self.flatten=nn.Flatten()
        self.act=nn.ReLU()
        self.hidden=nn.Linear(128*7*7,100)
        self.output=nn.Linear(100,10)


    def forward(self,input_data):
        out = self.cnn1(input_data)
        out = self.act(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.act(out)
        out = self.maxpool2(out)
        out = self.flatten(out)
        out = self.hidden(out)
        out = self.act(out)
        out = self.output(out)
        return out

# 加载数据
train_data = MNIST(
    root="../data/mnist",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = MNIST(
    root="../data/mnist",
    train=False,
    download=True,
    transform=ToTensor(),
)

train_dl=DataLoader(train_data,batch_size=32,shuffle=True)
test_dl=DataLoader(test_data,batch_size=32,shuffle=True)

model=customize_cnn().to('cuda')
optimizer=Adam(model.parameters())
loss_fn=CrossEntropyLoss()

writer = SummaryWriter()
writer.add_graph(model,torch.rand(size=(10,1,28,28),device='cuda'))
train_loss,test_loss,test_acc=0,0,0

for epoch in range(5):
    model.train()
    pbar=tqdm(train_dl,desc=f'epoch{epoch+1}')
    for img,lbl in pbar:
        img,lbl=img.to('cuda'),lbl.to('cuda')
        logits=model(img)
        loss=loss_fn(logits,lbl)
        model.zero_grad()
        loss.backward()
        optimizer.step()  # 更新参数
        writer.add_scalar('train_loss',loss,train_loss)
        train_loss+=1
        pbar.set_postfix(loss=loss.item())
    model.eval()
    correct=0
    total_loss=0
    total=0
    with torch.no_grad():
        for img,lbl in test_dl:
            img, lbl = img.to('cuda'), lbl.to('cuda')
            logits=model(img)
            loss = loss_fn(logits,lbl)
            total_loss+=loss.item()
            correct+=(logits.argmax(dim=-1)==lbl).sum().item()
            total+=lbl.size(0)

        # print('acc',correct/total)
        # print('loss',total_loss/len(test_dl))
        writer.add_scalar('test_loss',total_loss/len(test_dl),test_loss)
        writer.add_scalar('test_loss_acc',correct/len(test_data),test_acc)
        test_acc+=1
        test_loss+=1

