# SoftMax  将数取指数并求和，求出每个数的占比即Softmax    exp(x)/(exp(x1) +exp(x2) + ……)

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size=64
transforms=transforms.Compose([
    # Image-> 28*28 pixel∈{0,……,255}   Tensor-> 1*28*28 ,pixle∈[0,1](区间)
    transforms.ToTensor(),  # 图像先变成张量
    transforms.Normalize(mean=0.1307,std=0.3081)  # 标准化
                            #均值mean     #标准差 std
])

train_dataset=datasets.MNIST(root='./dataset/mnist',train=True,transform=transforms,download=False)
test_dataset=datasets.MNIST(root='./dataset/mnist',train=False,transform=transforms,download=False)

train_loader =DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
# 测试中不对数据进行打乱 以保证在测试结果输出的数据集顺序一致
test_loader =DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=4)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=torch.nn.Linear(784,512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self,x):
        x=x.view(-1,784)   # -1:会自动填充样本的数量   784: 将 1*28*28的张量全连接成784的张量(把平面变成一条线)
        x=F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return  self.l5(x)

model=Net()
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


criterion = torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.7)

def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader):
        inputs,target=data
        optimizer.zero_grad()


        outputs=model(inputs)
        loss=criterion(outputs,target)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        if batch_idx%300 ==299:
            print(f'[{epoch},{batch_idx+1}] \tloss:{running_loss/300}')
            running_loss=0.0

def test():
    correct=0
    total=0
    with torch.no_grad():   #使得with中的代码不进行梯度的计算
        for data in test_loader:
            images,labels=data
            outputs=model(images)
            # max 返回两个值(value 和 index) 因为不需要value 所以用 _
            _,predicted=torch.max(outputs.data,dim=1)
            total += labels.size(0)  # labels是一个N*1的矩阵  N为种类的总数
            correct+=(predicted==labels).sum().item()
        print(f"Accurary on test set:{100*correct/total}%")

if __name__ =="__main__":
    for epoch in range(15):
        train(epoch)
        test()
