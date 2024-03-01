# 使用GPU进行运算 将模型和数据都放到GPU上

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5) # 1*28*28--> 10*24*24
        self.conv2=torch.nn.Conv2d(10,20,kernel_size=5) #10*12*12 -->20*8*8
        self.pooling=torch.nn.MaxPool2d(kernel_size=2,stride=1,padding=0)  # 10*24*24->10*12*12   20*8*8-->20*4*4

        self.fc=torch.nn.Linear(320,10)

    def forward(self,x):
        batch_size=x.size(0)
        # F里面也有max_pool2d的方法，不过要传入input 这种要传入参数的可以写到forward里面
        # self.fpooling=F.max_pool2d(x,kernel_size=2,stride=1,padding=1)
        # self.fpooling=F.relu(self.conv1(x))
        x=self.pooling(F.relu(self.conv1(x))) #1*28*28  --> 10*24*24  --> 10*24*24-->  10*12*12
        x=self.pooling(F.relu(self.conv2(x))) #10*12*12 --> 20*8*8    --> 20*8*8  -->  20*4*4
        # -1代表动态调整这个维度上的元素个数，以保证元素的总数不变
        # 将x设为batch_size行
        x=x.view(batch_size,-1)  # 将 20*4*4-->320
        return self.fc(x)  # 320-->10

model=Net()

# 在GPU中跑
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

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

criterion = torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.7)

def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader):
        inputs,target=data
        # 把输入和输出也放到GPU上
        inputs,target=inputs.to(device),target.to(device)


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

            # 把输入和输出也放到GPU上
            images, labels = images.to(device), labels.to(device)

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