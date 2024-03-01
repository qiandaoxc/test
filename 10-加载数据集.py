import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset  # Dataset是抽象类，不能够实例化，只能继承
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        xy=np.loadtxt(filepath,delimiter=',',dtype=np.float32)  # 在糖尿病数据集中是 N*9的矩阵
        self.len=xy.shape[0] # xy.shape-->得到[N,9]  取数组的第0位得到数据的样本个数

        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    # 下面这两个函数必须要有的
    def __getitem__(self, item):  # 使数据能够支持下标操作  通过下标访问数据
        return self.x_data[item],self.y_data[item]

    def __len__(self):  # 得到数据条数
        return self.len

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1=torch.nn.Linear(8,6)
        self.linear2=torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()  # 这是一个类

    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x= self.sigmoid(self.linear2(x))
        x=F.sigmoid(self.linear3(x))
        return x

model=Model()
dataset =DiabetesDataset('dataset/diabetes/diabetes.csv')

# 数据加载
# dataset:数据集对象   batch_size:小批量的容量    shuffle:是否将数据集打乱    num_workers: 读取数据采用的进程数量
train_loader=DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)
# train_loader (如果num_workers不为零) 只能封装在main函数中使用---->if __name__ == '__main__':

criterion=torch.nn.BCELoss(size_average=False)
optimizer=torch.optim.Rprop(model.parameters(),lr=0.01)
e_list=[]
l_list=[]
if __name__ == '__main__':
    for epoch in range(100):
        # enumerate 将可迭代对象转换一个枚举类型，返回下标和数据 0代表起始下标
        for i ,data in enumerate(train_loader,0):
            inputs,labels=data  # 得到的是Xi 和Yi
            y_pred=model(inputs)
            loss=criterion(y_pred,labels)
            print(epoch,i,loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        e_list.append(epoch)
        l_list.append(loss.item())

    plt.plot(e_list,l_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BCE")
    plt.show()