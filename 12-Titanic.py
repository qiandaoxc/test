import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
# 用这个模型预测不了 MaxScore-->0.66028
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()

        self.linear_1=torch.nn.Linear(5,1)

        self.sigmoid=torch.nn.Sigmoid()

    def forward(self,x):
        x=self.sigmoid(self.linear_1(x))
        return x

class TitanicData(Dataset):
    def __init__(self,filepath):
        xy=np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        self.x_data=torch.from_numpy(xy[:,:-1])
        self.y_data=torch.from_numpy(xy[:,[-1]])
        # 样本数量
        self.len=xy.shape[0]
        print(self.len)
    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len

model=Model()
dataset=TitanicData('dataset/titanic/train.csv')

train_loader=DataLoader(dataset=dataset,shuffle=True,batch_size=8,num_workers=0)

criterion=torch.nn.BCELoss(reduction='sum')
optimizer=torch.optim.Rprop(model.parameters(),lr=0.1)

e_list=[]
l_list=[]
# 训练模型
for epoch in range(175):
    for i, data in enumerate(train_loader):
        inputs, labels = data
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)  # 计算损失
        optimizer.zero_grad()  # 梯度归零
        loss.backward()   # 反向传播
        optimizer.step()  # 更新梯度

    print(epoch,loss.item())
    e_list.append(epoch)
    l_list.append(loss.item())

plt.plot(e_list,l_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("BCE")
plt.show()

test=np.loadtxt('test.csv',delimiter=',',dtype=np.float32)
x_test=torch.from_numpy(test[:,:])

y_pred=model(x_test)
Survived_list=[]
for i in y_pred:
    print(i.item())
    if i.item()>0.5:
        Survived_list.append(1)
    else:
        Survived_list.append(0)
output = pd.DataFrame({'PassengerId': range(892,1310), 'Survived': Survived_list})
output.to_csv('submission.csv', index=False)