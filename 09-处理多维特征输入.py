import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

xy=np.loadtxt('diabetes.csv',delimiter=',',dtype=np.float32)
x_data=torch.from_numpy(xy[:-1,:-1])  #矩阵保留所有行，但是去掉最后一列
y_data=torch.from_numpy(xy[:-1,[-1]])  #[-1]-->表示最后拿出来是一个矩阵而不是向量
x_test=torch.from_numpy(xy[[-1],:-1])
y_test=torch.from_numpy(xy[[-1],[-1]])
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

criterion=torch.nn.BCELoss(size_average=False)

optimizer=torch.optim.Rprop(model.parameters(),lr=0.01)

e_list=[]
l_list=[]
for epoch in range(1000):
    y_pred=model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())

    e_list.append(epoch)
    l_list.append(loss.item())

    optimizer.zero_grad()   # 梯度归零
    loss.backward()   # 反向传播
    optimizer.step()  # 自动更新权重


y_trained_pred=model(x_test)
print('y_test：',y_test.item())
print('y_test_pred',y_trained_pred.item())
plt.plot(e_list,l_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("BCE")
plt.show()