# # 下载数据集
# import torchvision
# train_set = torchvision.datasets.MNIST(root='./dataset/mnist',train=True,download=True)  #root :文件加载的位置 train: 下载训练集/验证集  download：是否下载
# test_set = torchvision.datasets.MNIST(root='/dataset/mnist',train=False,download=True)

# 分类问题

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


x_data=torch.Tensor([[1.0],[2.0],[3.0]])
y_data=torch.Tensor([[0],[0],[1]])

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel,self).__init__()
        self.linear=torch.nn.Linear(1,1)

    def forward(self,x):
        # 激活函数，将得到的值放在[0,1]之间
        y_pred=F.sigmoid(self.linear(x))   # 这是一个函数
        return y_pred

model=LogisticRegressionModel()

#损失函数用BCE--->二分类
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

print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())

# 测试模型
x_test=torch.Tensor([4.0])
y_pred=model(x_test)
print('y_pred=',y_pred.data)

plt.plot(e_list,l_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("BCE")
plt.show()