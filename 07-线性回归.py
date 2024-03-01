import torch
import matplotlib.pyplot as plt
x_data=torch.Tensor([[1.0],[2.0],[3.0]])
y_data=torch.Tensor([[2.0],[4.0],[6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)  # 输入输出都为 1

    def forward(self,x):
        y_pred = self.linear(x)  # 可以直接调用的  __call__
        return y_pred

model=LinearModel()   #model 也是可以直接调用的
# 损失值计算
criterion = torch.nn.MSELoss(reduction='sum')
# 优化器
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)   # lr-->learning rate
# optimizer=torch.optim.ASGD(model.parameters(),lr=0.01)
# optimizer = torch.optim.Rprop(model.parameters(),lr=0.01)
e_list=[]
l_list=[]
for epoch in range(100):
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
plt.title("SGD")
plt.show()