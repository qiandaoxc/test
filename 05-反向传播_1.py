# y=w*x

import torch
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

# Tensor变量，一个是data 一个是gradient
w=torch.Tensor([1.0])
# 计算梯度
w.requires_grad=True

def forward(x):
    # x自动转换成Tensor进行计算，但是x不是Tensor
    return x*w

def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)**2

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        l=loss(x,y)
        # 反向传播 将梯度保存到Tensor  反向传播之后计算图就释放了 标量才能backward
        l.backward()
        print("\tgrad:",x,y,w.grad.item())
        
        # 取data不会建立计算图  更新权重得用data
        w.data=w.data-0.01*w.grad.data

        # 将梯度清零  否则下次梯度更新后会累加
        w.grad.data.zero_()

    print("progress:",epoch,l.item())

