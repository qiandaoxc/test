import matplotlib.pyplot as plt
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

w=1.0

def forward(x):
    return x*w


# cost即MSE
def cost(xs,ys):
    cost=0
    for x,y in zip(xs,ys):
        y_pred=forward(x)
        cost+=(y-y_pred)**2
    return cost/len(xs)


def gradient(xs,ys):  # 计算梯度不会产生依赖
    grad=0
    for x,y in zip(xs,ys):
        grad+=2*x*(w*x-y)
    return grad/len(xs)
cost_list=[]

epoch_list=[]
for epoch in range(100):
    cost_val=cost(x_data,y_data)
    grad_val=gradient(x_data,y_data)

    cost_list.append(cost_val)
    epoch_list.append(epoch)
    w-=0.01*grad_val  # 0.01 is study rate

plt.plot(epoch_list,cost_list)
plt.ylabel("Cost")
plt.xlabel("Epoch")
plt.show()

# 若最后不收敛可能是学习率太大了