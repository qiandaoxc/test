# 用单个样本计算的损失作为梯下降的梯度
# 在每一个epoch中，其实是要计算所有样本的梯度，而非只使用一个样本
# 随机梯度算法要依次计算梯度，不同样本之间计算梯度会产生依赖
# 而梯度算法是直接加和，不同样本之间计算梯度没有依赖
import matplotlib.pyplot as plt
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

w=1.0

def forward(x):
    return x*w

def loss(x,y):
    y_pred=forward(x)
    return (y-y_pred)**2
# 单个样本的梯度
def gradient(x,y):
    return 2*x*(x*w-y)

loss_list=[]
w_list=[]
epoch_list=[]
for epoch in range(100):
    for x,y in zip(x_data,y_data):
        
        loss_val=loss(x,y)
        grad_val=gradient(x,y)

        w_list.append(w)
        loss_list.append(loss_val)

        w-=0.01*grad_val  # 梯度计算需要w，因为不同样本之间计算梯度产生依赖，不能并行


plt.plot(w_list,loss_list)
plt.xlabel("Weight")
plt.ylabel("Loss")
plt.show()

