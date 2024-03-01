import numpy as np
import matplotlib.pyplot as plt

# eg1:  y=2x
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]


# 定义前向传播过程--> 线性的
def forward(x):
    # return x*w
    return x*w

# 定义损失函数
def loss(x,y):
    y_pred=forward(x)
    return (y-y_pred)*(y-y_pred)

w_list=[]
mse_list=[]  # 平均平方损失

for w in np.arange(0.0,4.1,0.1):
    print("weight:",w)
    loss_sum=0
    for x_val,y_val in zip(x_data,y_data):
        y_pred=forward(x_val)
        loss_sum+=loss(x_val,y_val)
    print("MSE:",loss_sum/3)
    w_list.append(w)
    mse_list.append(loss_sum/3)
    
plt.plot(w_list,mse_list)
plt.ylabel("MSE")
plt.xlabel("weight")
plt.show()
