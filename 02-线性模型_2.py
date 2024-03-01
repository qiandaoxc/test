import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# eg2:  y=3x+1

x_data=[1.0,2.0,3.0]
y_data=[3.0,5.0,7.0]

def forward(x):
    return x*w+b

def loss(x,y):
    y_pred=forward(x)
    return (y-y_pred)*(y-y_pred)

w_list=[]
b_list=[]
mse_list=[]

for w in np.arange(0.0,4.0,0.1):
    for b in np.arange(0.0,4.0,0.1):
        loss_sum=0
        for x_val,y_val in zip(x_data,y_data):
            y_pred=forward(x_val)
            loss_sum+=loss(x_val,y_val)
        if b not in b_list:
            b_list.append(b)
        mse_list.append(loss_sum/3)
    w_list.append(w)


 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
w_list,b_list=np.meshgrid(w_list,b_list)
mse_list=np.array(mse_list).reshape(len(w_list),len(b_list))
ax.plot_surface(w_list, b_list, mse_list)

ax.set_xlabel('Weight')
ax.set_ylabel('Bias')
ax.set_zlabel('MSE')

plt.show()

 

