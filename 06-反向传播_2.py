# y=w1*x*x + w2*x +b

import torch

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

rate=0.01
w_1=torch.Tensor([0.1])
w_2=torch.Tensor([0.1])
b=torch.Tensor([0.1])
w_1.requires_grad=True
w_2.requires_grad=True
b.requires_grad=True
def forward(x):
    return w_1*x*x+w_2*x+b

def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)**2

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        l=loss(x,y)
        l.backward()
        print(f'\tgrad: w_1:{w_1.grad.item()},w_2:{w_2.grad.item()},b:{b.grad.item()}')
        w_1.data=w_1.data-w_1.grad.data*rate
        w_2.data=w_2.data-w_2.grad.data*rate
        b.data=b-b.grad.data*rate

        w_1.grad.data.zero_()
        w_2.grad.data.zero_()
        b.grad.data.zero_()

    print(f"Epoch:{epoch},Loss:{l.item()}")

print(f'w-1:{w_1.item()},w-2:{w_2.item()},bias:{b.item()}')