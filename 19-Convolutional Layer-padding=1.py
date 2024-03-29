import  torch
input=[3,4,6,5,7,
       2,4,6,8,2,
       1,6,7,8,4,
       9,7,4,6,2,
       3,7,5,4,1]
# B C H W
input=torch.Tensor(input).view(1,1,5,5)
# padding=1 ---> 加一圈0   原本输入4*4  --> 5*5
conv_layer=torch.nn.Conv2d(1,1,kernel_size=3,padding=1,bias=False)
kernel=torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3)

# 卷积层的权重进行初始化
conv_layer.weight.data=kernel.data

output=conv_layer(input)
print(output.data)