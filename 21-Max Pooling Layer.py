import  torch

# 对于 5*5的矩阵 对于kernel_size=2的maxpooling 它舍弃了最后一列
input=[3,4,6,5,9,
       2,4,6,8,2,
       1,6,7,8,4,
       9,7,4,6,2,
       3,7,5,4,1]
# B C H W
input=torch.Tensor(input).view(1,1,5,5)
# 这里的kernel_size=2 同时代表stride也是2 它不会重复取样
# MaxPool2d取的是Max值，也有AvgPool2d
maxpooling_layer=torch.nn.MaxPool2d(kernel_size=2)
output=maxpooling_layer(input)
print(output.data)