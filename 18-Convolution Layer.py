import torch
in_channels,out_channels=10,5
width=100;height=100
kernel_size=3  #或者(5 x 5) /(5 x 3)  的都行  这个代表卷积核的形状
batch_size=1
# 进行标准正态的取样  B,C,H,W
input=torch.randn(batch_size,
                  in_channels,
                  width,
                  height)

conv_layer=torch.nn.Conv2d(in_channels,
                           out_channels,
                           kernel_size=kernel_size)

output=conv_layer(input)
print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)