import torch

y=torch.LongTensor([0])   #交叉熵损失需要长的张量

z=torch.Tensor([[0.2,0.1,-0.1]])
criterion=torch.nn.CrossEntropyLoss()  #-->相当于 SoftmaxLog + NLLLoss
loss=criterion(z,y)
print(loss.item())