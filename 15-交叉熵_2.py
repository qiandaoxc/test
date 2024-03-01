import torch
criterion =torch.nn.CrossEntropyLoss()
c_2=torch.nn.NLLLoss()

y=torch.LongTensor([2,0,1])
y_1=torch.LongTensor([2,0,1])

z_1=torch.Tensor([[4,0.2,0.3],
                [0.5,0.6,0.7],
                [0.8,0.3,0.9]])
z_2=torch.Tensor([[0.1,0.2,0.3],
                [0.8,0.6,0.7],
                [0.3,0.9,0.4]])

l1=criterion(z_1,y)
l2=criterion(z_2,y)

logsoftmax=torch.nn.LogSoftmax(dim=1)
z_3=logsoftmax(z_2)
print(z_3)

l3=c_2(z_3,y_1)
print(f'loss_1:{l1.data},\nloss_2:{l2.data},\nloss_3:{l3.data}')

# CrossEntropyLoss=LogSoftmax+NLLLoss