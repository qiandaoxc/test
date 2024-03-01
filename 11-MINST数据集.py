from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

##### 针对MNIST数据集还没写出对应的训练模型




train_dataset=datasets.MNIST(root='./dataset/mnist',train=True,transform=transforms.ToTensor(),download=False)
test_dataset=datasets.MNIST(root='./dataset/mnist',train=False,transform=transforms.ToTensor(),download=False)

train_loader =DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
# 测试中不对数据进行打乱 以保证在测试结果输出的数据集顺序一致
test_loader =DataLoader(dataset=test_dataset,batch_size=32,shuffle=False)

criterion=torch.nn.BCELoss(size_average=False)
optimizer=torch.optim.Rprop(model.parameters(),lr=0.01)
e_list=[]
l_list=[]
if __name__ == '__main__':
    for epoch in range(100):
        for batch_idx,(input,target) in enumerate(train_loader):
            y_pred=model(input)
            loss=criterion(y_pred,target)
            print(epoch,batch_idx,loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        e_list.append(epoch)
        l_list.append(loss.item())

    plt.plot(e_list, l_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BCE")
    plt.show()
