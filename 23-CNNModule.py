import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim


ConvdCha=[10,20,10]
ksize=[2,5,5]

class Modle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv1=torch.nn.Conv2d(1,ConvdCha[0],kernel_size=ksize[0],padding=1)
        self.Conv2=torch.nn.Conv2d(ConvdCha[0],ConvdCha[1],kernel_size=ksize[1],padding=4)
        self.Conv3=torch.nn.Conv2d(ConvdCha[1],ConvdCha[2],kernel_size=ksize[2],padding=4)
        self.pooling=torch.nn.MaxPool2d(2)
        self.fc1=torch.nn.Linear(360,120)
        self.fc2=torch.nn.Linear(120,60)
        self.fc3=torch.nn.Linear(60,10)

    def forward(self,x):
        batch_size=x.size(0)
        x=F.relu(self.pooling(self.Conv1(x)))
        x=F.relu(self.pooling(self.Conv2(x)))
        x=F.relu(self.pooling(self.Conv3(x)))
        x=x.view(batch_size,-1)
        # 查看x的 batch_size 和 channel
        print(x.shape)
        exit(0)
        x=self.fc1(x)
        x=self.fc2(x)
        return self.fc3(x)


model = Modle()

batch_size = 64
transforms = transforms.Compose([
    # Image-> 28*28 pixel∈{0,……,255}   Tensor-> 1*28*28 ,pixle∈[0,1](区间)
    transforms.ToTensor(),  # 图像先变成张量
    transforms.Normalize(mean=0.1307, std=0.3081)  # 标准化
    # 均值mean     #标准差 std
])

train_dataset = datasets.MNIST(root='./dataset/mnist', train=True, transform=transforms, download=False)
test_dataset = datasets.MNIST(root='./dataset/mnist', train=False, transform=transforms, download=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# 测试中不对数据进行打乱 以保证在测试结果输出的数据集顺序一致
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.7)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print(f'[{epoch},{batch_idx + 1}] \tloss:{running_loss / 300}')
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 使得with中的代码不进行梯度的计算
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            # max 返回两个值(value 和 index) 因为不需要value 所以用 _
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)  # labels是一个N*1的矩阵  N为种类的总数
            correct += (predicted == labels).sum().item()
        print(f"Accurary on test set:{100 * correct / total}%")


if __name__ == "__main__":
    for epoch in range(15):
        train(epoch)
        test()