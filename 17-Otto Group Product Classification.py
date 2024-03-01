import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
class OTTO(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=torch.nn.Linear(93,64)
        self.l2=torch.nn.Linear(64,36)
        self.l3=torch.nn.Linear(36,18)
        self.l4=torch.nn.Linear(18,9)

    def forward(self,x):
        x=F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.l4(x)

class OttoDatasetTrain(Dataset):
    def __init__(self,filepath):
        super().__init__()
        # 在relu中，x必须是float
        xy=np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        self.len=xy.shape[0]
        self.x_data=torch.from_numpy(xy[:,1:-1])
        # 在交叉熵中,targets必须是Long
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.int64)
        self.y_data=torch.from_numpy(xy[:,-1])


    def __len__(self):
        return self.len
    def __getitem__(self, item):
        return self.x_data[item],self.y_data[item]


class OttoDatasetTest(Dataset):
    def __init__(self, filepath):
        super().__init__()
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 1:])

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.x_data[item]

model=OTTO()
train_dataset=OttoDatasetTrain('./dataset/otto/train.csv')
train_loader=DataLoader(dataset=train_dataset,
                        batch_size=32,
                        shuffle=True,
                        num_workers=4)

test_dataset=OttoDatasetTest('./dataset/otto/test.csv')
test_loader=DataLoader(dataset=test_dataset,
                        batch_size=32,
                        shuffle=False,
                        num_workers=4)
criterion=torch.nn.CrossEntropyLoss()
optim=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    running_loss=0
    for i,data in enumerate(train_loader):
        inputs,check=data
        y_pred=model(inputs)
        optim.zero_grad()
        loss=criterion(y_pred,check)
        running_loss+=loss.item()
        loss.backward()
        optim.step()
        if i%300 ==299:
            print(f'[{epoch},{i+1}] \tloss:{running_loss/300}')
            running_loss=0.0
def test():
    output_list=[]
    class_list=[]
    with torch.no_grad():  # 使得with中的代码不进行梯度的计算
        for data in test_loader:
            outputs = model(data)
            # max 返回两个值(value 和 index) 因为不需要value 所以用 _
            _, predicted = torch.max(outputs.data, dim=1)
            output_list.append(predicted.tolist())
        for item in output_list:
            for i in item:
                class_list.append(i)  # 从0开始
        cl=[]
        for index in class_list:
            k=[0,0,0,0,0,0,0,0,0]
            k[index]=1
            cl.append(k)

        cl=np.array(cl)
        output = pd.DataFrame({'id':range(1,144369),
                               'Class_1':cl[:,0].tolist(),
                               'Class_2':cl[:,1].tolist(),
                               'Class_3':cl[:,2].tolist(),
                               'Class_4':cl[:,3].tolist(),
                                'Class_5':cl[:,4].tolist(),
                                'Class_6':cl[:,5].tolist(),
                                'Class_7':cl[:,6].tolist(),
                                'Class_8':cl[:,7].tolist(),
                                'Class_9':cl[:,8].tolist()},dtype=np.int64)
        output.to_csv('./dataset/otto/Submission.csv', index=False)
if __name__ =="__main__":
    for epoch in range(7):
        train(epoch)
    test()

