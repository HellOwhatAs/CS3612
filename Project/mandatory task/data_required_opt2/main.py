import os
import numpy as np
from dataset import get_data, normalize

from tqdm import tqdm
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from matplotlib import pyplot as plt

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 9, padding=4)      # 1*32*32 -> 6*32*32
        self.pool = nn.MaxPool2d(2, 2)                  # 6*32*32 -> 6*16*16
        self.conv2 = nn.Conv2d(6, 16, 7, padding=3)     # 6*16*16 -> 16*16*16
                                                        # 16*16*16 -> 16*8*8
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)    # 16*8*8 -> 32*8*8
                                                        # 32*8*8 -> 32*4*4
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train_and_test(netclass: type, *,epochs: int = 200):
    net = netclass().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    with tqdm(range(epochs), netclass.__name__) as tbar:
        for epoch in tbar:
            running_loss = 0.0
            for inputs, labels in trainloader:

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            tbar.set_postfix(loss = running_loss / len(trainloader))
            tbar.update()
            running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return net, correct / total

if __name__ == '__main__':
    ######################## Get train/test dataset ########################
    X_train, X_test, Y_train, Y_test = get_data('dataset')
    ########################################################################
    # 以上加载的数据为 numpy array格式
    # 如果希望使用pytorch或tensorflow等库，需要使用相应的函数将numpy arrray转化为tensor格式
    # 以pytorch为例：
    #   使用torch.from_numpy()函数可以将numpy array转化为pytorch tensor
    #
    # Hint:可以考虑使用torch.utils.data中的class来简化划分mini-batch的操作，比如以下class：
    #   https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset
    #   https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    ########################################################################

    ########################################################################
    ######################## Implement you code here #######################
    ########################################################################

    batch_size = 32
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainloader = torch.utils.data.DataLoader(list(zip(torch.from_numpy(X_train).to(device), torch.from_numpy(Y_train).to(device))), batch_size=batch_size,
                                            shuffle=True)

    testloader = torch.utils.data.DataLoader(list(zip(torch.from_numpy(X_test).to(device), torch.from_numpy(Y_test).to(device))), batch_size=batch_size,
                                            shuffle=False)
    
    print(train_and_test(LeNet)[1], train_and_test(MyNet)[1])