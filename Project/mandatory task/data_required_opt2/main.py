import os
import numpy as np
from dataset import get_data, normalize

from tqdm import tqdm
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Callable, Union, Tuple, Any
from collections import defaultdict

from matplotlib import pyplot as plt

class callback:
    def __init__(self, func: Callable[['callback', Any], None]):
        self.func = func
        self.data = defaultdict(list)
    def __call__(self, *args):
        return self.func(self, *args)
    def __getitem__(self, *args):
        return self.data.__getitem__(*args)
    def __delitem__(self, *args):
        self.data.__delitem__(*args)
    def clear(self):
        self.data.clear()

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
    
def calc_acc(net: nn.Module, dataloader: torch.utils.data.DataLoader) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def calc_test(net: nn.Module, testloader: torch.utils.data.DataLoader, criterion: nn.CrossEntropyLoss) -> Tuple[float, float]:
    '''
    Return: test_loss, test_acc
    '''
    running_loss = 0.0

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(testloader), correct / total

def train_and_test(netclass: type, trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader, device,* ,epochs: int = 200,
                   loss_callback: Union[None, Callable[[float, float], None]] = None,
                   acc_callback: Union[None, Callable[[float, float], None]] = None):
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

            train_loss = running_loss / len(trainloader)
            test_loss, test_acc = calc_test(net, testloader, criterion)
            if loss_callback: loss_callback(train_loss, test_loss)
            if acc_callback: acc_callback(calc_acc(net, trainloader), test_acc)

            tbar.set_postfix(train_loss = train_loss, test_loss = test_loss)
            tbar.update()
            running_loss = 0.0

    return net

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
    
    le_loss_cb, le_acc_cb, my_loss_cb, my_acc_cb = [callback(lambda self, a, b: (self['train'].append(a), self['test'].append(b))) for _ in range(4)]
    lenet = train_and_test(LeNet, trainloader, testloader, device, acc_callback = le_acc_cb, loss_callback = le_loss_cb)
    mynet = train_and_test(MyNet, trainloader, testloader, device, acc_callback = my_acc_cb, loss_callback = my_loss_cb)

    plt.figure()
    plt.title('le')
    plt.plot(le_loss_cb['train'])
    plt.plot(le_loss_cb['test'])

    plt.figure()
    plt.title('le')
    plt.plot(le_acc_cb['train'])
    plt.plot(le_acc_cb['test'])

    plt.figure()
    plt.title('my')
    plt.plot(my_loss_cb['train'])
    plt.plot(my_loss_cb['test'])

    plt.figure()
    plt.title('my')
    plt.plot(my_acc_cb['train'])
    plt.plot(my_acc_cb['test'])

    plt.show()