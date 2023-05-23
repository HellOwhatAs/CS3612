import os
import numpy as np
from dataset import get_data, normalize

import torch
import torch.utils.data
from cnns import callback, LeNet, MyNet, train_and_test

from matplotlib import pyplot as plt



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