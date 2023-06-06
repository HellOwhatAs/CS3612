<center><h1>Final Project</h1></center>
<div align=right>520030910246 薛家奇</div>

## 1. Mandatory Task
### 1.1 Implement LeNet
```py
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
```
![](mandatory%20task/assets/lenet.svg)

### 1.2 Design and Implement My Neural Network
```py
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
```
![](mandatory%20task/assets/mynet.svg)

### 1.3 Training and Testing
Using `torch.optim.SGD` with `lr = 0.001` and `momentum = 0.9` as optimizer, the training and testing loss of two net are:

|LeNet|MyNet|
|-|-|
![](./mandatory%20task/assets/lenet_loss.svg) | ![](./mandatory%20task/assets/mynet_loss.svg)

And the training and testing accuracy of two net are:

|LeNet|MyNet|
|-|-|
|![](./mandatory%20task/assets/lenet_acc.svg) | ![](./mandatory%20task/assets/mynet_acc.svg)|
</center>

### 1.4 Result Visualization
#### 1.4.1 PCA
The result of LeNet:

|Scatter|Picture|
|-|-|
|![](./mandatory%20task/assets/_pca_lenet_conv.svg) | ![](./mandatory%20task/assets/pca_lenet_conv.svg)|
|![](./mandatory%20task/assets/_pca_lenet_fc.svg) | ![](./mandatory%20task/assets/pca_lenet_fc.svg)|
|![](./mandatory%20task/assets/_pca_lenet_final.svg) | ![](./mandatory%20task/assets/pca_lenet_final.svg)|
</center>

The result of MyNet:
|Scatter|Picture|
|-|-|
|![](./mandatory%20task/assets/_pca_mynet_conv.svg) | ![](./mandatory%20task/assets/pca_mynet_conv.svg)|
|![](./mandatory%20task/assets/_pca_mynet_fc.svg) | ![](./mandatory%20task/assets/pca_mynet_fc.svg)|
|![](./mandatory%20task/assets/_pca_mynet_final.svg) | ![](./mandatory%20task/assets/pca_mynet_final.svg)|



#### 1.4.2 t-SNE
The result of LeNet:
|Scatter|Picture|
|-|-|
|![](./mandatory%20task/assets/_tsne_lenet_conv.svg) | ![](./mandatory%20task/assets/tsne_lenet_conv.svg)|
|![](./mandatory%20task/assets/_tsne_lenet_fc.svg) | ![](./mandatory%20task/assets/tsne_lenet_fc.svg)|
|![](./mandatory%20task/assets/_tsne_lenet_final.svg) | ![](./mandatory%20task/assets/tsne_lenet_final.svg)|

The result of MyNet:
|Scatter|Picture|
|-|-|
|![](./mandatory%20task/assets/_tsne_mynet_conv.svg) | ![](./mandatory%20task/assets/tsne_mynet_conv.svg)|
|![](./mandatory%20task/assets/_tsne_mynet_fc.svg) | ![](./mandatory%20task/assets/tsne_mynet_fc.svg)|
|![](./mandatory%20task/assets/_tsne_mynet_final.svg) | ![](./mandatory%20task/assets/tsne_mynet_final.svg)|


## 2. Optional Task 2
### 2.1 Grad-CAM
|Original Image|
|-|
|![](./optional%20task2/assets/cat_dog.png)|


||Boxer|Tiger Cat|
|:-:|-|-|
|**resnet34**|![](./optional%20task2/assets/gradcam_resnet34_boxer.png) | ![](./optional%20task2/assets/gradcam_resnet34_cat.png)|
|**vgg19**|![](./optional%20task2/assets/gradcam_vgg19_boxer.png) | ![](./optional%20task2/assets/gradcam_vgg19_cat.png)|

### 2.2 Integrated Gradients
||Viaduct|Fireboat|
|:-:|-|-|
|**Original Image**|![](./optional%20task2/assets/viaduct.jpg) | ![](./optional%20task2/assets/fireboat.jpg)|
|**resnet34**|![](./optional%20task2/assets/integrated_gradients_resnet34_viaduct.png) | ![](./optional%20task2/assets/integrated_gradients_resnet34_fireboat.png)|
|**vgg19**|![](./optional%20task2/assets/integrated_gradients_vgg19_viaduct.png) | ![](./optional%20task2/assets/integrated_gradients_vgg19_fireboat.png)|
