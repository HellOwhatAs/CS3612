import torch, torchvision
import numpy as np, cv2
import matplotlib.pyplot as plt
from PIL import Image

class GradCAM:
    def __init__(self, model: torch.nn.Module, hook_layer: torch.nn.Module):
        self.model = model
        self.gradient: torch.Tensor
        self.activation: torch.Tensor
        hook_layer.register_forward_hook(
            lambda model, input, output: (
                output.register_hook(lambda x: (setattr(self, 'gradient', x), None)[1]),
                setattr(self, 'activation', output.detach()), None
            )[2]
        )

    def __call__(self, input_tensor: torch.Tensor, label: int = None):
        output = self.model(input_tensor.unsqueeze(0))[0]
        label = label if label else torch.argmax(output)
        output[label].backward()

        activation = self.activation.squeeze(0)
        weighted_activation = torch.zeros(activation.shape)
        for idx, (weight, activation) in enumerate(zip(torch.mean(self.gradient, [0, 2, 3]), activation)):
            weighted_activation[idx] = weight * activation

        heatmap = torch.maximum(torch.mean(weighted_activation, 0).detach().cpu(), torch.tensor(0))
        heatmap /= torch.max(heatmap) + 1e-10
        return heatmap.numpy()
    
def apply_heatmap(img: np.ndarray, heatmap: np.ndarray, *, heatmap_ratio: float = 0.6):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, img.shape[:2])), cv2.COLORMAP_JET)
    return cv2.cvtColor(np.uint8(heatmap * heatmap_ratio + img * (1 - heatmap_ratio)), cv2.COLOR_BGR2RGB)
    

if __name__ == '__main__':
    model = torchvision.models.resnet34()
    model.load_state_dict(torch.load('./resnet34-b627a593.pth'))
    model.eval()
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(240),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open('./cat_dog.png')
    input_tensor = transform(image)
    tiger_cat_label = 282
    boxer_label = 242

    gradcam = GradCAM(model, model.layer4[-1].conv2)

    plt.figure()
    plt.imshow(apply_heatmap(np.array(image), gradcam(input_tensor, boxer_label)))
    plt.figure()
    plt.imshow(apply_heatmap(np.array(image), gradcam(input_tensor, tiger_cat_label)))
    plt.show()