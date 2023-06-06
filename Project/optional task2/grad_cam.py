import torch, torchvision
import numpy as np, cv2
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
        for idx, wa in enumerate(w * a for w, a in zip(torch.mean(self.gradient, [0, 2, 3]), activation)):
            weighted_activation[idx] = wa

        heatmap = torch.maximum(torch.mean(weighted_activation, 0).detach().cpu(), torch.tensor(0))
        heatmap /= torch.max(heatmap) + 1e-10
        return heatmap.numpy()
    
def apply_heatmap(img: np.ndarray, heatmap: np.ndarray, *, heatmap_ratio: float = 0.6):
    heatmap = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, img.shape[:2])), cv2.COLORMAP_JET)
    return np.uint8(heatmap * heatmap_ratio + img * (1 - heatmap_ratio))
    

if __name__ == '__main__':
    model = torchvision.models.resnet34()
    model.load_state_dict(torch.load('./resnet34-b627a593.pth'))
    model.eval()
    model2 = torchvision.models.vgg19(weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1)
    model2.eval()
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(240),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open('./assets/cat_dog.png')
    input_tensor = transform(image)
    boxer_label, tiger_cat_label = 242, 282 # http://befree2008.github.io/2018/10/05/20181005_ImageNet1000分类名称和编号

    gradcam = GradCAM(model, model.layer4[2].conv2)
    gradcam2 = GradCAM(model2, model2.features[34])

    cv2.imwrite('./assets/gradcam_resnet34_boxer.png', apply_heatmap(np.array(image), gradcam(input_tensor, boxer_label)))
    cv2.imwrite('./assets/gradcam_resnet34_cat.png', apply_heatmap(np.array(image), gradcam(input_tensor, tiger_cat_label)))
    cv2.imwrite('./assets/gradcam_vgg19_boxer.png', apply_heatmap(np.array(image), gradcam2(input_tensor, boxer_label)))
    cv2.imwrite('./assets/gradcam_vgg19_cat.png', apply_heatmap(np.array(image), gradcam2(input_tensor, tiger_cat_label)))