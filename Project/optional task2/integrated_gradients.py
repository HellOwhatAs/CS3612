import torch
from PIL import Image
from tqdm import tqdm
import torchvision
import cv2, numpy as np

class IntegratedGradients:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.model.eval()

    def attribute(self, input_tensor: torch.Tensor, baselines: torch.Tensor = None, target: int = None, n_steps: int = 300):
        if baselines is None: baselines = torch.zeros_like(input_tensor).to(input_tensor.device)
        if target is None: target = torch.argmax(self.model(input_tensor), 1).item()
        mean_grad = torch.zeros_like(input_tensor).to(input_tensor.device)
        for i in tqdm(range(n_steps), f'IntegratedGradients(target = {target})'):
            x = baselines + (i + 1) / n_steps * (input_tensor - baselines)
            x.requires_grad_()
            output = self.model(x)[0, target]
            grad = torch.autograd.grad(output, x)[0]
            mean_grad += grad / n_steps
        return (input_tensor - baselines) * mean_grad


if __name__ == "__main__":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = torchvision.models.resnet34().to(device)
    model.load_state_dict(torch.load('./resnet34-b627a593.pth'))
    model.eval()
    model2 = torchvision.models.vgg19(weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1).to(device)
    model2.eval()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(240),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open('./assets/fireboat.jpg')
    image2 = Image.open('./assets/viaduct.jpg')
    image_np, image2_np = np.array(image), np.array(image2)
    input_tensor = transform(image).to(device)
    input_tensor2 = transform(image2).to(device)

    ig = IntegratedGradients(model)

    output = ig.attribute(input_tensor.unsqueeze(0))
    output = output.squeeze(0).abs()
    output /= output.max()
    mask = np.sum(output.permute(1, 2, 0).cpu().numpy(), 2)[:,:,np.newaxis] + 0.1
    cv2.imwrite('./assets/integrated_gradients_resnet34_fireboat.png', np.uint8(mask * image_np))

    output = ig.attribute(input_tensor2.unsqueeze(0))
    output = output.squeeze(0).abs()
    output /= output.max()
    mask = np.sum(output.permute(1, 2, 0).cpu().numpy(), 2)[:,:,np.newaxis] + 0.1
    cv2.imwrite('./assets/integrated_gradients_resnet34_viaduct.png', np.uint8(mask * image2_np))

    ig2 = IntegratedGradients(model2)

    output = ig2.attribute(input_tensor.unsqueeze(0))
    output = output.squeeze(0).abs()
    output /= output.max()
    mask = np.sum(output.permute(1, 2, 0).cpu().numpy(), 2)[:,:,np.newaxis] + 0.1
    cv2.imwrite('./assets/integrated_gradients_vgg19_fireboat.png', np.uint8(mask * image_np))

    output = ig2.attribute(input_tensor2.unsqueeze(0))
    output = output.squeeze(0).abs()
    output /= output.max()
    mask = np.sum(output.permute(1, 2, 0).cpu().numpy(), 2)[:,:,np.newaxis] + 0.1
    cv2.imwrite('./assets/integrated_gradients_vgg19_viaduct.png', np.uint8(mask * image2_np))