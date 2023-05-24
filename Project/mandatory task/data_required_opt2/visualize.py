import svgwrite, numpy as np, cv2, base64, torch

def images2svg(svg_filename: str, images: np.ndarray, pos: np.ndarray, *, width_pixels: int = 1600, imsize: float = 0.02):
    height_pixels = round(width_pixels * (pos[:, 1].max() - pos[:, 1].min()) / (pos[:, 0].max() - pos[:, 0].min()))
    dwg = svgwrite.Drawing(svg_filename, size=(width_pixels, height_pixels))
    dwg.viewbox(0, 0, width_pixels, height_pixels)
    dwg.add(dwg.rect(size=('100%', '100%'), fill="lightgrey", class_='background'))
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].min()
        pos[:, i] /= pos[:, i].max()
    for image, (x, y) in zip(images, pos):
        dwg.add(
            dwg.image(
                'data:image/png;base64,' + base64.b64encode(cv2.imencode('.png', np.uint8(image))[1]).decode('utf-8'),
                insert = (f"{x * (1 - imsize) * 100}%", f"{y * (1 - imsize) * 100}%"),
                size = (f"{imsize * 100}%", ) * 2
            )
        )
    return dwg

def pca(X, no_dims: int = 2):
    d = X.shape[1]
    X -= torch.mean(X, 0)
    l_c, M_c = torch.linalg.eig(torch.mm(X.t(), X))
    i = 0
    while i < d:
        if l_c.imag[i] != 0:
            M_c[:, i+1] = M_c[:, i]
            i += 1
        i += 1
    return torch.mm(X, M_c[:, 0: no_dims].real)

if __name__ == '__main__':

    from dataset import get_data
    from cnns import LeNet, MyNet
    X_train, X_test, Y_train, Y_test = get_data('dataset')
    lenet: LeNet = torch.load('lenet.pth')
    mynet: MyNet = torch.load('mynet.pth')

    X_in = torch.from_numpy(X_test).to(next(lenet.parameters()).device)
    images = np.squeeze(X_test * 255)

    with torch.no_grad():
        conv1_out = lenet.pool(torch.nn.functional.relu(lenet.conv1(X_in)))
        fc1 = torch.nn.functional.relu(lenet.fc1(torch.flatten(lenet.pool(torch.nn.functional.relu(lenet.conv2(conv1_out))), 1)))
        final = lenet.fc3(torch.nn.functional.relu(lenet.fc2(fc1)))

    images2svg("./assets/pca_lenet_conv.svg", images, pca(torch.flatten(conv1_out, 1)).cpu().numpy()).save(pretty=True)
    images2svg("./assets/pca_lenet_fc.svg", images, pca(fc1).cpu().numpy()).save(pretty=True)
    images2svg("./assets/pca_lenet_final.svg", images, pca(final).cpu().numpy()).save(pretty=True)