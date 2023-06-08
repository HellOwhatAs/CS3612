import svgwrite, numpy as np, cv2, base64, torch
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def images2svg(svg_filename: str, images: np.ndarray, pos: np.ndarray, *, width_pixels: int = 1600, imsize: float = 0.02):
    height_pixels = round(width_pixels * (pos[:, 1].max() - pos[:, 1].min()) / (pos[:, 0].max() - pos[:, 0].min()))
    dwg = svgwrite.Drawing(svg_filename, size=(width_pixels, height_pixels))
    dwg.viewbox(0, 0, width_pixels, height_pixels)
    # dwg.add(dwg.rect(size=('100%', '100%'), fill="lightgrey", class_='background'))
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].min()
        pos[:, i] /= pos[:, i].max()
    for image, (x, y) in zip(images, pos):
        dwg.add(
            dwg.image(
                'data:image/png;base64,' + base64.b64encode(cv2.imencode('.png', np.uint8(image))[1]).decode('utf-8'),
                insert = (f"{x * (1 - imsize) * width_pixels}", f"{(1 - y) * (1 - imsize) * height_pixels}"),
                size = (f"{imsize * min(width_pixels, height_pixels)}", ) * 2
            )
        )
    return dwg

def pca(X: torch.Tensor, num_dims: int = 2):
    d = X.shape[1]
    X -= torch.mean(X, 0)
    l_c, M_c = torch.linalg.eig(torch.mm(X.t(), X))
    i = 0
    while i < d:
        if l_c.imag[i] != 0:
            M_c[:, i+1] = M_c[:, i]
            i += 1
        i += 1
    return torch.mm(X, M_c[:, 0: num_dims].real)

def calc_prob(X: torch.Tensor, eps: float = 1e-5, perplexity: float = 30.0, max_iter: int = 50):
    device = X.device
    n = X.shape[0]
    ret, beta = torch.zeros(n, n).to(device), torch.ones(n, 1).to(device)
    log_perplexity = (torch.ones(1) * torch.log(torch.tensor(perplexity))).to(device)
    data = (-2 * torch.mm(X, X.t()) + (X ** 2).sum(1)).t() + (X ** 2).sum(1)
    for i in range(n):
        datai = data[i, [j for j in range(n) if j != i]]
        P = torch.exp(-datai * beta[i])
        P_sum = P.sum()
        H = torch.log(P_sum) + beta[i] * torch.sum(datai * P) / P_sum
        P /= P_sum
        min_, max_ = None, None
        diff_H = H - log_perplexity
        for _ in range(max_iter):
            if torch.abs(diff_H) <= eps: break
            if diff_H > 0:
                min_ = beta[i].clone()
                beta[i] = beta[i] * 2 if max_ is None else (beta[i] + max_) / 2
            else:
                max_ = beta[i].clone()
                beta[i] = beta[i] / 2 if min_ is None else (beta[i] + min_) / 2
            P = torch.exp(-datai * beta[i])
            P_sum = P.sum()
            H = torch.log(P_sum) + beta[i] * torch.sum(datai * P) / P_sum
            P /= P_sum
            diff_H = H - log_perplexity
        ret[i, [j for j in range(n) if j != i]] = P
    return ret

def tsne(X: torch.Tensor, num_dims: int = 2, *, max_iter: int = 1000, perplexity: float = 20.0,
        eps: float = 1e-5, gain_min: float = 0.01, float_min: float = 1e-12):
    device = X.device
    n = X.shape[0]
    ret = torch.randn(n, num_dims).to(device)
    dY, iY = torch.zeros(n, num_dims).to(device), torch.zeros(n, num_dims).to(device)
    gains = torch.ones(n, num_dims).to(device)
    prob = calc_prob(X, eps, perplexity)
    prob += prob.clone().t()
    prob = torch.max(prob * 4 / prob.sum(), torch.tensor(float_min))
    with tqdm(range(max_iter), 't-SNE') as tbar:
        for iter in tbar:
            num = 1 / (1 + (-2 * torch.mm(ret, ret.t()) + (ret ** 2).sum(1)).t() + (ret ** 2).sum(1))
            num.diagonal()[:] = 0
            n_num = num / torch.sum(num)
            n_num = torch.max(n_num, torch.tensor(float_min))
            diff = prob - n_num
            if iter == 100: prob /= 4
            for i in range(n):
                diff_num_product = (diff[:, i] * num[:, i]).repeat(num_dims, 1).t()
                diff_y_product = diff_num_product * (ret[i, :] - ret)
                sum_diff_y_product = torch.sum(diff_y_product, 0)
                dY[i, :] = sum_diff_y_product
            mask = (dY > 0) != (iY > 0)
            gains[mask] += 0.2
            gains[~mask] *= 0.8
            gains = torch.max(gains, torch.tensor(gain_min))
            iY = (0.5 + 0.3 * (iter < 20)) * iY - 500 * (gains * dY)
            ret += iY - torch.mean(ret + iY, 0)
            tbar.set_postfix(error = torch.sum(prob * torch.log(prob / n_num)).item())
            tbar.update()
    return ret

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
    conv1_out = torch.flatten(conv1_out, 1)

    pos = pca(conv1_out).cpu().numpy()
    images2svg("./assets/pca_lenet_conv.svg", images, pos).save(pretty=True)
    plt.scatter(pos[:, 0], pos[:, 1], c = Y_test)
    plt.savefig("./assets/_pca_lenet_conv.svg")
    plt.cla()

    pos = pca(fc1).cpu().numpy()
    images2svg("./assets/pca_lenet_fc.svg", images, pos).save(pretty=True)
    plt.scatter(pos[:, 0], pos[:, 1], c = Y_test)
    plt.savefig("./assets/_pca_lenet_fc.svg")
    plt.cla()

    pos = pca(final).cpu().numpy()
    images2svg("./assets/pca_lenet_final.svg", images, pos).save(pretty=True)
    plt.scatter(pos[:, 0], pos[:, 1], c = Y_test)
    plt.savefig("./assets/_pca_lenet_final.svg")
    plt.cla()

    pos = tsne(conv1_out).cpu().numpy()
    images2svg("./assets/tsne_lenet_conv.svg", images, pos).save(pretty=True)
    plt.scatter(pos[:, 0], pos[:, 1], c = Y_test)
    plt.savefig("./assets/_tsne_lenet_conv.svg")
    plt.cla()

    pos = tsne(fc1).cpu().numpy()
    images2svg("./assets/tsne_lenet_fc.svg", images, pos).save(pretty=True)
    plt.scatter(pos[:, 0], pos[:, 1], c = Y_test)
    plt.savefig("./assets/_tsne_lenet_fc.svg")
    plt.cla()

    pos = tsne(final).cpu().numpy()
    images2svg("./assets/tsne_lenet_final.svg", images, pos).save(pretty=True)
    plt.scatter(pos[:, 0], pos[:, 1], c = Y_test)
    plt.savefig("./assets/_tsne_lenet_final.svg")
    plt.cla()

    with torch.no_grad():
        conv1_out = mynet.pool(torch.nn.functional.relu(mynet.conv1(X_in)))
        conv2_out = mynet.pool(torch.nn.functional.relu(mynet.conv2(conv1_out)))
        conv3_out = mynet.pool(torch.nn.functional.relu(mynet.conv3(conv2_out)))
        fc1 = torch.nn.functional.relu(mynet.fc1(torch.flatten(conv3_out, 1)))
        fc2 = torch.nn.functional.relu(mynet.fc2(fc1))
        final = mynet.fc3(fc2)
    conv3_out = torch.flatten(conv3_out, 1)

    pos = pca(conv3_out).cpu().numpy()
    images2svg("./assets/pca_mynet_conv.svg", images, pos).save(pretty=True)
    plt.scatter(pos[:, 0], pos[:, 1], c = Y_test)
    plt.savefig("./assets/_pca_mynet_conv.svg")
    plt.cla()

    pos = pca(fc2).cpu().numpy()
    images2svg("./assets/pca_mynet_fc.svg", images, pos).save(pretty=True)
    plt.scatter(pos[:, 0], pos[:, 1], c = Y_test)
    plt.savefig("./assets/_pca_mynet_fc.svg")
    plt.cla()

    pos = pca(final).cpu().numpy()
    images2svg("./assets/pca_mynet_final.svg", images, pos).save(pretty=True)
    plt.scatter(pos[:, 0], pos[:, 1], c = Y_test)
    plt.savefig("./assets/_pca_mynet_final.svg")
    plt.cla()

    pos = tsne(conv3_out).cpu().numpy()
    images2svg("./assets/tsne_mynet_conv.svg", images, pos).save(pretty=True)
    plt.scatter(pos[:, 0], pos[:, 1], c = Y_test)
    plt.savefig("./assets/_tsne_mynet_conv.svg")
    plt.cla()

    pos = tsne(fc2).cpu().numpy()
    images2svg("./assets/tsne_mynet_fc.svg", images, pos).save(pretty=True)
    plt.scatter(pos[:, 0], pos[:, 1], c = Y_test)
    plt.savefig("./assets/_tsne_mynet_fc.svg")
    plt.cla()

    pos = tsne(final).cpu().numpy()
    images2svg("./assets/tsne_mynet_final.svg", images, pos).save(pretty=True)
    plt.scatter(pos[:, 0], pos[:, 1], c = Y_test)
    plt.savefig("./assets/_tsne_mynet_final.svg")
    plt.cla()