import svgwrite, numpy as np, cv2, base64

def images2svg(svg_filename: str, images: np.ndarray, pos: np.ndarray, *, width_pixels: int = 1000, imsize: float = 0.02):
    height_pixels = round(width_pixels * (pos[:, 1].max() - pos[:, 1].min()) / (pos[:, 0].max() - pos[:, 0].min()))
    dwg = svgwrite.Drawing(svg_filename, size=(width_pixels, height_pixels))
    dwg.viewbox(0, 0, width_pixels, height_pixels)
    dwg.add(dwg.rect(size=('100%', '100%'), fill="lightgrey", class_='background'))
    for i in range(pos.shape[1]):
        pos[:, i] += pos[:, i].min()
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

if __name__ == '__main__':

    from dataset import get_data
    X_train, X_test, Y_train, Y_test = get_data('dataset')

    X_train_pos = np.random.random((X_train.shape[0], 2))

    images2svg("example_image.svg", np.squeeze(X_train * 255), X_train_pos).save(pretty=True)