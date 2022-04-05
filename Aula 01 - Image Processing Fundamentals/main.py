import numpy as np
import imageio
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load image from disk
    filename1 = str(input())
    filename2 = str(input())
    img1 = imageio.imread(filename1)
    img2 = imageio.imread(filename2)

    print(img1)
    print(type(img1))
    print(img1.shape)

    plt.imshow(img1, cmap="gray")

    assert img1.shape == img2.shape

    img_sub = np.zeros(img1.shape, dtype=float)

    for x in range(img1.shape[0]):
        for y in range(img1.shape[1]):
            img_sub[x, y] = float(img1[x, y]) - float(img2[x, y])

    plt.imshow(img_sub, cmap="gray")
    plt.colorbar()

    # Normalize difference matrix
    imax = np.max(img_sub)
    imin = np.min(img_sub)

    img_sub_norm = (img_sub - imin) / (imax - imin)
    plt.imshow(img_sub_norm, cmap="gray")
    img_sub_norm = (img_sub_norm * 255).astype(np.uint8)

    imageio.imwrite("diff.jpg", img_sub_norm)

