import numpy as np


def im2col(image, kernel_size, stride):  # X.shape=(N,C,H,W)

    # intuitive ways, many for loops -> slow
    col_image = []
    for b in range(image.shape[0]):
        for i in range(0, image.shape[2] - kernel_size + 1, stride):
            for j in range(0, image.shape[3] - kernel_size + 1, stride):
                col = image[b, :, i:i + kernel_size, j:j + kernel_size].reshape([-1])
                col_image.append(col)
    col_image = np.array(col_image)
    return col_image

#X = np.random.randn(1,1,10,10)
#print(im2col(X,3,1))