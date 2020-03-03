import numpy as np
from utils import im2col


class Parameter():
    def __init__(self, tensor):
        self.tensor = tensor
        self.gradient = np.zeros_like(self.tensor)


class Layer:
    def __init__(self):
        self.parameters = []

    def forward(self, X):
        return X, lambda D: D

    def build_param(self, tensor):
        param = Parameter(tensor)
        self.parameters.append(param)
        return param

    def update(self, optimizer):
        for param in self.parameters:
            optimizer.update(param)


class Linear(Layer):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.weights = self.build_param(np.random.randn(inputs, outputs) * np.sqrt(1 / inputs))
        self.bias = self.build_param(np.zeros(outputs))

    def forward(self, X):  # X.shape=(N,k)
        def backward(D):  # D=dL/dz, z=wx+b
            self.weights.gradient += X.T @ D
            self.bias.gradient += D.sum(axis=0)
            dLdX = D @ self.weights.tensor.T
            return dLdX

        return X @ self.weights.tensor + self.bias.tensor, backward  # make everything ready


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        k = 1 / (kernel_size*kernel_size*in_channels)  # initialization, same with pytorch
        self.weights = self.build_param(np.random.uniform(-np.sqrt(k), np.sqrt(k), (out_channels, in_channels, kernel_size, kernel_size)))
        self.bias = self.build_param(np.random.uniform(-np.sqrt(k), np.sqrt(k), out_channels))

    def forward(self, X):  # X.shape=(N,C,H,W)
        col_weights = self.weights.tensor.transpose(1, 2, 3, 0).reshape((-1, self.out_channels))
        self.col_image = im2col(X, self.kernel_size, self.stride)
        conv_ret = self.col_image @ col_weights + self.bias.tensor
        conv_ret = conv_ret.reshape((X.shape[0], self.out_channels, (X.shape[2]-self.kernel_size)//self.stride+1, (X.shape[3]-self.kernel_size)//self.stride+1))

        def backward(D):
            self.weights.gradient += (self.col_image.T @ D.transpose(3, 1, 2, 0).reshape(-1,self.out_channels)).reshape(self.weights.tensor.shape)
            self.bias.gradient += D.transpose(3, 1, 2, 0).reshape(-1, self.out_channels).sum(axis=0)
            pad_D = np.pad(D, ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1),
                               (self.kernel_size - 1, self.kernel_size - 1)), 'constant', constant_values=0)
            flip_W = self.weights.tensor[::-1]
            flip_W = flip_W.swapaxes(1, 2)
            col_flip_W = flip_W.reshape((-1, self.in_channels))
            col_pad_D = im2col(pad_D, self.kernel_size, self.stride)
            dX = col_pad_D @ col_flip_W
            dX = dX.reshape(X.shape)
            return dX
        return conv_ret, backward


class MaxPooling2D(Layer):
    def __init__(self, kernel_size=2, stride=2):  # no pad
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, X):
        X_tmp = X
        N, C, H, W = X.shape
        out_H = np.floor(1 + (H - self.kernel_size) / self.stride).astype(int)
        out_W = np.floor(1 + (W - self.kernel_size) / self.stride).astype(int)
        Y = np.zeros((N, C, out_H, out_W))
        for m in range(N):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        i0, i1 = i * self.stride, (i * self.stride) + self.kernel_size
                        j0, j1 = j * self.stride, (j * self.stride) + self.kernel_size
                        Y[m, c, i, j] = np.max(X[m, c, i0:i1, j0:j1])

        def backward(D):
            dX = np.zeros_like(X_tmp)
            for m in range(N):
                for c in range(C):
                    for i in range(out_H):
                        for j in range(out_W):
                            i0, i1 = i * self.stride, (i * self.stride) + self.kernel_size
                            j0, j1 = j * self.stride, (j * self.stride) + self.kernel_size
                            xi = X[m, c, i0:i1, j0:j1]
                            mask = np.zeros_like(xi).astype(bool)
                            x, y = np.argwhere(xi == np.max(xi))[0]
                            mask[x, y] = True
                            dX[m, c, i0:i1, j0:j1] += mask * D[m, c, i, j]
            return dX

        return Y, backward


class AvgPooling2D(Layer):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, X):
        X_tmp = X
        N, C, H, W = X.shape
        out_H = np.floor(1 + (H - self.kernel_size) / self.stride).astype(int)
        out_W = np.floor(1 + (W - self.kernel_size) / self.stride).astype(int)
        Y = np.zeros((N, C, out_H, out_W))
        for m in range(N):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        i0, i1 = i * self.stride, (i * self.stride) + self.kernel_size
                        j0, j1 = j * self.stride, (j * self.stride) + self.kernel_size
                        Y[m, c, i, j] = np.mean(X[m, c, i0:i1, j0:j1])

        def backward(D):
            dX = np.zeros_like(X_tmp)
            for m in range(N):
                for c in range(C):
                    for i in range(out_H):
                        for j in range(out_W):
                            i0, i1 = i * self.stride, (i * self.stride) + self.kernel_size
                            j0, j1 = j * self.stride, (j * self.stride) + self.kernel_size
                            frame = np.ones((self.kernel_size, self.kernel_size)) * D[m, c, i, j]
                            dX[m, c, i0:i1, j0:j1] += frame / np.prod((self.kernel_size, self.kernel_size))
            return dX

        return Y, backward


class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        def backward(D):
            return D.reshape(X.shape)

        return X.reshape((X.shape[0], -1)), backward


class Sequential(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        for layer in layers:
            self.parameters.extend(layer.parameters)

    def forward(self, X):
        backprops = []
        Y = X
        for layer in self.layers:
            Y, backprop = layer.forward(Y)  # Y:output of each layer
            backprops.append(backprop)

        def backward(D):
            for backprop in reversed(backprops):
                D = backprop(D)
            return D

        return Y, backward
