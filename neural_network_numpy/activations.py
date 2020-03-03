import numpy as np
from layers import Layer


class ReLu(Layer):
    def forward(self, X):
        mask = X > 0
        return X * mask, lambda D: D * mask


class Sigmoid(Layer):
    def forward(self, X):
        S = 1 / (1 + np.exp(-X))

        def backward(D):
            return D * S * (1 - S)

        return S, backward

