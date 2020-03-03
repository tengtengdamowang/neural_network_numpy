import numpy as np


class SGDOptimizer:
    def __init__(self, lr=0.1, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = {}

    def update(self, param):
        if param not in self.v:
            self.v[param] = np.zeros_like(param.tensor)
        self.v[param] = self.momentum * self.v[param] + (1 - self.momentum) * param.gradient
        param.tensor -= self.lr * self.v[param]
        param.gradient.fill(0)


class RmsPropOptimizer:
    def __init__(self, lr=0.1, beta=0.9, epsilon=1e-8):
        self.lr = lr
        self.beta = beta
        self.s = {}
        self.epsilon = epsilon

    def update(self, param):
        if param not in self.s:
            self.s[param] = np.zeros_like(param.tensor)
        self.s[param] = self.beta * self.s[param] + (1 - self.beta) * np.square(param.gradient)
        param.tensor -= self.lr * param.gradient / np.sqrt(self.s[param] + self.epsilon)
        param.gradient.fill(0)


class AdaGradOptimizer:
    def __init__(self, lr=0.1, epsilon=1e-10):
        self.lr = lr
        self.grad_sum = {}
        self.epsilon = epsilon

    def update(self, param):
        if param not in self.grad_sum:
            self.grad_sum[param] = np.zeros_like(param.tensor)
        self.grad_sum[param] += np.square(param.gradient)
        param.tensor -= self.lr * param.gradient / np.sqrt(self.grad_sum[param] + self.epsilon)
        param.gradient.fill(0)


class AdamOptimizer:
    def __init__(self, lr=0.1, momentum=0.9, beta=0.999, epsilon=1e-8):
        self.lr = lr
        self.momentum = momentum
        self.beta = beta
        self.s = {}
        self.v = {}
        self.epsilon = epsilon
        self.iter_counter = 1

    def update(self, param):
        if param not in self.s:
            self.s[param] = np.zeros_like(param.tensor)
            self.v[param] = np.zeros_like(param.tensor)
        self.s[param] = self.beta * self.s[param] + (1 - self.beta) * np.square(param.gradient)
        self.v[param] = self.momentum * self.v[param] + (1 - self.momentum) * param.gradient
        self.s[param] = self.s[param] / (1 - np.power(self.beta, self.iter_counter))
        self.v[param] = self.v[param] / (1 - np.power(self.momentum, self.iter_counter))
        param.tensor -= self.lr * self.v[param] / (np.sqrt(self.s[param]) + self.epsilon)
        param.gradient.fill(0)
        self.iter_counter += 1
