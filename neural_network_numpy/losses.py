import numpy as np


def mse_loss(y_pred, y):
    return 0.5 * np.linalg.norm(y_pred - y) ** 2, y_pred - y


def ce_loss(y_pred, y):  # y.shape=(N,k) N是样本数 k是类别
    num = np.exp(y_pred)
    prob = num / num.sum(axis=1).reshape(-1, 1)
    eps = np.finfo(float).eps
    cross_entropy = -np.sum(y * np.log(prob + eps))
    return cross_entropy/ len(y), (prob - y) / len(y)

