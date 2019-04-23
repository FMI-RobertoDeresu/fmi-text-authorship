import numpy as np


def linear(x, y):
    k = np.dot(x, y.T)
    return k


def binary(x, y):
    x_bin = np.where(x > 0, 1, 0)
    y_bin = np.where(y > 0, 1, 0)
    k = np.dot(x_bin, y_bin.T)
    return k


def intersection(x, y):
    k = np.zeros((x.shape[0], y.shape[0]), dtype=np.float64)
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            x_mask = np.where(x[i] <= y[j], 1, 0)
            y_mask = np.logical_not(x_mask).astype(np.int)
            k[i, j] = np.sum(x[i] * x_mask + y[j] * y_mask)

    return k
