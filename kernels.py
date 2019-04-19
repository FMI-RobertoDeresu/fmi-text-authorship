import math
import numpy as np


def linear(x, y):
    k = np.dot(x, y.T)
    return k


def linear_normalized(x, y):
    k = np.zeros((x.shape[0], y.shape[0]), dtype=np.float64)
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            k[i, j] = linear(x[i], y[j]) / math.sqrt(linear(x[i], x[i]) * linear(y[j], y[j]))

    return k


def binary(x, y):
    x_bin = np.where(np.array(x) > 0, 1, 0)
    y_bin = np.where(np.array(y) > 0, 1, 0)
    k = np.dot(x_bin, y_bin.T)
    return k
