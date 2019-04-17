import math
import numpy as np


def linear(x, y):
    k = np.dot(x, y.T)
    return k


def linear_normalized(x, y):
    k = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            k[i, j] = linear(x[i], y[j]) / math.sqrt(linear(x[i], x[i]) * linear(y[j], y[j]))

    return k
