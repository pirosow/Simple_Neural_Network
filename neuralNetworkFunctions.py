import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidDerivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(x, 0)


def reluDerivative(x):
    return np.where(x > 0, 1, 0)


def squaredError(y_pred, y):
    return (y_pred - y) ** 2


def squaredErrorDerivative(y_pred, y):
    return 2 * (y_pred - y)