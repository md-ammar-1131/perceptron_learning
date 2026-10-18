"""Activation functions and their derivatives.

Each *_grad function takes the ACTIVATION OUTPUT `a` (not the pre-activation z).
This is convenient because during backpropagation we have already stored `a`.
"""
import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)  # avoid overflow in exp
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_grad(a):
    # d sigmoid / dz = a * (1 - a)
    return a * (1.0 - a)


def tanh(z):
    return np.tanh(z)


def tanh_grad(a):
    # d tanh / dz = 1 - a^2
    return 1.0 - a ** 2


def relu(z):
    return np.maximum(0.0, z)


def relu_grad(a):
    # a > 0 exactly when z > 0
    return (a > 0).astype(float)


def softmax(z):
    """Row-wise softmax. Subtracting the max keeps exp() numerically stable."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (tanh, tanh_grad),
    "relu": (relu, relu_grad),
}
