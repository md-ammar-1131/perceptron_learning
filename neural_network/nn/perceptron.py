"""Rosenblatt's perceptron: a single neuron with a step activation.

Learning rule (applied one example at a time):
    prediction = 1 if w.x + b >= 0 else 0
    w <- w + lr * (target - prediction) * x
    b <- b + lr * (target - prediction)

It converges only if the data is linearly separable (perceptron convergence theorem).
"""
import numpy as np


class Perceptron:
    def __init__(self, n_inputs, lr=0.1):
        self.w = np.zeros(n_inputs)
        self.b = 0.0
        self.lr = lr

    def predict(self, X):
        return (X @ self.w + self.b >= 0).astype(int)

    def fit(self, X, y, epochs=20):
        """Train and return the number of misclassified examples in each epoch."""
        mistakes_per_epoch = []
        for _ in range(epochs):
            mistakes = 0
            for xi, target in zip(X, y):
                pred = int(xi @ self.w + self.b >= 0)
                error = target - pred
                if error != 0:
                    self.w += self.lr * error * xi
                    self.b += self.lr * error
                    mistakes += 1
            mistakes_per_epoch.append(mistakes)
            if mistakes == 0:  # converged
                break
        return mistakes_per_epoch
