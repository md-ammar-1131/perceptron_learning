"""Gradient checking: compare backprop gradients with numerical gradients.

Numerical gradient (centered difference):
    dL/dw ~= (L(w + eps) - L(w - eps)) / (2 * eps)

If backpropagation is correct, the relative error should be tiny (about 1e-7 or less).
Run:  python gradient_check.py
"""
import numpy as np

from nn import MLP
from nn.losses import one_hot


def numerical_gradient(model, X, y_onehot, param, eps=1e-5):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]
        param[idx] = original + eps
        loss_plus = -np.mean(np.sum(y_onehot * np.log(model.forward(X) + 1e-12), axis=1))
        param[idx] = original - eps
        loss_minus = -np.mean(np.sum(y_onehot * np.log(model.forward(X) + 1e-12), axis=1))
        param[idx] = original  # restore
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-12)


def main():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 5))
    y = rng.integers(0, 3, size=8)
    y1h = one_hot(y, 3)

    for activation in ["sigmoid", "tanh", "relu"]:
        model = MLP([5, 6, 4, 3], activation=activation, seed=2)
        # Biases start at 0, which can put a ReLU unit exactly on its kink (z = 0), where the
        # numerical gradient is not defined. Use small random biases to avoid that.
        for i in range(model.n_layers):
            model.b[i] = rng.normal(0, 0.1, size=model.b[i].shape)
        model.forward(X)
        dW, db = model.backward(y1h)

        worst = 0.0
        for i in range(model.n_layers):
            worst = max(worst,
                        relative_error(dW[i], numerical_gradient(model, X, y1h, model.W[i])),
                        relative_error(db[i], numerical_gradient(model, X, y1h, model.b[i])))
        status = "PASS" if worst < 1e-6 else "FAIL"
        print(f"{activation:8s} worst relative error = {worst:.2e}  [{status}]")


if __name__ == "__main__":
    main()
